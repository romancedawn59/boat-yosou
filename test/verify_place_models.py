# -*- coding: utf-8 -*-
"""2着・3着を直接予測するモデル vs Harville導出の比較(第1段)
   (2026-07-22ケンさん指示「2着専用モデルの土台を作る」)

    py -X utf8 test/verify_place_models.py

■ 何を確かめるか
現行は「1着になる確率」だけをLightGBMで学習し、2着・3着は計算式
(Harville法+Benter割引 λ=0.70/μ=0.50)で導出している。つまり2着・3着は
誰も直接予測していない。
一方でうちの主力買い目は2着・3着の精度に依存している:
  - 3連複3〜4点(3着以内の組み合わせ)= 買い目金額の半分以上
  - 3連単④⑤(1位予想を2着に置く形)= 超混戦帯で823%の稼ぎ頭
さらに超混戦帯では1位予想が「勝てないが75%は3着以内に残る」と判明しており
(test/verify_axis_diversify.py)、この"勝ち切れずに沈む"性質こそエッジの源泉。
そこを計算式の仮定に委ねたままでよいのかを検証する。

■ 比較するもの(同じ特徴量・同じfoldで公平に)
  A. 現行: 1着モデル + Harville導出
       - Harvilleの「2着になる確率」  = Σ_a P(a-i-*)
       - Harvilleの「3着以内に入る確率」= Σ P(i-*-*)+P(*-i-*)+P(*-*-i)
  B. 直接予測: ラベルを変えて学習した専用モデル
       - place2モデル: arrival_order == 2
       - top3モデル  : arrival_order <= 3

■ 【事前登録】第1段の判定基準(満たさなければ第2段へ進まない)
  1. 直接予測モデルのAUCが、Harville導出値のAUCを上回る
  2. 較正が同等以上(予測確率と実際の率のズレが小さい)
  3. Brierスコアが改善する
※AUCだけ良くても採用しない。検証⑫で「AUCは上がったが回収率は悪化」を
  経験しているため、第2段(買い目への反映・回収率で判定)まで通って初めて採用。

■ 第2段(第1段を通過した場合のみ・別スクリプト)
  新しい確率で3連複/3連単を組み直し、最大1発除き回収率・fold安定性・DDで判定。
  ここで初めてボックスやフォーメーションを再検証する意味が生まれる
  (これまでの構成17案は全て「今の確率」を前提にした変形だった)。

■ 注意
  本番コードには一切触れない(v1凍結の原則: 新規ファイル+test/出力のみ)。
  学習は3モデル×5fold=15回でそれなりに時間がかかる。
"""
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
import predictors as P
from backtest import N_FOLDS, PARAMS, TEST_START
from config import DB_PATH
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set

# 直接予測するラベル(列名, 説明, 作り方)
LABELS = [
    ("is_winner", "1着", lambda s: (s == 1).astype(int)),
    ("is_place2", "2着", lambda s: (s == 2).astype(int)),
    ("is_top3", "3着以内", lambda s: (s <= 3).astype(int)),
]


def train_with_label(train_df: pd.DataFrame, label_col: str) -> lgb.Booster:
    """backtest.train_foldと同じ手順でラベルだけ差し替えて学習する"""
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    train_set = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr[label_col],
                            categorical_feature=CATEGORICAL_FEATURES)
    valid_set = lgb.Dataset(va[FEATURE_COLUMNS], label=va[label_col],
                            reference=train_set)
    return lgb.train(PARAMS, train_set, valid_sets=[valid_set],
                     num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def auc(y_true, y_score) -> float:
    """順位ベースのAUC(sklearn非依存)"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # 同順位は平均ランクに均す
    s = pd.Series(y_score)
    ranks = s.rank(method="average").to_numpy()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier(y_true, y_prob) -> float:
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def calibration_gap(y_true, y_prob, bins=10) -> float:
    """較正誤差(予測確率と実際の率の差の重み付き平均)。小さいほど良い"""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    total, gap = 0, 0.0
    for i in range(bins):
        m = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        n = int(m.sum())
        if n == 0:
            continue
        gap += n * abs(y_prob[m].mean() - y_true[m].mean())
        total += n
    return gap / total if total else float("nan")


def main() -> None:
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)
    conn.close()
    for col, _desc, fn in LABELS:
        df[col] = fn(df["arrival_order"])

    dates = sorted(df[df["date"] >= TEST_START]["date"].unique())
    fold_size = len(dates) // N_FOLDS
    boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]
    print(f"評価期間 {dates[0]}〜{dates[-1]}({len(dates)}日) / {N_FOLDS}fold")

    rows = []   # 1行=1艇。予測と実績をためる
    for i in range(N_FOLDS):
        f_start, f_end = boundaries[i], boundaries[i + 1]
        train_df = df[df["date"] < f_start]
        fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
        if fold_df.empty or train_df.empty:
            continue
        print(f"fold{i+1} 学習中(3モデル)...", flush=True)
        for col, desc, _fn in LABELS:
            booster = train_with_label(train_df, col)
            fold_df[f"pred_{col}"] = booster.predict(fold_df[FEATURE_COLUMNS])

        # Harville導出値(現行のやり方)をレースごとに計算する
        for rid, g in fold_df.groupby("race_id"):
            g = g.sort_values("pred_is_winner", ascending=False)
            ranked = [{"lane": int(r["lane"]), "prob": float(r["pred_is_winner"])}
                      for _, r in g.iterrows()]
            probs = P.normalize_probs(ranked)
            if len(probs) < 4:
                continue
            tri = P.trifecta_probs(probs)
            h_place2 = defaultdict(float)   # その艇が2着になる確率
            h_top3 = defaultdict(float)     # その艇が3着以内に入る確率
            for (a, b, c), p in tri.items():
                h_place2[b] += p
                h_top3[a] += p
                h_top3[b] += p
                h_top3[c] += p
            for _, r in g.iterrows():
                lane = int(r["lane"])
                rows.append({
                    "fold": i + 1, "race_id": rid, "lane": lane,
                    "y_win": int(r["is_winner"]), "y_place2": int(r["is_place2"]),
                    "y_top3": int(r["is_top3"]),
                    "m_win": float(probs.get(lane, 0.0)),
                    "m_place2": float(r["pred_is_place2"]),
                    "m_top3": float(r["pred_is_top3"]),
                    "h_place2": float(h_place2.get(lane, 0.0)),
                    "h_top3": float(h_top3.get(lane, 0.0)),
                })

    res = pd.DataFrame(rows)
    print(f"\n評価行数 {len(res):,}(艇単位) / レース {res['race_id'].nunique():,}\n")

    def report(name, y_col, direct_col, harville_col):
        y = res[y_col]
        print(f"=== {name} ===")
        print(f"{'手法':<26}{'AUC':>9}{'Brier':>10}{'較正誤差':>10}{'平均予測':>10}"
              f"{'実際':>8}")
        for label, col in ((f"A 現行(Harville導出)", harville_col),
                           (f"B 直接予測モデル", direct_col)):
            p = res[col]
            print(f"{label:<26}{auc(y, p):>9.4f}{brier(y, p):>10.4f}"
                  f"{calibration_gap(y, p):>10.4f}{p.mean():>10.3f}{y.mean():>8.3f}")
        a_auc, b_auc = auc(y, res[harville_col]), auc(y, res[direct_col])
        a_br, b_br = brier(y, res[harville_col]), brier(y, res[direct_col])
        a_cg, b_cg = (calibration_gap(y, res[harville_col]),
                      calibration_gap(y, res[direct_col]))
        print(f"→ 基準1 AUC   : {'○ 直接予測が上' if b_auc > a_auc else '× 改善せず'}"
              f"({b_auc - a_auc:+.4f})")
        print(f"→ 基準2 較正   : {'○ 同等以上' if b_cg <= a_cg else '× 悪化'}"
              f"({b_cg - a_cg:+.4f})")
        print(f"→ 基準3 Brier  : {'○ 改善' if b_br < a_br else '× 改善せず'}"
              f"({b_br - a_br:+.4f})")
        print(f"→ 判定: "
              f"{'第2段へ進む価値あり' if (b_auc > a_auc and b_cg <= a_cg and b_br < a_br) else '見送り(Harvilleで十分)'}\n")

    report("3着以内(3連複の土台)", "y_top3", "m_top3", "h_top3")
    report("2着(3連単④⑤の土台)", "y_place2", "m_place2", "h_place2")

    # 超混戦帯だけを取り出した比較(エッジの本体で効くかを見る)
    top1 = res.sort_values("m_win", ascending=False).groupby("race_id").head(1)
    kon_ids = set(top1[top1["m_win"] < 0.20]["race_id"])
    kon = res[res["race_id"].isin(kon_ids)]
    if len(kon) > 0:
        print(f"=== 参考: 超混戦帯のみ({kon['race_id'].nunique():,}レース) ===")
        for name, y_col, d_col, h_col in (
                ("3着以内", "y_top3", "m_top3", "h_top3"),
                ("2着", "y_place2", "m_place2", "h_place2")):
            ya, yb = auc(kon[y_col], kon[h_col]), auc(kon[y_col], kon[d_col])
            print(f"{name:<8} Harville AUC {ya:.4f} / 直接予測 AUC {yb:.4f} "
                  f"({yb - ya:+.4f})")

    out = Path(__file__).with_name("place_models_predictions.csv")
    res.to_csv(out, index=False, encoding="utf-8")
    print(f"\n生データ: {out}")


if __name__ == "__main__":
    main()
