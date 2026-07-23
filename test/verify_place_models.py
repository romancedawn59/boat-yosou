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
  B. 直接予測(素): ラベルを変えて学習した専用モデル
       - place2モデル: arrival_order == 2 / top3モデル: arrival_order <= 3
  C. 直接予測+レース内正規化: Bの弱点(各艇を独立に評価するため「2着は1艇/
       3着以内は3艇」の制約を知らず、強い艇が揃うと全員高く出てAUCが鈍る)を、
       レース内で合計1(2着)/合計3(3着以内)に整えることで補えるか試す

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

■ 検証の枠組みは2本立て(このプロジェクトの標準手順)
  1. walk-forward(backtest.py同一fold・全期間)
  2. 2026-05-01〜06-30の固定期間(5/1以前で1回だけ学習)
     → 検証⑦⑧⑨・verify_ken_v2_202605_06.py・verify_konsen_202605_06.pyと
       同じ土俵。片方だけで良く見えるものは期間依存を疑う
  両方で基準を満たして初めて第2段へ進む。

■ 注意
  本番コードには一切触れない(v1凍結の原則: 新規ファイル+test/出力のみ)。
  学習は3モデル×(5fold+固定1)=18回でそれなりに時間がかかる。
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

# 固定期間シミュレーションの評価区間(既存の5〜6月系列と同じ土俵にする)
FIXED_START, FIXED_END = "2026-05-01", "2026-06-30"

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

    res_wf = run_walkforward(df)
    res_fx = run_fixed(df, FIXED_START, FIXED_END)

    for title, res in (("枠組み1: walk-forward(全期間・fold別学習)", res_wf),
                       (f"枠組み2: {FIXED_START}〜{FIXED_END} 固定"
                        "(5/1以前で1回だけ学習)", res_fx)):
        print("\n" + "=" * 68)
        print(f"■ {title}")
        print(f"  評価行数 {len(res):,}(艇単位) / "
              f"レース {res['race_id'].nunique():,}")
        print("=" * 68)
        report_all(res)

    out = Path(__file__).with_name("place_models_predictions.csv")
    res_wf.assign(枠組み="walkforward").to_csv(out, index=False, encoding="utf-8")
    out_fx = Path(__file__).with_name("place_models_predictions_202605_06.csv")
    res_fx.assign(枠組み="fixed").to_csv(out_fx, index=False, encoding="utf-8")
    print(f"\n生データ: {out}\n          {out_fx}")


def _collect(train_df: pd.DataFrame, eval_df: pd.DataFrame, tag) -> list:
    """3モデルを学習してeval_dfを予測し、Harville導出値と並べた行を返す"""
    eval_df = eval_df.copy()
    for col, _desc, _fn in LABELS:
        booster = train_with_label(train_df, col)
        eval_df[f"pred_{col}"] = booster.predict(eval_df[FEATURE_COLUMNS])

    rows = []
    for rid, g in eval_df.groupby("race_id"):
        g = g.sort_values("pred_is_winner", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred_is_winner"])}
                  for _, r in g.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        # 超混戦の判定は本番select_shobushoと同じく「正規化前の生の1着予測値」の
        # 最大値で行う(正規化後で判定すると値が変わり超混戦が激減するバグがあった)
        top_raw = float(g["pred_is_winner"].iloc[0])

        # Harville導出値(現行のやり方)
        tri = P.trifecta_probs(probs)
        h_place2 = defaultdict(float)   # その艇が2着になる確率
        h_top3 = defaultdict(float)     # その艇が3着以内に入る確率
        for (a, b, c), p in tri.items():
            h_place2[b] += p
            h_top3[a] += p
            h_top3[b] += p
            h_top3[c] += p

        # レース内正規化: 直接予測モデルは各艇を独立に評価するため「2着は1艇/
        # 3着以内は3艇」という制約を知らず、強い艇が揃うと全員高く出てAUCが鈍る。
        # レース内で合計を1(2着)/3(3着以内)に整え、順位づけの弱点を補えるか見る
        s2 = g["pred_is_place2"].sum()
        s3 = g["pred_is_top3"].sum()
        norm2 = {int(r["lane"]): float(r["pred_is_place2"]) / s2 if s2 else 0.0
                 for _, r in g.iterrows()}
        norm3 = {int(r["lane"]): float(r["pred_is_top3"]) * 3 / s3 if s3 else 0.0
                 for _, r in g.iterrows()}

        for _, r in g.iterrows():
            lane = int(r["lane"])
            rows.append({
                "fold": tag, "race_id": rid, "lane": lane, "top_raw": top_raw,
                "y_win": int(r["is_winner"]), "y_place2": int(r["is_place2"]),
                "y_top3": int(r["is_top3"]),
                "m_win": float(probs.get(lane, 0.0)),
                "m_place2": float(r["pred_is_place2"]),
                "m_top3": float(r["pred_is_top3"]),
                "n_place2": norm2[lane],          # レース内正規化した直接予測(2着)
                "n_top3": norm3[lane],            # レース内正規化した直接予測(3着以内)
                "h_place2": float(h_place2.get(lane, 0.0)),
                "h_top3": float(h_top3.get(lane, 0.0)),
            })
    return rows


def run_walkforward(df: pd.DataFrame) -> pd.DataFrame:
    """backtest.pyと同一foldのウォークフォワード(学習期間に未来を含めない)"""
    dates = sorted(df[df["date"] >= TEST_START]["date"].unique())
    fold_size = len(dates) // N_FOLDS
    boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]
    print(f"\n[walk-forward] 評価 {dates[0]}〜{dates[-1]}"
          f"({len(dates)}日) / {N_FOLDS}fold")
    rows = []
    for i in range(N_FOLDS):
        f_start, f_end = boundaries[i], boundaries[i + 1]
        train_df = df[df["date"] < f_start]
        fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)]
        if fold_df.empty or train_df.empty:
            continue
        print(f"  fold{i+1} 学習中(3モデル)...", flush=True)
        rows += _collect(train_df, fold_df, i + 1)
    return pd.DataFrame(rows)


def run_fixed(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """固定期間シミュレーション(検証⑦⑧⑨・verify_ken_v2_202605_06.pyと同じ枠組み)。
    startより前の全データで1回だけ学習し、start〜endを評価する。
    既存の5〜6月系列と同じ土俵の数字になるため、walk-forwardの補完として見る"""
    train_df = df[df["date"] < start]
    eval_df = df[(df["date"] >= start) & (df["date"] <= end)]
    print(f"\n[{start}〜{end} 固定] 学習 {len(train_df):,}行 / "
          f"評価 {len(eval_df):,}行")
    print("  学習中(3モデル)...", flush=True)
    return pd.DataFrame(_collect(train_df, eval_df, "fixed"))


def report_all(res: pd.DataFrame) -> None:
    def report(name, y_col, harville_col, direct_col, norm_col):
        y = res[y_col]
        print(f"=== {name} ===")
        print(f"{'手法':<28}{'AUC':>9}{'Brier':>10}{'較正誤差':>10}{'平均予測':>10}"
              f"{'実際':>8}")
        methods = (("A 現行(Harville導出)", harville_col),
                   ("B 直接予測(素)", direct_col),
                   ("C 直接予測+レース内正規化", norm_col))
        m = {}
        for label, col in methods:
            p = res[col]
            m[label] = {"auc": auc(y, p), "brier": brier(y, p),
                        "cg": calibration_gap(y, p)}
            print(f"{label:<28}{m[label]['auc']:>9.4f}{m[label]['brier']:>10.4f}"
                  f"{m[label]['cg']:>10.4f}{p.mean():>10.3f}{y.mean():>8.3f}")
        base = m["A 現行(Harville導出)"]
        # BとCそれぞれを現行と照合し、3基準(AUC↑・較正同等以上・Brier改善)を見る
        for label in ("B 直接予測(素)", "C 直接予測+レース内正規化"):
            v = m[label]
            c1, c2, c3 = (v["auc"] > base["auc"], v["cg"] <= base["cg"],
                          v["brier"] < base["brier"])
            print(f"  {label}: AUC{'○' if c1 else '×'}({v['auc']-base['auc']:+.4f}) "
                  f"較正{'○' if c2 else '×'}({v['cg']-base['cg']:+.4f}) "
                  f"Brier{'○' if c3 else '×'}({v['brier']-base['brier']:+.4f}) "
                  f"→ {'第2段へ' if (c1 and c2 and c3) else '基準未達'}")
        print()

    report("3着以内(3連複の土台)", "y_top3", "h_top3", "m_top3", "n_top3")
    report("2着(3連単④⑤の土台)", "y_place2", "h_place2", "m_place2", "n_place2")

    # 超混戦帯だけを取り出した比較(エッジの本体で効くか)。判定は本番と同じく
    # 正規化前の生1着予測値(top_raw)で行う
    kon = res[res["top_raw"] < 0.20]
    nk = kon["race_id"].nunique()
    if nk >= 30:
        print(f"=== 参考: 超混戦帯のみ({nk:,}レース) ===")
        for name, y_col, h_col, d_col, n_col in (
                ("3着以内", "y_top3", "h_top3", "m_top3", "n_top3"),
                ("2着", "y_place2", "h_place2", "m_place2", "n_place2")):
            print(f"{name:<8} Harville {auc(kon[y_col], kon[h_col]):.4f} / "
                  f"直接 {auc(kon[y_col], kon[d_col]):.4f} / "
                  f"正規化 {auc(kon[y_col], kon[n_col]):.4f}  "
                  f"(較正 H{calibration_gap(kon[y_col], kon[h_col]):.4f} "
                  f"→ 正規化{calibration_gap(kon[y_col], kon[n_col]):.4f})")
    elif nk > 0:
        print(f"=== 超混戦帯は{nk}レースのみ→サンプル不足で判定しない ===")


if __name__ == "__main__":
    main()
