# -*- coding: utf-8 -*-
"""v2特徴量アブレーション: KR指数(Elo) + 伸びギャップ をモデルに注入する

    py -X utf8 test/verify_v2_features_ablation.py

【事前登録(実行前に固定・2026-07-29)】
研究期(7/24-29)で「単因子としては効く」と実測された2つを、手動コンボではなく
モデルの特徴量として与え、LightGBMに掛け算を探させる。

対象アーム(この4つだけ。実行後にアームを増やさない):
  A  base           = 現行FEATURE_COLUMNS(31特徴量)
  B  +KR            = base + kr_rating(レース前Elo) + kr_rel(レース内平均との差)
  C  +GAP           = base + rising_gap(直近90日実測2連対率 − 番組表2連率)
  D  +BOTH          = base + 上記4列

主要指標(採否はこれで決める):
  5場スコープ・ken現行構成(実運用と同じ買い方)の全fold合計回収率
合格基準(事前固定):
  (1) baseを +5.0pt 以上上回る、かつ
  (2) 5fold中4fold以上でbaseを上回る(1発依存の排除)
  両方を満たしたアームのみv2本採用の候補とする。片方だけなら「保留・継続観測」。

副次指標(参考表示のみ。これで採否を決めない):
  全場AUC / モデル1位の1着的中率 / バケット別回収率 / 新特徴量のimportance

注意: マスを見てから基準を動かさない(Phase 0の教訓)。
出力: test/verify_v2_features_ablation_results.json
"""
import datetime as dt
import json
import sqlite3
import sys
from bisect import bisect_left
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import BUCKETS, KEN_STRATEGY, PARAMS, TEST_START, N_FOLDS, bucket_of
from config import DB_PATH, TARGET_VENUE_CODES
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set

K_ELO = 3.0            # verify_rating_v1.py と同じ更新幅
GAP_WINDOW_DAYS = 90   # verify_ef_rising.py と同じ窓
GAP_MIN_RACES = 12     # 同上(母数不足はNaN=LightGBMが欠損として扱う)

KR_COLS = ["kr_rating", "kr_rel"]
GAP_COLS = ["rising_gap"]

ARMS = {
    "A base": [],
    "B +KR": KR_COLS,
    "C +GAP": GAP_COLS,
    "D +BOTH": KR_COLS + GAP_COLS,
}
RESULT_JSON = r"Y:\マイドライブ\boat\test\verify_v2_features_ablation_results.json"


# ---------------------------------------------------------------- 特徴量の生成
def build_extra_features() -> pd.DataFrame:
    """(race_id, lane)キーで kr_rating / kr_rel / rising_gap を返す。

    どちらも「そのレースより前の結果」だけから作る(時系列順に逐次更新 /
    日付で厳密に前方カット)ため walk-forward で安全。
    """
    raw = sqlite3.connect(DB_PATH)
    races_date = dict(raw.execute("SELECT race_id, date FROM races"))
    lane_racer, printed = {}, {}
    for rid, lane, reg, n2 in raw.execute(
            "SELECT race_id, lane, reg_no, national_2rate FROM entries"):
        lane_racer[(rid, lane)] = reg
        printed[(rid, lane)] = n2
    finish = defaultdict(dict)
    for rid, lane, ao in raw.execute(
            "SELECT race_id, lane, arrival_order FROM results "
            "WHERE arrival_order IS NOT NULL"):
        finish[rid][lane] = ao
    raw.close()

    # --- KR指数(Elo。1レース=15対戦を時系列順に処理) ---
    R = defaultdict(lambda: 1500.0)
    order = sorted(finish.keys(), key=lambda r: (races_date.get(r, ""), r))
    kr_rows = []
    for rid in order:
        if not races_date.get(rid):
            continue
        boats = [(lane, ao) for lane, ao in finish[rid].items()
                 if (rid, lane) in lane_racer]
        if len(boats) < 2:
            continue
        regs = {lane: lane_racer[(rid, lane)] for lane, _ in boats}
        rates = {lane: R[regs[lane]] for lane, _ in boats}   # 更新前=レース前の値
        mean = sum(rates.values()) / len(rates)
        for lane, r in rates.items():
            kr_rows.append((rid, lane, r, r - mean))
        for i in range(len(boats)):
            for j in range(i + 1, len(boats)):
                li, ai = boats[i]
                lj, aj = boats[j]
                ri, rj = regs[li], regs[lj]
                e = 1 / (1 + 10 ** ((R[rj] - R[ri]) / 400))
                s = 1.0 if ai < aj else 0.0
                R[ri] += K_ELO * (s - e)
                R[rj] -= K_ELO * (s - e)

    kr = pd.DataFrame(kr_rows, columns=["race_id", "lane", "kr_rating", "kr_rel"])

    # --- 伸びギャップ(直近90日実測2連対率 − 番組表2連率) ---
    hist = defaultdict(list)
    rows = []
    for rid, per_lane in finish.items():
        d = races_date.get(rid)
        if not d:
            continue
        for lane, ao in per_lane.items():
            reg = lane_racer.get((rid, lane))
            if reg:
                rows.append((d, reg, ao))
    rows.sort()
    for d, reg, ao in rows:
        hist[reg].append((d, 1 if ao <= 2 else 0))
    hist_dates = {reg: [x[0] for x in v] for reg, v in hist.items()}

    gap_rows = []
    for (rid, lane), reg in lane_racer.items():
        d = races_date.get(rid)
        n2 = printed.get((rid, lane))
        if d is None or n2 is None or reg not in hist:
            continue
        dates = hist_dates[reg]
        hi = bisect_left(dates, d)                       # 当日以降は使わない
        d0 = (dt.date.fromisoformat(d)
              - dt.timedelta(days=GAP_WINDOW_DAYS)).isoformat()
        lo = bisect_left(dates, d0)
        seg = hist[reg][lo:hi]
        if len(seg) < GAP_MIN_RACES:
            continue
        gap = sum(t for _, t in seg) / len(seg) - n2 / 100.0
        gap_rows.append((rid, lane, gap))
    gp = pd.DataFrame(gap_rows, columns=["race_id", "lane", "rising_gap"])

    return kr.merge(gp, on=["race_id", "lane"], how="outer")


# ---------------------------------------------------------------- 学習・評価
def train_fold(train_df: pd.DataFrame, cols: list[str]) -> lgb.Booster:
    """backtest.train_fold と同じ手順(末尾10%で早期停止)を任意の特徴量列で行う"""
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr, va = train_df[train_df["date"] < cutoff], train_df[train_df["date"] >= cutoff]
    train_set = lgb.Dataset(tr[cols], label=tr["is_winner"],
                            categorical_feature=CATEGORICAL_FEATURES)
    valid_set = lgb.Dataset(va[cols], label=va["is_winner"], reference=train_set)
    return lgb.train(PARAMS, train_set, valid_sets=[valid_set], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def auc_of(y: np.ndarray, p: np.ndarray) -> float:
    """順位ベースのAUC(sklearn非依存。同値はタイ補正あり)"""
    ranks = pd.Series(p).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)

    actual = defaultdict(dict)
    for rid, lane, o in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id = res.race_id "
        "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,),
    ):
        actual[rid][o] = lane
    payout_map = defaultdict(dict)
    for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,),
    ):
        payout_map[rid][(bt, comb)] = amt or 0
    conn.close()

    print("追加特徴量を生成中(KR指数のElo逐次更新 + 伸びギャップ)...", flush=True)
    extra = build_extra_features()
    df = df.merge(extra, on=["race_id", "lane"], how="left")
    cov = {c: float(df[c].notna().mean()) for c in KR_COLS + GAP_COLS}
    print("  被覆率: " + " / ".join(f"{c}={v:.1%}" for c, v in cov.items()), flush=True)

    test_df = df[df["date"] >= TEST_START]
    dates = sorted(test_df["date"].unique())
    fold_size = len(dates) // N_FOLDS
    boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

    # {(arm, scope, bucket): [投資, 回収, レース数, 的中数]}  scope: 5場/全場
    total = defaultdict(lambda: [0, 0, 0, 0])
    fold_roi = defaultdict(dict)     # {arm: {fold: 5場ken回収率}}
    diag = defaultdict(lambda: {"auc": [], "top1": [], "imp": {}})

    for i in range(N_FOLDS):
        f_start, f_end = boundaries[i], boundaries[i + 1]
        train_df = df[df["date"] < f_start]
        fold_base = df[(df["date"] >= f_start) & (df["date"] < f_end)]
        print(f"\n--- fold{i+1} {f_start}〜{fold_base['date'].max()} "
              f"(学習 {len(train_df):,}行 / 検証 {len(fold_base):,}行) ---", flush=True)

        for arm, add_cols in ARMS.items():
            cols = FEATURE_COLUMNS + add_cols
            booster = train_fold(train_df, cols)
            fold_df = fold_base.copy()
            fold_df["pred"] = booster.predict(fold_df[cols])

            diag[arm]["auc"].append(
                auc_of(fold_df["is_winner"].to_numpy(), fold_df["pred"].to_numpy()))
            gains = dict(zip(booster.feature_name(),
                             booster.feature_importance("gain")))
            g_sum = sum(gains.values()) or 1
            for c in add_cols:
                diag[arm]["imp"].setdefault(c, []).append(gains.get(c, 0) / g_sum)

            stake_f = ret_f = 0
            top1_hit = top1_n = 0
            for rid, g in fold_df.groupby("race_id"):
                if 1 not in actual[rid] or not payout_map[rid]:
                    continue
                g_sorted = g.sort_values("pred", ascending=False)
                b = bucket_of(g["pred"].max())
                scopes = ["全場"]
                if int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES:
                    scopes.append("5場")
                top1_n += 1
                top1_hit += int(int(g_sorted["lane"].iloc[0]) == actual[rid][1])

                ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                          for _, r in g_sorted.iterrows()]
                probs = P.normalize_probs(ranked)
                if not probs:
                    continue
                plan = P.ken_portfolio(b, ranked, P.picks_yamada(probs),
                                       P.picks_katsu(probs))
                if not plan:
                    continue
                stake = sum(yen for _, _, yen, _ in plan)
                ret = sum(payout_map[rid].get((bt, comb), 0) * yen // 100
                          for bt, comb, yen, _src in plan)
                for scope in scopes:
                    s = total[(arm, scope, b)]
                    s[0] += stake
                    s[1] += ret
                    s[2] += 1
                    s[3] += 1 if ret else 0
                if "5場" in scopes:
                    stake_f += stake
                    ret_f += ret

            diag[arm]["top1"].append(top1_hit / max(1, top1_n))
            roi = ret_f / stake_f if stake_f else float("nan")
            fold_roi[arm][i + 1] = roi
            print(f"  {arm:<9} 5場ken回収率 {roi:7.1%}  (投資{stake_f:,}円)", flush=True)

    # ---------------------------------------------------------------- 判定
    def scope_roi(arm, scope):
        st = sum(total[(arm, scope, b)][0] for b in BUCKETS)
        rt = sum(total[(arm, scope, b)][1] for b in BUCKETS)
        return rt / st if st else float("nan")

    base_roi = scope_roi("A base", "5場")
    print("\n" + "=" * 66)
    print(f"===== 主要指標: 5場・{KEN_STRATEGY} 全fold合計回収率 =====")
    verdict = {}
    for arm in ARMS:
        roi = scope_roi(arm, "5場")
        wins = sum(1 for f, v in fold_roi[arm].items()
                   if v > fold_roi["A base"][f])
        d = (roi - base_roi) * 100
        ok = (arm != "A base" and d >= 5.0 and wins >= 4)
        hold = (arm != "A base" and not ok and (d >= 5.0 or wins >= 4))
        verdict[arm] = ("採用候補" if ok else "保留・継続観測" if hold
                        else "—" if arm == "A base" else "不合格")
        print(f"  {arm:<9} {roi:7.1%}  base差 {d:+5.1f}pt  "
              f"base超えfold {wins}/{N_FOLDS}  → {verdict[arm]}")

    print("\n----- 副次指標(参考。採否には使わない) -----")
    for arm in ARMS:
        au = np.nanmean(diag[arm]["auc"])
        t1 = np.mean(diag[arm]["top1"])
        imp = " ".join(f"{c}={np.mean(v):.1%}" for c, v in diag[arm]["imp"].items())
        print(f"  {arm:<9} AUC {au:.4f}  1位的中 {t1:.1%}  全場回収率 "
              f"{scope_roi(arm,'全場'):7.1%}  gain寄与 {imp}")

    print("\n----- バケット別(5場・参考) -----")
    for b in BUCKETS:
        line = []
        for arm in ARMS:
            st, rt, n, h = total[(arm, "5場", b)]
            line.append(f"{arm}={rt/st:.1%}" if st else f"{arm}=—")
        n = total[("A base", "5場", b)][2]
        print(f"  [{b}] {n}R  " + "  ".join(line))

    out = {
        "pre_registered": {
            "arms": {k: v for k, v in ARMS.items()},
            "primary": "5場・ken現行構成の全fold合計回収率",
            "pass_rule": "base比 +5.0pt以上 かつ base超えfoldが4/5以上",
        },
        "coverage": cov,
        "roi_5venues": {a: scope_roi(a, "5場") for a in ARMS},
        "roi_all": {a: scope_roi(a, "全場") for a in ARMS},
        "fold_roi_5venues": {a: fold_roi[a] for a in ARMS},
        "auc": {a: float(np.nanmean(diag[a]["auc"])) for a in ARMS},
        "top1": {a: float(np.mean(diag[a]["top1"])) for a in ARMS},
        "gain_share": {a: {c: float(np.mean(v)) for c, v in diag[a]["imp"].items()}
                       for a in ARMS},
        "verdict": verdict,
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {RESULT_JSON}")


if __name__ == "__main__":
    main()
