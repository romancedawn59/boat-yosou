# -*- coding: utf-8 -*-
"""検証: 場スコープ(5場/他19場/全場)×1位勝率帯のウォークフォワード比較CLI

    py -X utf8 test/verify_scope_bands.py

2026-07-18の「購入対象を全競艇場にするか?」への回答に使った買い方表の再現スクリプト。
設計はbacktest.py/検証⑪と同一のウォークフォワード(2025-12-01以降を5分割、
各foldはそれより前の全データで学習)。買い目は全行共通で検証済みV2構成+C。
1日上限10レースのcapは適用しない(帯そのものの性質を測るため)。
帯境界(20/25/30/35%)は5場の負け分析由来の固定値であり、成績を見た最適化はしない。
他19場は帯の発見に使っていない外部標本で、超混戦帯のエッジ再現性の検証を兼ねる。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import challengers as CH
import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

BANDS = [(0.00, 0.20, "〜20%"), (0.20, 0.25, "20〜25%"),
         (0.25, 0.30, "25〜30%"), (0.30, 0.35, "30〜35%")]

conn = db.connect(DB_PATH)
df = build_training_set(conn)
actual = defaultdict(dict)
for rid, lane, order in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,)):
    actual[rid][order] = lane
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]
n_days = len(dates)

stats = defaultdict(lambda: {"n": 0, "hits": 0, "junto": 0, "stake": 0, "ret": 0, "max_ret": 0})
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()  # 全場
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])

    for rid, g in fold_df.groupby("race_id"):
        if 1 not in actual[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        top = ranked[0]["prob"]
        if top >= 0.35:
            continue
        band = next(label for lo, hi, label in BANDS if lo <= top < hi)
        scope = "5場" if int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES else "他19場"
        plan = P.ken_portfolio("荒れ注意", ranked, [], P.picks_katsu(probs))
        pay = payout_map[rid]
        stake = sum(y for _, _, y, _ in plan)
        ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
        res = actual[rid]
        santan = pay.get(("3連単", f"{res.get(1)}-{res.get(2)}-{res.get(3)}"), 0)
        out = CH.classify_outcome(stake, ret, santan)
        for key in ((scope, band), (scope, "全帯"), ("全場", band), ("全場", "全帯")):
            s = stats[key]
            s["n"] += 1
            s["stake"] += stake
            s["ret"] += ret
            s["max_ret"] = max(s["max_ret"], ret)
            if ret:
                s["hits"] += 1
                if out == "順当":
                    s["junto"] += 1

print(f"\n=== 場スコープ×帯(walk-forward {n_days}日) ===")
print(f"{'スコープ':<7}{'帯':<9}{'R数':>6}{'R/日':>6}{'的中率':>8}{'回収率':>8}{'最大1発除き':>10}{'ガミ率':>8}")
for scope in ("5場", "他19場", "全場"):
    for _lo, _hi, band in BANDS + [(0, 0, "全帯")]:
        s = stats[(scope, band)]
        if not s["n"]:
            continue
        roi = s["ret"] / s["stake"]
        roi_ex = (s["ret"] - s["max_ret"]) / s["stake"]
        junto = s["junto"] / s["hits"] if s["hits"] else 0
        print(f"{scope:<7}{band:<9}{s['n']:>6}{s['n']/n_days:>6.1f}"
              f"{s['hits']/s['n']:>8.1%}{roi:>8.1%}{roi_ex:>10.1%}{junto:>8.1%}")
