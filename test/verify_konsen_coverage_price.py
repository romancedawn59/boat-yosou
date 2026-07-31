# -*- coding: utf-8 -*-
"""超混戦「少額広範囲」の値段: 網を広げた先の海のROI(2026-08-01ケンさん発案の決着)

    py -X utf8 test/verify_konsen_coverage_price.py

■ 問い
超混戦の決着はモデル順位・市場人気とも全域に散らばる(23R観測)。
なら少額広範囲(点数を拡げる)で捉えられないか?

■ 方法(事前登録)
超混戦帯(1位生値20%未満)walk-forwardで、各カバレッジのROIを測る:
  1. 3連単 全120通り × 100円(完全な網=全レースを必ず「捕まえる」)
  2. 3連複 全20通り × 100円(順不同の完全な網)
  3. 現行5点(案1の形・比較用に各100円換算)
  4. 現行5点以外の全て(単118点+複17点)=「広げた先の海」
判定: 4が100%を大きく下回るなら「広げる=薄めた海水を買う」で決着。
"""
import sys
from collections import defaultdict
from itertools import combinations, permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

conn = db.connect(DB_PATH)
df = build_training_set(conn)
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

agg = defaultdict(lambda: [0, 0, 0])   # {アーム: [st, rt, hitR]}
n = 0
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        if not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 6 or ranked[0]["prob"] >= 0.20:
            continue
        n += 1
        lanes = [r["lane"] for r in ranked]
        r1, r2, r3, r4, r5 = lanes[:5]

        def trio(a, b, c):
            s = sorted([a, b, c])
            return f"{s[0]}={s[1]}={s[2]}"
        core = {("3連単", f"{r3}-{r1}-{r2}"), ("3連単", f"{r4}-{r1}-{r2}"),
                ("3連複", trio(r1, r2, r3)), ("3連複", trio(r1, r2, r4)),
                ("3連複", trio(r3, r4, r5))}
        pay = payout_map[rid]
        arms = {
            "全3連単120点": [("3連単", f"{a}-{b}-{c}")
                          for a, b, c in permutations(lanes, 3)],
            "全3連複20点": [("3連複", trio(a, b, c))
                          for a, b, c in combinations(lanes, 3)],
            "現行5点(各100円換算)": list(core),
        }
        arms["5点以外の海(135点)"] = [x for x in arms["全3連単120点"]
                                   + arms["全3連複20点"] if x not in core]
        for name, bets in arms.items():
            st = 100 * len(bets)
            rt = sum(pay.get(b, 0) for b in bets)
            a = agg[name]
            a[0] += st
            a[1] += rt
            a[2] += 1 if rt else 0

print(f"\n超混戦帯: {n:,}R")
print(f"{'カバレッジ':<18}{'投資/R':>9}{'何か当たる率':>10}{'回収率':>8}")
for name, (st, rt, hit) in agg.items():
    print(f"{name:<18}{st//n:>8,}円{hit/n:>10.1%}{rt/st:>8.1%}")
