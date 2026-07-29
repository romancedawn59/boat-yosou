# -*- coding: utf-8 -*-
"""r1圏外保険ブロック(r2〜r5の3連複4点)の単体成績

    py -X utf8 test/verify_r1_insurance.py

verify_r1_dependency.pyの続き: r1圏外の72.3%はtop3⊆{r2..r5}で、
その日の決着は53%が55倍超の大穴帯だった。では{r2,r3,r4,r5}の
3連複4点(各100円)を保険ブロックとして買った場合の単体回収率は?
本命帯・超混戦帯それぞれで、月次も出す。
"""
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

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

ctxs = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        if 1 not in actual[rid] or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]),
                     "top": P.normalize_probs(ranked)[ranked[0]["lane"]]
                     if False else ranked[0]["prob"],
                     "ranked": ranked})


def block_slots(c):
    lanes = [r["lane"] for r in c["ranked"]]
    r2, r3, r4, r5 = lanes[1:5]
    out = []
    for trio in combinations([r2, r3, r4, r5], 3):
        s = sorted(trio)
        out.append(("3連複", f"{s[0]}={s[1]}={s[2]}"))
    return out


for scope_name, lo, hi in (("本命帯(20-35%)", 0.20, 0.35),
                           ("超混戦帯(<20%)", 0.0, 0.20)):
    sel = [c for c in ctxs if lo <= c["top"] < hi]
    n = len(sel)
    st = rt = hits = 0
    pays = []
    monthly = defaultdict(lambda: [0, 0])
    for c in sel:
        pay = payout_map[c["rid"]]
        got_any = 0
        for bt, comb in block_slots(c):
            st += 100
            got = pay.get((bt, comb), 0)
            rt += got
            got_any += got
            monthly[c["date"][:7]][0] += 100
            monthly[c["date"][:7]][1] += got
        if got_any:
            hits += 1
            pays.append(got_any)
    print(f"\n=== {scope_name}: {n:,}R × 保険複4点(400円) ===")
    if not st:
        continue
    print(f"  的中率: {hits/n:.1%}  平均的中額: "
          f"{sum(pays)/len(pays) if pays else 0:,.0f}円")
    print(f"  投資{st:,}円 回収{rt:,}円 回収率{rt/st:.1%}")
    print(f"  月次: " + "  ".join(
        f"{m[-2:]}月{v[1]/v[0]:.0%}" for m, v in sorted(monthly.items()) if v[0]))
