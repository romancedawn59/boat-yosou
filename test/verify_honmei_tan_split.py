# -*- coding: utf-8 -*-
"""本命帯: E/F単200円を「2並び×100円」に分割する案の検証(2026-08-04ケンさん発案)

    py -X utf8 test/verify_honmei_tan_split.py

問い: E/F単はトリオ的中後の1/6を狙っている。2並び買って1/3にした方が得では?
前提: E/Fは無作為の1/6ではなく差されマス(検証済みの安売りセル)の狙撃。
事前登録アーム(1,000円不変):
  現行: E(3位-1位-2位)200 / F(4位-1位-2位)200
  分割イ(入替型): E100+3位-2位-1位100 / F100+4位-2位-1位100
  分割ロ(確率型): E100+Eトリオ内確率最上位(E除く)100 / F側も同様
参考表示(採否に使わない): 両トリオ全12並びの単体ROI。
判定: 分割が現行をROIと最大1発除きの両方で上回る場合のみ9/1候補。
"""
import sys
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
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

ctxs = []
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
        if int(g["venue_code"].iloc[0]) not in TARGET_VENUE_CODES:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5 or not (0.20 <= ranked[0]["prob"] < 0.30):
            continue
        ctxs.append({"rid": rid, "date": g["date"].iloc[0],
                     "top": ranked[0]["prob"], "ranked": ranked})

by_day = defaultdict(list)
for c in ctxs:
    by_day[c["date"]].append(c)
sel = []
for d, cs in by_day.items():
    cs.sort(key=lambda c: c["top"])
    sel.extend(cs[:6])
n = len(sel)
print(f"\n本命選別再現: {n:,}R")

# 参考: 12並びの単体ROI
slot = defaultdict(lambda: [0, 0, 0])
for c in sel:
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    pay = payout_map[c["rid"]]
    lbl = {r1: "r1", r2: "r2", r3: "r3", r4: "r4"}
    for members, tag in (((r1, r2, r3), "E"), ((r1, r2, r4), "F")):
        for a, b, cc in permutations(members):
            key = tag + ":" + "-".join(lbl[x] for x in (a, b, cc))
            got = pay.get(("3連単", f"{a}-{b}-{cc}"), 0)
            s = slot[key]
            s[0] += 100
            s[1] += got
            if got:
                s[2] += 1
print("\n--- 参考(採否に使わない): 12並びの単体ROI(各100円) ---")
for key, (st, rt, h) in sorted(slot.items(), key=lambda x: -x[1][1]/x[1][0]):
    print(f"  {key:<12} 的中{h:>3}本 回収率{rt/st:>7.1%}")

def plans(c):
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    probs = P.normalize_probs(c["ranked"])
    tri = P.trifecta_probs(probs)

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"
    fuku = [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
            ("3連複", trio(r1, r3, r4), 100), ("3連複", trio(r2, r3, r4), 100)]
    E, F = f"{r3}-{r1}-{r2}", f"{r4}-{r1}-{r2}"

    def top_excl(members, exclude):
        cands = sorted(((o, p) for o, p in tri.items()
                        if set(o) == set(members)), key=lambda x: -x[1])
        for (a, b, cc), _p in cands:
            cb = f"{a}-{b}-{cc}"
            if cb != exclude:
                return cb
    return {
        "現行(E200/F200)": fuku + [("3連単", E, 200), ("3連単", F, 200)],
        "分割イ(入替型)": fuku + [("3連単", E, 100),
                              ("3連単", f"{r3}-{r2}-{r1}", 100),
                              ("3連単", F, 100),
                              ("3連単", f"{r4}-{r2}-{r1}", 100)],
        "分割ロ(確率型)": fuku + [("3連単", E, 100),
                              ("3連単", top_excl((r1, r2, r3), E), 100),
                              ("3連単", F, 100),
                              ("3連単", top_excl((r1, r2, r4), F), 100)],
    }

agg = defaultdict(lambda: [0, 0, 0, 0, 0])
for c in sel:
    pay = payout_map[c["rid"]]
    for name, bets in plans(c).items():
        st = sum(y for _, _, y in bets)
        rt = sum(pay.get((bt, cb), 0) * y // 100 for bt, cb, y in bets)
        a = agg[name]
        a[0] += st
        a[1] += rt
        if rt and rt < st:
            a[2] += 1
        elif rt:
            a[3] += 1
        a[4] = max(a[4], rt)
print("\n--- 構成比較(1,000円・レース単位) ---")
for name, (st, rt, gm, pl, best) in agg.items():
    print(f"  {name:<14} 回収率{rt/st:>7.1%} ガミ率{gm/n:>6.1%} プラス率{pl/n:>6.1%} "
          f"除き{(rt-best)/st:>7.1%} 損益{rt-st:+,}円")
