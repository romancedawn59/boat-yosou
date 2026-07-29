# -*- coding: utf-8 -*-
"""事前登録テスト: E/F差され単×頭の艇が伸び盛り(2026-07-29)

    py -X utf8 test/verify_ef_rising.py

仮説(事前固定): 超混戦帯のE/F単(r3-r1-r2 / r4-r1-r2)で、頭のr3/r4の選手の
伸びギャップ(直近90日実測2連対率−番組表2連率)が+10pt超なら、
「並びの安売り2.48倍×古いアンカー」の複利で回収率が通常E/Fを上回る。
これ以外のマスは見ない(多重検定の回避)。本命帯は参考表示のみ。
"""
import sqlite3
import sys
from bisect import bisect_left
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

# --- 伸びギャップの下ごしらえ(fold外・履歴のみ使用) ---
raw = sqlite3.connect(DB_PATH)
races_date = dict(raw.execute("SELECT race_id, date FROM races"))
lane_racer = {}
printed = {}
for rid, lane, reg, n2 in raw.execute(
        "SELECT race_id, lane, reg_no, national_2rate FROM entries"):
    lane_racer[(rid, lane)] = reg
    printed[(rid, lane)] = n2
hist = defaultdict(list)
rows = []
for rid, lane, ao in raw.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    d = races_date.get(rid)
    reg = lane_racer.get((rid, lane))
    if d and reg:
        rows.append((d, reg, ao))
rows.sort()
for d, reg, ao in rows:
    hist[reg].append((d, 1 if ao <= 2 else 0))
raw.close()


def gap_of(rid, lane):
    import datetime as dt
    reg = lane_racer.get((rid, lane))
    n2 = printed.get((rid, lane))
    d = races_date.get(rid)
    if reg is None or n2 is None or d is None:
        return None
    h = hist[reg]
    dates = [x[0] for x in h]
    hi = bisect_left(dates, d)
    d0 = (dt.date.fromisoformat(d) - dt.timedelta(days=90)).isoformat()
    lo = bisect_left(dates, d0)
    seg = h[lo:hi]
    if len(seg) < 12:
        return None
    return sum(t for _, t in seg) / len(seg) - n2 / 100.0


# --- walk-forward ---
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
        if len(ranked) < 4:
            continue
        ctxs.append({"rid": rid, "top": ranked[0]["prob"],
                     "lanes": [r["lane"] for r in ranked]})

for scope_name, lo, hi in (("★超混戦帯(事前登録の本勝負)", 0.0, 0.20),
                           ("本命帯(参考)", 0.20, 0.35)):
    sel = [c for c in ctxs if lo <= c["top"] < hi]
    agg = {"伸び盛り(+10pt超)": [0, 0, 0], "それ以外": [0, 0, 0]}
    for c in sel:
        r1, r2, r3, r4 = c["lanes"][:4]
        pay = payout_map[c["rid"]]
        for head in (r3, r4):
            comb = f"{head}-{r1}-{r2}"
            got = pay.get(("3連単", comb), 0) * 2  # 200円
            g = gap_of(c["rid"], head)
            key = "伸び盛り(+10pt超)" if (g is not None and g > 0.10) else "それ以外"
            a = agg[key]
            a[0] += 200
            a[1] += got
            if got:
                a[2] += 1
    print(f"\n=== {scope_name}: {len(sel):,}R ===")
    for key, (st, rt, h) in agg.items():
        if st:
            print(f"  E/F単・頭が{key}: {st//200:,}点 投資{st:,}円 "
                  f"回収{rt:,}円 回収率{rt/st:.1%} 的中{h}本")
