# -*- coding: utf-8 -*-
"""「1号(r1)が絡まない着順予想」の条件付きモード検証(2026-07-29ケンさん発案)

    py -X utf8 test/verify_no_r1_mode.py

モデルのP(r1が3着以内)でレースをバケット分けし、疑い度が高いレースでの
「r1抜き構成」の成績を測る。
  (i) 複ブロック: {r2,r3,r4,r5}の3連複4点×100円
  (ii) 単r1抜き: r1を含まない3連単の確率上位6点×100円(=1号抜き着順予想)
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
                     "top": ranked[0]["prob"],
                     "probs": P.normalize_probs(ranked), "ranked": ranked})

BUCKETS = [(0.00, 0.60, "疑い濃(P<60%)"), (0.60, 0.70, "疑い中(60-70%)"),
           (0.70, 0.80, "並(70-80%)"), (0.80, 1.01, "信頼(80%+)")]

for scope_name, lo, hi in (("本命帯(20-35%)", 0.20, 0.35),
                           ("超混戦帯(<20%)", 0.0, 0.20)):
    sel = [c for c in ctxs if lo <= c["top"] < hi]
    print(f"\n{'='*70}\n=== {scope_name}: {len(sel):,}R ===")
    agg = defaultdict(lambda: {"n": 0, "out": 0,
                               "st_f": 0, "rt_f": 0, "h_f": 0,
                               "st_t": 0, "rt_t": 0, "h_t": 0})
    for c in sel:
        lanes = [r["lane"] for r in c["ranked"]]
        r1 = lanes[0]
        tri = P.trifecta_probs(c["probs"])
        p_r1 = sum(p for k, p in tri.items() if r1 in k)
        lbl = next(l for blo, bhi, l in BUCKETS if blo <= p_r1 < bhi)
        a = agg[lbl]
        a["n"] += 1
        res = actual[c["rid"]]
        top3 = {res.get(1), res.get(2), res.get(3)}
        if r1 not in top3:
            a["out"] += 1
        pay = payout_map[c["rid"]]
        # (i) 複ブロック4点
        got_any = 0
        for trio in combinations(lanes[1:5], 3):
            s = sorted(trio)
            comb = f"{s[0]}={s[1]}={s[2]}"
            a["st_f"] += 100
            got = pay.get(("3連複", comb), 0)
            a["rt_f"] += got
            got_any += got
        if got_any:
            a["h_f"] += 1
        # (ii) 単r1抜き上位6点
        cands = sorted(((k, p) for k, p in tri.items() if r1 not in k),
                       key=lambda kv: -kv[1])[:6]
        got_any = 0
        for (x, y, z), _p in cands:
            a["st_t"] += 100
            got = pay.get(("3連単", f"{x}-{y}-{z}"), 0)
            a["rt_t"] += got
            got_any += got
        if got_any:
            a["h_t"] += 1

    print(f"{'バケット':<14}{'R数':>6}{'r1圏外(実測)':>11}"
          f"{'複4点回収':>10}{'複的中':>7}{'単6点回収':>10}{'単的中':>7}")
    for _lo, _hi, lbl in BUCKETS:
        a = agg.get(lbl)
        if not a or not a["n"]:
            continue
        print(f"{lbl:<14}{a['n']:>6,}{a['out']/a['n']:>11.1%}"
              f"{a['rt_f']/max(1,a['st_f']):>10.1%}{a['h_f']/a['n']:>7.1%}"
              f"{a['rt_t']/max(1,a['st_t']):>10.1%}{a['h_t']/a['n']:>7.1%}")
