# -*- coding: utf-8 -*-
"""フォーメーション均等額化のROI影響(2026-07-31ケンさん発案・買い方の手間削減)

    py -X utf8 test/verify_formation_amounts.py

■ 問い
現構成は実質「3連複フォーメーション+[3位,4位]-1位-2位の3連単フォーメーション」。
金額を均等にすればフォーメーション一括入力できて手間が激減する。
傾斜配分(検証で決めた300/200等)を均等にするとROIはいくら変わるか。

■ 事前登録アーム
超混戦(1,000円スケールで比較・2,000円化は単純2倍なのでROI不変):
  K1 案1傾斜: A複300 B複200 E単200 F単200 G複100
  K2 均等200: A複200 B複200 E単200 F単200 G複200(フォーメーション入力可)
本命(5場20-30%cap6):
  H1 現行: 複200/200/100+単200/200+保険複100(計1,000円)
  H2 均等200: 複4点(1=2=3,1=2=4,1=3=4,2=3=4)各200+単2点各200(計1,200円)
判定: 均等形が傾斜形と回収率で-2pt以内なら「等価=手間削減を優先して採用可」。
それ以上悪化するなら傾斜維持。
"""
import sys
from collections import defaultdict

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

konsen_ctx, honmei_ctx = [], []
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
        if len(ranked) < 5:
            continue
        top = ranked[0]["prob"]
        c = {"rid": rid, "date": g["date"].iloc[0], "top": top,
             "lanes": [r["lane"] for r in ranked]}
        if top < 0.20:
            konsen_ctx.append(c)
        elif (top < 0.30
              and int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES):
            honmei_ctx.append(c)

by_day = defaultdict(list)
for c in honmei_ctx:
    by_day[c["date"]].append(c)
honmei_sel = []
for d, cs in by_day.items():
    cs.sort(key=lambda c: c["top"])
    honmei_sel.extend(cs[:6])


def trio(a, b, x):
    s = sorted([a, b, x])
    return f"{s[0]}={s[1]}={s[2]}"


def bets_konsen(c, equal):
    r1, r2, r3, r4, r5 = c["lanes"][:5]
    w = (200, 200, 200, 200, 200) if equal else (300, 200, 200, 200, 100)
    return [("3連複", trio(r1, r2, r3), w[0]), ("3連複", trio(r1, r2, r4), w[1]),
            ("3連単", f"{r3}-{r1}-{r2}", w[2]), ("3連単", f"{r4}-{r1}-{r2}", w[3]),
            ("3連複", trio(r3, r4, r5), w[4])]


def bets_honmei(c, equal):
    r1, r2, r3, r4 = c["lanes"][:4]
    if equal:
        return [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
                ("3連複", trio(r1, r3, r4), 200), ("3連複", trio(r2, r3, r4), 200),
                ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200)]
    return [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
            ("3連複", trio(r1, r3, r4), 100), ("3連複", trio(r2, r3, r4), 100),
            ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200)]


for band, ctxs, fn in (("超混戦(全場)", konsen_ctx, bets_konsen),
                       ("本命(5場cap6)", honmei_sel, bets_honmei)):
    print(f"\n=== {band}: {len(ctxs):,}R ===")
    for equal, label in ((False, "傾斜(現行)"), (True, "均等200(フォーメーション形)")):
        st = rt = gami = plus = 0
        for c in ctxs:
            pay = payout_map[c["rid"]]
            bs = fn(c, equal)
            s = sum(y for _, _, y in bs)
            r = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y in bs)
            st += s
            rt += r
            if r and r < s:
                gami += 1
            elif r >= s and r:
                plus += 1
        print(f"  {label:<24} 投資/R{st//len(ctxs):>6,}円 回収率{rt/st:>7.1%} "
              f"ガミ率{gami/len(ctxs):>6.1%} プラス率{plus/len(ctxs):>6.1%} 損益{rt-st:+,}円")
