# -*- coding: utf-8 -*-
"""超混戦: トリオ単BOX(1,400円)+600円の使い道 三択シミュレーション(2026-08-01)

    py -X utf8 test/verify_konsen_box_plus600.py

■ ケンさんの三択(ベース=⑫: A単BOX600+B単BOX600+G複200)
  ① A単BOXを1,200円に倍厚(各200円)
  ② {2,3,4位}トリオ単BOX600円を追加
  ③ おすすめ枠: E/F差され単に各+300円(ボックスに差され傾斜を戻す。
     E/Fは実測全マス中の最高値帯のため理論本命)
比較: ⑫ベース1,400円 / ①②③各2,000円 / 現行(案1×2)2,000円
月次学習8か月(2025-12〜2026-07)・超混戦帯(1位生値20%未満)。
"""
import sys
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07"]

conn = db.connect(DB_PATH)
df = build_training_set(conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

agg = defaultdict(lambda: [0, 0])   # {(arm, month): [st, rt]}
n = 0
for m in MONTHS:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, g in month_df.groupby("race_id"):
        if not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5 or ranked[0]["prob"] >= 0.20:
            continue
        n += 1
        lanes = [r["lane"] for r in ranked]
        r1, r2, r3, r4, r5 = lanes[:5]

        def trio(a, b, c):
            s = sorted([a, b, c])
            return f"{s[0]}={s[1]}={s[2]}"

        def box(members, yen):
            return [("3連単", f"{a}-{b}-{c}", yen)
                    for a, b, c in permutations(members)]
        gfuku = [("3連複", trio(r3, r4, r5), 200)]
        boxA = box((r1, r2, r3), 100)
        boxB = box((r1, r2, r4), 100)
        ef300 = [("3連単", f"{r3}-{r1}-{r2}", 300),
                 ("3連単", f"{r4}-{r1}-{r2}", 300)]
        arms = {
            "⑫BOXベース(1,400円)": boxA + boxB + gfuku,
            "①A-BOX倍厚(2,000円)": box((r1, r2, r3), 200) + boxB + gfuku,
            "②+234位BOX(2,000円)": boxA + boxB + box((r2, r3, r4), 100) + gfuku,
            "③+E/F各300(2,000円)": boxA + boxB + ef300 + gfuku,
            "現行 案1×2(2,000円)": [
                ("3連複", trio(r1, r2, r3), 600), ("3連複", trio(r1, r2, r4), 400),
                ("3連単", f"{r3}-{r1}-{r2}", 400), ("3連単", f"{r4}-{r1}-{r2}", 400),
                ("3連複", trio(r3, r4, r5), 200)],
        }
        pay = payout_map[rid]
        for arm, bets in arms.items():
            merged = defaultdict(int)
            for bt, comb, y in bets:
                merged[(bt, comb)] += y
            st = sum(merged.values())
            rt = sum(pay.get(k, 0) * y // 100 for k, y in merged.items())
            a = agg[(arm, m)]
            a[0] += st
            a[1] += rt

ARMS = ("⑫BOXベース(1,400円)", "①A-BOX倍厚(2,000円)", "②+234位BOX(2,000円)",
        "③+E/F各300(2,000円)", "現行 案1×2(2,000円)")
print(f"\n超混戦帯: {n:,}R")
print(f"\n{'月':<9}" + "".join(f"{a[:9]:<12}" for a in ARMS))
for m in MONTHS:
    row = f"{m:<9}"
    for arm in ARMS:
        st, rt = agg[(arm, m)]
        row += f"{(rt/st if st else 0):>9.1%}   "
    print(row)
print("\n===== 合計 =====")
for arm in ARMS:
    st = sum(agg[(arm, m)][0] for m in MONTHS)
    rt = sum(agg[(arm, m)][1] for m in MONTHS)
    lows = min((agg[(arm, m)][1] / agg[(arm, m)][0])
               for m in MONTHS if agg[(arm, m)][0])
    print(f"  {arm:<20} 投資{st:>11,}円 回収率{rt/st:>7.1%} "
          f"損益{rt-st:>+11,}円 最低月{lows:>7.1%}")
