# -*- coding: utf-8 -*-
"""超混戦: ケンさんの「トリオ単ボックス1,400円」案の試算(2026-08-01)

    py -X utf8 test/verify_konsen_trio_box_plan.py

■ 案(ケンさん)
  Aトリオ{1,2,3位}の3連単ボックス6点×100円 = 600円
  Bトリオ{1,2,4位}の3連単ボックス6点×100円 = 600円
  G複 3=4=5位 200円
  計13点1,400円(複を持たず、トリオを単ボックスで丸ごと)

■ 比較(同一レース・月次学習8か月)
  案1×2(現行2,000円) / 家族拡張(2,400円) / トリオ単ボックス(1,400円)
■ 予想される論点: ボックスは各トリオの高価値マス(E/F/確率上位)と
  低価値マス(残り3並び)を等額で買う。複 vs 単ボックスは堅め帯で複優位が実証済み。
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

agg = defaultdict(lambda: [0, 0, 0, 0])   # {(arm, month): [st, rt, R, hitR]}
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
        probs = P.normalize_probs(ranked)
        tri = P.trifecta_probs(probs)

        def trio(a, b, c):
            s = sorted([a, b, c])
            return f"{s[0]}={s[1]}={s[2]}"

        def next2(members, exclude):
            cands = sorted(((o, p) for o, p in tri.items()
                            if set(o) == set(members)), key=lambda x: -x[1])
            out = []
            for (a, b, c), _p in cands:
                cb = f"{a}-{b}-{c}"
                if cb != exclude:
                    out.append(cb)
                if len(out) == 2:
                    break
            return out

        base = [("3連複", trio(r1, r2, r3), 600), ("3連複", trio(r1, r2, r4), 400),
                ("3連単", f"{r3}-{r1}-{r2}", 400), ("3連単", f"{r4}-{r1}-{r2}", 400),
                ("3連複", trio(r3, r4, r5), 200)]
        ext = base + [("3連単", cb, 100)
                      for cb in next2((r1, r2, r3), f"{r3}-{r1}-{r2}")
                      + next2((r1, r2, r4), f"{r4}-{r1}-{r2}")]
        box = ([("3連単", f"{a}-{b}-{c}", 100)
                for a, b, c in permutations((r1, r2, r3))]
               + [("3連単", f"{a}-{b}-{c}", 100)
                  for a, b, c in permutations((r1, r2, r4))]
               + [("3連複", trio(r3, r4, r5), 200)])
        pay = payout_map[rid]
        for arm, bets in (("案1×2(2,000円)", base), ("家族拡張(2,400円)", ext),
                          ("トリオ単BOX(1,400円)", box)):
            st = sum(y for _, _, y in bets)
            rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y in bets)
            a = agg[(arm, m)]
            a[0] += st
            a[1] += rt
            a[2] += 1
            a[3] += 1 if rt else 0

ARMS = ("案1×2(2,000円)", "家族拡張(2,400円)", "トリオ単BOX(1,400円)")
print(f"\n超混戦帯: {n:,}R")
print(f"\n{'月':<9}" + "".join(f"{a:<18}" for a in ARMS))
for m in MONTHS:
    row = f"{m:<9}"
    for arm in ARMS:
        st, rt, _r, _h = agg[(arm, m)]
        row += f"{(rt/st if st else 0):>10.1%}        "
    print(row)
print("\n===== 合計 =====")
for arm in ARMS:
    st = sum(agg[(arm, m)][0] for m in MONTHS)
    rt = sum(agg[(arm, m)][1] for m in MONTHS)
    hr = sum(agg[(arm, m)][3] for m in MONTHS)
    print(f"  {arm:<18} 投資{st:>11,}円 回収率{rt/st:>7.1%} "
          f"損益{rt-st:>+11,}円 何か当たる率{hr/n:.1%}")
