# -*- coding: utf-8 -*-
"""9/1再判定: 超混戦⑬「BOX+差され傾斜」vs 紙上対抗案(2026-08の実戦レースで)

    py -X utf8 test/judge_0901_konsen.py

8月の配信済みpicks(docs/data/picks_2026-08-*.json)の超混戦レースに対し、
実際の払戻で各構成を採点する(ライブ紙上対決・学習不要)。
比較: ⑬現行(A/B単BOX+E/F傾斜+G複=2,000円) / 案1×2(複厚2,000円) /
      ⑫BOX素(1,400円) / 参考: 旧Q案×2(2,000円)
"""
import glob
import json
import sys
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
import db
from config import DB_PATH

conn = db.connect(DB_PATH)
payout = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date LIKE '2026-08%'"):
    payout[rid][(bt, comb)] = amt or 0
arrivals = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date LIKE '2026-08%' AND res.arrival_order IS NOT NULL"):
    arrivals[rid][ao] = lane
conn.close()


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plans(lanes):
    r1, r2, r3, r4, r5 = lanes[:5]
    abox = [f"{p[0]}-{p[1]}-{p[2]}" for p in permutations((r1, r2, r3))]
    bbox = [f"{p[0]}-{p[1]}-{p[2]}" for p in permutations((r1, r2, r4))]
    e, f = f"{r3}-{r1}-{r2}", f"{r4}-{r1}-{r2}"
    p13 = ([("3連単", c, 100) for c in abox] +
           [("3連単", c, 100) for c in bbox] +
           [("3連単", e, 300), ("3連単", f, 300),
            ("3連複", trio(r3, r4, r5), 200)])
    an1x2 = [("3連複", trio(r1, r2, r3), 600), ("3連複", trio(r1, r2, r4), 400),
             ("3連単", e, 400), ("3連単", f, 400),
             ("3連複", trio(r3, r4, r5), 200)]
    p12 = ([("3連単", c, 100) for c in abox] +
           [("3連単", c, 100) for c in bbox] +
           [("3連複", trio(r3, r4, r5), 200)])
    q2 = [("3連複", trio(r1, r2, r3), 400), ("3連複", trio(r1, r2, r4), 200),
          ("3連複", trio(r1, r3, r4), 200), ("3連複", trio(r2, r3, r4), 200),
          ("3連単", e, 400), ("3連単", f, 400),
          ("3連複", trio(r3, r4, r5), 200)]
    return {"⑬現行BOX+傾斜(2000)": p13, "案1×2複厚(2000)": an1x2,
            "⑫BOX素(1400)": p12, "旧Q案×2(2000)": q2}


agg = defaultdict(lambda: [0, 0, 0])
details = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-08-*.json")):
    p = json.load(open(path, encoding="utf-8"))
    for r in p["races"]:
        if r.get("shobusho") != "超混戦":
            continue
        rid = r["race_id"]
        if rid not in payout:
            continue
        lanes = [lane for lane, _prob in r["ranked"]]
        if len(lanes) < 5:
            continue
        a = arrivals.get(rid, {})
        res = f"{a.get(1)}-{a.get(2)}-{a.get(3)}"
        day = {}
        for name, plan in plans(lanes).items():
            st = sum(y for _, _, y in plan)
            rt = sum(payout[rid].get((bt, c), 0) * y // 100
                     for bt, c, y in plan)
            g = agg[name]
            g[0] += st
            g[1] += rt
            if rt:
                g[2] += 1
            day[name] = rt - st
        details.append((rid, res, day))

print(f"2026年8月・超混戦の実戦レース: {len(details)}R\n")
print(f"{'構成':<22}{'投資':>9}{'回収':>9}{'的中':>5}{'回収率':>8}{'損益':>10}")
for name, (st, rt, h) in agg.items():
    print(f"{name:<22}{st:>8,}円{rt:>8,}円{h:>5}{rt/st:>8.1%}{rt-st:>+9,}円")

print("\n--- レース明細(損益: ⑬ / 案1×2) ---")
for rid, res, day in details:
    d, vc, rno = rid.split("_")
    print(f"  {d[4:6]}/{d[6:]} 場{vc} {int(rno)}R 結果{res}  "
          f"⑬{day['⑬現行BOX+傾斜(2000)']:+,}円  案1×2{day['案1×2複厚(2000)']:+,}円")
