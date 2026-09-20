# -*- coding: utf-8 -*-
"""1位勝率20〜22.5%帯の紙上記録(2026-09-21から買い目報告に上げない帯)

    py -X utf8 test/track_band_2000_2250.py

本番配信picks JSONから、この帯のレースを月ごとに紙上採点する(各100円)。
  ドンピシャ r1-r2-r3 / 4位割り込み r1-r4-r3 / 1-2-4位の3連単BOX(6点) / 現行6行1,000円
判定の目安(事前登録): 変更後(2026-09-22〜)の対象5場で30R以上たまった時点で、
r1-r4-r3 か 1-2-4位BOX が回収率120%超かつ的中3本以上なら買い方として再検討。
ドンピシャが回収率100%超に戻っていたら除外そのものを見直す。
"""
import glob
import json
import sqlite3
from collections import defaultdict
from itertools import permutations

FIVE = {3, 4, 8, 13, 20}
CHANGE_DATE = "2026-09-22"

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-07-21' "
        "AND p.bet_type IN ('3連単','3連複')"):
    pay[rid][(bt, comb)] = amt or 0
conn.close()


def trio(*x):
    s = sorted(x)
    return ("3連複", f"{s[0]}={s[1]}={s[2]}")


def plans(l):
    r1, r2, r3, r4 = l[:4]
    return {
        "ドンピシャ": [("3連単", f"{r1}-{r2}-{r3}", 100)],
        "r1-r4-r3": [("3連単", f"{r1}-{r4}-{r3}", 100)],
        "1-2-4位BOX": [("3連単", f"{a}-{b}-{c}", 100)
                      for a, b, c in permutations((r1, r2, r4))],
        "現行6行": [(*trio(r1, r2, r3), 200), (*trio(r2, r3, r4), 100),
                  ("3連単", f"{r1}-{r2}-{r3}", 100), ("3連単", f"{r3}-{r1}-{r2}", 200),
                  ("3連単", f"{r4}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r2}-{r1}", 200)],
    }


agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # 区分→構成→[R,的中,投資,回収]
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if not r.get("ranked") or len(r["ranked"]) < 4 or r["race_id"] not in pay:
            continue
        if not (0.20 <= r["ranked"][0][1] < 0.225):
            continue
        scope = "対象5場" if r["venue_code"] in FIVE else "他19場"
        era = "変更後" if pk["date"] >= CHANGE_DATE else "変更前"
        for key in (f"{scope} {pk['date'][:7]}", f"{scope} {era}計"):
            for name, bets in plans([x[0] for x in r["ranked"]]).items():
                a = agg[key][name]
                g = sum(pay[r["race_id"]].get((bt, cb), 0) * y // 100
                        for bt, cb, y in bets)
                a[0] += 1
                a[1] += g > 0
                a[2] += sum(y for *_x, y in bets)
                a[3] += g

for key in sorted(agg):
    print(f"\n― {key} ―")
    for name, (n, h, st, rt) in agg[key].items():
        print(f"  {name:<12}{n:>4}R 的中{h:>3} 回収率{rt / st:>7.1%} 損益{rt - st:>+8,}円")
