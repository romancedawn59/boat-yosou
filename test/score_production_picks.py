# -*- coding: utf-8 -*-
"""本番配信の予想そのもので採点する(検証と本番を同じものにする・2026-09-21ケンさん承認)

    py -X utf8 test/score_production_picks.py [開始日 既定2026-07-21] [終了日]

採点に使う予想は docs/data/picks_YYYY-MM-DD.json(その朝に実際に配信した全レースの
確率と順位)だけ。後から作り直した予想は一切使わない。全24場同列・返還反映。
買い方: ドンピシャ1点500円 / 現行6行1,000円 / 中間案1,000円 / 複勝r1 500円
"""
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

START = sys.argv[1] if len(sys.argv) > 1 else "2026-07-21"
END = sys.argv[2] if len(sys.argv) > 2 else "2099-12-31"
BANDS = [("〜22.5%", 0, .225), ("22.5〜30%", .225, .30), ("30〜40%", .30, .40),
         ("40〜55%", .40, .55), ("55%〜", .55, 1.01)]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>=? "
        "AND p.bet_type IN ('3連単','3連複','複勝')", (START,)):
    pay[rid][(bt, comb)] = max(pay[rid].get((bt, comb), 0), amt or 0)
fin = defaultdict(dict)
for rid, lane, ao, st in conn.execute(
        "SELECT e.race_id, e.lane, res.arrival_order, res.st_time FROM entries e "
        "JOIN races r ON r.race_id=e.race_id "
        "LEFT JOIN results res ON res.race_id=e.race_id AND res.lane=e.lane "
        "WHERE r.date>=?", (START,)):
    fin[rid][lane] = (ao, st)
conn.close()


def trio(*x):
    s = sorted(x)
    return ("3連複", f"{s[0]}={s[1]}={s[2]}")


def plans(l):
    r1, r2, r3, r4 = l[:4]
    don = ("3連単", f"{r1}-{r2}-{r3}")
    return {
        "ドンピシャ1点500円": [(*don, 500)],
        "現行6行1,000円": [(*trio(r1, r2, r3), 200), (*trio(r2, r3, r4), 100), (*don, 100),
                        ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200),
                        ("3連単", f"{r4}-{r2}-{r1}", 200)],
        "中間案1,000円": [(*trio(r1, r2, r3), 100), (*trio(r2, r3, r4), 100), (*don, 300),
                       ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 100),
                       ("3連単", f"{r4}-{r2}-{r1}", 200)],
        "複勝r1 500円": [("複勝", str(r1), 500)],
    }


def lanes_of(bt, comb):
    return {int(x) for x in comb.replace("=", "-").split("-")}


agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))  # 区分→買い方→[R,払戻あり,投資,回収,最大]
n_races = 0
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    d = os.path.basename(path)[6:16]
    if not (START <= d <= END):
        continue
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        rid = r["race_id"]
        f = fin.get(rid)
        if not r.get("ranked") or len(r["ranked"]) < 4 or rid not in pay or not f \
                or sum(1 for v in f.values() if v[0] is not None) < 3:
            continue
        n_races += 1
        refund = {l for l, (ao, st) in f.items() if ao is None and st is None}
        p1 = r["ranked"][0][1]
        band = next(b[0] for b in BANDS if b[1] <= p1 < b[2])
        era = "v2.1(〜8/31)" if d < "2026-09-01" else "v2.2(9/1〜)"
        for name, bets in plans([x[0] for x in r["ranked"]]).items():
            st = sum(y for *_x, y in bets)
            g = sum(y if lanes_of(bt, cb) & refund else pay[rid].get((bt, cb), 0) * y // 100
                    for bt, cb, y in bets)
            for key in (f"帯 {band}", f"帯 {band} / {d[:7]}", f"全帯 / {era}"):
                a = agg[key][name]
                a[0] += 1
                a[1] += g > 0
                a[2] += st
                a[3] += g
                a[4] = max(a[4], g)

print(f"本番配信の予想で採点: {START}〜 / {n_races:,}R(全24場)")
for key in sorted(agg):
    print(f"\n― {key} ―")
    for name, (n, h, st, rt, best) in agg[key].items():
        print(f"  {name:<14}{n:>6,}R 払戻あり{h / n:>6.1%} 回収率{rt / st:>7.1%} "
              f"損益{rt - st:>+10,}円 最大1本除く{(rt - best) / st:>7.1%}")
