# -*- coding: utf-8 -*-
"""事前宣言した買い方の紙上採点(2026-09-21ケンさん承認)

    py -X utf8 test/track_preregistered.py

test/preregistered.jsonl に宣言した買い方を、宣言日以降の本番配信picks JSONだけで採点する。
宣言の追加は1行足してコミットする(gitの履歴が「結果を見る前に宣言した」証拠になる)。
既存の行は書き換えない。
判定の目安: 50R以上で 回収率100%超かつ的中3本以上 → 昇格候補 / 50R以上で的中0〜1本 → 打ち切り。
"""
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation_kit import wilson  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
decls = [json.loads(x) for x in open(os.path.join(HERE, "preregistered.jsonl"), encoding="utf-8")
         if x.strip()]
start = min(d["declared"] for d in decls)

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>=?", (start,)):
    pay[rid][(bt, comb)] = max(pay[rid].get((bt, comb), 0), amt or 0)
fin = defaultdict(dict)
for rid, lane, ao, st in conn.execute(
        "SELECT e.race_id, e.lane, res.arrival_order, res.st_time FROM entries e "
        "JOIN races r ON r.race_id=e.race_id "
        "LEFT JOIN results res ON res.race_id=e.race_id AND res.lane=e.lane "
        "WHERE r.date>=?", (start,)):
    fin[rid][lane] = (ao, st)
conn.close()

agg = {d["name"]: [0, 0, 0, 0] for d in decls}   # R, 的中, 投資, 回収
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    day = os.path.basename(path)[6:16]
    if day < start:
        continue
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid, f = r["race_id"], fin.get(r["race_id"])
        if not r.get("ranked") or len(r["ranked"]) < 6 or rid not in pay or not f \
                or sum(1 for v in f.values() if v[0] is not None) < 3:
            continue
        lanes = [x[0] for x in r["ranked"]]
        refund = {l for l, (ao, st) in f.items() if ao is None and st is None}
        p1 = r["ranked"][0][1]
        for d in decls:
            if day < d["declared"] or not (d["band"][0] <= p1 < d["band"][1]):
                continue
            if d["venues"] != "all" and r["venue_code"] not in d["venues"]:
                continue
            pick = [lanes[k - 1] for k in d["pattern"]]
            comb = "-".join(map(str, pick))
            got = d["yen"] if set(pick) & refund else \
                pay[rid].get((d["ticket"], comb), 0) * d["yen"] // 100
            a = agg[d["name"]]
            a[0] += 1
            a[1] += got > d["yen"]
            a[2] += d["yen"]
            a[3] += got

print(f"事前宣言の紙上採点(宣言日 {start} 以降の本番配信だけ)")
print(f"{'宣言した買い方':<44}{'R数':>5}{'的中':>5}{'発生確率(90%幅)':>20}{'回収率':>8}{'損益':>10}  判定")
for d in decls:
    n, h, st, rt = agg[d["name"]]
    if not n:
        print(f"{d['name']:<44}{0:>5}  (対象レースなし)")
        continue
    lo, hi = wilson(h, n)
    verdict = ("昇格候補" if n >= 50 and rt > st and h >= 3
               else "打ち切り" if n >= 50 and h <= 1 else "観察中")
    print(f"{d['name']:<44}{n:>5}{h:>5}{h / n:>8.1%} ({lo:.1%}〜{hi:.1%}){rt / st:>8.1%}"
          f"{rt - st:>+9,}円  {verdict}")
