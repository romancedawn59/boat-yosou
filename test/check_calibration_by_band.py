# -*- coding: utf-8 -*-
"""確率の確かさの点検(帯ごと・月ごと)(2026-09-21ケンさん承認)

    py -X utf8 test/check_calibration_by_band.py

本番配信picks JSONの確率(全艇)を実際の1着と突き合わせる。
  ① 全艇: 確率の区間ごとに「出した確率の平均」と「実際の1着率」
  ② 予想1位: 1位勝率の帯ごとに同じ比較(買うレースを選ぶ数字なので重要)
  v2.1(〜8/31)とv2.2(9/1〜)を分けて出す。差が±2pt・z±2を超えたら補正を検討。
"""
import glob
import json
import math
import os
import sqlite3
from collections import defaultdict

BINS = [0, .05, .10, .15, .20, .225, .25, .275, .30, .35, .40, .50, .60, .75, 1.01]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
winner = {rid: lane for rid, lane in conn.execute(
    "SELECT res.race_id, res.lane FROM results res JOIN races r ON r.race_id=res.race_id "
    "WHERE r.date>='2026-07-21' AND res.arrival_order=1")}
conn.close()

agg = {k: defaultdict(lambda: [0, 0.0, 0]) for k in
       ("全艇 v2.1", "全艇 v2.2", "予想1位 v2.1", "予想1位 v2.2")}
ll = defaultdict(lambda: [0.0, 0])
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    d = os.path.basename(path)[6:16]
    if d < "2026-07-21":
        continue
    era = "v2.1" if d < "2026-09-01" else "v2.2"
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        w = winner.get(r["race_id"])
        if not r.get("ranked") or len(r["ranked"]) < 6 or w is None:
            continue
        for i, (lane, p) in enumerate(r["ranked"]):
            b = next(j for j in range(len(BINS) - 1) if BINS[j] <= p < BINS[j + 1])
            for key in ([f"全艇 {era}"] + ([f"予想1位 {era}"] if i == 0 else [])):
                a = agg[key][b]
                a[0] += 1
                a[1] += p
                a[2] += lane == w
            pp = min(max(p, 1e-6), 1 - 1e-6)
            ll[d[:7]][0] -= math.log(pp if lane == w else 1 - pp)
            ll[d[:7]][1] += 1

for key, bins in agg.items():
    print(f"\n===== {key} =====")
    print(f"{'確率の区間':<14}{'件数':>7}{'出した確率':>10}{'実際の1着率':>11}{'差':>8}{'z':>7}")
    for b in sorted(bins):
        n, sp, h = bins[b]
        if n < 30:
            continue
        p, o = sp / n, h / n
        z = (o - p) / math.sqrt(p * (1 - p) / n)
        flag = " ←要注意" if abs(z) >= 2 and abs(o - p) >= 0.02 else ""
        print(f"{BINS[b]:>5.1%}〜{BINS[b + 1]:>5.1%}{n:>8,}{p:>10.1%}{o:>11.1%}{(o - p) * 100:>+7.1f}pt{z:>7.2f}{flag}")
print("\n月別のlogloss(小さいほど良い): " + " / ".join(
    f"{m} {v[0] / v[1]:.4f}" for m, v in sorted(ll.items())))
