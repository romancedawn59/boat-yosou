# -*- coding: utf-8 -*-
"""3連単1点買い(500円)を並び別に試算(2026-09-21ケンさん指示)
ドンピシャ(r1-r2-r3)の次に来やすい並び=ドンピシャ2位・3位…を同じ計算方式で。

    py -X utf8 test/sim_single_point_ranks_0921.py

選別 = 対象5場 × 1位勝率22.5〜30% × 勝率の低い順 × 1日4R(sim_honmei_kaikata_0921.pyと同じ)。
並びの順位 = 本命帯の台帳(全24場・2026-01〜04)での発生回数の多い順(結果を見る前に固定)。
"""
import csv
import glob
import json
import sqlite3
from collections import Counter, defaultdict

LEDGER = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
          r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad\honmei_results_2026.csv")
VEN = {"桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
       "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
       "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
       "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24}
FIVE = {3, 4, 8, 13, 20}
YEN = 500
TOPN = 10

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay3 = {}
for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-01-01' "
        "AND p.bet_type='3連単'"):
    pay3[rid] = (comb, amt or 0)
conn.close()

freq = Counter()
pool = defaultdict(list)
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    res = [int(x) for x in r["結果(3連単)"].split("-")]
    if len(pred) < 6 or len(res) < 3:
        continue
    if r["日付"] < "2026-05-01":
        freq[tuple(pred.index(l) + 1 for l in res)] += 1
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    if r["対象5場"] and rid in pay3:
        pool[r["日付"]].append((float(r["モデル1位勝率"].rstrip("%")), pred, rid))
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if r.get("ranked") and len(r["ranked"]) >= 6 and r["venue_code"] in FIVE \
                and r["race_id"] in pay3:
            pool[pk["date"]].append((r["ranked"][0][1] * 100,
                                     [x[0] for x in r["ranked"]], r["race_id"]))

sel = []
for d, rs in pool.items():
    for _p1, pred, rid in sorted(x for x in rs if 22.5 <= x[0] < 30)[:4]:
        sel.append((d, pred, rid))
PERIODS = [("全期間", "2026-01", "2026-99"), ("1〜4月", "2026-01", "2026-04"),
           ("5〜8月", "2026-05", "2026-08"), ("9月", "2026-09", "2026-09")]
print(f"選別レース: {len(sel)}R / 1点{YEN}円")
print(f"{'順位':<4}{'並び':<10}{'的中':>5}{'的中率':>7}{'平均配当':>9}"
      + "".join(f"{p[0]:>9}" for p in PERIODS) + f"{'損益':>11}{'最大1本除く':>10}")
for i, (pt, _c) in enumerate(freq.most_common(TOPN), 1):
    agg = {p[0]: [0, 0] for p in PERIODS}
    hits = []
    for d, pred, rid in sel:
        comb, amt = pay3[rid]
        g = amt * YEN // 100 if comb == "-".join(str(pred[k - 1]) for k in pt) else 0
        if g:
            hits.append(amt)
        for pn, a, b in PERIODS:
            if a <= d[:7] <= b:
                agg[pn][0] += YEN
                agg[pn][1] += g
    st, rt = agg["全期間"]
    best = max(hits, default=0) * YEN // 100
    cells = "".join(f"{(v[1] / v[0] if v[0] else 0):>9.1%}" for v in agg.values())
    print(f"{i:<4}r{pt[0]}-r{pt[1]}-r{pt[2]:<4}{len(hits):>5}{len(hits) / len(sel):>7.1%}"
          f"{(sum(hits) // len(hits) if hits else 0):>8,}円{cells}{rt - st:>+10,}円"
          f"{(rt - best) / st:>10.1%}")
