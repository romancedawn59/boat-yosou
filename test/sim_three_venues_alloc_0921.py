# -*- coding: utf-8 -*-
"""若松・尼崎・丸亀の本命レースにドンピシャ1位・2位・3位の3連単3点買い: 予算配分の試算
(2026-09-21ケンさん指示)

    py -X utf8 test/sim_three_venues_alloc_0921.py

1位=r1-r2-r3 / 2位=r1-r2-r4 / 3位=r1-r3-r2。2026-01〜08は月次walk-forward台帳、09月は本番配信。
帯 (a) 1位勝率22.5〜30%(今の本命帯) / (b) 20〜35%(標本を増やした参考)。
配分は結果を見る前に決めた型を並べる(100円単位・1レース500〜1,500円)。
注意: 3場は今日の表を見て選んだ場なので、数字は後出しの上限値として読むこと。
"""
import csv
import glob
import json
import sqlite3
from collections import defaultdict

LEDGER = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
          r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad\honmei_results_2026.csv")
VEN = {"桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
       "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
       "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
       "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24}
NAME = {v: k for k, v in VEN.items()}
THREE = (20, 13, 15)
TICKETS = [(1, 2, 3), (1, 2, 4), (1, 3, 2)]
ALLOCS = [("500円 300/100/100", (300, 100, 100)), ("500円 500/0/0", (500, 0, 0)),
          ("1,000円 均等寄り 400/300/300", (400, 300, 300)),
          ("1,000円 500/300/200", (500, 300, 200)), ("1,000円 600/200/200", (600, 200, 200)),
          ("1,000円 800/100/100", (800, 100, 100)), ("1,000円 1000/0/0", (1000, 0, 0)),
          ("1,500円 900/300/300", (900, 300, 300)), ("1,500円 700/400/400", (700, 400, 400))]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay3 = {}
for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-01-01' "
        "AND p.bet_type='3連単'"):
    pay3[rid] = (comb, amt or 0)
conn.close()

races = []   # (date, venue, p1, [各券の100円払戻])
def add(d, v, p1, pred, rid):
    if v in THREE and len(pred) >= 4 and rid in pay3:
        comb, amt = pay3[rid]
        races.append((d, v, p1, [amt if comb == "-".join(str(pred[k - 1]) for k in t) else 0
                                 for t in TICKETS]))


for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    add(r["日付"], VEN[r["場"]], float(r["モデル1位勝率"].rstrip("%")),
        [int(x) for x in r["予想(1位→6位)"].split("-")], rid)
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if r.get("ranked"):
            add(pk["date"], r["venue_code"], r["ranked"][0][1] * 100,
                [x[0] for x in r["ranked"]], r["race_id"])

for title, lo, hi in (("(a) 1位勝率22.5〜30%", 22.5, 30), ("(b) 1位勝率20〜35%", 20, 35)):
    band = [x for x in races if lo <= x[2] < hi]
    print(f"\n===== {title}: 3場計{len(band)}R =====")
    print("― 券ごとの実力(100円あたり) ―")
    print(f"{'範囲':<8}{'R数':>5}" + "".join(f"{'  ' + str(i) + '位 的中/回収率':>18}" for i in (1, 2, 3))
          + f"{'3点どれか的中':>12}")
    for label, flt in [("3場計", lambda v: True)] + [(NAME[v], (lambda vv: lambda v: v == vv)(v))
                                                  for v in THREE]:
        s = [x for x in band if flt(x[1])]
        cells = "".join(f"{sum(1 for x in s if x[3][i]):>9}本 {sum(x[3][i] for x in s) / (len(s) * 100):>6.0%}"
                        for i in range(3))
        anyhit = sum(1 for x in s if any(x[3])) / len(s)
        print(f"{label:<8}{len(s):>5}{cells}{anyhit:>12.1%}")
    print("― 配分別(3場計) ―")
    print(f"{'配分 1位/2位/3位':<26}{'回収率':>8}{'損益':>11}{'最大1本除く':>10}{'1〜4月':>8}{'5〜9月':>8}"
          + "".join(f"{NAME[v]:>8}" for v in THREE))
    for name, al in ALLOCS:
        def roi(s):
            st = len(s) * sum(al)
            return sum(x[3][i] * al[i] // 100 for x in s for i in range(3)) / st if st else 0
        gains = [sum(x[3][i] * al[i] // 100 for i in range(3)) for x in band]
        st = len(band) * sum(al)
        h1 = [x for x in band if x[0] < "2026-05-01"]
        h2 = [x for x in band if x[0] >= "2026-05-01"]
        print(f"{name:<26}{sum(gains) / st:>8.1%}{sum(gains) - st:>+10,}円"
              f"{(sum(gains) - max(gains)) / st:>10.1%}{roi(h1):>8.0%}{roi(h2):>8.0%}"
              + "".join(f"{roi([x for x in band if x[1] == v]):>8.0%}" for v in THREE))
