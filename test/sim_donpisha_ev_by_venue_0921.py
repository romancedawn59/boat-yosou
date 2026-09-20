# -*- coding: utf-8 -*-
"""場ごとのドンピシャ1点500円の期待値 = 発生確率 × 的中時の払戻(2026-09-21ケンさん指示)
「現行5場」の区別は付けず、全24場を同列に並べる。

    py -X utf8 test/sim_donpisha_ev_by_venue_0921.py [並び 例: 1-2-4]
    並び = 予想順位の並び。省略時は 1-2-3(ドンピシャ)。1-2-4 = ドンピシャ2位

2026-01〜08は月次walk-forward台帳、09月は本番配信picks。
帯は (a) 1位勝率22.5〜30% と (b) 20〜35%(標本を増やした版) の2通り。
期待値(円) = 発生確率 × 的中時の平均払戻(500円分)。1Rの期待損益 = 期待値 − 500円。
発生確率の幅 = ウィルソンの90%区間(標本の少なさによるブレの目安)。
"""
import csv
import glob
import json
import math
import sqlite3
import sys
from collections import defaultdict

LEDGER = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
          r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad\honmei_results_2026.csv")
VEN = {"桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
       "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
       "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
       "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24}
NAME = {v: k for k, v in VEN.items()}
YEN = 500
PT = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1-2-3").split("-")]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay3 = {}
for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-01-01' "
        "AND p.bet_type='3連単'"):
    pay3[rid] = (comb, amt or 0)
conn.close()

races = []   # (date, venue, p1, 的中時の100円払戻 or 0)
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    pred = r["予想(1位→6位)"].split("-")
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    if len(pred) >= max(PT) and rid in pay3:
        comb, amt = pay3[rid]
        races.append((r["日付"], VEN[r["場"]], float(r["モデル1位勝率"].rstrip("%")),
                      amt if comb == "-".join(pred[k - 1] for k in PT) else 0))
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if r.get("ranked") and len(r["ranked"]) >= max(PT) and r["race_id"] in pay3:
            comb, amt = pay3[r["race_id"]]
            mine = "-".join(str(r["ranked"][k - 1][0]) for k in PT)
            races.append((pk["date"], r["venue_code"], r["ranked"][0][1] * 100,
                          amt if comb == mine else 0))


def wilson(h, n, z=1.645):
    if not n:
        return 0, 0
    p = h / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return max(0, (c - w) / d), (c + w) / d


for title, lo, hi in (("(a) 1位勝率22.5〜30%", 22.5, 30), ("(b) 1位勝率20〜35%", 20, 35)):
    sub = [x for x in races if lo <= x[2] < hi]
    print(f"\n===== {title}・予想{"-".join(map(str, PT))}位の1点{YEN}円 / 全{len(sub):,}R =====")
    print(f"{'場':<6}{'R数':>5}{'的中':>4}{'発生確率':>8}{'(90%幅)':>14}{'的中時払戻':>10}"
          f"{'期待値':>8}{'1R損益':>8}{'通算損益':>11}{'1〜4月':>8}{'5〜9月':>8}")
    rows = []
    for v in NAME:
        s = [x for x in sub if x[1] == v]
        if not s:
            continue
        hits = [x[3] for x in s if x[3]]
        n, h = len(s), len(hits)
        avg = sum(hits) / h * YEN / 100 if h else 0
        ev = h / n * avg
        half = []
        for a, b in (("2026-01", "2026-04"), ("2026-05", "2026-09")):
            t = [x for x in s if a <= x[0][:7] <= b]
            half.append(sum(x[3] for x in t) / (len(t) * 100) if t else 0)
        rows.append((ev, v, n, h, avg, half))
    for ev, v, n, h, avg, half in sorted(rows, reverse=True):
        l, u = wilson(h, n)
        print(f"{NAME[v]:<6}{n:>5}{h:>4}{h / n:>8.1%}  ({l:>4.1%}〜{u:>5.1%}){avg:>9,.0f}円"
              f"{ev:>7,.0f}円{ev - YEN:>+7,.0f}円{(ev - YEN) * n:>+10,.0f}円"
              f"{half[0]:>8.0%}{half[1]:>8.0%}")
    n, hits = len(sub), [x[3] for x in sub if x[3]]
    avg = sum(hits) / len(hits) * YEN / 100
    print(f"{'全場計':<6}{n:>5}{len(hits):>4}{len(hits) / n:>8.1%}{'':>16}{avg:>9,.0f}円"
          f"{len(hits) / n * avg:>7,.0f}円{len(hits) / n * avg - YEN:>+7,.0f}円")
