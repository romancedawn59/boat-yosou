# -*- coding: utf-8 -*-
"""3連単1点買い(500円)を対象外の場まで広げた試算(2026-09-21ケンさん指示)

    py -X utf8 test/sim_single_point_allvenues_0921.py

帯 = 1位勝率22.5〜30%。2026-01〜08は月次walk-forward台帳、09月は本番配信picks。
① 範囲別(対象5場/他19場/全24場・帯の全レース)に並び上位5種の1点買い
② 全24場で1日の上限をかけた場合(勝率の低い順)のドンピシャ
③ 場ごとのドンピシャ(前半1〜4月/後半5〜9月)
並びの順位は本命帯台帳(全24場・2026-01〜04)の発生回数順で固定。
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
NAME = {v: k for k, v in VEN.items()}
FIVE = {3, 4, 8, 13, 20}
YEN = 500

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
races = []   # (date, venue, p1, pred, rid)
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    res = [int(x) for x in r["結果(3連単)"].split("-")]
    if len(pred) < 6 or len(res) < 3:
        continue
    if r["日付"] < "2026-05-01":
        freq[tuple(pred.index(l) + 1 for l in res)] += 1
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    if rid in pay3:
        races.append((r["日付"], VEN[r["場"]], float(r["モデル1位勝率"].rstrip("%")),
                      pred, rid))
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if r.get("ranked") and len(r["ranked"]) >= 6 and r["race_id"] in pay3:
            races.append((pk["date"], r["venue_code"], r["ranked"][0][1] * 100,
                          [x[0] for x in r["ranked"]], r["race_id"]))
band = [x for x in races if 22.5 <= x[2] < 30]
TOP = [pt for pt, _ in freq.most_common(5)]
PERIODS = [("全期間", "2026-01", "2026-99"), ("1〜4月", "2026-01", "2026-04"),
           ("5〜8月", "2026-05", "2026-08"), ("9月", "2026-09", "2026-09")]


def score(sub, pt):
    agg = {p[0]: [0, 0] for p in PERIODS}
    hits = []
    for d, _v, _p, pred, rid in sub:
        comb, amt = pay3[rid]
        ok = comb == "-".join(str(pred[k - 1]) for k in pt)
        if ok:
            hits.append(amt)
        for pn, a, b in PERIODS:
            if a <= d[:7] <= b:
                agg[pn][0] += YEN
                agg[pn][1] += amt * YEN // 100 if ok else 0
    return agg, hits


def line(label, sub, pt):
    agg, hits = score(sub, pt)
    st, rt = agg["全期間"]
    best = max(hits, default=0) * YEN // 100
    cells = "".join(f"{(v[1] / v[0] if v[0] else 0):>9.1%}" for v in agg.values())
    print(f"{label:<16}{len(sub):>6}{len(hits):>5}{len(hits) / max(1, len(sub)):>7.1%}"
          f"{(sum(hits) // len(hits) if hits else 0):>8,}円{cells}{rt - st:>+11,}円"
          f"{(rt - best) / max(1, st):>10.1%}")


HEAD = (f"{'':<16}{'R数':>6}{'的中':>5}{'的中率':>7}{'平均配当':>9}"
        + "".join(f"{p[0]:>9}" for p in PERIODS) + f"{'損益':>12}{'最大1本除く':>10}")

print("===== ① 範囲別・帯の全レース・1点500円 =====")
for sname, flt in (("対象5場", lambda v: v in FIVE), ("他19場", lambda v: v not in FIVE),
                   ("全24場", lambda v: True)):
    sub = [x for x in band if flt(x[1])]
    print(f"\n― {sname} ―\n{HEAD}")
    for i, pt in enumerate(TOP, 1):
        line(f"{i}位 r{pt[0]}-r{pt[1]}-r{pt[2]}", sub, pt)

print(f"\n===== ② 全24場・勝率の低い順に1日の上限をかけたドンピシャ =====\n{HEAD}")
by_day = defaultdict(list)
for x in band:
    by_day[x[0]].append(x)
for cap in (4, 8, 16):
    sub = [x for rs in by_day.values() for x in sorted(rs, key=lambda y: y[2])[:cap]]
    line(f"1日{cap}Rまで", sub, (1, 2, 3))

print("\n===== ③ 場ごとのドンピシャ(1点500円) =====")
print(f"{'場':<8}{'R数':>5}{'的中':>5}{'的中率':>7}{'平均配当':>9}{'全期間':>9}{'1〜4月':>9}{'5〜9月':>9}{'損益':>11}")
rows = []
for v in sorted(NAME):
    sub = [x for x in band if x[1] == v]
    if not sub:
        continue
    agg, hits = score(sub, (1, 2, 3))
    late = [agg["5〜8月"][0] + agg["9月"][0], agg["5〜8月"][1] + agg["9月"][1]]
    avg = sum(hits) // len(hits) if hits else 0
    rows.append((agg["全期間"][1] / agg["全期間"][0], v, len(sub), len(hits), agg, late, avg))
for roi, v, n, h, agg, late, avg in sorted(rows, reverse=True):
    e = agg["1〜4月"]
    mark = "★" if v in FIVE else ""
    print(f"{NAME[v] + mark:<8}{n:>5}{h:>5}{h / n:>7.1%}{avg:>8,}円{roi:>9.1%}{(e[1] / e[0] if e[0] else 0):>9.1%}"
          f"{(late[1] / late[0] if late[0] else 0):>9.1%}"
          f"{agg['全期間'][1] - agg['全期間'][0]:>+10,}円")
print("(★=対象5場)")
