# -*- coding: utf-8 -*-
"""本命レースの買い方 再試算(2026-09-21ケンさん指示)
条件: 1レース500〜1,500円 / 1日8,000円まで / 1位勝率20〜22.5%は買わない

    py -X utf8 test/sim_honmei_kaikata_0921.py

選別 = 対象5場 × 1位勝率の帯 × 勝率の低い順 × 1日上限R。
2026-01〜08は月次walk-forward台帳、09月は本番配信picks(帯に入る5場の全レース)。
構成はどれも結果を見る前に固定。1〜4月/5〜8月/9月で採点。
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
FIVE = {3, 4, 8, 13, 20}
DAY_LIMIT = 8000

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-01-01' "
        "AND p.bet_type IN ('3連単','3連複')"):
    pay[rid][(bt, comb)] = amt or 0
conn.close()


def T(l, *ix):
    s = sorted(l[i - 1] for i in ix)
    return ("3連複", f"{s[0]}={s[1]}={s[2]}")


def S(l, a, b, c):
    return ("3連単", f"{l[a - 1]}-{l[b - 1]}-{l[c - 1]}")


PLANS = {
    "現行6行 1,000円": lambda l: [(*T(l, 1, 2, 3), 200), (*T(l, 2, 3, 4), 100),
                               (*S(l, 1, 2, 3), 100), (*S(l, 3, 1, 2), 200),
                               (*S(l, 4, 1, 2), 200), (*S(l, 4, 2, 1), 200)],
    "中間案 1,000円": lambda l: [(*T(l, 1, 2, 3), 100), (*T(l, 2, 3, 4), 100),
                              (*S(l, 1, 2, 3), 300), (*S(l, 3, 1, 2), 200),
                              (*S(l, 4, 1, 2), 100), (*S(l, 4, 2, 1), 200)],
    "ドンピシャ1点 500円": lambda l: [(*S(l, 1, 2, 3), 500)],
    "ドンピシャ300+複200 500円": lambda l: [(*S(l, 1, 2, 3), 300), (*T(l, 1, 2, 3), 200)],
    "ドンピシャ500+複500 1,000円": lambda l: [(*S(l, 1, 2, 3), 500), (*T(l, 1, 2, 3), 500)],
    "順当寄せ 1,000円": lambda l: [(*S(l, 1, 2, 3), 300), (*S(l, 1, 2, 4), 100),
                               (*S(l, 1, 3, 2), 100), (*S(l, 2, 1, 3), 100),
                               (*T(l, 1, 2, 3), 200), (*S(l, 4, 1, 2), 200)],
    "厚張り 1,500円": lambda l: [(*S(l, 1, 2, 3), 500), (*T(l, 1, 2, 3), 300),
                              (*S(l, 1, 2, 4), 100), (*S(l, 1, 3, 2), 100),
                              (*S(l, 3, 1, 2), 200), (*S(l, 4, 1, 2), 100),
                              (*S(l, 4, 2, 1), 200)],
}

pool = defaultdict(list)
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    if not r["対象5場"]:
        continue
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    if rid in pay and len(pred) >= 4:
        pool[r["日付"]].append((float(r["モデル1位勝率"].rstrip("%")), pred, rid))
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if r.get("ranked") and len(r["ranked"]) >= 4 and r["venue_code"] in FIVE \
                and r["race_id"] in pay:
            pool[pk["date"]].append((r["ranked"][0][1] * 100,
                                     [x[0] for x in r["ranked"]], r["race_id"]))

SELECTS = [("旧: 20〜30%・1日4R", 20, 30, 4), ("22.5〜30%・1日4R", 22.5, 30, 4),
           ("22.5〜30%・1日8R", 22.5, 30, 8), ("22.5〜27.5%・1日8R", 22.5, 27.5, 8)]
PERIODS = [("全期間", "2026-01-01", "2026-12-31"), ("1〜4月", "2026-01-01", "2026-04-30"),
           ("5〜8月", "2026-05-01", "2026-08-31"), ("9月", "2026-09-01", "2026-09-30")]

for sname, lo, hi, cap in SELECTS:
    print(f"\n===== 選別: {sname} =====")
    print(f"{'構成':<22}{'R数':>5}{'的中率':>7}" + "".join(f"{p[0]:>9}" for p in PERIODS)
          + f"{'損益(全期間)':>12}{'最大1本除く':>10}{'1日最大':>8}")
    for name, fn in PLANS.items():
        unit = sum(y for *_x, y in fn([1, 2, 3, 4, 5, 6]))
        n_day = min(cap, DAY_LIMIT // unit)
        agg = {p[0]: [0, 0] for p in PERIODS}
        n = hit = best = 0
        for d, rs in pool.items():
            cand = sorted(x for x in rs if lo <= x[0] < hi)[:n_day]
            for _p1, pred, rid in cand:
                bets = fn(pred)
                g = sum(pay[rid].get((bt, cb), 0) * y // 100 for bt, cb, y in bets)
                n += 1
                hit += g > 0
                best = max(best, g)
                for pn, a, b in PERIODS:
                    if a <= d <= b:
                        agg[pn][0] += unit
                        agg[pn][1] += g
        st, rt = agg["全期間"]
        cells = "".join(f"{(v[1] / v[0] if v[0] else 0):>9.1%}" for v in agg.values())
        print(f"{name:<22}{n:>5}{hit / max(1, n):>7.1%}{cells}{rt - st:>+11,}円"
              f"{(rt - best) / max(1, st):>10.1%}{n_day * unit:>7,}円")
