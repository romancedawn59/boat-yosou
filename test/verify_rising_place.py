# -*- coding: utf-8 -*-
"""伸び盛りの現金化場所は「絡み」か(2026-07-30ケンさん発案)

    py -X utf8 test/verify_rising_place.py

仮説(事前固定): 伸び盛り(+10pt超)の艇は1着(単勝=市場の得意分野)より
2-3着への食い込み(絡み=市場の弱点)で安売りされている。
複勝(2着以内)と3連複(その艇絡みの10点流し)の回収率で、
バケット間の差が単勝(横ばい64.6% vs 伸び盛り77.3%)より開くはず。
"""
import sqlite3
import datetime as dt
from bisect import bisect_left
from collections import defaultdict

DB = r"Y:\マイドライブ\boat\boat.db"
c = sqlite3.connect(DB)

races_date = dict(c.execute("SELECT race_id, date FROM races"))
lane_racer = {}
printed = {}
for rid, lane, reg, n2 in c.execute(
        "SELECT race_id, lane, reg_no, national_2rate FROM entries"):
    lane_racer[(rid, lane)] = reg
    printed[(rid, lane)] = n2

finish = defaultdict(dict)          # rid -> {order: lane}
rows = []
for rid, lane, ao in c.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    d = races_date.get(rid)
    reg = lane_racer.get((rid, lane))
    if d and reg:
        rows.append((d, rid, lane, reg, ao))
        finish[rid][ao] = lane
rows.sort()
hist = defaultdict(list)
for d, rid, lane, reg, ao in rows:
    hist[reg].append((d, 1 if ao <= 2 else 0))

fuku_pay = defaultdict(dict)        # 複勝: rid -> {lane_str: amt}
for rid, comb, amt in c.execute(
        "SELECT race_id, combination, amount_yen FROM payouts "
        "WHERE bet_type='複勝'"):
    fuku_pay[rid][comb] = amt or 0
trio_pay = {}                       # 3連複: rid -> (comb, amt)
for rid, comb, amt in c.execute(
        "SELECT race_id, combination, amount_yen FROM payouts "
        "WHERE bet_type='3連複'"):
    trio_pay[rid] = (comb, amt or 0)
c.close()


def gap_of(reg, before_date):
    h = hist[reg]
    dates = [x[0] for x in h]
    hi = bisect_left(dates, before_date)
    d0 = (dt.date.fromisoformat(before_date) -
          dt.timedelta(days=90)).isoformat()
    lo = bisect_left(dates, d0)
    seg = h[lo:hi]
    if len(seg) < 12:
        return None
    n2 = printed.get_last if False else None
    return sum(t for _, t in seg) / len(seg)


BUCKETS = [(-1.0, -0.10, "急落"), (-0.10, -0.03, "下降"),
           (-0.03, 0.03, "横ばい"), (0.03, 0.10, "上昇"),
           (0.10, 2.0, "★伸び盛り")]
agg = defaultdict(lambda: {"n": 0, "top2": 0, "top3": 0,
                           "st_f": 0, "rt_f": 0,
                           "st_t": 0, "rt_t": 0})

for d, rid, lane, reg, ao in rows:
    n2 = printed.get((rid, lane))
    if n2 is None:
        continue
    rr = gap_of(reg, d)
    if rr is None:
        continue
    gap = rr - n2 / 100.0
    lbl = next(l for lo, hi, l in BUCKETS if lo <= gap < hi)
    a = agg[lbl]
    a["n"] += 1
    if ao <= 2:
        a["top2"] += 1
    if ao <= 3:
        a["top3"] += 1
    # 複勝100円
    a["st_f"] += 100
    if ao <= 2:
        a["rt_f"] += fuku_pay.get(rid, {}).get(str(lane), 0)
    # 3連複・この艇絡み10点流し(各100円)
    a["st_t"] += 1000
    tp = trio_pay.get(rid)
    if tp and ao <= 3:
        comb, amt = tp
        if str(lane) in comb.split("="):
            a["rt_t"] += amt

print("=== 伸びギャップ×絡み空間(事前登録テスト) ===")
print(f"{'バケット':<12}{'n':>8}{'2連対率':>8}{'3連対率':>8}"
      f"{'複勝回収':>9}{'3連複流し回収':>12}")
for _lo, _hi, lbl in BUCKETS:
    a = agg[lbl]
    if not a["n"]:
        continue
    print(f"{lbl:<12}{a['n']:>8,}{a['top2']/a['n']:>8.1%}"
          f"{a['top3']/a['n']:>8.1%}{a['rt_f']/a['st_f']:>9.1%}"
          f"{a['rt_t']/a['st_t']:>12.1%}")
print("\n参考(昨日の単勝): 横ばい64.6% → 伸び盛り77.3%(差+12.7pt)")
