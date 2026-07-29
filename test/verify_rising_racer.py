# -*- coding: utf-8 -*-
"""伸び盛り選手仮説の検証(2026-07-29ケンさん発案)

    py -X utf8 test/verify_rising_racer.py

伸びギャップ = 直近90日の実測2連対率 − 番組表の全国2連率(市場のアンカー)。
①ギャップは当該レースの成績を予測するか(勢いの実在)
②ギャップ上位の選手の単勝を買うと市場価格を上回るか(市場の気づき遅れ)
級別も併記(B級で伸びギャップ大=「市場が気づく前」の候補)。
"""
import sqlite3
from bisect import bisect_left
from collections import defaultdict

DB = r"Y:\マイドライブ\boat\boat.db"
c = sqlite3.connect(DB)

# 選手ごとの全出走履歴(日付順)
hist = defaultdict(list)   # reg -> [(date, top2)]
races_date = {}
for rid, d in c.execute("SELECT race_id, date FROM races"):
    races_date[rid] = d
lane_racer = {}
printed = {}
klass = {}
for rid, lane, reg, n2, kl in c.execute(
        "SELECT race_id, lane, reg_no, national_2rate, racer_class FROM entries"):
    lane_racer[(rid, lane)] = reg
    printed[(rid, lane)] = n2
    klass[(rid, lane)] = kl

results = []
for rid, lane, ao in c.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    d = races_date.get(rid)
    reg = lane_racer.get((rid, lane))
    if d and reg:
        results.append((d, rid, lane, reg, ao))
results.sort()
for d, rid, lane, reg, ao in results:
    hist[reg].append((d, 1 if ao <= 2 else 0))

# 単勝払戻
win_pay = {}
for rid, comb, amt in c.execute(
        "SELECT race_id, combination, amount_yen FROM payouts "
        "WHERE bet_type='単勝'"):
    win_pay[(rid, comb)] = amt or 0
c.close()


def recent_rate(reg, before_date, days=90, min_n=12):
    h = hist[reg]
    dates = [x[0] for x in h]
    hi = bisect_left(dates, before_date)
    import datetime as dt
    d0 = (dt.date.fromisoformat(before_date) -
          dt.timedelta(days=days)).isoformat()
    lo = bisect_left(dates, d0)
    seg = h[lo:hi]
    if len(seg) < min_n:
        return None
    return sum(t for _, t in seg) / len(seg)


BUCKETS = [(-1.0, -0.10, "急落(-10pt以下)"), (-0.10, -0.03, "下降"),
           (-0.03, 0.03, "横ばい"), (0.03, 0.10, "上昇"),
           (0.10, 2.0, "★伸び盛り(+10pt超)")]
agg = defaultdict(lambda: {"n": 0, "win": 0, "top2": 0, "st": 0, "rt": 0})
agg_b = defaultdict(lambda: {"n": 0, "win": 0, "top2": 0, "st": 0, "rt": 0})

for d, rid, lane, reg, ao in results:
    n2 = printed.get((rid, lane))
    if n2 is None:
        continue
    rr = recent_rate(reg, d)
    if rr is None:
        continue
    gap = rr - n2 / 100.0
    lbl = next(l for lo, hi, l in BUCKETS if lo <= gap < hi)
    pay = win_pay.get((rid, str(lane)), 0) if ao == 1 else 0
    for target in ([agg[lbl]] +
                   ([agg_b[lbl]] if (klass.get((rid, lane)) or "").startswith("B")
                    else [])):
        target["n"] += 1
        target["st"] += 100
        target["rt"] += pay
        if ao == 1:
            target["win"] += 1
        if ao <= 2:
            target["top2"] += 1

print("=== 伸びギャップ(直近90日実測−番組表2連率)バケット別 ===")
print(f"{'バケット':<18}{'n':>8}{'勝率':>7}{'2連対率':>8}{'単勝回収率':>10}")
for _lo, _hi, lbl in BUCKETS:
    a = agg[lbl]
    if not a["n"]:
        continue
    print(f"{lbl:<18}{a['n']:>8,}{a['win']/a['n']:>7.1%}"
          f"{a['top2']/a['n']:>8.1%}{a['rt']/a['st']:>10.1%}")

print("\n=== 同・B級選手のみ(「市場が気づく前」候補) ===")
print(f"{'バケット':<18}{'n':>8}{'勝率':>7}{'2連対率':>8}{'単勝回収率':>10}")
for _lo, _hi, lbl in BUCKETS:
    a = agg_b[lbl]
    if not a["n"]:
        continue
    print(f"{lbl:<18}{a['n']:>8,}{a['win']/a['n']:>7.1%}"
          f"{a['top2']/a['n']:>8.1%}{a['rt']/a['st']:>10.1%}")
