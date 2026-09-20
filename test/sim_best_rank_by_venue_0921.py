# -*- coding: utf-8 -*-
"""場ごとに「ドンピシャX位」のどれが一番勝てるか(2026-09-21ケンさん指示)

    py -X utf8 test/sim_best_rank_by_venue_0921.py

ドンピシャX位 = 本命帯台帳(全24場・2026-01〜04)で発生回数がX番目に多い並び(上位10種)。
1点500円の期待値 = 発生確率 × 的中時の平均払戻(500円分)。500円超でプラス。
2026-01〜08は月次walk-forward台帳、09月は本番配信picks。
① 場×X位の期待値一覧と、場ごとの最良X位(全期間=後出しの参考値)
② 後出しなしの確認: 1〜4月の期待値で場ごとに最良X位を決め、5〜9月にその1点を買う
   比較: 全場ドンピシャ1位固定 / 場ごとの最良X位 / 最良X位が1〜4月に黒字の場だけ
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
arr = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id=res.race_id WHERE r.date>='2026-09-01' "
        "AND res.arrival_order IS NOT NULL"):
    arr[rid][lane] = ao
conn.close()

freq = Counter()
races = []   # (date, venue, p1, 結果の並び(予想順位), 100円払戻)
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    res = [int(x) for x in r["結果(3連単)"].split("-")]
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    if len(pred) < 6 or len(res) < 3 or rid not in pay3:
        continue
    pt = tuple(pred.index(l) + 1 for l in res)
    if r["日付"] < "2026-05-01":
        freq[pt] += 1
    races.append((r["日付"], VEN[r["場"]], float(r["モデル1位勝率"].rstrip("%")), pt,
                  pay3[rid][1]))
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        rid = r["race_id"]
        a = arr.get(rid)
        if not r.get("ranked") or len(r["ranked"]) < 6 or rid not in pay3 \
                or not a or len(a) < 3:
            continue
        pred = [x[0] for x in r["ranked"]]
        top3 = sorted(a, key=a.get)[:3]
        races.append((pk["date"], r["venue_code"], r["ranked"][0][1] * 100,
                      tuple(pred.index(l) + 1 for l in top3), pay3[rid][1]))
RANKS = [pt for pt, _ in freq.most_common(TOPN)]
print("ドンピシャX位の定義: " + " / ".join(
    f"{i}位=r{p[0]}-r{p[1]}-r{p[2]}" for i, p in enumerate(RANKS, 1)))


def ev(sub, pt):
    """(期待値円, 的中数)"""
    if not sub:
        return 0.0, 0
    hits = [x[4] for x in sub if x[3] == pt]
    return sum(hits) * YEN / 100 / len(sub), len(hits)


for title, lo, hi in (("(a) 1位勝率22.5〜30%", 22.5, 30), ("(b) 1位勝率20〜35%", 20, 35)):
    band = [x for x in races if lo <= x[2] < hi]
    print(f"\n===== {title}: 場×ドンピシャX位の期待値(1点{YEN}円・500円超でプラス) =====")
    print(f"{'場':<6}{'R数':>5}" + "".join(f"{str(i) + '位':>7}" for i in range(1, TOPN + 1))
          + f"{'最良':>6}{'期待値':>8}{'的中':>4}{'2番手':>6}")
    rows = []
    for v in NAME:
        s = [x for x in band if x[1] == v]
        if not s:
            continue
        evs = [ev(s, pt) for pt in RANKS]
        order = sorted(range(TOPN), key=lambda i: -evs[i][0])
        rows.append((evs[order[0]][0], v, len(s), evs, order))
    for best, v, n, evs, order in sorted(rows, reverse=True):
        print(f"{NAME[v]:<6}{n:>5}" + "".join(f"{e:>7,.0f}" for e, _h in evs)
              + f"{order[0] + 1:>5}位{best:>7,.0f}円{evs[order[0]][1]:>4}{order[1] + 1:>5}位")
    tot = [ev(band, pt) for pt in RANKS]
    print(f"{'全場計':<6}{len(band):>5}" + "".join(f"{e:>7,.0f}" for e, _h in tot))

    # ② 後出しなし: 1〜4月で決めて5〜9月に買う
    tr = [x for x in band if x[0] < "2026-05-01"]
    te = [x for x in band if x[0] >= "2026-05-01"]
    choice, plus = {}, set()
    for v in NAME:
        s = [x for x in tr if x[1] == v]
        if not s:
            continue
        evs = [ev(s, pt)[0] for pt in RANKS]
        i = max(range(TOPN), key=lambda k: evs[k])
        choice[v] = i
        if evs[i] > YEN:
            plus.add(v)
    print(f"\n― ② 1〜4月で場ごとの最良X位を決め、5〜9月に1点{YEN}円({len(te):,}R) ―")
    same = sum(1 for v in choice
               if choice[v] == max(range(TOPN), key=lambda k: ev(
                   [x for x in te if x[1] == v], RANKS[k])[0]))
    print(f"  1〜4月の最良X位が5〜9月も最良だった場: {same}/{len(choice)}場")
    for label, pick in (("全場ドンピシャ1位で固定", lambda v: 0),
                        ("場ごとの最良X位", lambda v: choice.get(v)),
                        ("最良X位が1〜4月に黒字の場だけ", lambda v: choice[v] if v in plus else None)):
        n = h = rt = best = 0
        for _d, v, _p, pt, amt in te:
            i = pick(v)
            if i is None:
                continue
            n += 1
            if pt == RANKS[i]:
                g = amt * YEN // 100
                h += 1
                rt += g
                best = max(best, g)
        st = n * YEN
        print(f"  {label:<18}{n:>5}R 的中{h:>3} 回収率{rt / max(1, st):>7.1%} "
              f"損益{rt - st:>+10,}円 最大1本除く{(rt - best) / max(1, st):>7.1%}")
