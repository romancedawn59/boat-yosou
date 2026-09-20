# -*- coding: utf-8 -*-
"""ドンピシャ(予想1-2-3位そのまま)に厚く張る構成の検証
(2026-09-21ケンさん発案「ドンピシャ半分+3連複半分」「1〜3位の3連単BOX200円×6」
 「1位ガチガチの帯でこそドンピシャに厚く」)

    py -X utf8 test/verify_donpisha.py

■ ① 選別レースでの構成対決(2026-01〜08台帳+09月本番配信)
選別=対象5場×1位勝率20〜30%×勝率の低い順×1日4Rまで
(日次予算による減枠は超混戦3R以上の日だけなので、ここでは4R固定で近似)。
構成はどれも結果を見る前に固定。前半(1〜4月)/後半(5〜8月)/9月(本番)で採点。
■ ② 帯別のドンピシャ(本番配信2026-07-21〜09-19・全帯)
1位勝率の帯ごとに、ドンピシャの的中率と100円買いの回収率を測る。
■ 合格基準: 現行6行を全期間で上回り、かつ前半・後半・9月のうち2期間以上で
現行以上、かつ最大1本除外後も現行以上。
"""
import csv
import glob
import json
import sqlite3
from collections import defaultdict
from itertools import permutations

LEDGER = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
          r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad\honmei_results_2026.csv")
VEN = {"桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
       "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
       "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
       "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24}
FIVE = {3, 4, 8, 13, 20}

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-01-01' "
        "AND p.bet_type IN ('3連単','3連複')"):
    pay[rid][(bt, comb)] = amt or 0
arr = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id=res.race_id WHERE r.date>='2026-07-21' "
        "AND res.arrival_order IS NOT NULL"):
    arr[rid][lane] = ao
conn.close()


def T(l, *ix):
    s = sorted(l[i - 1] for i in ix)
    return ("3連複", f"{s[0]}={s[1]}={s[2]}")


def S(l, a, b, c):
    return ("3連単", f"{l[a - 1]}-{l[b - 1]}-{l[c - 1]}")


def box123(l, yen):
    return [("3連単", f"{a}-{b}-{c}", yen) for a, b, c in permutations(l[:3])]


PLANS = {
    "現行6行 1,000円": lambda l: [(*T(l, 1, 2, 3), 200), (*T(l, 2, 3, 4), 100),
                               (*S(l, 1, 2, 3), 100), (*S(l, 3, 1, 2), 200),
                               (*S(l, 4, 1, 2), 200), (*S(l, 4, 2, 1), 200)],
    "ケン案1 ドンピシャ500+複500": lambda l: [(*S(l, 1, 2, 3), 500), (*T(l, 1, 2, 3), 500)],
    "ケン案2 1〜3位BOX各200(1,200円)": lambda l: box123(l, 200),
    "折衷 ドンピシャ500+他5並び各100": lambda l: ([(*S(l, 1, 2, 3), 400)] + box123(l, 100)),
    "ドンピシャ1点 1,000円": lambda l: [(*S(l, 1, 2, 3), 1000)],
    "中間 ドンピシャ300(複100・4位頭各200→計1,000)": lambda l: [
        (*T(l, 1, 2, 3), 100), (*T(l, 2, 3, 4), 100), (*S(l, 1, 2, 3), 300),
        (*S(l, 3, 1, 2), 200), (*S(l, 4, 1, 2), 100), (*S(l, 4, 2, 1), 200)],
}

# --- 選別レース(台帳01〜08) + 9月本番 ---
pool = defaultdict(list)
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    p1 = float(r["モデル1位勝率"].rstrip("%"))
    if not r["対象5場"] or p1 >= 30:
        continue
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    if rid in pay and len(pred) >= 4:
        pool[r["日付"]].append((p1, pred, rid))
sel = []
for d, rs in pool.items():
    for p1, pred, rid in sorted(rs)[:4]:
        sel.append((d, pred, rid))
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if r.get("shobusho") == "本命" and r["ranked"][0][1] >= 0.20 \
                and r["race_id"] in pay:
            sel.append((pk["date"], [x[0] for x in r["ranked"]], r["race_id"]))

periods = [("全期間", "2026-01-01", "2026-12-31"), ("1〜4月", "2026-01-01", "2026-04-30"),
           ("5〜8月", "2026-05-01", "2026-08-31"), ("9月(本番配信)", "2026-09-01", "2026-09-30")]
print("===== ① 選別レースでの構成対決 =====")
for pname, lo, hi in periods:
    sub = [x for x in sel if lo <= x[0] <= hi]
    print(f"\n― {pname}: {len(sub)}R ―")
    print(f"{'構成':<26}{'投資':>10}{'回収率':>8}{'損益':>10}{'的中率':>7}{'ガミ':>5}{'最大1本除く':>10}")
    for name, fn in PLANS.items():
        st = rt = hit = gami = best = 0
        for _d, pred, rid in sub:
            bets = fn(pred)
            s = sum(y for *_x, y in bets)
            g = sum(pay[rid].get((bt, cb), 0) * y // 100 for bt, cb, y in bets)
            st += s
            rt += g
            hit += g > 0
            gami += 0 < g < s
            best = max(best, g)
        print(f"{name:<26}{st:>9,}円{rt / st:>8.1%}{rt - st:>+9,}円"
              f"{hit / len(sub):>7.1%}{gami:>5}{(rt - best) / st:>10.1%}")

# --- ② 帯別のドンピシャ(本番配信・全帯) ---
print("\n===== ② 1位勝率の帯別: ドンピシャの的中率と100円買いの回収率"
      "(本番配信7/21〜9/19) =====")
bands = [("超混戦 <20%", 0, .20), ("本命 20-30%", .20, .30), ("30-35%", .30, .35),
         ("標準 35-50%", .35, .50), ("堅め 50-65%", .50, .65),
         ("堅め 65-75%", .65, .75), ("ガチガチ 75%+", .75, 1.01)]
agg = {b[0]: {"all": [0, 0, 0], "five": [0, 0, 0]} for b in bands}
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-0[789]-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if not r.get("ranked") or len(r["ranked"]) < 3:
            continue
        rid = r["race_id"]
        a = arr.get(rid)
        if not a or len(a) < 3 or rid not in pay:
            continue
        lanes = [x[0] for x in r["ranked"]]
        comb = f"{lanes[0]}-{lanes[1]}-{lanes[2]}"
        got = pay[rid].get(("3連単", comb), 0)
        p1 = r["ranked"][0][1]
        for name, lo, hi in bands:
            if lo <= p1 < hi:
                for key in (["all", "five"] if r["venue_code"] in FIVE else ["all"]):
                    g = agg[name][key]
                    g[0] += 1
                    g[1] += got > 0
                    g[2] += got
print(f"{'帯':<16}{'全24場R':>8}{'的中率':>7}{'回収率':>8}{'的中時平均':>10} | "
      f"{'5場R':>6}{'的中率':>7}{'回収率':>8}")
for name, *_ in bands:
    a, f = agg[name]["all"], agg[name]["five"]
    if not a[0]:
        continue
    avg = a[2] // a[1] if a[1] else 0
    print(f"{name:<16}{a[0]:>8,}{a[1] / a[0]:>7.1%}{a[2] / (a[0] * 100):>8.1%}"
          f"{avg:>9,}円 | {f[0]:>6,}{(f[1] / f[0] if f[0] else 0):>7.1%}"
          f"{(f[2] / (f[0] * 100) if f[0] else 0):>8.1%}")
