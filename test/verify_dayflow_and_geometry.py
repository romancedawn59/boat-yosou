# -*- coding: utf-8 -*-
"""案B 当日の水面傾向 / 案C 枠の位置関係 の検証
(2026-09-21ケンさん指示「今までから逸脱した全く新たな買い目予想アプローチ」)

    py -X utf8 test/verify_dayflow_and_geometry.py

■ 案B 当日の水面傾向で切り替える(昼に後半だけ買う方式は成り立つか)
同じ場・同じ日の前半(1〜6R)で1号艇が何回勝ったかを、その場の平常値と比べて
「イン弱い日/平常/イン強い日」に分ける。後半(7〜12R)の1号艇の勝率・
3連単配当・単勝1号艇の回収率が、前半の傾向を引き継ぐかを測る。
事前登録: 後半の1号艇勝率(場の平常値との差)が弱い日と強い日で5pt以上開き、
かつ どちらかの日の単純な買い方(単勝1号艇/1号艇を外した3連単BOX)が100%超。

■ 案C 枠の位置関係で相手を決める
勝った艇の「1つ外の枠(マーク位置)」「1つ内の枠」が2着に来る率を実測し、
予想順位を頭の候補にだけ使って相手を枠の位置で選ぶ構成を、
本命帯の台帳(全24場×1位勝率20〜30%・2026-01〜08)で現行6行と対決させる。
  幾何構成1,000円: 頭=予想1位と2位。各頭について2着=外隣と内隣の枠、
  3着=残りの予想最上位 → 最大4点を各200円 + ドンピシャ200円
事前登録: 全24場で現行6行を+5pt以上上回り、対象5場で100%超、
かつ前半(1〜4月)・後半(5〜8月)の両方で現行以上。
"""
import csv
import sqlite3
from collections import Counter, defaultdict
from itertools import permutations

DB = r"file:Y:\マイドライブ\boat\boat.db?mode=ro"
LEDGER = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
          r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad\honmei_results_2026.csv")
VEN = {"桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
       "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
       "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
       "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24}

conn = sqlite3.connect(DB, uri=True, timeout=120)
top3 = {}
arr = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id=res.race_id WHERE r.date>='2026-01-01' "
        "AND res.arrival_order IS NOT NULL"):
    arr[rid][lane] = ao
meta = {rid: (d, v, n) for rid, d, v, n in conn.execute(
    "SELECT race_id, date, venue_code, race_no FROM races WHERE date>='2026-01-01'")}
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-01-01' "
        "AND p.bet_type IN ('3連単','3連複','単勝')"):
    pay[rid][(bt, comb)] = amt or 0
conn.close()
for rid, a in arr.items():
    if len(a) >= 3:
        top3[rid] = sorted(a, key=a.get)[:3]

# ============ 案B ============
print("===== 案B 当日の水面傾向(前半1〜6R → 後半7〜12R) =====")
by_day = defaultdict(dict)
for rid, t in top3.items():
    d, v, n = meta[rid]
    by_day[(d, v)][n] = rid
venue_base = defaultdict(lambda: [0, 0])
for rid, t in top3.items():
    v = meta[rid][1]
    venue_base[v][0] += 1
    venue_base[v][1] += t[0] == 1
stat = defaultdict(lambda: defaultdict(float))
for (d, v), rs in by_day.items():
    front = [rs[n] for n in range(1, 7) if n in rs]
    back = [rs[n] for n in range(7, 13) if n in rs]
    if len(front) < 5 or len(back) < 4:
        continue
    base = venue_base[v][1] / venue_base[v][0]
    dev = sum(top3[r][0] == 1 for r in front) - base * len(front)
    bucket = ("イン弱い日" if dev <= -1.5 else "イン強い日" if dev >= 1.5
              else "平常")
    s = stat[bucket]
    s["days"] += 1
    for r in back:
        t = top3[r]
        comb = "-".join(map(str, t))
        s["n"] += 1
        s["inn_win"] += t[0] == 1
        s["inn_win_expect"] += base
        s["pay3"] += pay[r].get(("3連単", comb), 0)
        s["tan1"] += pay[r].get(("単勝", "1"), 0) if t[0] == 1 else 0
        box = {f"{a}-{b}-{c}" for a, b, c in permutations((2, 3, 4))}
        s["box234"] += pay[r].get(("3連単", comb), 0) if comb in box else 0
print(f"{'前半の傾向':<10}{'日数':>6}{'後半R':>7}{'後半の1号艇勝率':>13}{'場の平常値':>10}"
      f"{'差':>7}{'3連単平均配当':>12}{'単勝1号艇':>9}{'2-3-4BOX':>9}")
for b in ("イン弱い日", "平常", "イン強い日"):
    s = stat[b]
    n = s["n"]
    print(f"{b:<10}{s['days']:>6.0f}{n:>7.0f}{s['inn_win'] / n:>13.1%}"
          f"{s['inn_win_expect'] / n:>10.1%}"
          f"{(s['inn_win'] - s['inn_win_expect']) / n * 100:>+6.1f}pt"
          f"{s['pay3'] / n:>11,.0f}円{s['tan1'] / (n * 100):>9.1%}"
          f"{s['box234'] / (n * 600):>9.1%}")

# ============ 案C ============
print("\n===== 案C 枠の位置関係 =====")
print("全レース(2026年): 勝った艇の枠ごとに、2着がどの位置関係の艇だったか")
rel = defaultdict(Counter)
for rid, t in top3.items():
    w, s2 = t[0], t[1]
    rel[w]["n"] += 1
    rel[w]["外隣(w+1)"] += s2 == w + 1
    rel[w]["内隣(w-1)"] += s2 == w - 1
    rel[w]["1号艇"] += s2 == 1 and w != 1 and w != 2
print(f"{'勝者の枠':<8}{'回数':>7}{'2着=外隣':>9}{'2着=内隣':>9}{'2着=1号艇(隣以外)':>16}")
for w in range(1, 7):
    c = rel[w]
    n = c["n"]
    print(f"{w}号艇{'':<4}{n:>7,}{c['外隣(w+1)'] / n:>9.1%}{c['内隣(w-1)'] / n:>9.1%}"
          f"{c['1号艇'] / n:>16.1%}")


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def h1000(l):
    r1, r2, r3, r4 = l[:4]
    return [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r2, r3, r4), 100),
            ("3連単", f"{r1}-{r2}-{r3}", 100), ("3連単", f"{r3}-{r1}-{r2}", 200),
            ("3連単", f"{r4}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r2}-{r1}", 200)]


def geometry(l):
    bets = [("3連単", f"{l[0]}-{l[1]}-{l[2]}", 200)]
    for head in l[:2]:
        for nb in (head + 1, head - 1):
            if 1 <= nb <= 6:
                third = next(x for x in l if x not in (head, nb))
                bets.append(("3連単", f"{head}-{nb}-{third}", 200))
    return bets


def score(bets, p):
    m = defaultdict(int)
    for bt, comb, y in bets:
        m[(bt, comb)] += y
    return sum(m.values()), sum(p.get(k, 0) * y // 100 for k, y in m.items())


ledger = []
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    if float(r["モデル1位勝率"].rstrip("%")) >= 30:
        continue
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    if len(pred) < 6:
        continue
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    if rid in pay:
        ledger.append((r["日付"], bool(r["対象5場"]), pred, pay[rid]))
print(f"\n本命帯の台帳 {len(ledger):,}R(全24場×1位勝率20〜30%)で構成対決")
print(f"{'範囲':<16}{'構成':<12}{'投資':>11}{'回収率':>8}{'的中率':>8}")
for label, flt in (("全24場", lambda x: True), ("対象5場", lambda x: x[1]),
                   ("全24場・1〜4月", lambda x: x[0] < "2026-05-01"),
                   ("全24場・5〜8月", lambda x: x[0] >= "2026-05-01")):
    sub = [x for x in ledger if flt(x)]
    for name, fn in (("現行6行", h1000), ("幾何構成", geometry)):
        st = rt = hit = 0
        for _d, _f, pred, p in sub:
            s, g = score(fn(pred), p)
            st += s
            rt += g
            hit += g > 0
        print(f"{label:<16}{name:<12}{st:>10,}円{rt / st:>8.1%}{hit / len(sub):>8.1%}")
