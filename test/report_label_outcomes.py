# -*- coding: utf-8 -*-
"""本命・超混戦ラベルのレースの結果台帳: ①人気が来たか ②配当 ③買い目の的中
(2026-09-20ケンさん要望「検討方法のデータ蓄積として知りたい」)

    py -X utf8 test/report_label_outcomes.py 2026-09   # 月指定(省略時は当月)

配信済みpicks JSONのラベル(本命/超混戦)が付いたレースについて、
- 確定オッズが未取得なら公式サイトから取得して odds_final に保存(蓄積・冪等)
- 当選3連単が確定オッズで何番人気だったか、配当、配信買い目が的中したかを一覧
- 超混戦は🧪専用順位⑬の的中も併記
人気帯の区切り: 1-5番人気=人気決着 / 6-20=中穴 / 21-60=穴 / 61以降=大穴。
"""
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db
from collect_final_odds import collect
from config import DB_PATH, PROJECT_DIR, VENUE_NAMES, jst_today

month = sys.argv[1] if len(sys.argv) > 1 else jst_today().isoformat()[:7]
today = jst_today().isoformat()

labeled = []
for path in sorted(glob.glob(str(PROJECT_DIR / "docs" / "data"
                                 / f"picks_{month}-*.json"))):
    p = json.load(open(path, encoding="utf-8"))
    for r in p["races"]:
        if r.get("shobusho") in ("本命", "超混戦"):
            labeled.append((p["date"], r))

conn = db.connect(DB_PATH)
targets = [
    (r["race_id"], r["venue_code"], r["race_no"], d) for d, r in labeled
    if d < today and not conn.execute(
        "SELECT 1 FROM odds_final WHERE race_id=? LIMIT 1",
        (r["race_id"],)).fetchone()
]
if targets:
    print(f"確定オッズを取得: {len(targets)}レース", flush=True)
    collect(conn, targets)


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan13(l):
    r1, r2, r3, r4, r5 = l[:5]
    b = [("3連単", f"{a}-{bb}-{cc}", 100) for a, bb, cc in permutations((r1, r2, r3))]
    b += [("3連単", f"{a}-{bb}-{cc}", 100) for a, bb, cc in permutations((r1, r2, r4))]
    return b + [("3連単", f"{r3}-{r1}-{r2}", 300), ("3連単", f"{r4}-{r1}-{r2}", 300),
                ("3連複", trio(r3, r4, r5), 200)]


def score(bets, pay):
    m = defaultdict(int)
    for bt, comb, y in bets:
        m[(bt, comb)] += y
    return sum(m.values()), sum(pay.get(k, 0) * y // 100 for k, y in m.items())


def zone(rank):
    return ("人気決着" if rank <= 5 else "中穴" if rank <= 20
            else "穴" if rank <= 60 else "大穴")


rows = []
for d, r in labeled:
    rid = r["race_id"]
    pay = {(bt, comb): amt or 0 for bt, comb, amt in conn.execute(
        "SELECT bet_type, combination, amount_yen FROM payouts WHERE race_id=?",
        (rid,))}
    arr = dict(conn.execute(
        "SELECT lane, arrival_order FROM results WHERE race_id=? "
        "AND arrival_order IS NOT NULL", (rid,)))
    if not pay or len(arr) < 3:
        continue
    res = "-".join(str(l) for l in sorted(arr, key=arr.get)[:3])
    amt = pay.get(("3連単", res), 0)
    odds = dict(conn.execute(
        "SELECT combination, odds FROM odds_final WHERE race_id=? "
        "AND bet_type='3連単' AND odds IS NOT NULL", (rid,)))
    rank = (sorted(odds.values()).index(odds[res]) + 1) if res in odds else None
    p1 = r["ranked"][0][1]
    lanes = [x[0] for x in r["ranked"]]
    band = "超混戦" if p1 < 0.20 else "本命"
    st, rt = score([(bt, comb, y) for bt, comb, y, _s in r["ken"]], pay)
    t3 = r.get("top3_order")
    rt3 = score(plan13(t3), pay)[1] if band == "超混戦" and t3 and len(t3) >= 5 else None
    rows.append(dict(d=d, v=VENUE_NAMES.get(r["venue_code"]), rn=r["race_no"],
                     band=band, p1=p1, pred="-".join(map(str, lanes)), res=res,
                     amt=amt, rank=rank, st=st, rt=rt, rt3=rt3))
conn.close()

print(f"\n===== {month} ラベル別の結果台帳 =====")
print(f"{'日付':<6}{'場':<5}{'R':>2} {'帯':<4}{'1位':>4} {'予想':<12}{'結果':<7}"
      f"{'3連単配当':>9}{'人気':>5} {'区分':<5}{'買い目':>7}")
for x in rows:
    rk = f"{x['rank']}" if x["rank"] else "-"
    zn = zone(x["rank"]) if x["rank"] else "-"
    hit = f"○{x['rt']:,}" if x["rt"] else "×"
    print(f"{x['d'][5:]:<6}{x['v']:<5}{x['rn']:>2} {x['band']:<4}{x['p1']:>4.0%} "
          f"{x['pred']:<12}{x['res']:<7}{x['amt']:>8,}円{rk:>5} {zn:<5}{hit:>7}")

for band in ("本命", "超混戦"):
    sub = [x for x in rows if x["band"] == band]
    if not sub:
        continue
    n = len(sub)
    amts = sorted(x["amt"] for x in sub)
    ranks = [x["rank"] for x in sub if x["rank"]]
    zc = Counter(zone(k) for k in ranks)
    hits = [x for x in sub if x["rt"]]
    st = sum(x["st"] for x in sub)
    rt = sum(x["rt"] for x in sub)
    print(f"\n【{band}】{n}R")
    print(f"  ①人気: 当選組の人気順 中央値{statistics.median(ranks):.0f}番人気 / "
          + " / ".join(f"{z}{zc[z]}R({zc[z] / len(ranks):.0%})"
                       for z in ("人気決着", "中穴", "穴", "大穴")))
    print(f"  ②配当: 中央値{amts[n // 2]:,}円 / 平均{sum(amts) // n:,}円 / "
          f"55倍以上{sum(1 for a in amts if a >= 5500)}R / 万舟{sum(1 for a in amts if a >= 10000)}R")
    print(f"  ③的中: 配信買い目 {len(hits)}/{n}R({len(hits) / n:.0%}) "
          f"投資{st:,}円 回収{rt:,}円 回収率{rt / st:.1%} "
          f"(うちガミ{sum(1 for x in hits if x['rt'] < x['st'])}R)")
    for z in ("人気決着", "中穴", "穴", "大穴"):
        zs = [x for x in sub if x["rank"] and zone(x["rank"]) == z]
        if zs:
            zh = sum(1 for x in zs if x["rt"])
            print(f"     {z}のとき: 的中{zh}/{len(zs)}R "
                  f"回収{sum(x['rt'] for x in zs):,}円/投資{sum(x['st'] for x in zs):,}円")
    if band == "超混戦":
        s3 = [x for x in sub if x["rt3"] is not None]
        if s3:
            h3 = sum(1 for x in s3 if x["rt3"])
            print(f"  (🧪専用順位⑬: 的中{h3}/{len(s3)}R 回収{sum(x['rt3'] for x in s3):,}円"
                  f"/投資{len(s3) * 2000:,}円)")
