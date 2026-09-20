# -*- coding: utf-8 -*-
"""逸れ地図 vs 市場: 予想順位からの逸れ方を、市場はどこで見誤っているか
(2026-09-21ケンさん発案「逸れ具合を信頼する」の検証)

    py -X utf8 test/verify_deviation_vs_market.py

■ 問い
予想順位(1着確率順)を固定し、決着がそこからどう逸れるかの発生率を信頼して
買う——これが利益になるのは「自分の発生率」と「市場の見積もり」が食い違う場所が
あるときだけ。食い違いは実在するか。

■ 対象(事前登録)
全24場×モデル1位勝率20〜30%で、3連単オッズの記録(締切15分前スナップショット
中心)が110通り以上あるレース。予想順位は2026-01〜08が月次walk-forwardの台帳
(list_honmei_monthly_2026.py)、09月は本番配信のpicks JSON。

■ A. 市場の見積もり vs 実際
市場の想定確率 q = (1/オッズ) を1レース内で合計1に正規化(控除を除去)。
各目を予想順位空間の並び(r_i-r_j-r_k)に写し、グループごとに
実際の発生数Oと市場の想定数E=Σqを比べる。z=(O-E)/sqrt(Σq(1-q))。

■ B. 逸れ地図×オッズの買い方(walk-forward)
各月、その月より前の台帳レース(オッズ不要・同帯)だけで並び120通りの発生率を
推定(平滑化あり)。当月の各レースで EV=発生率×オッズ が基準以上の目を各100円。
払戻は確定配当(payouts)。基準=1.0/1.2/1.5/2.0。
■ 合格基準: 基準1.2以上のどれかで 回収率105%超 かつ 最大1本除外後も100%超
かつ 評価月の過半で100%超。満たさなければ「逸れ具合は市場に織り込み済み」と結論。
"""
import csv
import glob
import json
import math
import sqlite3
from collections import Counter, defaultdict
from itertools import permutations

LEDGER = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
          r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad\honmei_results_2026.csv")
VEN = {"桐生": 1, "戸田": 2, "江戸川": 3, "平和島": 4, "多摩川": 5, "浜名湖": 6,
       "蒲郡": 7, "常滑": 8, "津": 9, "三国": 10, "びわこ": 11, "住之江": 12,
       "尼崎": 13, "鳴門": 14, "丸亀": 15, "児島": 16, "宮島": 17, "徳山": 18,
       "下関": 19, "若松": 20, "芦屋": 21, "福岡": 22, "唐津": 23, "大村": 24}
PATTERNS = list(permutations(range(1, 7), 3))
THRS = (1.0, 1.2, 1.5, 2.0)
MIN_HISTORY = 500


def r1_fate(pt):
    if pt[0] == 1:
        return "予想1位が1着"
    if pt[1] == 1:
        return "予想1位が2着"
    if pt[2] == 1:
        return "予想1位が3着"
    return "予想1位が圏外"


def groups_of(pt):
    return [f"勝者=予想{pt[0]}位", r1_fate(pt),
            "予想5・6位が3着内に絡む" if max(pt) >= 5 else "上位4艇で決着"]


def pattern_of(pred, comb):
    a, b, c = (int(x) for x in comb.split("-"))
    return (pred.index(a) + 1, pred.index(b) + 1, pred.index(c) + 1)


races = []   # (date, rid, pred順, 結果の並び(順位空間))
for r in csv.DictReader(open(LEDGER, encoding="utf-8-sig")):
    if float(r["モデル1位勝率"].rstrip("%")) >= 30:
        continue
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    res = [int(x) for x in r["結果(3連単)"].split("-")]
    if len(pred) < 6:
        continue
    rid = (r["日付"].replace("-", "") + "_" + f"{VEN[r['場']]:02d}" + "_"
           + f"{int(r['R']):02d}")
    races.append((r["日付"], rid, pred, tuple(pred.index(l) + 1 for l in res)))

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
arr = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id=res.race_id WHERE r.date>='2026-09-01' "
        "AND res.arrival_order IS NOT NULL"):
    arr[rid][lane] = ao
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if not r.get("ranked") or len(r["ranked"]) < 6:
            continue
        if not (0.20 <= r["ranked"][0][1] < 0.30):
            continue
        a = arr.get(r["race_id"])
        if not a or len(a) < 3:
            continue
        pred = [x[0] for x in r["ranked"]]
        top3 = sorted(a, key=a.get)[:3]
        races.append((pk["date"], r["race_id"], pred,
                      tuple(pred.index(l) + 1 for l in top3)))
races.sort()
print(f"同帯レース(発生率の母集団): {len(races):,}R "
      f"({races[0][0]}〜{races[-1][0]})", flush=True)

odds = defaultdict(dict)
snap = {}
want = {x[1] for x in races}
for rid, comb, o, fa in conn.execute(
        "SELECT race_id, combination, odds, fetched_at FROM odds "
        "WHERE bet_type='3連単' AND odds IS NOT NULL AND odds > 0"):
    if rid in want:
        odds[rid][comb] = o
        snap[rid] = fa
pay3 = {}
for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-05-01' "
        "AND p.bet_type='3連単'"):
    pay3[rid] = (comb, amt or 0)
conn.close()

ev_set = [x for x in races if len(odds.get(x[1], {})) >= 110 and x[1] in pay3]
n_final = sum(1 for x in ev_set if snap.get(x[1]) == "final-backfill")
print(f"オッズ記録ありの評価対象: {len(ev_set)}R "
      f"(うち確定オッズの遡及分{n_final}R・残りは締切前スナップショット)")

# ---------- A. 市場の見積もり vs 実際 ----------
grp_O = Counter()
grp_E = Counter()
grp_V = Counter()
pat_O = Counter()
pat_E = Counter()
for d, rid, pred, actual in ev_set:
    inv = {cb: 1 / o for cb, o in odds[rid].items()}
    tot = sum(inv.values())
    for cb, v in inv.items():
        q = v / tot
        try:
            pt = pattern_of(pred, cb)
        except ValueError:
            continue
        for k in groups_of(pt):
            grp_E[k] += q
            grp_V[k] += q * (1 - q)
        pat_E[pt] += q
    for k in groups_of(actual):
        grp_O[k] += 1
    pat_O[actual] += 1

print("\n===== A. 市場の想定 vs 実際(予想順位の空間) =====")
print(f"{'グループ':<22}{'実際':>6}{'市場の想定':>10}{'実際/想定':>9}{'z':>7}")
order = ([f"勝者=予想{k}位" for k in range(1, 7)]
         + ["予想1位が1着", "予想1位が2着", "予想1位が3着", "予想1位が圏外",
            "上位4艇で決着", "予想5・6位が3着内に絡む"])
for k in order:
    O, E = grp_O[k], grp_E[k]
    z = (O - E) / math.sqrt(grp_V[k]) if grp_V[k] else 0
    flag = (" ←市場が過小評価" if z >= 2
            else " ←市場が過大評価" if z <= -2 else "")
    print(f"{k:<22}{O:>6}{E:>10.1f}{O / E:>9.2f}{z:>7.2f}{flag}")
print("\n並び別(市場の想定数が多い順・上位16):")
for pt, E in sorted(pat_E.items(), key=lambda kv: -kv[1])[:16]:
    O = pat_O[pt]
    print(f"  r{pt[0]}-r{pt[1]}-r{pt[2]}: 実際{O:>3} / 市場想定{E:>5.1f} "
          f"(比{O / E:>4.2f})")

# ---------- B. 逸れ地図×オッズ(walk-forward) ----------
print("\n===== B. 逸れ地図×オッズの買い方(発生率は当月より前のみで推定) =====")
months = sorted({d[:7] for d, *_ in ev_set})
res = {t: defaultdict(lambda: [0, 0, 0, 0]) for t in THRS}  # 月→[目数,投資,回収,的中]
best = {t: 0 for t in THRS}
eval_months = set()
for m in months:
    hist = [x for x in races if x[0] < f"{m}-01"]
    if len(hist) < MIN_HISTORY:
        continue
    eval_months.add(m)
    cnt = Counter(x[3] for x in hist)
    ph = {pt: (cnt[pt] + 0.5) / (len(hist) + 60) for pt in PATTERNS}
    for d, rid, pred, actual in ev_set:
        if d[:7] != m:
            continue
        win_comb, win_amt = pay3[rid]
        for cb, o in odds[rid].items():
            try:
                pt = pattern_of(pred, cb)
            except ValueError:
                continue
            ev = ph[pt] * o
            for t in THRS:
                if ev >= t:
                    a = res[t][m]
                    a[0] += 1
                    a[1] += 100
                    if cb == win_comb:
                        a[2] += win_amt
                        a[3] += 1
                        best[t] = max(best[t], win_amt)

n_eval = sum(1 for x in ev_set if x[0][:7] in eval_months)
print(f"評価レース: {n_eval}R / 評価月: {sorted(eval_months)}")
print(f"{'基準':<8}{'買った目':>8}{'1Rあたり':>8}{'的中':>6}{'投資':>10}{'回収':>10}"
      f"{'回収率':>8}{'最大1本除く':>10}  月別回収率")
for t in THRS:
    b = sum(v[0] for v in res[t].values())
    st = sum(v[1] for v in res[t].values())
    rt = sum(v[2] for v in res[t].values())
    h = sum(v[3] for v in res[t].values())
    if not st:
        continue
    mon = " ".join(f"{m[5:]}月{v[2] / v[1]:.0%}"
                   for m, v in sorted(res[t].items()) if v[1])
    print(f"EV≥{t:<5}{b:>8,}{b / max(1, n_eval):>8.1f}{h:>6}{st:>9,}円{rt:>9,}円"
          f"{rt / st:>8.1%}{(rt - best[t]) / (st - 100):>10.1%}  {mon}")
all_rt = sum(pay3[x[1]][1] for x in ev_set if x[0][:7] in eval_months)
print(f"(参考) 全120通りを毎回買うと回収率 {all_rt / max(1, n_eval * 12000):.1%}")
