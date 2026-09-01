# -*- coding: utf-8 -*-
"""タイプ別に効く線の詳細シミュレーション(選別レース・候補線70本×8タイプ)"""
import csv, sqlite3
from collections import defaultdict, Counter
from itertools import permutations, combinations
BASE = r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad"
VEN = {"桐生":1,"戸田":2,"江戸川":3,"平和島":4,"多摩川":5,"浜名湖":6,"蒲郡":7,"常滑":8,"津":9,"三国":10,"びわこ":11,"住之江":12,"尼崎":13,"鳴門":14,"丸亀":15,"児島":16,"宮島":17,"徳山":18,"下関":19,"若松":20,"芦屋":21,"福岡":22,"唐津":23,"大村":24}
hon = list(csv.DictReader(open(BASE + r"\honmei_results_2026.csv", encoding="utf-8-sig")))
kon = list(csv.DictReader(open(BASE + r"\konsen_results_2026.csv", encoding="utf-8-sig")))
def rid_of(r): return r["日付"].replace("-","") + "_" + f"{VEN[r['場']]:02d}" + "_" + f"{int(r['R']):02d}"
# --- 選別の再現: 日ごとに超混戦数→残予算→take, 5場×20-30%を低い順 ---
konsen_per_day = Counter(r["日付"] for r in kon)
pool = defaultdict(list)
for r in hon:
    if r["対象5場"] and float(r["モデル1位勝率"].rstrip("%")) < 30:
        pool[r["日付"]].append(r)
sel = []
for d, rs in pool.items():
    take = min(4, max(0, (10200 - 2000 * konsen_per_day.get(d, 0)) // 1400))
    sel += sorted(rs, key=lambda r: float(r["モデル1位勝率"].rstrip("%")))[:take]
rids = [rid_of(r) for r in sel]
c = sqlite3.connect(r"Y:\マイドライブ\boat\boat.db")
pm = defaultdict(dict); tech = {}; nonfin = set()
for i in range(0, len(rids), 800):
    ch = rids[i:i+800]; ph = ",".join("?"*len(ch))
    for rid, bt, comb, amt in c.execute(f"SELECT race_id, bet_type, combination, amount_yen FROM payouts WHERE race_id IN ({ph}) AND bet_type IN ('3連単','3連複')", ch):
        pm[rid][(bt, comb)] = amt or 0
    for rid, t in c.execute(f"SELECT race_id, winning_technique_number FROM races WHERE race_id IN ({ph})", ch):
        tech[rid] = t
    for (rid,) in c.execute(f"SELECT DISTINCT race_id FROM results WHERE race_id IN ({ph}) AND arrival_order IS NULL", ch):
        nonfin.add(rid)
c.close()
def typ(pred, res, rid):
    rs = set(res); t = tech.get(rid)
    if res == pred[:3]: return "T0ドンピシャ"
    if rid in nonfin: return "T1事故"
    if pred[0] not in rs and t in (3,4): return "T2まくられ"
    if pred[0] != res[0] and pred[0] in rs and t == 2: return "T3差され"
    if pred[0] != 1 and res[0] == 1 and t == 1: return "T4イン逃げ"
    if rs == set(pred[:3]): return "T5順序違い"
    if rs <= set(pred[:4]): return "T6 4位滑込"
    return "T7下位奇襲"
TYPES = ["T0ドンピシャ","T1事故","T2まくられ","T3差され","T4イン逃げ","T5順序違い","T6 4位滑込","T7下位奇襲"]
def trio(a,b,c2):
    s = sorted([a,b,c2]); return f"{s[0]}={s[1]}={s[2]}"
# 候補線(予想順位空間): 3連単 r1..r5の順列60 + 3連複 r1..r5の組10
cands = [("3連単", p) for p in permutations(range(5), 3)] + [("3連複", cmb) for cmb in combinations(range(5), 3)]
def label(bt, ranks):
    return (f"単 r{ranks[0]+1}-r{ranks[1]+1}-r{ranks[2]+1}" if bt == "3連単" else f"複 r{ranks[0]+1}r{ranks[1]+1}r{ranks[2]+1}")
stat = {k: {"st":0,"rt":0,"hit":0,"bytype":Counter(),"rtbytype":Counter(),"mon":defaultdict(lambda:[0,0])} for k in cands}
type_n = Counter(); months = set()
n_scored = 0
for r in sel:
    rid = rid_of(r); pay = pm.get(rid)
    if not pay: continue
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    res = [int(x) for x in r["結果(3連単)"].split("-")]
    if len(pred) < 5: continue
    n_scored += 1
    ty = typ(pred, res, rid); type_n[ty] += 1
    m = r["日付"][:7]; months.add(m)
    for bt, ranks in cands:
        lanes = [pred[i] for i in ranks]
        comb = f"{lanes[0]}-{lanes[1]}-{lanes[2]}" if bt == "3連単" else trio(*lanes)
        got = pay.get((bt, comb), 0)
        s = stat[(bt, ranks)]
        s["st"] += 100; s["rt"] += got; s["mon"][m][0] += 100; s["mon"][m][1] += got
        if got:
            s["hit"] += 1; s["bytype"][ty] += 1; s["rtbytype"][ty] += got
print(f"選別レース再現: {n_scored}R")
print("タイプ分布(選別上): " + " / ".join(f"{t}{type_n[t]}R({type_n[t]/n_scored:.0%})" for t in TYPES))
print("\n===== 候補線70本の成績(各100円・回収率降順・上位25) =====")
print(f"{'線':<16}{'的中':>5}{'回収率':>8}{'100%超月':>8}  的中の内訳(タイプ別) ")
rows_out = []
for k, s in stat.items():
    roi = s["rt"]/s["st"] if s["st"] else 0
    ok = sum(1 for v in s["mon"].values() if v[0] and v[1]/v[0] > 1)
    rows_out.append((roi, label(*k), s["hit"], ok, s["bytype"], k))
rows_out.sort(key=lambda x: -x[0])
for roi, lab, hit, ok, bt, k in rows_out[:25]:
    det = " ".join(f"{t[:5]}{n}" for t, n in bt.most_common(4))
    print(f"{lab:<16}{hit:>5}{roi:>8.1%}{ok:>6}/8   {det}")
print("\n===== タイプ別: そのタイプを最も効率よく拾う線(タイプ内回収額上位3) =====")
for t in TYPES:
    best = sorted(((s["rtbytype"][t], label(*k), s["bytype"][t], s["rt"]/s["st"]) for k, s in stat.items() if s["bytype"][t]), reverse=True)[:3]
    print(f"{t}({type_n[t]}R): " + " / ".join(f"{lab}[{n}回・線全体{roi:.0%}]" for _, lab, n, roi in best))
# 頑健線から静的構成を組む(複≤2・1000/2000)
robust = [(roi, lab, k) for roi, lab, hit, ok, bt, k in rows_out if hit >= 12 and roi > 1.0 and ok >= 4]
print("\n===== 頑健線(的中12回以上・回収率100%超・100%超月4以上) =====")
for roi, lab, k in robust:
    print(f"  {lab:<16}{roi:>7.1%}")
