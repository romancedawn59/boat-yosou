# -*- coding: utf-8 -*-
"""タイプ配分型構成の分割検証: 前半(1-4月)で線を選び後半(5-8月)で採点"""
import csv, sqlite3
from collections import defaultdict, Counter
from itertools import permutations, combinations
BASE = r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad"
VEN = {"桐生":1,"戸田":2,"江戸川":3,"平和島":4,"多摩川":5,"浜名湖":6,"蒲郡":7,"常滑":8,"津":9,"三国":10,"びわこ":11,"住之江":12,"尼崎":13,"鳴門":14,"丸亀":15,"児島":16,"宮島":17,"徳山":18,"下関":19,"若松":20,"芦屋":21,"福岡":22,"唐津":23,"大村":24}
hon = list(csv.DictReader(open(BASE + r"\honmei_results_2026.csv", encoding="utf-8-sig")))
kon = list(csv.DictReader(open(BASE + r"\konsen_results_2026.csv", encoding="utf-8-sig")))
def rid_of(r): return r["日付"].replace("-","") + "_" + f"{VEN[r['場']]:02d}" + "_" + f"{int(r['R']):02d}"
kpd = Counter(r["日付"] for r in kon)
pool = defaultdict(list)
for r in hon:
    if r["対象5場"] and float(r["モデル1位勝率"].rstrip("%")) < 30:
        pool[r["日付"]].append(r)
sel = []
for d, rs in pool.items():
    take = min(4, max(0, (10200 - 2000 * kpd.get(d, 0)) // 1400))
    sel += sorted(rs, key=lambda r: float(r["モデル1位勝率"].rstrip("%")))[:take]
rids = [rid_of(r) for r in sel]
c = sqlite3.connect(r"Y:\マイドライブ\boat\boat.db")
pm = defaultdict(dict)
for i in range(0, len(rids), 800):
    ch = rids[i:i+800]; ph = ",".join("?"*len(ch))
    for rid, bt, comb, amt in c.execute(f"SELECT race_id, bet_type, combination, amount_yen FROM payouts WHERE race_id IN ({ph}) AND bet_type IN ('3連単','3連複')", ch):
        pm[rid][(bt, comb)] = amt or 0
c.close()
def trio(a,b,c2):
    s = sorted([a,b,c2]); return f"{s[0]}={s[1]}={s[2]}"
cands = [("3連単", p) for p in permutations(range(5), 3)] + [("3連複", cmb) for cmb in combinations(range(5), 3)]
def lab(bt, rk): return (f"単r{rk[0]+1}-r{rk[1]+1}-r{rk[2]+1}" if bt=="3連単" else f"複r{rk[0]+1}r{rk[1]+1}r{rk[2]+1}")
def line_ret(r, bt, rk):
    pred = [int(x) for x in r["予想(1位→6位)"].split("-")]
    lanes = [pred[i] for i in rk]
    comb = f"{lanes[0]}-{lanes[1]}-{lanes[2]}" if bt=="3連単" else trio(*lanes)
    return pm[rid_of(r)].get((bt, comb), 0)
sel = [r for r in sel if pm.get(rid_of(r)) and len(r["予想(1位→6位)"].split("-")) >= 5]
first = [r for r in sel if r["日付"] < "2026-05-01"]
second = [r for r in sel if r["日付"] >= "2026-05-01"]
print(f"選別{len(sel)}R = 前半{len(first)}R + 後半{len(second)}R")
# 前半で線をランク(的中5回以上・回収率順)
fs = {}
for k in cands:
    st = len(first)*100; rt = sum(line_ret(r, *k) for r in first); hit = sum(1 for r in first if line_ret(r,*k))
    fs[k] = (rt/st, hit)
ranked = [k for k, (roi, hit) in sorted(fs.items(), key=lambda kv: -kv[1][0]) if hit >= 5 and roi > 1.0]
def pick(n_units):
    """複は最大2本の制約で上位から100円ずつ"""
    chosen, fuku = [], 0
    for k in ranked:
        if k[0] == "3連複":
            if fuku >= 2: continue
            fuku += 1
        chosen.append(k)
        if len(chosen) >= n_units: break
    return chosen
auto1000 = [(k, 100) for k in pick(10)]
auto2000 = [(k, 100) for k in pick(20)]
print("\n前半で選ばれた線(auto1000):", " ".join(lab(*k) for k, _ in auto1000))
# 手組み(原則+頑健線): タイプ配分型1000
manual1000 = [(("3連単",(0,1,2)),100), (("3連単",(2,0,1)),200), (("3連単",(3,1,0)),200),
              (("3連単",(0,3,1)),100), (("3連単",(1,0,4)),100), (("3連単",(1,2,4)),100),
              (("3連単",(0,4,2)),100), (("3連複",(1,2,3)),100)]
h1000 = [(("3連複",(0,1,2)),200), (("3連複",(1,2,3)),100), (("3連単",(0,1,2)),100),
         (("3連単",(2,0,1)),200), (("3連単",(3,0,1)),200), (("3連単",(3,1,0)),200)]
nine = [(("3連複",(0,1,2)),200), (("3連複",(0,1,3)),200), (("3連複",(0,2,3)),100),
        (("3連単",(2,0,1)),200), (("3連単",(3,0,1)),200), (("3連複",(1,2,3)),100),
        (("3連単",(2,1,0)),100), (("3連単",(3,1,0)),300)]
def evalp(plan, rows):
    st = rt = 0; mon = defaultdict(lambda: [0,0])
    for r in rows:
        m = r["日付"][:7]
        for k, y in plan:
            got = line_ret(r, *k) * y // 100
            st += y; rt += got; mon[m][0] += y; mon[m][1] += got
    ok = sum(1 for v in mon.values() if v[1] > v[0])
    return rt/st, rt-st, ok, len(mon)
print("\n===== 分割検証(線の選択は前半のみ・採点は後半5-8月) =====")
print(f"{'構成':<22}{'前半(参考)':>10}{'後半(本番)':>10}{'後半損益':>10}{'後半100%超月':>8}")
for name, plan in (("現行9行1,400", nine), ("★H静的1000", h1000), ("手組みタイプ配分1000", manual1000),
                   ("auto1000(前半最適)", auto1000), ("auto2000(前半最適)", auto2000)):
    a = evalp(plan, first); b = evalp(plan, second)
    print(f"{name:<22}{a[0]:>10.1%}{b[0]:>10.1%}{b[1]:>+9,}円{b[2]:>5}/{b[3]}")
print("\n(auto=前半の成績上位線を機械的に選んだ構成。後半で崩れれば『線の拾い上げは幻』の証明)")
