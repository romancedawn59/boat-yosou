# -*- coding: utf-8 -*-
"""全24場×1位勝率22.5〜30%×3連単ドンピシャ(r1-r2-r3)1点の再検証(2026-09-22ケンさん指示)

    py -X utf8 test/reverify_donpisha_band_0922.py

① 1年分のwalk-forward台帳(2025-10〜2026-09)を月別・帯の細分・最大1本除外・信頼区間で見直す
② 本番配信の予想だけ(2026-07-21〜09-21)で同じ数字を出す(9/21=全24場選別の初日を含む)
③ 帯の優位が偶然と区別できるか: 帯ラベルをシャッフルして「22.5〜30%帯の回収率以上」が出る確率
④ 市場との比較: 確定オッズがある帯内レースで、ドンピシャの目の市場想定確率と実際の的中率
"""
import glob, json, os, random, sqlite3
from collections import defaultdict
import pandas as pd

YEN = 500
conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
pay3 = {r: (c, a or 0) for r, c, a in conn.execute("SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連単'")}
fin = defaultdict(dict)
for rid, lane, ao, st in conn.execute("SELECT e.race_id, e.lane, res.arrival_order, res.st_time FROM entries e JOIN races r ON r.race_id=e.race_id LEFT JOIN results res ON res.race_id=e.race_id AND res.lane=e.lane WHERE r.date>='2026-07-21'"):
    fin[rid][lane] = (ao, st)
ofin = defaultdict(dict)
for rid, comb, o in conn.execute("SELECT race_id, combination, odds FROM odds_final WHERE bet_type='3連単' AND odds>0"):
    ofin[rid][comb] = o
conn.close()

def got(pred, refund, rid):
    comb, amt = pay3[rid]
    top = pred[:3]
    if set(top) & refund: return YEN, False
    hit = comb == "-".join(map(str, top))
    return (amt * YEN // 100 if hit else 0), hit

def summarize(rows, label):
    """rows: (month, p1, got, hit)"""
    n = len(rows); ret = sum(r[2] for r in rows); h = sum(r[3] for r in rows); best = max((r[2] for r in rows), default=0)
    import math
    p = h / max(1, n); z = 1.645
    lo = max(0, (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)) if n else 0
    hi = ((p + z*z/(2*n) + z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)) if n else 0
    print(f"{label:<26}{n:>6,}R 的中{h:>4}({p:.1%} 幅{lo:.1%}〜{hi:.1%}) 回収率{ret/(n*YEN):>7.1%} 損益{ret-n*YEN:>+10,}円 最大1本除く{(ret-best)/(n*YEN):>6.1%}")

# ① 1年台帳
L = pd.read_csv(r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv", encoding="utf-8-sig", dtype={"refund": str})
wf = []
for rid, m, p1, pred, rf in zip(L["race_id"], L["month"], L["p1"], L["pred"], L["refund"].fillna("")):
    if rid not in pay3: continue
    g, h = got([int(x) for x in pred.split("-")], {int(x) for x in rf.split("-") if x}, rid)
    wf.append((m, p1, g, h))
band = [r for r in wf if .225 <= r[1] < .30]
print("===== ① 1年分のwalk-forward台帳(2025-10〜2026-09) =====")
summarize(band, "22.5〜30%帯 全体")
for lo, hi in ((.20,.225),(.225,.25),(.25,.275),(.275,.30),(.30,.325),(.325,.35)):
    summarize([r for r in wf if lo <= r[1] < hi], f"  細分 {lo:.1%}〜{hi:.1%}")
print("― 月別 ―")
for m in sorted({r[0] for r in band}):
    summarize([r for r in band if r[0] == m], f"  {m}")
n_plus = sum(1 for m in sorted({r[0] for r in band}) if sum(r[2] for r in band if r[0]==m) > YEN*sum(1 for r in band if r[0]==m))
print(f"  黒字の月: {n_plus}/{len({r[0] for r in band})}")

# ③ シャッフル検定(帯ラベル): 20〜35%を2.5%刻みの6帯に分け、22.5〜30%(2帯の合算)以上の回収率が偶然で出る確率
pool = [r for r in wf if .20 <= r[1] < .35]
def roi_of(rows): return sum(r[2] for r in rows) / (len(rows)*YEN)
real = roi_of(band)
labels = [min(5, int((r[1]-.20)/.025)) for r in pool]
rng = random.Random(7); ge = 0; N = 2000
for _ in range(N):
    rng.shuffle(labels)
    # 隣り合う2帯の合算(5通り)のうち最大の回収率
    by = defaultdict(list)
    for lab, r in zip(labels, pool): by[lab].append(r)
    best = max(roi_of(by[k]+by[k+1]) for k in range(5) if by[k] and by[k+1])
    ge += best >= real
print(f"\n===== ③ 帯の優位は偶然か(20〜35%の6帯をシャッフル・隣接2帯の最良が{real:.1%}以上になる確率) ===== {ge/N:.1%}")

# ② 本番配信
print("\n===== ② 本番配信の予想だけ(2026-07-21〜09-21) =====")
prod = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    d = os.path.basename(path)[6:16]
    if d < "2026-07-21": continue
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid, f = r["race_id"], fin.get(r["race_id"])
        if not r.get("ranked") or len(r["ranked"]) < 3 or rid not in pay3 or not f or sum(1 for v in f.values() if v[0] is not None) < 3: continue
        refund = {l for l, (ao, st) in f.items() if ao is None and st is None}
        g, h = got([x[0] for x in r["ranked"]], refund, rid)
        prod.append((d[:7], r["ranked"][0][1], g, h, rid, r.get("shobusho")))
pb = [r for r in prod if .225 <= r[1] < .30]
summarize(pb, "22.5〜30%帯 全体")
for m in sorted({r[0] for r in pb}): summarize([r for r in pb if r[0]==m], f"  {m}")
summarize([r for r in pb if r[5] == "本命"], "  うち配信で「本命」の印")
d21 = [r for r in prod if r[4].startswith("20260921") and r[5] == "本命"]
print("  9/21(全24場選別の初日)の本命4R: " + " / ".join(f"{r[4][9:11]}場{int(r[4][12:14])}R {'的中' if r[3] else '外れ'}({r[2]:,}円)" for r in d21))

# ④ 市場との比較(確定オッズあり・帯内)
print("\n===== ④ 市場との比較(確定オッズがある帯内レース) =====")
rows = []
for rid, m, p1, pred, rf in zip(L["race_id"], L["month"], L["p1"], L["pred"], L["refund"].fillna("")):
    if not (.225 <= p1 < .30) or rid not in ofin or len(ofin[rid]) < 110 or rid not in pay3: continue
    top = "-".join(pred.split("-")[:3]); o = ofin[rid]
    q = (1/o[top]) / sum(1/x for x in o.values()) if top in o else None
    if q is None: continue
    rows.append((m, q, o[top], pay3[rid][0] == top))
n = len(rows); h = sum(r[3] for r in rows)
print(f"  {n:,}R: ドンピシャの目の市場想定確率 平均{sum(r[1] for r in rows)/n:.2%}(想定的中{sum(r[1] for r in rows):.1f}本) / 実際の的中{h}本({h/n:.2%}) / 実際÷市場 {h/sum(r[1] for r in rows):.2f}")
print(f"  確定オッズでの回収率(1点{YEN}円): {sum(r[2]*YEN for r in rows if r[3])/(n*YEN):.1%} / 平均オッズ{sum(r[2] for r in rows)/n:.1f}倍")
for lo, hi in ((0,.03),(.03,.05),(.05,.08),(.08,1)):
    g = [r for r in rows if lo <= r[1] < hi]
    if g: print(f"    市場想定{lo:.0%}〜{hi:.0%}: {len(g):>4}R 想定{sum(r[1] for r in g):.1f}本 実際{sum(r[3] for r in g)}本 回収率{sum(r[2]*YEN for r in g if r[3])/(len(g)*YEN):.0%}")
