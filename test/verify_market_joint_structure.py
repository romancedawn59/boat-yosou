# -*- coding: utf-8 -*-
"""市場の同時分布に決まり手の構造(外艇同士の連結など)が入っているか(2026-09-21ケンさん指示)

    py -X utf8 test/verify_market_joint_structure.py

モデルは使わない。3連単オッズ(全120通り)と着順だけで測る。
  市場の想定確率 q(a-b-c) = (1/オッズ) をレース内で合計1に整えたもの
  市場の1着確率   P(a)    = Σ q(a-*-*)
  市場の2連単確率 M(a→b)  = Σ_c q(a-b-c)
  独立の式(PL)    PL(a→b) = P(a) × P(b) / (1 − P(a))   ← 1着確率だけから組み立てた場合
1着艇×2着艇の各マスで
  実測/PL : 実際の回数 ÷ PLの想定回数(=構造の大きさ)
  市場/PL : 市場の想定回数 ÷ PLの想定回数(=市場がその構造をどこまで織り込んでいるか)
  実測/市場: 1より大きければ市場がまだ低く見ている(穴の候補)。zはその有意さ。
オッズは odds(締切15分前)と odds_final(確定)の両方を使い、同じレースは確定を優先。
"""
import math
import sqlite3
from collections import defaultdict

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
odds = defaultdict(dict)
src = {}
for tbl in ("odds", "odds_final"):        # 後から読む odds_final が上書き=確定を優先
    for rid, comb, o in conn.execute(
            f"SELECT race_id, combination, odds FROM {tbl} WHERE bet_type='3連単' "
            f"AND odds IS NOT NULL AND odds > 0"):
        odds[rid][comb] = o
        src[rid] = tbl
res = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT race_id, lane, arrival_order FROM results WHERE arrival_order IN (1, 2)"):
    res[rid][ao] = lane
pay = {rid: (c, a or 0) for rid, c, a in conn.execute(
    "SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='2連単'")}
conn.close()

O = defaultdict(int)
E_pl = defaultdict(float)
E_mk = defaultdict(float)
V_mk = defaultdict(float)
n = n_final = 0
for rid, o in odds.items():
    if len(o) < 110 or 1 not in res.get(rid, {}) or 2 not in res[rid]:
        continue
    inv = {cb: 1 / x for cb, x in o.items()}
    tot = sum(inv.values())
    P = defaultdict(float)
    M = defaultdict(float)
    for cb, v in inv.items():
        a, b, _c = (int(x) for x in cb.split("-"))
        P[a] += v / tot
        M[(a, b)] += v / tot
    n += 1
    n_final += src[rid] == "odds_final"
    O[(res[rid][1], res[rid][2])] += 1
    for a in range(1, 7):
        for b in range(1, 7):
            if a != b and P[a] < 1:
                E_pl[(a, b)] += P[a] * P[b] / (1 - P[a])
                E_mk[(a, b)] += M[(a, b)]
                V_mk[(a, b)] += M[(a, b)] * (1 - M[(a, b)])

print(f"対象: {n:,}R(うち確定オッズ{n_final:,}R・残りは締切15分前)")
for title, f in (("実測/PL(構造の大きさ)", lambda k: O[k] / E_pl[k]),
                 ("市場/PL(市場の織り込み)", lambda k: E_mk[k] / E_pl[k]),
                 ("実測/市場(1超=市場がまだ低く見ている)", lambda k: O[k] / E_mk[k])):
    print(f"\n===== {title} =====")
    print("1着＼2着" + "".join(f"{b:>8}号艇" for b in range(1, 7)))
    for a in range(1, 7):
        print(f"{a}号艇    " + "".join(
            f"{'—':>10}" if a == b else f"{f((a, b)):>10.2f}" for b in range(1, 7)))

print("\n===== 実測と市場の差が大きいマス(|z|の大きい順・上位12) =====")
print(f"{'マス':<10}{'実測':>6}{'市場の想定':>10}{'PLの想定':>10}{'実測/PL':>9}{'市場/PL':>9}{'実測/市場':>10}{'z':>7}")
cells = sorted(((O[k] - E_mk[k]) / math.sqrt(V_mk[k]), k) for k in E_mk if V_mk[k] > 0)
for z, k in sorted(cells, key=lambda x: -abs(x[0]))[:12]:
    print(f"{k[0]}→{k[1]:<8}{O[k]:>6}{E_mk[k]:>10.1f}{E_pl[k]:>10.1f}{O[k] / E_pl[k]:>9.2f}"
          f"{E_mk[k] / E_pl[k]:>9.2f}{O[k] / E_mk[k]:>10.2f}{z:>7.2f}")

print("\n===== 注目マス: 外艇が勝ったときの外艇2着 =====")
for k in ((4, 5), (4, 6), (5, 6), (5, 4), (6, 5), (3, 4), (3, 5), (2, 3)):
    z = (O[k] - E_mk[k]) / math.sqrt(V_mk[k])
    print(f"  {k[0]}→{k[1]}: 実測{O[k]:>4}回 / 実測÷PL {O[k] / E_pl[k]:.2f} / 市場÷PL {E_mk[k] / E_pl[k]:.2f}"
          f" / 実測÷市場 {O[k] / E_mk[k]:.2f} (z={z:+.2f})")
