# -*- coding: utf-8 -*-
"""オリジナル選手レーティング第一版(2026-07-29ケンさん発案「FIFAランキング的」)

    py -X utf8 test/verify_rating_v1.py

Elo方式(6艇の着順=15組の対戦として逐次更新・時系列順=walk-forward安全)。
検証(事前設計):
①級別とレーティングの逆転がどれくらい起きているか
②「レーティングはレース内1位だが級別は格下(B級)」の艇の単勝成績
  vs「レーティング1位かつA1」(市場も評価済み)の艇 — ラベル税の実測
"""
import sqlite3
from collections import defaultdict

DB = r"Y:\マイドライブ\boat\boat.db"
K = 3.0          # 1対戦あたりの更新幅(1レースで最大15対戦)
BURN_END = "2025-11-01"   # ここまでは学習のみ(バーンイン)

c = sqlite3.connect(DB)
races_date = dict(c.execute("SELECT race_id, date FROM races"))
lane_racer = {}
klass = {}
for rid, lane, reg, kl in c.execute(
        "SELECT race_id, lane, reg_no, racer_class FROM entries"):
    lane_racer[(rid, lane)] = reg
    klass[(rid, lane)] = kl
name_of = dict(c.execute(
    "SELECT reg_no, racer_name FROM entries GROUP BY reg_no"))
finish = defaultdict(dict)
for rid, lane, ao in c.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    finish[rid][lane] = ao
win_pay = {}
for rid, comb, amt in c.execute(
        "SELECT race_id, combination, amount_yen FROM payouts "
        "WHERE bet_type='単勝'"):
    win_pay[(rid, comb)] = amt or 0
c.close()

R = defaultdict(lambda: 1500.0)
order = sorted(finish.keys(), key=lambda r: (races_date.get(r, ""), r))

stats = {"逆転あり": 0, "対象R": 0}
money = {"B級だがレート1位": [0, 0, 0], "A1でレート1位": [0, 0, 0]}

for rid in order:
    d = races_date.get(rid)
    if not d:
        continue
    boats = [(lane, ao) for lane, ao in finish[rid].items()
             if (rid, lane) in lane_racer]
    if len(boats) < 6:
        continue
    regs = {lane: lane_racer[(rid, lane)] for lane, _ in boats}

    # --- 予測フェーズ(更新前のレートで判定) ---
    if d >= BURN_END:
        stats["対象R"] += 1
        rates = {lane: R[regs[lane]] for lane, _ in boats}
        cls = {lane: (klass.get((rid, lane)) or "?") for lane, _ in boats}
        a1_rates = [rates[l] for l in rates if cls[l] == "A1"]
        b_over = [l for l in rates
                  if cls[l].startswith("B") and a1_rates
                  and rates[l] > max(a1_rates)]
        if b_over:
            stats["逆転あり"] += 1
        top_lane = max(rates, key=rates.get)
        won = finish[rid].get(top_lane) == 1
        pay = win_pay.get((rid, str(top_lane)), 0) if won else 0
        key = ("B級だがレート1位" if cls[top_lane].startswith("B")
               else "A1でレート1位" if cls[top_lane] == "A1" else None)
        if key:
            m = money[key]
            m[0] += 100
            m[1] += pay
            if won:
                m[2] += 1

    # --- 更新フェーズ(Elo・15対戦) ---
    for i in range(len(boats)):
        for j in range(i + 1, len(boats)):
            li, ai = boats[i]
            lj, aj = boats[j]
            ri, rj = regs[li], regs[lj]
            e = 1 / (1 + 10 ** ((R[rj] - R[ri]) / 400))
            s = 1.0 if ai < aj else 0.0
            R[ri] += K * (s - e)
            R[rj] -= K * (s - e)

print(f"レーティング構築完了: 選手{len(R):,}人")
print(f"\n=== 現在のトップ10(KR指数) ===")
top = sorted(R.items(), key=lambda kv: -kv[1])[:10]
for reg, r in top:
    print(f"  {name_of.get(reg,'?')}: {r:,.0f}")

n = stats["対象R"]
print(f"\n=== ①級別逆転の実在(バーンイン後 {n:,}R) ===")
print(f"  「B級艇のレートがそのレースのA1最上位を上回る」レース: "
      f"{stats['逆転あり']:,}R ({stats['逆転あり']/max(1,n):.1%})")

print(f"\n=== ②ラベル税の実測(レース内レート1位艇の単勝100円) ===")
for key, (st, rt, w) in money.items():
    if st:
        print(f"  {key}: {st//100:,}回 勝率{w/(st//100):.1%} 単勝回収率{rt/st:.1%}")
