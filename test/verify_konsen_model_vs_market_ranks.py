# -*- coding: utf-8 -*-
"""超混戦の決着: モデル順位 vs 市場人気順位の分布(2026-08-01ケンさん発案)

    py -X utf8 test/verify_konsen_model_vs_market_ranks.py

■ 問い
超混戦の結果(1-2-3着)は、v2.1モデルの予想順位で見ると何位-何位-何位で、
市場(15分前オッズ)の人気順位で見ると何番人気-何番-何番だったのか。
例: 結果5-4-6 / モデル1位-2位-3位 / 市場6番-3番-5番人気。

■ 方法
- 市場人気: 15分前スナップショット(oddsテーブル)の3連単全組から
  各艇の勝利評価を合成(艇Xが頭の全組の1/オッズ合計)し順位化。
  単勝オッズは未収集のためこの合成値を人気の代理とする(公表の単勝人気と概ね一致)
- モデル順位: 月次再学習(本番同方式)の予測順位
- 対象: 2026-05〜07でスナップショットと予測が揃う超混戦レース(1位生値20%未満)
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

VN = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
      7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
      13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
      19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}

conn = db.connect(DB_PATH)
df = build_training_set(conn)

# 15分前スナップショットから各艇の市場勝率(合成)を作る
market_w = defaultdict(lambda: defaultdict(float))   # rid -> lane -> Σ1/odds
for rid, comb, odds in conn.execute(
        "SELECT race_id, combination, odds FROM odds "
        "WHERE bet_type = '3連単' AND odds > 0"):
    head = int(comb.split("-")[0])
    market_w[rid][head] += 1.0 / odds

res_top3 = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= '2026-05-01' AND res.arrival_order <= 3"):
    res_top3[rid][ao] = lane
payout_tri = {}
for rid, comb, amt in conn.execute(
    "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id "
    "WHERE r.date >= '2026-05-01' AND p.bet_type = '3連単'"):
    payout_tri[rid] = (comb, amt or 0)
race_venue = dict(conn.execute(
    "SELECT race_id, venue_code FROM races WHERE date >= '2026-05-01'"))
conn.close()

rows = []
for m in ("2026-05", "2026-06", "2026-07"):
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, g in month_df.groupby("race_id"):
        t3 = res_top3.get(rid, {})
        if len(t3) != 3 or rid not in market_w or len(market_w[rid]) < 6:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 6 or ranked[0]["prob"] >= 0.20:
            continue
        model_rank = {r["lane"]: i + 1 for i, r in enumerate(ranked)}
        mw = market_w[rid]
        market_rank = {lane: i + 1 for i, lane in
                       enumerate(sorted(mw, key=lambda l: -mw[l]))}
        finish = [t3[1], t3[2], t3[3]]
        comb, amt = payout_tri.get(rid, ("?", 0))
        rows.append({"rid": rid, "date": rid[:8], "venue": race_venue.get(rid),
                     "finish": finish,
                     "mrank": [model_rank.get(l, 9) for l in finish],
                     "krank": [market_rank.get(l, 9) for l in finish],
                     "pay": amt})

n = len(rows)
print(f"\n対象: 超混戦×スナップショットあり {n}R(2026-05〜07)")

print(f"\n===== 着順位置ごとの順位分布 =====")
print(f"{'':>12}" + "".join(f"{k}位/{k}番人気" .rjust(10) for k in range(1, 7)))
for pos, label in ((0, "1着"), (1, "2着"), (2, "3着")):
    mc = Counter(r["mrank"][pos] for r in rows)
    kc = Counter(r["krank"][pos] for r in rows)
    print(f"{label} モデル:  " + "".join(f"{mc.get(k,0)/n:>9.0%} " for k in range(1, 7)))
    print(f"{label} 市場:    " + "".join(f"{kc.get(k,0)/n:>9.0%} " for k in range(1, 7)))

agree_top = sum(1 for r in rows
                if r["mrank"][0] == 1 and r["krank"][0] == 1)
m_top = sum(1 for r in rows if r["mrank"][0] == 1)
k_top = sum(1 for r in rows if r["krank"][0] == 1)
both_rank1_same = sum(1 for r in rows if r["krank"][0] == r["mrank"][0])
print(f"\n勝者がモデル1位: {m_top/n:.0%} / 勝者が市場1番人気: {k_top/n:.0%} "
      f"/ 両方1位(順当中の順当): {agree_top/n:.0%}")

# E/F単(モデル3・4位頭×1位2着×2位3着)の的中がどんな人気だったか
ef = [r for r in rows if r["mrank"] in ([3, 1, 2], [4, 1, 2])]
if ef:
    kav = [r["krank"] for r in ef]
    print(f"\nE/F単的中型({len(ef)}R)の市場人気: " +
          " / ".join("-".join(map(str, k)) + f"番人気({r['pay']:,}円)"
                     for k, r in zip(kav, ef)))

print(f"\n===== 実例(ケンさん形式・払戻降順の上位12R) =====")
print(f"{'日付':<10}{'場':<5}{'結果':<8}{'モデル順位':<10}{'市場人気':<10}{'3連単':>9}")
for r in sorted(rows, key=lambda r: -r["pay"])[:12]:
    print(f"{r['date']:<10}{VN.get(r['venue'],'?'):<5}"
          f"{'-'.join(map(str, r['finish'])):<8}"
          f"{'-'.join(map(str, r['mrank'])):<10}"
          f"{'-'.join(map(str, r['krank'])):<10}{r['pay']:>8,}円")
