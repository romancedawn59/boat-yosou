# -*- coding: utf-8 -*-
"""市場との乖離はレース選別に使えるか(2026-07-21ケンさん提案)

    py -X utf8 test/verify_market_divergence.py

提案: 朝は超混戦だけ買う(現行維持)。それとは別に、オッズが出てから
「モデルの自信が高いのに配当が高い=市場と乖離している」レースを見つけて買えば
もっと勝てるのではないか。朝の買い目は動かさないので「リアルタイム昇格はしない」
という確定方針には抵触しない(別レイヤーの購入機会)。

検証⑥(不採用)との違い:
  検証⑥ = 目を選ぶ(EVが高い3連単だけ買う)→ 素通しより悪化
  今回   = レースを選ぶ(乖離が大きいレースを買う)→ 未検証
うちのエッジはレース選別にあり目の価格発見にはない、という整理と整合する。
検証⑪の「挑戦者①市場相違型」(閾値7が仮置きのまま判定不能)の本検証でもある。

方法:
- 2026-05-01より前の全データで1回学習し、5/1〜6/30を評価(固定期間)
- 市場のインプライド確率を3連単オッズから復元:
  艇iが1着の20通りについて Σ(1/オッズ) を取り、控除率で正規化する
- 乖離 = モデルの1位確率 / 市場の1位確率(>1ならモデルが強気=市場が過小評価)
- 乖離の大きさで層別し、各層の荒れ率・回収率・順当決着率を見る

【事前登録】見る指標と判定:
  1. 乖離が大きい層で、現行構成の最大1発除き回収率が全体平均を上回るか
  2. その層の万舟決着率が高いか(荒れの予測力があるか)
  3. サンプルが層あたり50R未満なら「傾向」として記録するに留め判定はしない
※231レースしかないため統計的な結論は出せない。方向性の確認が目的。
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

EVAL_START, EVAL_END = "2026-05-01", "2026-06-30"
TAKEOUT = 0.75
MANSHU_MIN = 10_000

conn = db.connect(DB_PATH)
df = build_training_set(conn)

actual = defaultdict(dict)
for rid, lane, order in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date BETWEEN ? AND ? AND res.arrival_order IS NOT NULL",
    (EVAL_START, EVAL_END)):
    actual[rid][order] = lane
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date BETWEEN ? AND ?",
    (EVAL_START, EVAL_END)):
    payout_map[rid][(bt, comb)] = amt or 0
# 3連単オッズ(この期間はfinal-backfill=最終オッズ)
odds_map = defaultdict(dict)
for rid, comb, o in conn.execute(
    "SELECT o.race_id, o.combination, o.odds FROM odds o "
    "JOIN races r ON r.race_id = o.race_id "
    "WHERE r.date BETWEEN ? AND ? AND o.bet_type = '3連単' AND o.odds > 0",
    (EVAL_START, EVAL_END)):
    odds_map[rid][comb] = o
conn.close()
print(f"オッズのあるレース: {len(odds_map):,}")

train_df = df[df["date"] < EVAL_START]
eval_df = df[(df["date"] >= EVAL_START) & (df["date"] <= EVAL_END)].copy()
print(f"学習 {len(train_df):,}行 / 評価 {len(eval_df):,}行")
booster = train_fold(train_df)
eval_df["pred"] = booster.predict(eval_df[FEATURE_COLUMNS])


def market_win_probs(od: dict[str, float]) -> dict[int, float]:
    """3連単オッズ全体から各艇の市場インプライド1着確率を復元する。
    艇iが1着の目について Σ(1/オッズ) を取り、全体で正規化(控除率を除去)"""
    raw = defaultdict(float)
    total = 0.0
    for comb, o in od.items():
        if o <= 0:
            continue
        try:
            a = int(comb.split("-")[0])
        except (ValueError, IndexError):
            continue
        raw[a] += 1.0 / o
        total += 1.0 / o
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


recs = []
for rid, g in eval_df.groupby("race_id"):
    res = actual[rid]
    od = odds_map.get(rid)
    if 1 not in res or 2 not in res or 3 not in res or not payout_map[rid] or not od:
        continue
    g_sorted = g.sort_values("pred", ascending=False)
    ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
              for _, r in g_sorted.iterrows()]
    probs = P.normalize_probs(ranked)
    if len(probs) < 4:
        continue
    mkt = market_win_probs(od)
    top_lane = ranked[0]["lane"]
    if top_lane not in mkt or mkt[top_lane] <= 0:
        continue
    model_p = probs[top_lane]
    mkt_p = mkt[top_lane]
    # 市場人気順(インプライド確率の降順)でモデル1位艇が何番人気か
    order = sorted(mkt, key=lambda k: -mkt[k])
    pop_rank = order.index(top_lane) + 1

    plan = P.ken_portfolio("荒れ注意", ranked, [], P.picks_katsu(probs))
    pay = payout_map[rid]
    stake = sum(y for _, _, y, _ in plan)
    ret = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _ in plan)
    finish = f"{res[1]}-{res[2]}-{res[3]}"
    santan = pay.get(("3連単", finish), 0)
    recs.append({
        "rid": rid, "top": ranked[0]["prob"], "model_p": model_p, "mkt_p": mkt_p,
        "ratio": model_p / mkt_p, "pop_rank": pop_rank,
        "stake": stake, "ret": ret, "santan": santan,
        "manshu": santan >= MANSHU_MIN,
        "junto": res[1] == top_lane,   # モデル1位が1着=順当
    })

print(f"評価対象 {len(recs)}レース\n")


def show(title, groups):
    print(f"=== {title} ===")
    print(f"{'層':<22}{'R数':>6}{'万舟率':>8}{'順当率':>8}{'的中率':>8}"
          f"{'回収率':>9}{'最大1発除き':>11}{'損益':>11}")
    for label, rs in groups:
        if not rs:
            continue
        stake = sum(r["stake"] for r in rs)
        ret = sum(r["ret"] for r in rs)
        mx = max((r["ret"] for r in rs), default=0)
        hits = sum(1 for r in rs if r["ret"])
        print(f"{label:<22}{len(rs):>6}"
              f"{sum(1 for r in rs if r['manshu'])/len(rs):>8.1%}"
              f"{sum(1 for r in rs if r['junto'])/len(rs):>8.1%}"
              f"{hits/len(rs):>8.1%}{ret/stake:>9.1%}{(ret-mx)/stake:>11.1%}"
              f"{ret-stake:>+10,}円")
    print()


# 乖離比率(モデル確率 / 市場確率)で層別
BANDS = [(0, 0.8, "0.8未満(市場が強気)"), (0.8, 1.0, "0.8-1.0"),
         (1.0, 1.25, "1.0-1.25"), (1.25, 1.6, "1.25-1.6(モデル強気)"),
         (1.6, 99, "1.6以上(大きく強気)")]
show("乖離比率別(モデルの1位確率 ÷ 市場の1位確率)",
     [(lbl, [r for r in recs if lo <= r["ratio"] < hi]) for lo, hi, lbl in BANDS])

# モデル1位艇が市場で何番人気か
show("モデル1位艇の市場人気順",
     [(f"{k}番人気", [r for r in recs if r["pop_rank"] == k]) for k in range(1, 7)])

# 参考: 全体
show("参考: 全体", [("全レース", recs)])

# 超混戦帯だけを取り出した場合
kon = [r for r in recs if r["top"] < 0.20]
if kon:
    show(f"参考: 超混戦帯のみ({len(kon)}R)",
         [(lbl, [r for r in kon if lo <= r["ratio"] < hi]) for lo, hi, lbl in BANDS])
