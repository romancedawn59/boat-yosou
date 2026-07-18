# -*- coding: utf-8 -*-
"""ケンさんv2案(2026-07-18発案)の検証CLI

    py -X utf8 test/verify_ken_v2.py

案: 「現行(5場×35%未満・1位勝率が低い順)を上位MAX6レースに絞り、
全場×1位勝率20%未満(超混戦)を出現するたび追加する」。
比較対象: 現行cap10 / 現行cap6 / 全場20%のみ / ケンさん案(cap6+超混戦の和集合)。
ウォークフォワードはbacktest.py同一fold・買い目は全ルール共通の検証済みV2構成+C。
日次損益系列から最大DD・最長連敗も算出する(v2判断会の主要材料)。
注意: DDはfoldモデルの揺らぎに敏感(現行cap10で-5万〜-10万の幅を観測)。点推定を過信しない。
"""
import sys
from collections import defaultdict
from itertools import groupby

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import challengers as CH
import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

conn = db.connect(DB_PATH)
df = build_training_set(conn)
actual = defaultdict(dict)
for rid, lane, order in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,)):
    actual[rid][order] = lane
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]
n_days = len(dates)

# 全レースのコンテキスト(全場)
ctxs = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        if 1 not in actual[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]),
                     "venue": int(g["venue_code"].iloc[0]),
                     "top": ranked[0]["prob"], "ranked": ranked, "probs": probs})
ctxs.sort(key=lambda c: (c["date"], c["rid"]))

graded = {}
def grade(c):
    if c["rid"] in graded:
        return graded[c["rid"]]
    plan = P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(c["probs"]))
    pay = payout_map[c["rid"]]
    stake = sum(y for _, _, y, _ in plan)
    ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
    res = actual[c["rid"]]
    santan = pay.get(("3連単", f"{res.get(1)}-{res.get(2)}-{res.get(3)}"), 0)
    graded[c["rid"]] = {"stake": stake, "ret": ret,
                        "out": CH.classify_outcome(stake, ret, santan)}
    return graded[c["rid"]]

def select_day(day_ctxs, rule):
    """ruleに従い1日分の購入レース集合を返す"""
    hon = sorted((c for c in day_ctxs
                  if c["venue"] in TARGET_VENUE_CODES and c["top"] < 0.35),
                 key=lambda c: c["top"])
    kon = [c for c in day_ctxs if c["top"] < 0.20]  # 全場×超混戦
    if rule == "現行cap10":
        return {c["rid"] for c in hon[:10]}
    if rule == "現行cap6":
        return {c["rid"] for c in hon[:6]}
    if rule == "全場20%のみ":
        return {c["rid"] for c in kon}
    if rule == "ケンさん案":  # 現行cap6 + 全場超混戦を都度追加(重複は1回)
        return {c["rid"] for c in hon[:6]} | {c["rid"] for c in kon}
    raise ValueError(rule)

RULES = ("現行cap10", "現行cap6", "全場20%のみ", "ケンさん案")
print(f"\n=== ケンさんv2案の比較(walk-forward {n_days}日・全fold) ===")
print(f"{'ルール':<10}{'R数':>6}{'R/日':>6}{'日予算平均':>9}{'的中率':>8}{'回収率':>8}"
      f"{'最大1発除き':>10}{'ガミ率':>7}{'損益':>12}{'最大DD':>10}{'最長連敗':>7}")
for rule in RULES:
    daily_pnl = {}
    tot = {"n": 0, "hits": 0, "junto": 0, "stake": 0, "ret": 0, "max_ret": 0}
    day_stakes = []
    for d, grp in groupby(ctxs, key=lambda c: c["date"]):
        day_list = list(grp)
        sel = select_day(day_list, rule)
        pnl = 0.0
        dstake = 0
        for c in day_list:
            if c["rid"] not in sel:
                continue
            g = grade(c)
            tot["n"] += 1
            tot["stake"] += g["stake"]
            tot["ret"] += g["ret"]
            tot["max_ret"] = max(tot["max_ret"], g["ret"])
            dstake += g["stake"]
            pnl += g["ret"] - g["stake"]
            if g["ret"]:
                tot["hits"] += 1
                if g["out"] == "順当":
                    tot["junto"] += 1
        daily_pnl[d] = pnl
        day_stakes.append(dstake)
    series = [daily_pnl[d] for d in sorted(daily_pnl)]
    roi = tot["ret"] / tot["stake"]
    roi_ex = (tot["ret"] - tot["max_ret"]) / tot["stake"]
    junto = tot["junto"] / tot["hits"] if tot["hits"] else 0
    avg_budget = sum(day_stakes) / len(day_stakes)
    print(f"{rule:<10}{tot['n']:>6}{tot['n']/n_days:>6.1f}{avg_budget:>8,.0f}円"
          f"{tot['hits']/tot['n']:>8.1%}{roi:>8.1%}{roi_ex:>10.1%}{junto:>7.1%}"
          f"{tot['ret']-tot['stake']:>+11,.0f}円{CH.max_drawdown(series):>9,.0f}円"
          f"{CH.longest_losing_streak(series):>6}日")
