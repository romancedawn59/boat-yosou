# -*- coding: utf-8 -*-
"""検証㉒: 一桁オッズ判定の新基準案「100円複2点のオッズ水準」(2026-08-09ケンさん発案)

    py -X utf8 test/verify_cheap_trio_odds_rule.py

■ ケンさんの案(事前登録)
⑰プランの3連複100円の2点(◎▲△複と保険複○▲△)の15分前オッズを見て:
  両方15倍以上(min≥15) → 購入確定
  両方10倍以上(10≤min<15) → 購入検討
  どちらかが一桁(min<10) → 見送り
現行基準(プラン複4点中の一桁がちょうど2点=○)との比較を同一レースで行う。

■ 方法
⑭検証(verify_odds_single_digit_rule.py)と同一の枠組み:
月次学習(2026-05〜08)×5場×スナップショットが揃うレース。
帯=30-35(要オッズ確認の対象帯)を主対象、20-30(本命)を参考。
プラン全体(⑰構成1,400円)を買った場合の回収率・ガミ率を区分別に測る。

■ 注意
⑭(35R)同様に区分ごとの標本は小さい。結論は「表示基準の変更に足るか」の
参考であり、同帯データの再利用(⑭→㉒)のため過剰適合に留意して読む。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

conn = db.connect(DB_PATH)
df = build_training_set(conn)
snap = defaultdict(dict)
for rid, comb, odds in conn.execute(
        "SELECT race_id, combination, odds FROM odds "
        "WHERE bet_type = '3連複' AND odds > 0"):
    snap[rid][comb] = odds
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-05-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

agg = defaultdict(lambda: [0, 0, 0, 0, 0])  # {(帯, 区分): [st, rt, n, gami, plus]}
for m in ("2026-05", "2026-06", "2026-07", "2026-08"):
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, g in month_df.groupby("race_id"):
        if not payout_map[rid] or rid not in snap:
            continue
        if int(g["venue_code"].iloc[0]) not in TARGET_VENUE_CODES:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        top = ranked[0]["prob"]
        band = ("30-35帯(要注目)" if 0.30 <= top < 0.35
                else "20-30帯(本命)" if 0.20 <= top < 0.30 else None)
        if band is None:
            continue
        plan = P.ken_portfolio("荒れ注意", ranked, [],
                               P.picks_katsu(P.normalize_probs(ranked)))
        fuku_all = [(comb, y) for bt, comb, y, _s in plan if bt == "3連複"]
        odds_all = [snap[rid].get(comb) for comb, _y in fuku_all]
        if any(o is None for o in odds_all):
            continue
        # 現行基準: 複4点中の一桁点数
        singles = sum(1 for o in odds_all if o < 10.0)
        cur = ("現行○(一桁2点)" if singles == 2
               else "現行△(0-1点)" if singles <= 1 else "現行×(3点+)")
        # 新基準: 100円複2点(◎▲△複・保険複)の最低オッズ
        cheap = [o for (c, y), o in zip(fuku_all, odds_all) if y == 100]
        if len(cheap) != 2:
            continue
        mn = min(cheap)
        new = ("新・購入確定(min15倍+)" if mn >= 15
               else "新・購入検討(min10-15)" if mn >= 10
               else "新・見送り(一桁あり)")
        pay = payout_map[rid]
        st = sum(y for _, _, y, _ in plan)
        rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _s in plan)
        for k in (cur, new):
            s = agg[(band, k)]
            s[0] += st
            s[1] += rt
            s[2] += 1
            if rt and rt < st:
                s[3] += 1
            elif rt:
                s[4] += 1

KEYS = ("新・購入確定(min15倍+)", "新・購入検討(min10-15)", "新・見送り(一桁あり)",
        "現行○(一桁2点)", "現行△(0-1点)", "現行×(3点+)")
for band in ("30-35帯(要注目)", "20-30帯(本命)"):
    print(f"\n=== {band}(5場・スナップショットあり・⑰構成1,400円) ===")
    for k in KEYS:
        st, rt, n, gami, plus = agg[(band, k)]
        if n:
            print(f"  {k:<20} {n:>4,}R 回収率{rt/st:>7.1%} ガミ率{gami/n:>6.1%} "
                  f"プラス率{plus/n:>6.1%} 損益{rt-st:+,}円")
