# -*- coding: utf-8 -*-
"""ケンさんの速報基準「3連複オッズ一桁が2つ(まで)なら購入チャンス」の検証
(2026-08-02発案・要オッズ確認フラグの実オッズ版)

    py -X utf8 test/verify_odds_single_digit_rule.py

■ 基準(事前登録)
プランの3連複4点(①②③+保険複)の15分前スナップショットオッズのうち、
一桁(10倍未満)の点数が2つ以下 → 「ガミりにくい形」=購入チャンス。
3つ以上一桁 → 全体が安い=ガミ地獄の形。
※オッズは締切までにさらに下がるのは織り込み済み(スナップショットは15分前)

■ 検証
帯: 30〜35%(要注目帯・本命候補外)を主対象、20〜30%(本命帯)を参考。
月次学習(2026-05〜07)×スナップショットが揃うレースで、
一桁点数別のROI・ガミ率を測る。
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
for m in ("2026-05", "2026-06", "2026-07"):
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
        fuku = [(comb, y) for bt, comb, y, _s in plan if bt == "3連複"]
        odds_known = [snap[rid].get(comb) for comb, _y in fuku]
        if any(o is None for o in odds_known):
            continue
        singles = sum(1 for o in odds_known if o < 10.0)
        key = f"一桁{singles}点" if singles <= 2 else "一桁3点以上"
        chance = "○チャンス(一桁≤2)" if singles <= 2 else "×見送り(一桁≥3)"
        pay = payout_map[rid]
        st = sum(y for _, _, y, _ in plan)
        rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _s in plan)
        for k in (chance, key):
            s = agg[(band, k)]
            s[0] += st
            s[1] += rt
            s[2] += 1
            if rt and rt < st:
                s[3] += 1
            elif rt:
                s[4] += 1

for band in ("30-35帯(要注目)", "20-30帯(本命)"):
    print(f"\n=== {band}(5場・スナップショットあり・v2.1構成) ===")
    for k in ("○チャンス(一桁≤2)", "×見送り(一桁≥3)",
              "一桁0点", "一桁1点", "一桁2点"):
        st, rt, n, gami, plus = agg[(band, k)]
        if n:
            print(f"  {k:<16} {n:>4,}R 回収率{rt/st:>7.1%} ガミ率{gami/n:>6.1%} "
                  f"プラス率{plus/n:>6.1%} 損益{rt-st:+,}円")
