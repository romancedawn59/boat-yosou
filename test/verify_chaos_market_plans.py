# -*- coding: utf-8 -*-
"""検証㉓: 総混沌形(独自カテゴリ)を勝てる買い方の探索・3案(2026-08-09ケンさん指示)

    py -X utf8 test/verify_chaos_market_plans.py

■ 問い(ケンさんの設計)
総混沌形(モデル30-35帯×市場がプラン複4点を安くしない=市場がモデルの見立てを
否定しているレース)は現行3構成すべてが水面下(35/36/77%)。
「市場側の情報に乗り換える」買い方なら勝てるか。予算1,000〜2,000円で3案。

■ 仮説の構造
総混沌形では市場が何か(展示・進入気配・機力)を知っている。ならば
15分前スナップショットの3連単オッズから市場の見立てを復元し、
(A)市場の順位で形を組む (B)市場とモデルの乖離が最大の艇に乗る
(C)市場の本線の並びをそのまま買う、の3経路を試す。

■ 事前登録(結果を見る前に固定)
- 総混沌形の定義: 30-35帯(モデル1位生値)×⑰プラン複4点の15分前オッズの
  一桁が0-1点。主集計=全24場(標本優先)・参考=5場
- 市場順位の復元: 各艇の市場勝率スコア=Σ(その艇が1着の3連単の1/オッズ)を正規化
- 案A「市場乗り換え⑰」1,400円: ⑰構成を市場順位の上位4艇で組む
- 案B「乖離艇軸複」1,200円: 軸=市場勝率−モデル勝率の差が最大の艇。
  相手=軸を除く市場上位4艇。3連複 軸-相手2艇 の6点×200円
- 案C「市場本線単」1,200円: 市場オッズ最低(=最有力)の3連単3点×400円
- 判定: これは発見データでの探索。回収率120%超の案のみ「前向き紙上追跡
  (8月〜)→10月判定」へ登録する。即採用はしない。全滅なら
  「総混沌形は現データでは攻略不能・見送り継続」と正直に結論する
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
snap_f = defaultdict(dict)
snap_t = defaultdict(dict)
for rid, bt, comb, odds in conn.execute(
        "SELECT race_id, bet_type, combination, odds FROM odds WHERE odds > 0"):
    (snap_t if bt == "3連単" else snap_f)[rid][comb] = odds
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-05-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan17(l):
    r1, r2, r3, r4 = l[:4]
    return [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
            ("3連複", trio(r1, r3, r4), 100),
            ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200),
            ("3連複", trio(r2, r3, r4), 100),
            ("3連単", f"{r3}-{r2}-{r1}", 100), ("3連単", f"{r4}-{r2}-{r1}", 100),
            ("3連単", f"{r4}-{r2}-{r1}", 200)]


agg = defaultdict(lambda: [0, 0, 0, 0])   # {(scope, arm): [st, rt, n, hit]}
for m in ("2026-05", "2026-06", "2026-07", "2026-08"):
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(m, flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, g in month_df.groupby("race_id"):
        if not payout_map[rid] or rid not in snap_f or rid not in snap_t:
            continue
        gs = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in gs.iterrows()]
        if len(ranked) < 5:
            continue
        top = ranked[0]["prob"]
        if not (0.30 <= top < 0.35):
            continue
        model_lanes = [r["lane"] for r in ranked]
        fuku = [comb for bt, comb, _y in plan17(model_lanes) if bt == "3連複"]
        odds_known = [snap_f[rid].get(c) for c in fuku]
        if any(o is None for o in odds_known):
            continue
        singles = sum(1 for o in odds_known if o < 10.0)
        if singles > 1:
            continue                      # 総混沌形のみ

        # 市場の見立てを復元
        tan = snap_t[rid]
        w = defaultdict(float)
        for comb, o in tan.items():
            w[int(comb.split("-")[0])] += 1.0 / o
        tot_w = sum(w.values()) or 1.0
        market_p = {l: w[l] / tot_w for l in w}
        market_lanes = sorted(market_p, key=market_p.get, reverse=True)
        if len(market_lanes) < 5:
            continue
        norm = P.normalize_probs(ranked)
        # 乖離艇=市場勝率−モデル勝率の差が最大の艇
        gap_lane = max(market_p, key=lambda l: market_p[l] - norm.get(l, 0.0))
        partners = [l for l in market_lanes if l != gap_lane][:4]
        planB = []
        for i in range(len(partners)):
            for j in range(i + 1, len(partners)):
                planB.append(("3連複", trio(gap_lane, partners[i], partners[j]), 200))
        top3tan = sorted(tan, key=tan.get)[:3]
        arms = {
            "A 市場乗り換え⑰(1,400円)": plan17(market_lanes),
            "B 乖離艇軸複(1,200円)": planB,
            "C 市場本線単(1,200円)": [("3連単", c, 400) for c in top3tan],
        }
        scopes = ["全24場"]
        if int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES:
            scopes.append("5場")
        pay = payout_map[rid]
        for arm, plan in arms.items():
            merged = defaultdict(int)
            for bt, comb, y in plan:
                merged[(bt, comb)] += y
            st = sum(merged.values())
            rt = sum(pay.get(k, 0) * y // 100 for k, y in merged.items())
            for sc in scopes:
                a = agg[(sc, arm)]
                a[0] += st
                a[1] += rt
                a[2] += 1
                a[3] += 1 if rt else 0

ARMS = ("A 市場乗り換え⑰(1,400円)", "B 乖離艇軸複(1,200円)", "C 市場本線単(1,200円)")
for sc in ("全24場", "5場"):
    print(f"\n=== 総混沌形・{sc} ===")
    for arm in ARMS:
        st, rt, n, hit = agg[(sc, arm)]
        if n:
            print(f"  {arm:<20} {n:>4}R 回収率{rt/st:>7.1%} 的中率{hit/n:>6.1%} "
                  f"損益{rt-st:+,}円")
print("\n判定(事前登録): 120%超の案のみ前向き紙上追跡へ。全滅なら見送り継続を結論")
