# -*- coding: utf-8 -*-
"""「要オッズ確認」フラグの選定基準検証(2026-08-02ケンさん発案)

    py -X utf8 test/verify_odds_check_flag.py

■ 構想
30〜35%帯(購入対象外の観測帯)のうち「全点が損益分岐線超えの形」のレースに
朝の時点で「要オッズ確認」の印を付け、昼のオッズタブで実オッズを目視→
買うかはケンさんの裁量(記録は裁量枠)。システムの自動オッズ判断はしない
(検証⑥でEVフィルタ棄却・朝買い原則は不変)。

■ 事前登録(実行前に固定)
帯: 5場・1位生値30〜35%(要注目帯)・v2.1本命構成1,000円
形の定義: プラン全点で「想定配当×賭金≥1,000円」
  = 200円点は自信≤16%(想定5倍≥) かつ 100円点は自信≤8%(想定10倍≥)
  (ケンさんの「100円複が10倍超ならガミりにくい」の一般化)
判定: 形ありが ①ガミ率で形なしより10pt以上低い かつ ②回収率100%超なら
フラグ採用(表示+紙上記録)。回収率90-100%は「表示のみ採用・裁量の入口」として採用
(このフラグは購入対象ではないため基準は緩め。実弾判断は9/1に紙上+裁量実績で)。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

conn = db.connect(DB_PATH)
df = build_training_set(conn)
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

split = {True: [0, 0, 0, 0, 0], False: [0, 0, 0, 0, 0]}  # [st, rt, n, gami, plus]
days = defaultdict(int)
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        if not payout_map[rid]:
            continue
        if int(g["venue_code"].iloc[0]) not in TARGET_VENUE_CODES:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5 or not (0.30 <= ranked[0]["prob"] < 0.35):
            continue
        probs = P.normalize_probs(ranked)
        plan = P.ken_portfolio("荒れ注意", ranked, [], P.picks_katsu(probs))
        confs = [(P.combo_prob(bt, comb, probs), y) for bt, comb, y, _s in plan]
        # 形: 全点で自信が分岐線以下(200円点16%・100円点8%)
        flag = all(cf <= (0.16 if y >= 200 else 0.08) for cf, y in confs)
        pay = payout_map[rid]
        st = sum(y for _, _, y, _ in plan)
        rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _s in plan)
        s = split[flag]
        s[0] += st
        s[1] += rt
        s[2] += 1
        if rt and rt < st:
            s[3] += 1
        elif rt >= st and rt:
            s[4] += 1
        if flag:
            days[g["date"].iloc[0]] += 1

print(f"\n=== 5場・30〜35%帯(要注目帯)・v2.1構成 ===")
for flag, label in ((True, "形あり(全点分岐超え)=要オッズ確認候補"), (False, "形なし")):
    st, rt, n, gami, plus = split[flag]
    if n:
        print(f"  {label:<28} {n:>4,}R 回収率{rt/st:>7.1%} ガミ率{gami/n:>6.1%} "
              f"プラス率{plus/n:>6.1%} 損益{rt-st:+,}円")
n_days = len(set(d for f in ({True}, ) for d in days))
if days:
    import statistics
    print(f"\n形ありの出現: {sum(days.values())}R / {len(days)}日 "
          f"(約{sum(days.values())/max(1,len(dates)):.1f}R/日)")
