# -*- coding: utf-8 -*-
"""堅め帯の本線600円: 「複400+山田単200」vs「3連単ボックス6点」vs「複のみ600」
(2026-07-30夜・ケンさん発案)

    py -X utf8 test/verify_katame_trio_box.py

■ 問い
堅め表示プランは本線トリオに 3連複400円+同トリオの山田単(確率最大の並び)200円
=計600円を張る。同じ600円なら3連単ボックス6点×100円の方が取りこぼしが
ないのではないか(2026-07-30丸亀11Rの実例: 現行形880円 vs ボックス1,810円)。

■ 論点の整理
昨夜の「複vsボックス」(超混戦・オラクル条件で複272%>箱263%)とは別の問い。
今回は「単1点への200円集中 vs 6並びへの分散」であり、勝敗は
「モデルの並び順位付けが市場より鋭いか」で決まる。既存知見は中立〜否定的
(市場レポート(c): 順列歪み比1.006・最有力順列の一致率63.6%、検証F: 確率上位の
単は市場と同じ買い方)だが、堅め帯では未測定。

■ 事前登録(実行前に固定)
対象: 全場・堅め帯(1位生値50%以上)。本線トリオ=P.trio_top(probs,2)[0]、
山田単=P.picks_yamada(probs)[0](現行コードそのまま)。
アーム(600円固定・この3つだけ):
  A 現行形: 複(本線トリオ)400円+山田単200円
  B ボックス: 本線トリオの3連単6点×100円
  C 複のみ: 複(本線トリオ)600円
判定: 回収率の高い順を報告。この帯は購入対象外(表示・裁量の器の話)なので
採否ではなく「裁量買いの推奨形」を決める材料とする。5場スコープも参考表示。
"""
import sys
from collections import defaultdict
from itertools import permutations

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

agg = defaultdict(lambda: [0, 0, 0, 0])   # {(scope,arm):[st,rt,n,hit]}
b0_out = 0
n_all = 0
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
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 4 or ranked[0]["prob"] < 0.50:
            continue
        probs = P.normalize_probs(ranked)
        trio_comb = P.trio_top(probs, 1)[0][0]          # '1=2=3'形式
        trio_lanes = [int(x) for x in trio_comb.split("=")]
        b0 = P.picks_yamada(probs)[0]                    # ('3連単','a-b-c',p)
        n_all += 1
        if set(int(x) for x in b0[1].split("-")) != set(trio_lanes):
            b0_out += 1
        arms = {
            "A 現行形(複400+単200)": [("3連複", trio_comb, 400), (b0[0], b0[1], 200)],
            "B ボックス6点": [("3連単", f"{a}-{b}-{c}", 100)
                            for a, b, c in permutations(trio_lanes)],
            "C 複のみ600": [("3連複", trio_comb, 600)],
        }
        scopes = ["全場"]
        if int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES:
            scopes.append("5場")
        pay = payout_map[rid]
        for arm, bets in arms.items():
            st = sum(y for _, _, y in bets)
            rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y in bets)
            for sc in scopes:
                a = agg[(sc, arm)]
                a[0] += st
                a[1] += rt
                a[2] += 1
                a[3] += 1 if rt else 0

print(f"\n堅め帯(1位50%以上)。山田単が本線トリオ外だった率: {b0_out/n_all:.1%}")
for sc in ("全場", "5場"):
    n = agg[(sc, "A 現行形(複400+単200)")][2]
    print(f"\n=== [{sc}] {n:,}R(各600円) ===")
    for arm in ("A 現行形(複400+単200)", "B ボックス6点", "C 複のみ600"):
        st, rt, _n, hit = agg[(sc, arm)]
        if st:
            print(f"  {arm:<20} 何か当たる率{hit/_n:>6.1%} 回収率{rt/st:>7.1%} "
                  f"損益{rt-st:+,}円")
