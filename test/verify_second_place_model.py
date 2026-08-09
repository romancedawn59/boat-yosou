# -*- coding: utf-8 -*-
"""検証⑲: 2着率・3着率の専用モデル(Benterの先)の候補精査(2026-08-09ケンさん指示)

    py -X utf8 test/verify_second_place_model.py

■ ケンさんの精査項目(現行の計算方法は変更しない・候補としての評価のみ)
① 算出できるのか(実装可否)
② その的中率(現行Harville+Benter導出との精度比較)
③ 的中した時の払戻額(経済性: 同条件の紙上買いでの回収率・平均配当)
④ トークン・演算負荷を鑑みて開催全レースに適用できるのか(実測時間)

■ 方法
walk-forward 5fold(backtest.py同一)。各foldで1着モデルに加えて
「2着か」「3着か」を正解ラベルにした同一特徴量のLightGBMを追加学習。
並び確率エンジン2種を比較:
  A) 現行: P1から Harville+Benter(λ0.70/μ0.50)で導出
  B) 候補: P(a,b,c)=P1(a)×[P2m(b)/Σ残]×[P3m(c)/Σ残](専用モデルの逐次正規化)
評価(fold2-5): 2着的中率(そのエンジンの2着候補1位が実際に2着)、
3連単log-loss、紙上買い(各エンジンの確率上位6点×100円)の的中率/回収率/平均配当。
帯別(超混戦<20%/本命20-35%/堅め50%+)も併記。

■ 事前登録(採否はここでは決めない)
「有望」と報告する条件: B)が (1)2着的中率で上回る (2)log-lossで上回る
(3)紙上買い回収率で悪化しない、の3つ全て。1つでも欠けたら課題を正直に記す。
9/1判断会の材料。購入ルール・本番コードは不変。
"""
import sys
import time
from collections import defaultdict
from math import log

import numpy as np

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set

conn = db.connect(DB_PATH)
df = build_training_set(conn)
res_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id WHERE r.date >= ?", (TEST_START,)):
    res_all[rid][lane] = ao
payout_map = {}
for rid, comb, amt in conn.execute(
    "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id "
    "WHERE p.bet_type = '3連単' AND r.date >= ?", (TEST_START,)):
    payout_map.setdefault(rid, {})[comb] = amt or 0
conn.close()

df["is_second"] = (df["arrival_order"] == 2).astype(int)
df["is_third"] = (df["arrival_order"] == 3).astype(int)


def train_pos(train_df, label):
    ds = lgb.Dataset(train_df[FEATURE_COLUMNS], label=train_df[label],
                     categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(
        {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
         "min_data_in_leaf": 50, "feature_fraction": 0.9, "verbosity": -1,
         "seed": 7},
        ds, num_boost_round=400)


test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

BANDS = [("超混戦(<20%)", 0.0, 0.20), ("本命帯(20-35%)", 0.20, 0.35),
         ("堅め(50%+)", 0.50, 1.01), ("全体", -1, 2.0)]
M = lambda: {"n": 0, "hit2": {"A": 0, "B": 0}, "ll": {"A": 0.0, "B": 0.0},
             "bet": {"A": [0, 0, 0], "B": [0, 0, 0]}}   # bet=[st, rt, hits]
agg = {name: M() for name, *_ in BANDS}
train_sec = predict_sec = 0.0
n_pred_rows = 0

for i in range(1, N_FOLDS):        # fold2-5を評価(fold1は学習専用)
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中(勝/2着/3着の3モデル)...", flush=True)
    t0 = time.perf_counter()
    b_win = train_fold(train_df)
    t_win = time.perf_counter() - t0
    t0 = time.perf_counter()
    b_2nd = train_pos(train_df, "is_second")
    b_3rd = train_pos(train_df, "is_third")
    train_sec += time.perf_counter() - t0
    print(f"  1着モデル{t_win:.0f}秒 / 2着+3着モデル追加{train_sec:.0f}秒(累計)",
          flush=True)

    t0 = time.perf_counter()
    fold_df["p1"] = b_win.predict(fold_df[FEATURE_COLUMNS])
    fold_df["p2m"] = b_2nd.predict(fold_df[FEATURE_COLUMNS])
    fold_df["p3m"] = b_3rd.predict(fold_df[FEATURE_COLUMNS])
    predict_sec += time.perf_counter() - t0
    n_pred_rows += len(fold_df)

    for rid, g in fold_df.groupby("race_id"):
        arr = {l: a for l, a in res_all.get(rid, {}).items() if a}
        pays = payout_map.get(rid)
        if len(arr) < 3 or not pays:
            continue
        top3 = sorted(arr, key=lambda l: arr[l])[:3]
        actual = tuple(top3)
        lanes = list(g["lane"].astype(int))
        if len(lanes) < 6:
            continue
        p1 = dict(zip(lanes, g["p1"]))
        ranked = [{"lane": l, "prob": p1[l]} for l in lanes]
        probs = P.normalize_probs(ranked)
        top_raw = max(p1.values())

        # A) 現行エンジン
        triA = P.trifecta_probs(probs)
        # B) 専用モデルエンジン(逐次正規化)
        p2m = dict(zip(lanes, g["p2m"]))
        p3m = dict(zip(lanes, g["p3m"]))
        triB = {}
        for a in lanes:
            d2 = sum(p2m[x] for x in lanes if x != a)
            if d2 <= 0:
                continue
            for b in lanes:
                if b == a:
                    continue
                d3 = sum(p3m[x] for x in lanes if x not in (a, b))
                if d3 <= 0:
                    continue
                for c in lanes:
                    if c in (a, b):
                        continue
                    triB[(a, b, c)] = probs[a] * (p2m[b] / d2) * (p3m[c] / d3)
        sB = sum(triB.values())
        triB = {k: v / sB for k, v in triB.items()}
        sA = sum(triA.values())
        triA = {k: v / sA for k, v in triA.items()}

        # 2着候補1位: A=P2周辺化 / B=モデル素点
        p2A = defaultdict(float)
        for (a, b, c), p in triA.items():
            p2A[b] += p
        top2A = max(p2A, key=p2A.get)
        top2B = max(p2m, key=p2m.get)

        for name, lo, hi in BANDS:
            if name != "全体" and not (lo <= top_raw < hi):
                continue
            m = agg[name]
            m["n"] += 1
            m["hit2"]["A"] += top2A == actual[1]
            m["hit2"]["B"] += top2B == actual[1]
            for eng, tri in (("A", triA), ("B", triB)):
                m["ll"][eng] += -log(max(tri.get(actual, 1e-9), 1e-9))
                top6 = sorted(tri, key=tri.get, reverse=True)[:6]
                st = 600
                rt = sum(pays.get("-".join(map(str, k)), 0) for k in top6)
                bet = m["bet"][eng]
                bet[0] += st
                bet[1] += rt
                bet[2] += 1 if rt else 0

print("\n===== ①算出可否 =====")
print("可能。既存の特徴量28本+着順ラベルだけで追加学習でき、本番コードの変更は不要")

print("\n===== ②的中率 / ③払戻(紙上・確率上位6点×100円) =====")
for name, *_ in BANDS:
    m = agg[name]
    if m["n"] == 0:
        continue
    print(f"\n[{name}] {m['n']:,}R")
    for eng, label in (("A", "現行Harville+Benter"), ("B", "専用モデル(候補)")):
        st, rt, hits = m["bet"][eng]
        avg_pay = (rt / hits) if hits else 0
        print(f"  {label:<22} 2着的中{m['hit2'][eng]/m['n']:>6.1%} "
              f"log-loss{m['ll'][eng]/m['n']:>6.3f} | "
              f"上位6点買い: 的中{hits/m['n']:>6.1%} 回収{rt/st:>7.1%} "
              f"的中時平均{avg_pay:,.0f}円")

print("\n===== ④演算負荷 =====")
print(f"追加学習(2着+3着モデル×4fold): {train_sec:.0f}秒 "
      f"(月次再学習に+{train_sec/4:.0f}秒/月)")
print(f"推論: {n_pred_rows:,}行を{predict_sec:.1f}秒 "
      f"→ 1日約170レース(1,000行)なら{predict_sec/n_pred_rows*1000:.2f}秒")
print("トークン消費: ゼロ(LLM不使用・ローカルLightGBMのみ)。全レース適用は余裕")
