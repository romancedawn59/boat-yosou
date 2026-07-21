# -*- coding: utf-8 -*-
"""配当帯を狙い撃つ買い方の検証(2026-07-21ケンさん発案「40倍想定を当てまくる」)

    py -X utf8 test/verify_payout_bands.py

問い: 買い目を「想定配当の帯」で選ぶと成績は上がるか。
今までの検証は買い目の形(構成9案)・買うレース(選別)・期待値フィルタ(検証⑥)を
動かしてきたが、「狙う配当帯を指定する」は未検証だった。

方法:
- 選別レース(超混戦帯 top<20% / 本命帯 top<30%)で3連単120通りの発生確率を計算
- 想定オッズ = 0.75 / 確率(控除率25%を織り込んだ市場想定値)
- 帯ごとに確率上位3点を1点100円で購入(=300円/レース)して回収率を測る
- 比較対象: 現行V2構成6点(1,000円)を同じレース集合に当てた場合

注意: 朝買いのため実オッズは使えない(使うと未来を見ることになる)。
モデル確率から想定配当を出しているので、市場が同じ目をどう評価したかとはズレる。
このズレ自体が検証⑥で「大穴領域は市場の方が正確」と出た論点。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

TAKEOUT = 0.75  # 3連単の払戻率(控除率25%)
# (下限倍, 上限倍, ラベル)
BANDS = [(5, 10, "5-10倍"), (10, 20, "10-20倍"), (20, 40, "20-40倍"),
         (40, 80, "40-80倍"), (80, 200, "80-200倍"), (200, 10**9, "200倍〜")]
POINTS = 3  # 各帯で買う点数(確率上位から)

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

ctxs = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        res = actual[rid]
        if 1 not in res or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        ctxs.append({"rid": rid, "top": ranked[0]["prob"], "fold": i + 1,
                     "ranked": ranked, "probs": probs})

def measure(scope_name, pred):
    sel = [c for c in ctxs if pred(c["top"])]
    print(f"\n=== {scope_name}({len(sel):,}レース) ===")
    print(f"{'狙う帯':<11}{'買った点':>9}{'的中':>6}{'的中率':>8}"
          f"{'平均払戻':>10}{'投資':>11}{'回収':>12}{'回収率':>9}")
    for lo, hi, lbl in BANDS:
        stake = ret = hits = pts = 0
        pays = []
        for c in sel:
            tri = P.trifecta_probs(c["probs"])
            # 想定オッズがこの帯に入る目を、確率上位からPOINTS点
            cands = sorted(
                ((k, p) for k, p in tri.items()
                 if p > 0 and lo <= TAKEOUT / p < hi),
                key=lambda x: -x[1])[:POINTS]
            if not cands:
                continue
            pay = payout_map[c["rid"]]
            for (a, b, cc), _p in cands:
                comb = f"{a}-{b}-{cc}"
                amt = pay.get(("3連単", comb), 0)
                stake += 100
                pts += 1
                if amt:
                    ret += amt
                    hits += 1
                    pays.append(amt)
        if not stake:
            continue
        avg = sum(pays) / len(pays) if pays else 0
        print(f"{lbl:<11}{pts:>9,}{hits:>6,}{hits/pts:>8.2%}{avg:>9,.0f}円"
              f"{stake:>10,}円{ret:>11,}円{ret/stake:>9.1%}")

    # 比較: 現行V2構成6点を同じレース集合に当てた場合
    stake = ret = hits = 0
    for c in sel:
        plan = P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(c["probs"]))
        pay = payout_map[c["rid"]]
        s = sum(y for _, _, y, _ in plan)
        r = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _ in plan)
        stake += s
        ret += r
        hits += 1 if r else 0
    print(f"{'現行V2構成6点':<11}{len(sel)*6:>9,}{hits:>6,}{hits/len(sel):>8.2%}"
          f"{'—':>10}{stake:>10,}円{ret:>11,}円{ret/stake:>9.1%}")

measure("超混戦帯(1位勝率20%未満)", lambda t: t < 0.20)
measure("本命帯(1位勝率30%未満)", lambda t: t < 0.30)


def robustness(scope_name, pred, lo, hi, lbl):
    """有望に見えた帯の頑健性: 最大1発除き・fold別・払戻分布を見る。
    少数の大穴に依存していないかを確認する(このプロジェクトの標準手順)"""
    sel = [c for c in ctxs if pred(c["top"])]
    per_fold = defaultdict(lambda: {"stake": 0, "ret": 0, "hits": 0})
    pays = []
    stake = ret = 0
    for c in sel:
        tri = P.trifecta_probs(c["probs"])
        cands = sorted(((k, p) for k, p in tri.items()
                        if p > 0 and lo <= TAKEOUT / p < hi),
                       key=lambda x: -x[1])[:POINTS]
        if not cands:
            continue
        pay = payout_map[c["rid"]]
        for (a, b, cc), _p in cands:
            amt = pay.get(("3連単", f"{a}-{b}-{cc}"), 0)
            stake += 100
            per_fold[c["fold"]]["stake"] += 100
            if amt:
                ret += amt
                pays.append(amt)
                per_fold[c["fold"]]["ret"] += amt
                per_fold[c["fold"]]["hits"] += 1
    print(f"\n--- 頑健性チェック: {scope_name} × {lbl} ---")
    print(f"投資{stake:,}円 回収{ret:,}円 回収率{ret/stake:.1%} 的中{len(pays)}件")
    pays.sort(reverse=True)
    for n in (1, 2, 3):
        if len(pays) >= n:
            r2 = ret - sum(pays[:n])
            print(f"  最大{n}発除き: {r2/stake:>8.1%}")
    print(f"  的中payoutの上位5件: {', '.join(f'{p:,}円' for p in pays[:5])}")
    print("  fold別:")
    for f in sorted(per_fold):
        v = per_fold[f]
        print(f"    fold{f}: 投資{v['stake']:>7,}円 回収{v['ret']:>9,}円 "
              f"{v['ret']/v['stake'] if v['stake'] else 0:>8.1%} "
              f"(的中{v['hits']}件)")

robustness("超混戦帯", lambda t: t < 0.20, 5, 10, "5-10倍")
robustness("本命帯", lambda t: t < 0.30, 5, 10, "5-10倍")
robustness("本命帯", lambda t: t < 0.30, 40, 80, "40-80倍(提案)")
