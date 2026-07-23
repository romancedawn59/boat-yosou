# -*- coding: utf-8 -*-
"""上位3艇が抜けて強いレースで3連単ボックス6点は有効か
   (2026-07-23ケンさん提案)

    py -X utf8 test/verify_top3_box.py

提案: 予測上位3艇が抜けて強い(4位以下との差が大きい)レースなら、
その3艇の3連単ボックス6点(3艇の並び全通り)に賭ける手が有効ではないか。

論点: 「上位3艇が抜ける」レースは1位勝率が高い=堅めに寄るはず。堅めは
現行では見送り(回収率77-93%)。ボックスは的中率を上げるが1着固定でないぶん
配当が薄い目も買う。市場が織り込んだ本命ボックスは配当が伸びない懸念(検証⑥)。

■ まず「抜け度」の定義と分布を見る:
  抜け度 = (1位+2位+3位の勝率合計) - (4位+5位+6位)  … 上位集中度
  または top3和 が高いほど「上位3艇で決まりやすい」

■ ボックスの検証(walk-forward・全24場):
  対象を「top3和」で層別し、各層で3連単ボックス6点(200円×6=1200円→
  比較のため100円×6=600円に正規化)の的中率・回収率・最大1発除きを見る。
  現行の荒れ注意構成(選別レースのみ)とは土俵が違うので、
  ここでは純粋に「ボックスという買い方」の帯別成績を測る。

【事前登録】判定基準:
  ある抜け度の層で、3連単ボックスの最大1発除き回収率が100%を超えるか。
  超える層があれば第2段(現行選別との組み合わせ)へ。なければ不採用。
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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

recs = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        res = actual[rid]
        if 1 not in res or 2 not in res or 3 not in res or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        lanes = [r["lane"] for r in ranked]
        r1, r2, r3 = lanes[:3]
        top3_sum = probs[r1] + probs[r2] + probs[r3]   # 上位3艇の勝率合計
        # 3連単ボックス6点(上位3艇の全順列)。1点100円=600円
        box = [(a, b, c) for a in (r1, r2, r3) for b in (r1, r2, r3)
               for c in (r1, r2, r3) if len({a, b, c}) == 3]
        pay = payout_map[rid]
        finish = (res[1], res[2], res[3])
        box_ret = sum(pay.get(("3連単", f"{a}-{b}-{c}"), 0)
                      for (a, b, c) in box if (a, b, c) == finish)
        # 決着が上位3艇の組だったか(ボックス的中)
        hit = finish in box
        recs.append({
            "top1": probs[r1], "top3_sum": top3_sum,
            "venue": int(g["venue_code"].iloc[0]),
            "box_stake": 600, "box_ret": box_ret, "hit": hit,
            "finish_santan": pay.get(("3連単", f"{res[1]}-{res[2]}-{res[3]}"), 0),
        })

print(f"\n対象 {len(recs):,}レース(全24場)\n")

# 抜け度(top3_sum)で層別
BANDS = [(0.0, 0.5, "〜50%(バラける)"), (0.5, 0.6, "50-60%"),
         (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"),
         (0.8, 0.9, "80-90%"), (0.9, 1.01, "90%〜(3艇に集中)")]

print("=== 3連単ボックス6点(600円)の成績を「上位3艇の勝率合計」で層別 ===")
print(f"{'上位3艇の勝率和':<16}{'R数':>7}{'的中率':>8}{'平均配当':>10}"
      f"{'回収率':>9}{'最大1発除き':>11}")
for lo, hi, lbl in BANDS:
    rs = [r for r in recs if lo <= r["top3_sum"] < hi]
    if not rs:
        continue
    stake = sum(r["box_stake"] for r in rs)
    ret = sum(r["box_ret"] for r in rs)
    mx = max((r["box_ret"] for r in rs), default=0)
    hits = sum(1 for r in rs if r["box_ret"] > 0)
    avg = (sum(r["finish_santan"] for r in rs if r["hit"]) / hits / 100
           if hits else 0)
    print(f"{lbl:<16}{len(rs):>7,}{hits/len(rs):>8.1%}{avg:>9.0f}倍"
          f"{ret/stake:>9.1%}{(ret-mx)/stake:>11.1%}")

print("\n=== 参考: 対象5場のみ ===")
for lo, hi, lbl in BANDS:
    rs = [r for r in recs
          if lo <= r["top3_sum"] < hi and r["venue"] in TARGET_VENUE_CODES]
    if len(rs) < 20:
        continue
    stake = sum(r["box_stake"] for r in rs)
    ret = sum(r["box_ret"] for r in rs)
    mx = max((r["box_ret"] for r in rs), default=0)
    hits = sum(1 for r in rs if r["box_ret"] > 0)
    print(f"{lbl:<16}{len(rs):>7,}{hits/len(rs):>8.1%}{'':>10}"
          f"{ret/stake:>9.1%}{(ret-mx)/stake:>11.1%}")

# ボックスが100%超になる層があるか
print("\n=== 判定 ===")
ok = []
for lo, hi, lbl in BANDS:
    rs = [r for r in recs if lo <= r["top3_sum"] < hi]
    if len(rs) < 50:
        continue
    stake = sum(r["box_stake"] for r in rs)
    ret = sum(r["box_ret"] for r in rs)
    mx = max((r["box_ret"] for r in rs), default=0)
    if (ret - mx) / stake > 1.0:
        ok.append(lbl)
if ok:
    print(f"最大1発除きで100%超の層: {', '.join(ok)} → 第2段へ")
else:
    print("どの層も最大1発除きで100%未満 → 3連単ボックスは不採用")
