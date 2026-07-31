# -*- coding: utf-8 -*-
"""超混戦の増資1,000円を「広げる」候補の最終測定(2026-08-01ケンさん発案)

    py -X utf8 test/verify_konsen_widen_slots.py

■ 問い
超混戦2,000円を「同じ5点を厚く(現行・案1×2)」でなく「点数を広げて」使えないか。

■ 既に測定済みの候補(再測定しない・番号は当時の検証)
  AトリオTop1単 204.6% / BトリオTop1単 276.8%(5fold)→ただし変形ハとして
  月次切りで基準割れ(-39,000円)=頑健性なし・9/1紙上判定中
  C複r1r3r4 95.9% / D複r2r3r4 97.6%(案1で廃止した赤字スロット)
  C帯差され単r3-r1-r4/r4-r1-r3 70.5%/72.6%(E/F安売りは移植不可)
  D帯素直単r2-r3-r4 233.8%(12マス事後選択のため不採用)

■ 今回の事前登録(未測定の自然延長2マスのみ。マスを増やさない)
  W1 単 r5-r1-r2 (E/F差され形の5位頭延長)
  W2 複 r1=r2=r5 (A/B本線複の5位延長)
基準: 頑健に194%(案1×2の限界効率=案1自体のROI)を超えない限り
「厚く」が増資の最適解。150%未満は即棄却。150-194%は紙上観測のみ。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

conn = db.connect(DB_PATH)
df = build_training_set(conn)
payout_map = defaultdict(dict)
races_month = {}
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
for rid, d in conn.execute(
        "SELECT race_id, date FROM races WHERE date >= ?", (TEST_START,)):
    races_month[rid] = d[:7]
conn.close()

test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

slots = {
    "W1 単 r5-r1-r2": lambda l: ("3連単", f"{l[4]}-{l[0]}-{l[1]}"),
    "W2 複 r1=r2=r5": lambda l: ("3連複", "=".join(map(str, sorted([l[0], l[1], l[4]])))),
    "(比較)E単 r3-r1-r2": lambda l: ("3連単", f"{l[2]}-{l[0]}-{l[1]}"),
}
agg = defaultdict(lambda: {"st": 0, "rt": 0, "hit": 0,
                           "monthly": defaultdict(lambda: [0, 0])})
n = 0
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
        if len(ranked) < 5 or ranked[0]["prob"] >= 0.20:
            continue
        n += 1
        lanes = [r["lane"] for r in ranked]
        pay = payout_map[rid]
        m = races_month.get(rid, "?")
        for name, fn in slots.items():
            bt, comb = fn(lanes)
            got = pay.get((bt, comb), 0)
            a = agg[name]
            a["st"] += 100
            a["rt"] += got
            if got:
                a["hit"] += 1
            mm = a["monthly"][m]
            mm[0] += 100
            mm[1] += got

print(f"\n超混戦帯: {n:,}R(各スロット100円)")
for name, a in agg.items():
    months = sorted(a["monthly"])
    pos = sum(1 for m in months if a["monthly"][m][1] > a["monthly"][m][0])
    print(f"  {name:<18} 的中{a['hit']:>3}本({a['hit']/n:.1%}) "
          f"回収率{a['rt']/a['st']:>7.1%} 黒字月{pos}/{len(months)}")
print("\n判定: 194%(案1の限界効率)超え=広げる価値あり / 150%未満=棄却")
