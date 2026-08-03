# -*- coding: utf-8 -*-
"""検証⑮: 「本命に吸われた超混戦帯」(5場×20%未満)は⑬構成にすべきか
(2026-08-03発見・2026-08-04実施)

    py -X utf8 test/verify_konsen_absorbed_honmei.py

5場×1位生値20%未満のレースは本命プールに真っ先に吸われ、本命構成1,000円で
買われる(尼崎3R 8/3・若松9R 7/29が実例)。同じ帯の他19場は⑬構成2,000円。
この帯のレースだけ取り出して両構成を直接対決させる。
判定: ⑬がROIと最大1発除きの両方で本命構成を上回れば「⑮: 帯優先で⑬適用」を
9/1に提案。金額は1,000/2,000で異なるためROIで比較し損益は参考。
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

sel = []
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
        if len(ranked) < 5 or ranked[0]["prob"] >= 0.20:
            continue
        sel.append({"rid": rid, "date": g["date"].iloc[0], "ranked": ranked})

n = len(sel)
n_days = len({c["date"] for c in sel})
print(f"\n5場×20%未満(本命吸収帯): {n:,}R / 全{len(dates)}日中{n_days}日に出現")

agg = defaultdict(lambda: [0, 0, 0, 0, 0])
for c in sel:
    probs = P.normalize_probs(c["ranked"])
    hon = P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(probs))
    kon = P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(probs),
                          konsen=True)
    pay = payout_map[c["rid"]]
    for name, plan in (("本命構成1,000円(現行)", hon), ("⑬構成2,000円", kon)):
        st = sum(y for _, _, y, _ in plan)
        rt = sum(pay.get((bt, cb), 0) * y // 100 for bt, cb, y, _s in plan)
        a = agg[name]
        a[0] += st
        a[1] += rt
        if rt and rt < st:
            a[2] += 1
        elif rt:
            a[3] += 1
        a[4] = max(a[4], rt)

for name, (st, rt, gm, pl, best) in agg.items():
    print(f"  {name:<18} 回収率{rt/st:>7.1%} ガミ率{gm/n:>6.1%} プラス率{pl/n:>6.1%} "
          f"除き{(rt-best)/st:>7.1%} 損益{rt-st:+,}円")
