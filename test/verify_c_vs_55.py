# -*- coding: utf-8 -*-
"""C枠(万舟圏100円) vs ケン55倍案(想定55倍圏の最有力1点200円)の直接対決

    py -X utf8 test/verify_c_vs_55.py

ケンさんの根拠(2026-07-29): 「100円×100倍で1万円を拾うより、朝の想定55倍に
200円(締切で41倍に下がる前提)の方が、1万円に届く確率が高い」。
想定55倍 ⇔ モデル確率 p <= 0.75/55 ≈ 1.36%。その条件で最も確率の高い
3連単1点に200円。現行C(picks_katsu: p<=万舟閾値の最有力1点100円)と比較する。
KPI: 回収率に加えて「そのスロットだけで1万円以上取れたレース数」。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

P55 = 0.75 / 55        # 想定55倍に相当するモデル確率(約1.36%)

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
        if 1 not in actual[rid] or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        probs = P.normalize_probs(ranked)
        ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]),
                     "top": ranked[0]["prob"], "probs": probs})


def slot_c(c):
    """現行C: 万舟圏の最有力1点(picks_katsuの1点目)"""
    picks = P.picks_katsu(c["probs"])
    return ("3連単", picks[0][1]) if picks else None


def slot_55(c):
    """ケン案: 想定55倍以上(p<=1.36%)で最も確率の高い3連単1点"""
    tri = P.trifecta_probs(c["probs"])
    cands = [(k, p) for k, p in tri.items() if p <= P55]
    if not cands:
        return None
    (a, b, x), _ = max(cands, key=lambda kv: kv[1])
    return ("3連単", f"{a}-{b}-{x}")


for scope_name, lo, hi in (("本命帯(20-35%)", 0.20, 0.35),
                           ("超混戦帯(<20%)", 0.0, 0.20)):
    sel = [c for c in ctxs if lo <= c["top"] < hi]
    n = len(sel)
    print(f"\n=== {scope_name}: {n:,}R ===")
    print(f"{'案':<24}{'金額':>5}{'的中率':>8}{'平均払戻':>10}"
          f"{'1万円超':>7}{'回収率':>8}")
    for name, fn, yen in (("現行C: 万舟圏1点", slot_c, 100),
                          ("ケン案: 55倍圏1点", slot_55, 200)):
        st = rt = hits = man = 0
        pays = []
        for c in sel:
            s = fn(c)
            if not s:
                continue
            bt, comb = s
            st += yen
            got = payout_map[c["rid"]].get((bt, comb), 0) * yen // 100
            rt += got
            if got:
                hits += 1
                pays.append(got)
                if got >= 10000:
                    man += 1
        if not st:
            continue
        nn = st // yen
        avg = sum(pays) / len(pays) if pays else 0
        print(f"{name:<24}{yen:>4}円{hits/nn:>8.2%}{avg:>9,.0f}円"
              f"{man:>6}回{rt/st:>8.1%}")
