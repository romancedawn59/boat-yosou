# -*- coding: utf-8 -*-
"""順位シミュレーションの精度上げ: 2着・3着の確率も別モデルで出して並びを作る
(2026-09-21ケンさん指示「即実行」)

    py -X utf8 test/sim_position_models_2026.py

■ 方法(本番忠実・事前登録)
月次walk-forward(2026-01〜08)。各月、その月より前の全データで
1着/2着/3着/3着内の4モデルを学習し、全entries(非完走含む)に確率を付ける。
対象=全24場×1着確率の最大が20〜35%×6艇。並びの作り方4種を同じレースで比べる。
  A 現行        : 1着確率の高い順
  B 位置別      : 1位=1着確率最大 / 2位=残りで2着確率最大 / 3位=残りで3着確率最大 /
                  4位以下=3着内確率順
  C 頭+3着内    : 1位=1着確率最大 / 2位以下=3着内確率順
  D 3着内順     : 全員を3着内確率順
採点は verify_exclude_then_pick.py と同じ型: 1〜4月の発生率上位K通りを
5〜8月に各100円(返還対象艇を含む券は返還)。
■ 合格基準(現行A比): 上位6点の的中率+2pt以上 かつ 全24場回収率+5pt以上
  かつ 対象5場で100%超。
"""
import sys
from collections import Counter, defaultdict
from itertools import permutations

import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH, TARGET_VENUE_CODES
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

MONTHS = [f"2026-{m:02d}" for m in range(1, 9)]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
train_all["is_2nd"] = (train_all["arrival_order"] == 2).astype(int)
train_all["is_3rd"] = (train_all["arrival_order"] == 3).astype(int)
train_all["is_top3"] = (train_all["arrival_order"] <= 3).astype(int)
eval_df = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2026-01-01' AND r.date < '2026-09-01'
""", conn)
eval_df = _encode(eval_df)
eval_df = eval_df.merge(compute_form_features(conn), on=["race_id", "lane"],
                        how="left")
eval_df = _attach_extra_features(eval_df, conn)
pay3 = {}
for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id "
        "WHERE r.date >= '2026-01-01' AND p.bet_type = '3連単'"):
    pay3[rid] = (comb, amt or 0)
conn.close()


def train(train_df, label):
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    ds = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr[label],
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(va[FEATURE_COLUMNS], label=va[label], reference=ds)
    return lgb.train(PARAMS, ds, valid_sets=[vs], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def orders(g):
    lanes = list(g["lane"].astype(int))
    pw = dict(zip(lanes, g["p_win"]))
    p2 = dict(zip(lanes, g["p_2nd"]))
    p3 = dict(zip(lanes, g["p_3rd"]))
    pt = dict(zip(lanes, g["p_top3"]))
    a = sorted(lanes, key=lambda l: -pw[l])
    head = a[0]
    rest = [l for l in lanes if l != head]
    b2 = max(rest, key=lambda l: p2[l])
    rest_b = [l for l in rest if l != b2]
    b3 = max(rest_b, key=lambda l: p3[l])
    b = [head, b2, b3] + sorted((l for l in rest_b if l != b3), key=lambda l: -pt[l])
    c = [head] + sorted(rest, key=lambda l: -pt[l])
    d = sorted(lanes, key=lambda l: -pt[l])
    return {"A 現行": a, "B 位置別": b, "C 頭+3着内": c, "D 3着内順": d}


recs = defaultdict(list)   # arm -> [(month, in5, pattern, amt, refund_ranks)]
for m in MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行 ×4モデル", flush=True)
    md = ev.copy()
    for col, label in (("p_win", "is_winner"), ("p_2nd", "is_2nd"),
                       ("p_3rd", "is_3rd"), ("p_top3", "is_top3")):
        md[col] = train(tr_df, label).predict(md[FEATURE_COLUMNS])
    p1 = md.groupby("race_id")["p_win"].max()
    band = set(p1[(p1 >= 0.20) & (p1 < 0.35)].index)
    for rid, g in md[md["race_id"].isin(band)].groupby("race_id"):
        if rid not in pay3 or len(g) < 6 or g["arrival_order"].notna().sum() < 3:
            continue
        comb, amt = pay3[rid]
        if not amt:
            continue
        res = [int(x) for x in comb.split("-")]
        refund = {int(l) for l, ao, st in zip(g["lane"], g["arrival_order"], g["st_time"])
                  if pd.isna(ao) and pd.isna(st)}
        in5 = int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES
        for arm, od in orders(g).items():
            recs[arm].append((m, in5, tuple(od.index(l) + 1 for l in res), amt,
                              frozenset(od.index(l) + 1 for l in refund)))

PATTERNS = list(permutations(range(1, 7), 3))
print(f"\n対象レース: {len(recs['A 現行']):,}R(全24場×1着確率最大20〜35%)")
print("\n===== 並びの当たり具合(2026-01〜08・walk-forward) =====")
print(f"{'並び':<12}{'1位が1着':>8}{'2位が2着':>8}{'3位が3着':>8}{'ドンピシャ':>8}"
      f"{'上位3艇で決着':>11}{'1位が3着内':>10}")
for arm, rs in recs.items():
    n = len(rs)
    pts = [x[2] for x in rs]
    print(f"{arm:<12}{sum(p[0] == 1 for p in pts) / n:>8.1%}"
          f"{sum(p[1] == 2 for p in pts) / n:>8.1%}{sum(p[2] == 3 for p in pts) / n:>8.1%}"
          f"{sum(p == (1, 2, 3) for p in pts) / n:>8.1%}"
          f"{sum(max(p) <= 3 for p in pts) / n:>11.1%}{sum(1 in p for p in pts) / n:>10.1%}")

print("\n===== 1〜4月の発生率上位K点を5〜8月に各100円 =====")
for k in (1, 3, 6, 10):
    print(f"\n― 上位{k}点 ―")
    print(f"{'並び':<12}{'的中率':>7}{'全24場':>8}{'最大1本除く':>10}{'対象5場':>8}{'5場R':>6}")
    for arm, rs in recs.items():
        tr = [x for x in rs if x[0] < "2026-05"]
        te = [x for x in rs if x[0] >= "2026-05"]
        cnt = Counter(x[2] for x in tr)
        buy = sorted(PATTERNS, key=lambda p: -cnt[p])[:k]
        st = rt = hit = best = st5 = rt5 = n5 = 0
        for _m, in5, pt, amt, rf in te:
            g = 0
            for p in buy:
                if set(p) & rf:
                    g += 100
                elif p == pt:
                    g += amt
                    hit += 1
                    best = max(best, amt)
            st += k * 100
            rt += g
            if in5:
                st5 += k * 100
                rt5 += g
                n5 += 1
        print(f"{arm:<12}{hit / len(te):>7.1%}{rt / st:>8.1%}{(rt - best) / st:>10.1%}"
              f"{rt5 / max(1, st5):>8.1%}{n5:>6}")
print("\n(合格基準: 現行A比で上位6点の的中率+2pt・全24場回収率+5pt・対象5場100%超)")
