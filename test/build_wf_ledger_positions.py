# -*- coding: utf-8 -*-
"""位置別の順位付き台帳を作る(1着確率・2着確率・3着以内確率)。消去パターン検証用。

    py -X utf8 test/build_wf_ledger_positions.py

オッズがある2026-05〜2026-09の各月を、その月より前の全データで学習した3モデルで予測する
(月次walk-forward・全entries=非完走艇も順位に含める)。9月も本番配信ではなく同じ方式で作る
(本番には2着確率モデルが無いため、3つの順位を同じ条件で揃える)。
出力: data_raw/wf_ledger_pos_202605_202609.csv
  race_id, month, venue, p1, win_order, p2_order, top3_order(いずれも確率の高い順の艇番), refund
"""
import sys

import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

MONTHS = ["2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
OUT = r"Y:\マイドライブ\boat\data_raw\wf_ledger_pos_202605_202609.csv"
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
train_all["is_2nd"] = (train_all["arrival_order"] == 2).astype(int)
train_all["is_top3"] = (train_all["arrival_order"] <= 3).astype(int)
ev_all = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2026-05-01'
""", conn)
ev_all = _encode(ev_all)
ev_all = ev_all.merge(compute_form_features(conn), on=["race_id", "lane"], how="left")
ev_all = _attach_extra_features(ev_all, conn)
conn.close()


def train(tr, label):
    tr = tr.sort_values("date")
    cutoff = tr["date"].iloc[int(len(tr) * 0.9)]
    a, b = tr[tr["date"] < cutoff], tr[tr["date"] >= cutoff]
    ds = lgb.Dataset(a[FEATURE_COLUMNS], label=a[label],
                     categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(PARAMS, ds, num_boost_round=500,
                     valid_sets=[lgb.Dataset(b[FEATURE_COLUMNS], label=b[label], reference=ds)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


rows = []
for m in MONTHS:
    tr = train_all[train_all["date"] < f"{m}-01"]
    ev = ev_all[ev_all["date"].str.startswith(m)].copy()
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr):,}行 ×3モデル / 予測{ev['race_id'].nunique():,}R", flush=True)
    for col, label in (("m_win", "is_winner"), ("m_place2", "is_2nd"), ("m_top3", "is_top3")):
        ev[col] = train(tr, label).predict(ev[FEATURE_COLUMNS])
    for rid, g in ev.groupby("race_id"):
        if len(g) < 6 or g["arrival_order"].notna().sum() < 3:
            continue
        refund = sorted(int(l) for l, ao, st in zip(g["lane"], g["arrival_order"], g["st_time"])
                        if pd.isna(ao) and pd.isna(st))

        def order(col):
            return "-".join(str(int(x)) for x in g.sort_values(col, ascending=False)["lane"])
        rows.append((rid, m, int(g["venue_code"].iloc[0]), float(g["m_win"].max()),
                     order("m_win"), order("m_place2"), order("m_top3"),
                     "-".join(map(str, refund))))
pd.DataFrame(rows, columns=["race_id", "month", "venue", "p1", "win_order", "p2_order",
                            "top3_order", "refund"]).to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"保存: {OUT} ({len(rows):,}R)", flush=True)
