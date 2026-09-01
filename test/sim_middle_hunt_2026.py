# -*- coding: utf-8 -*-
"""中間帯の狩場: 唯一プラス圏の「本命選別459R」の上で買い方を一斉検証
(2026-09-02ケンさん方針「頭としっぽはくれてやる。中間で勝てる買い方を探す」)

    py -X utf8 test/sim_middle_hunt_2026.py

■ 前提
クリーン8か月で100%を超えた唯一のレース選択=「5場×1位勝率20-30%×
勝率が低い順×日次予算内cap4」(現行9行構成で102.5%)。
このレース集合を固定し、買い方だけを入れ替えて一斉比較する。

■ アーム(事前登録・各レース同額換算はせず素の構成額で回収率比較)
  A 現行9行1,400円(基準・102.5%)
  B ドンピシャ1点 3連単r1-r2-r3 200円
  C 本線複のみ3点 700円(r1r2r3 300/r1r2r4 200/r1r3r4 200)
  D 単勝r1 200円
  E 2連単r1-r2 200円
  F 差され2点 3連単r3-r1-r2/r4-r1-r2 各200円
  G 消し自信絞り: 3着内モデルの下位2艇P(3着内)和が過去分布の中央値以下の
    レースだけ現行9行(出動半減・質の検証)
■ 判定: 回収率105%超かつ8か月中5か月以上100%超のアームを「9月紙上追跡」へ。
  同一データでの比較につき、即実弾昇格はしない(多重比較を自覚)。
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH, TARGET_VENUE_CODES
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

EVAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
               "2026-05", "2026-06", "2026-07", "2026-08"]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}
DAILY_BUDGET, KONSEN_UNIT, HONMEI_UNIT, CAP = 10200, 2000, 1400, 4

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
train_all["is_top3"] = (train_all["arrival_order"] <= 3).astype(int)
eval_df = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2026-01-01'
""", conn)
eval_df = _encode(eval_df)
eval_df = eval_df.merge(compute_form_features(conn), on=["race_id", "lane"],
                        how="left")
eval_df = _attach_extra_features(eval_df, conn)
paymap = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-01-01'"):
    paymap[rid][(bt, comb)] = amt or 0
conn.close()


def train_lgb(train_df, label):
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    ds = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr[label],
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(va[FEATURE_COLUMNS], label=va[label], reference=ds)
    return lgb.train(PARAMS, ds, valid_sets=[vs], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plans(l):
    r1, r2, r3, r4 = l[:4]
    nine = [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
            ("3連複", trio(r1, r3, r4), 100), ("3連単", f"{r3}-{r1}-{r2}", 200),
            ("3連単", f"{r4}-{r1}-{r2}", 200), ("3連複", trio(r2, r3, r4), 100),
            ("3連単", f"{r3}-{r2}-{r1}", 100), ("3連単", f"{r4}-{r2}-{r1}", 300)]
    return {
        "A 現行9行(基準)": nine,
        "B ドンピシャ1点": [("3連単", f"{r1}-{r2}-{r3}", 200)],
        "C 本線複3点": [("3連複", trio(r1, r2, r3), 300),
                       ("3連複", trio(r1, r2, r4), 200),
                       ("3連複", trio(r1, r3, r4), 200)],
        "D 単勝1位": [("単勝", str(r1), 200)],
        "E 2連単1-2位": [("2連単", f"{r1}-{r2}", 200)],
        "F 差され2点": [("3連単", f"{r3}-{r1}-{r2}", 200),
                       ("3連単", f"{r4}-{r1}-{r2}", 200)],
    }


def score(bets, pay, refund):
    merged = defaultdict(int)
    for bt, comb, y in bets:
        merged[(bt, comb)] += y
    st = rt = 0
    for (bt, comb), y in merged.items():
        sep = "-" if bt in ("3連単", "2連単") else "="
        members = ({int(comb)} if bt in ("単勝", "複勝")
                   else {int(x) for x in comb.split(sep)})
        st += y
        rt += y if members & refund else pay.get((bt, comb), 0) * y // 100
    return st, rt


agg = defaultdict(lambda: [0, 0, 0])   # {(アーム, 月): [st, rt, n]}
kesi_hist = []
for m in EVAL_MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行 ×2モデル", flush=True)
    win_model = train_lgb(tr_df, "is_winner")
    top3_model = train_lgb(tr_df, "is_top3")
    md = ev.copy()
    md["p_win"] = win_model.predict(md[FEATURE_COLUMNS])
    md["p_top3"] = top3_model.predict(md[FEATURE_COLUMNS])
    daily = defaultdict(lambda: {"konsen": 0, "pool": []})
    for rid, grp in md.groupby("race_id"):
        pay = paymap[rid]
        if not pay:
            continue
        gs = grp.sort_values("p_win", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        p1 = float(gs["p_win"].iloc[0])
        d = gs["date"].iloc[0]
        arr_n = grp["arrival_order"].notna().sum()
        if arr_n < 3:
            continue
        if p1 < 0.20 and len(lanes) >= 5:
            daily[d]["konsen"] += 1
            continue
        if (0.20 <= p1 < 0.30 and len(lanes) >= 4
                and int(gs["venue_code"].iloc[0]) in TARGET_VENUE_CODES):
            nonfin = {int(r["lane"]) for _, r in grp.iterrows()
                      if pd.isna(r["arrival_order"])}
            refund = {l for l in nonfin
                      if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
            gt = grp.sort_values("p_top3", ascending=False)
            pt = [float(x) for x in gt["p_top3"]]
            kesi = pt[4] + pt[5] if len(pt) >= 6 else pt[4] + 1.0
            daily[d]["pool"].append(
                {"p1": p1, "lanes": lanes, "pay": pay, "refund": refund,
                 "kesi": kesi})
    for d, info in daily.items():
        pool = sorted(info["pool"], key=lambda x: x["p1"])
        remaining = DAILY_BUDGET - KONSEN_UNIT * info["konsen"]
        take = min(CAP, max(0, remaining // HONMEI_UNIT))
        for x in pool[:take]:
            med = (np.median(kesi_hist) if len(kesi_hist) >= 30 else None)
            for arm, bets in plans(x["lanes"]).items():
                st, rt = score(bets, x["pay"], x["refund"])
                a = agg[(arm, m)]
                a[0] += st
                a[1] += rt
                a[2] += 1
            if med is not None and x["kesi"] <= med:
                st, rt = score(plans(x["lanes"])["A 現行9行(基準)"],
                               x["pay"], x["refund"])
                a = agg[("G 消し自信絞り9行", m)]
                a[0] += st
                a[1] += rt
                a[2] += 1
            kesi_hist.append(x["kesi"])

print("\n===== 中間帯の狩場: 選別459Rの上での買い方一斉比較 =====")
ARMS = ["A 現行9行(基準)", "B ドンピシャ1点", "C 本線複3点", "D 単勝1位",
        "E 2連単1-2位", "F 差され2点", "G 消し自信絞り9行"]
for arm in ARMS:
    tot = [0, 0, 0]
    ok_m = 0
    line = []
    for m in EVAL_MONTHS:
        a = agg[(arm, m)]
        for i in range(3):
            tot[i] += a[i]
        if a[0]:
            roi = a[1] / a[0]
            ok_m += roi > 1
            line.append(f"{m[5:]}月{roi:>5.0%}")
    st, rt, n = tot
    if not st:
        continue
    mark = "★" if rt / st > 1.05 and ok_m >= 5 else " "
    print(f"\n{mark}{arm}: {n}R 回収率{rt / st:.1%} 損益{rt - st:+,}円 "
          f"100%超の月{ok_m}/8")
    print("  " + " ".join(line))
print("\n(判定: 105%超かつ5か月以上100%超=★を9月紙上追跡へ。同一データ比較に"
      "つき即実弾昇格はしない)")
