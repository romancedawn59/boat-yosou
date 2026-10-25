# -*- coding: utf-8 -*-
"""案1: レース内で6艇を競わせる学習(2026-10-25ケンさん承認・結果を見る前に基準を固定)

    py -X utf8 test/sim_race_ranking_model.py

比較するモデル(月次walk-forward 2026-05〜10、各月はその月より前の全データで学習):
  A そのまま        現行と同じ1艇ずつの二値分類(1着確率)
  B 合計で割る      Aの確率をレース内で合計1に正規化
  C 較正            Aの確率・合計・順位・枠から実際の1着率を当てる小さなLightGBM(前月までで学習)
  D ランキング      lambdarank(1レースを1グループ・6艇)。スコアをレース内softmaxで確率に
  E ランキング+相対 Dに「レース内の相対特徴量」(平均との差・レース内順位・1号艇との差)を追加
採点(全部出す):
  ① レース内で正規化した1着確率のlogloss
  ② 市場の暗黙確率(3連単オッズ・確定優先/無ければ締切15分前)のloglossとの差(主基準)
  ③ 同じレース(Aの1位勝率22.5〜30%)でのドンピシャ率と1点500円の回収率(並びの質の比較)
     + 各モデル自身の1位確率の順位で、Aの帯と同じ順位範囲を取った「自分の帯」の回収率
  ④ 前月のモデルと当月のモデルで、同じレースの予想1-2-3位が一致する率(安定度)
合格: ②がBより縮まり、③(同じレース)のドンピシャ率がAを下回らない。
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

MONTHS = [f"2026-{m:02d}" for m in range(5, 11)]
P_BIN = {"objective": "binary", "metric": "auc", "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31}
P_RANK = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [1], "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31, "lambdarank_truncation_level": 6}
REL_BASE = ["national_win_rate", "local_win_rate", "motor_2rate", "avg_st", "kr",
            "form_last10_win_rate", "racer_class_ord", "nobi_gap"]
REL_COLS = [f"{c}_{s}" for c in REL_BASE for s in ("dmean", "rrank", "dlane1")]
COLS_E = FEATURE_COLUMNS + REL_COLS
CAL_FEATS = ["p_raw", "p_sum", "p_prop", "rank", "lane"]
PCOL = {"A そのまま": "p_An", "B 合計で割る": "p_B", "C 較正": "p_C",
        "D ランキング": "p_D", "E ランキング+相対": "p_E"}

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
ev_all = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2026-05-01'
""", conn)
ev_all = _encode(ev_all).merge(compute_form_features(conn), on=["race_id", "lane"], how="left")
ev_all = _attach_extra_features(ev_all, conn)
pay3 = {r: (cb, a or 0) for r, cb, a in conn.execute(
    "SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連単'")}
odds = {}
for tbl, cond in (("odds", "fetched_at != 'final-backfill'"), ("odds_final", "1=1")):
    tmp = defaultdict(dict)
    for r, cb, o in conn.execute(f"SELECT race_id, combination, odds FROM {tbl} "
                                 f"WHERE bet_type='3連単' AND odds > 0 AND {cond}"):
        tmp[r][cb] = o
    for r, d in tmp.items():
        if len(d) >= 110:
            odds[r] = d
conn.close()
mkt = {}
for r, o in odds.items():
    tot = sum(1 / x for x in o.values())
    q = defaultdict(float)
    for cb, x in o.items():
        q[int(cb[0])] += (1 / x) / tot
    mkt[r] = dict(q)


def add_relative(df):
    df = df.copy()
    g = df.groupby("race_id")
    for c in REL_BASE:
        df[f"{c}_dmean"] = df[c] - g[c].transform("mean")
        df[f"{c}_rrank"] = g[c].rank(ascending=False, method="min")
        lane1 = df[df["lane"] == 1].set_index("race_id")[c]
        df[f"{c}_dlane1"] = df[c] - df["race_id"].map(lane1)
    return df


def fit_bin(tr, cols):
    tr = tr.sort_values("date")
    cut = tr["date"].iloc[int(len(tr) * .9)]
    a, b = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    ds = lgb.Dataset(a[cols], label=a["is_winner"], categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(P_BIN, ds, 500, valid_sets=[lgb.Dataset(b[cols], label=b["is_winner"], reference=ds)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def fit_rank(tr, cols):
    tr = tr[tr.groupby("race_id")["lane"].transform("count") == 6].sort_values(["date", "race_id", "lane"])
    cut = tr["date"].iloc[int(len(tr) * .9)]
    a, b = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    ds = lgb.Dataset(a[cols], label=a["is_winner"], group=a.groupby("race_id", sort=False).size().values,
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(b[cols], label=b["is_winner"], group=b.groupby("race_id", sort=False).size().values,
                     reference=ds)
    return lgb.train(P_RANK, ds, 500, valid_sets=[vs], callbacks=[lgb.early_stopping(30, verbose=False)])


def softmax_by_race(df, col):
    s = df[col] - df.groupby("race_id")[col].transform("max")
    e = np.exp(s)
    return e / e.groupby(df["race_id"]).transform("sum")


def ll(p, w):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(np.where(w == 1, np.log(p), np.log(1 - p))))


train_all = add_relative(train_all)
ev_all = add_relative(ev_all)
recs, cal_hist = [], []
prev_models = None
stab = defaultdict(lambda: [0, 0])
CUR = {"A そのまま": "p_A", "D ランキング": "p_D", "E ランキング+相対": "p_E"}
for m in MONTHS:
    tr = train_all[train_all["date"] < f"{m}-01"]
    ev = ev_all[ev_all["date"].str.startswith(m)].copy()
    ev = ev[ev.groupby("race_id")["lane"].transform("count") == 6]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr):,}行 / 予測{ev['race_id'].nunique():,}R", flush=True)
    mA, mD, mE = fit_bin(tr, FEATURE_COLUMNS), fit_rank(tr, FEATURE_COLUMNS), fit_rank(tr, COLS_E)
    ev["p_A"] = mA.predict(ev[FEATURE_COLUMNS])
    ev["p_sum"] = ev.groupby("race_id")["p_A"].transform("sum")
    ev["p_B"] = ev["p_A"] / ev["p_sum"]
    ev["sD"] = mD.predict(ev[FEATURE_COLUMNS])
    ev["p_D"] = softmax_by_race(ev, "sD")
    ev["sE"] = mE.predict(ev[COLS_E])
    ev["p_E"] = softmax_by_race(ev, "sE")
    ev["p_raw"], ev["p_prop"] = ev["p_A"], ev["p_B"]
    ev["rank"] = ev.groupby("race_id")["p_A"].rank(ascending=False, method="first")
    ev["win"] = (ev["arrival_order"] == 1).astype(int)
    if cal_hist:
        H = pd.concat(cal_hist)
        cal = lgb.train({"objective": "binary", "verbosity": -1, "learning_rate": .03, "num_leaves": 7,
                         "min_data_in_leaf": 150, "seed": 7}, lgb.Dataset(H[CAL_FEATS], label=H["win"]), 300)
        ev["p_C"] = cal.predict(ev[CAL_FEATS])
        ev["p_C"] = ev["p_C"] / ev.groupby("race_id")["p_C"].transform("sum")
    else:
        ev["p_C"] = ev["p_B"]
    cal_hist.append(ev[CAL_FEATS + ["win"]].copy())
    if prev_models:
        for name, (pm, cols) in prev_models.items():
            tmp = ev.assign(prev=pm.predict(ev[cols]))
            for _rid, g in tmp.groupby("race_id"):
                a = list(g.sort_values(CUR[name], ascending=False)["lane"].iloc[:3])
                b = list(g.sort_values("prev", ascending=False)["lane"].iloc[:3])
                stab[name][0] += 1
                stab[name][1] += a == b
    prev_models = {"A そのまま": (mA, FEATURE_COLUMNS), "D ランキング": (mD, FEATURE_COLUMNS),
                   "E ランキング+相対": (mE, COLS_E)}
    recs.append(ev[["race_id", "lane", "win", "p_A", "p_B", "p_C", "p_D", "p_E"]].assign(month=m))

R = pd.concat(recs)
R["p_An"] = R["p_A"] / R.groupby("race_id")["p_A"].transform("sum")
R["q"] = [mkt.get(r, {}).get(l, np.nan) for r, l in zip(R["race_id"], R["lane"])]
has_m = R["q"].notna()
llq = ll(R.loc[has_m, "q"].values, R.loc[has_m, "win"].values)
print(f"\n===== 採点(2026-05〜10・{R['race_id'].nunique():,}R / 市場比較 {R.loc[has_m, 'race_id'].nunique():,}R) =====")
print(f"{'モデル':<16}{'①logloss':>10}{'②市場比較のlogloss':>14}{'市場':>8}{'差(モデル−市場)':>12}")
for name, col in PCOL.items():
    lm = ll(R.loc[has_m, col].values, R.loc[has_m, "win"].values)
    print(f"{name:<16}{ll(R[col].values, R['win'].values):>10.4f}{lm:>14.4f}{llq:>8.4f}{lm - llq:>+12.4f}")

p1A = R.groupby("race_id")["p_A"].max()
band = set(p1A[(p1A >= .225) & (p1A < .30)].index)
rankA = p1A.rank()
lo, hi = rankA[list(band)].min(), rankA[list(band)].max()
print(f"\n③ 同じレース(Aの1位勝率22.5〜30%・{len(band):,}R)での並び / 自分の帯(1位確率の順位がAの帯と同じ範囲)")
print(f"{'モデル':<16}{'ドンピシャ':>8}{'率':>7}{'回収率':>8}{'1位の1着率':>9} | {'自分の帯R':>8}{'ドンピシャ':>7}{'回収率':>8}")


def score(sub, col):
    don = ret = w1 = 0
    for rid, g in sub.groupby("race_id"):
        gs = g.sort_values(col, ascending=False)
        top = list(gs["lane"].iloc[:3])
        cb, amt = pay3.get(rid, ("", 0))
        w1 += int(gs["win"].iloc[0])
        if cb == "-".join(map(str, top)):
            don += 1
            ret += amt * 5
    n = sub["race_id"].nunique()
    return don, ret / (n * 500), w1 / n, n


for name, col in PCOL.items():
    don, roi, w1, n = score(R[R["race_id"].isin(band)], col)
    rk = R.groupby("race_id")[col].max().rank()
    own = set(rk[(rk >= lo) & (rk <= hi)].index)
    don2, roi2, _w, n2 = score(R[R["race_id"].isin(own)], col)
    print(f"{name:<16}{don:>8}{don / n:>7.1%}{roi:>8.1%}{w1:>9.1%} | {n2:>8,}{don2:>7}{roi2:>8.1%}")
print("\n④ 前月モデルとの予想1-2-3位の一致率: " + " / ".join(f"{k} {v[1] / v[0]:.1%}" for k, v in stab.items()))
