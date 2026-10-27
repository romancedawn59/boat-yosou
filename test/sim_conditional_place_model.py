# -*- coding: utf-8 -*-
"""案2: 1着が決まった後の2着・3着を条件付きで学ぶ(2026-10-27ケンさん承認・基準は結果を見る前に固定)

    py -X utf8 test/sim_conditional_place_model.py

現行(A)は1着確率の順にr1〜r6を並べる。案2(F)は
  r1 = 1着確率が最大の艇(現行と同じ)
  r2 = 「aが1着のとき、bが2着になる確率」モデル P2(b|a) で a=r1 としたときの最大の艇
  r3 = 「a1着・b2着のとき、cが3着になる確率」モデル P3(c|a,b) の最大の艇
P2/P3 は実際の1着(・2着)を条件にした行で学習し、予測時は予想のr1(・r2)を条件に使う。
特徴量 = 対象艇の現行33本 + 条件艇との差(枠・勝率・KR・ST・モーター・級) + 条件艇の枠。
月次walk-forward 2026-05〜10。評価は 22.5〜30%帯(Aの1位勝率)のドンピシャ率・1点500円の回収率 を主、
2着・3着の的中率(r1が1着だったレースに限る)と全帯のドンピシャ率を従とする。
合格: 帯のドンピシャ率がAを+0.5pt以上上回り、回収率もAを上回る。
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
DIFF = ["national_win_rate", "local_win_rate", "kr", "avg_st", "motor_2rate", "racer_class_ord", "nobi_gap"]

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
conn.close()
train_all = train_all[train_all.groupby("race_id")["lane"].transform("count") == 6]


def pair_frame(df, cond_lane_col):
    """df(1艇1行)に、同じレースの条件艇(cond_lane_colで指定した枠)の特徴を結合して差を作る"""
    cond = df[["race_id", "lane"] + DIFF].rename(columns={"lane": "c_lane", **{c: f"c_{c}" for c in DIFF}})
    out = df.merge(cond, left_on=["race_id", cond_lane_col], right_on=["race_id", "c_lane"], how="inner")
    out = out[out["lane"] != out["c_lane"]].copy()
    for c in DIFF:
        out[f"d_{c}"] = out[c] - out[f"c_{c}"]
    out["d_lane"] = out["lane"] - out["c_lane"]
    out["outer_of_cond"] = (out["lane"] > out["c_lane"]).astype(int)
    return out


PAIR_FEATS = FEATURE_COLUMNS + ["c_lane", "d_lane", "outer_of_cond"] + [f"d_{c}" for c in DIFF]


def fit(tr, cols, label):
    tr = tr.sort_values("date")
    cut = tr["date"].iloc[int(len(tr) * .9)]
    a, b = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    ds = lgb.Dataset(a[cols], label=a[label], categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(P_BIN, ds, 500, valid_sets=[lgb.Dataset(b[cols], label=b[label], reference=ds)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


# 学習用の条件付き行: 実際の1着を条件に(2着モデル)、実際の1着と2着を条件に(3着モデル)
win_lane = train_all[train_all["arrival_order"] == 1][["race_id", "lane"]].rename(columns={"lane": "w_lane"})
sec_lane = train_all[train_all["arrival_order"] == 2][["race_id", "lane"]].rename(columns={"lane": "s_lane"})
T2 = pair_frame(train_all.merge(win_lane, on="race_id"), "w_lane")
T2["is_2nd"] = (T2["arrival_order"] == 2).astype(int)
T3 = train_all.merge(win_lane, on="race_id").merge(sec_lane, on="race_id")
T3 = T3[(T3["lane"] != T3["w_lane"])]
T3 = pair_frame(T3, "s_lane")                       # 条件=2着艇との差(1着艇との差も足す)
w = train_all[["race_id", "lane"] + DIFF].rename(columns={"lane": "w_lane2", **{c: f"w_{c}" for c in DIFF}})
T3 = T3.merge(w, left_on=["race_id", "w_lane"], right_on=["race_id", "w_lane2"], how="inner")
for c in DIFF:
    T3[f"dw_{c}"] = T3[c] - T3[f"w_{c}"]
T3["dw_lane"] = T3["lane"] - T3["w_lane"]
T3["is_3rd"] = (T3["arrival_order"] == 3).astype(int)
TRI_FEATS = PAIR_FEATS + ["w_lane", "dw_lane"] + [f"dw_{c}" for c in DIFF]
print(f"2着モデルの学習行 {len(T2):,} / 3着モデルの学習行 {len(T3):,}", flush=True)

res = defaultdict(lambda: defaultdict(int))
for m in MONTHS:
    tr = train_all[train_all["date"] < f"{m}-01"]
    ev = ev_all[ev_all["date"].str.startswith(m)].copy()
    ev = ev[ev.groupby("race_id")["lane"].transform("count") == 6]
    ev = ev[ev.groupby("race_id")["arrival_order"].transform(lambda s: s.notna().sum()) >= 3]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr):,}行 / 予測{ev['race_id'].nunique():,}R", flush=True)
    mW = fit(tr, FEATURE_COLUMNS, "is_winner")
    m2 = fit(T2[T2["date"] < f"{m}-01"], PAIR_FEATS, "is_2nd")
    m3 = fit(T3[T3["date"] < f"{m}-01"], TRI_FEATS, "is_3rd")
    ev["p_win"] = mW.predict(ev[FEATURE_COLUMNS])
    r1 = ev.loc[ev.groupby("race_id")["p_win"].idxmax(), ["race_id", "lane", "p_win"]].rename(
        columns={"lane": "w_lane", "p_win": "p1"})
    E2 = pair_frame(ev.merge(r1[["race_id", "w_lane"]], on="race_id"), "w_lane")
    E2["p2"] = m2.predict(E2[PAIR_FEATS])
    r2 = E2.loc[E2.groupby("race_id")["p2"].idxmax(), ["race_id", "lane"]].rename(columns={"lane": "s_lane"})
    E3 = ev.merge(r1[["race_id", "w_lane"]], on="race_id").merge(r2, on="race_id")
    E3 = E3[E3["lane"] != E3["w_lane"]]
    E3 = pair_frame(E3, "s_lane")
    E3 = E3.merge(w, left_on=["race_id", "w_lane"], right_on=["race_id", "w_lane2"], how="inner")
    for c in DIFF:
        E3[f"dw_{c}"] = E3[c] - E3[f"w_{c}"]
    E3["dw_lane"] = E3["lane"] - E3["w_lane"]
    E3["p3"] = m3.predict(E3[TRI_FEATS])
    r3 = E3.loc[E3.groupby("race_id")["p3"].idxmax(), ["race_id", "lane"]].rename(columns={"lane": "t_lane"})
    F = r1.merge(r2, on="race_id").merge(r3, on="race_id").set_index("race_id")
    for rid, g in ev.groupby("race_id"):
        gs = g.sort_values("p_win", ascending=False)
        A = [int(x) for x in gs["lane"].iloc[:3]]
        if rid not in F.index:
            continue
        Fo = [int(F.loc[rid, "w_lane"]), int(F.loc[rid, "s_lane"]), int(F.loc[rid, "t_lane"])]
        p1 = float(F.loc[rid, "p1"])
        act = {int(l): int(ao) for l, ao in zip(g["lane"], g["arrival_order"]) if pd.notna(ao)}
        order = sorted(act, key=act.get)
        cb, amt = pay3.get(rid, ("", 0))
        for name, o in (("A 現行", A), ("F 条件付き", Fo)):
            for scope in (["全帯"] + (["帯22.5〜30%"] if .225 <= p1 < .30 else [])):
                r = res[(name, scope)]
                r["n"] += 1
                hit = cb == "-".join(map(str, o))
                r["don"] += hit
                r["ret"] += amt * 5 if hit else 0
                r["best"] = max(r["best"], amt * 5 if hit else 0)
                if order and order[0] == o[0]:
                    r["w1"] += 1
                    r["s2"] += len(order) > 1 and order[1] == o[1]
                    r["s23"] += len(order) > 2 and order[1] == o[1] and order[2] == o[2]
                r["set3"] += set(o) == set(order[:3])
        res[("共通", "全帯")]["same_order"] += A == Fo
        res[("共通", "全帯")]["n"] += 1

print("\n===== 採点(2026-05〜10) =====")
print(f"{'モデル':<10}{'範囲':<12}{'R数':>7}{'ドンピシャ':>7}{'率':>7}{'回収率':>8}{'最大1本除く':>10}{'1位が1着':>8}{'→2着も':>8}{'→2・3着も':>9}{'上位3艇の顔ぶれ':>12}")
for (name, scope), r in sorted(res.items()):
    if name == "共通":
        continue
    n = r["n"]
    print(f"{name:<10}{scope:<12}{n:>7,}{r['don']:>7}{r['don'] / n:>7.2%}{r['ret'] / (n * 500):>8.1%}"
          f"{(r['ret'] - r['best']) / (n * 500):>10.1%}{r['w1'] / n:>8.1%}{r['s2'] / max(1, r['w1']):>8.1%}"
          f"{r['s23'] / max(1, r['w1']):>9.1%}{r['set3'] / n:>12.1%}")
c = res[("共通", "全帯")]
print(f"\nAとFで予想1-2-3位が同じレース: {c['same_order'] / c['n']:.1%}")
a, f = res[("A 現行", "帯22.5〜30%")], res[("F 条件付き", "帯22.5〜30%")]
ok = (f["don"] - a["don"]) / a["n"] >= .005 and f["ret"] > a["ret"]
print(f"判定: {'合格' if ok else '不合格'}(帯のドンピシャ率 A {a['don'] / a['n']:.2%} → F {f['don'] / f['n']:.2%}、回収率 A {a['ret'] / (a['n'] * 500):.1%} → F {f['ret'] / (f['n'] * 500):.1%})")
