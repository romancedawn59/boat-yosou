# -*- coding: utf-8 -*-
"""外れ方タイプ連動の適応型構成+T2/T7予報の検証(2026-09-02ケンさん指示)

    py -X utf8 test/sim_type_aware_2026.py

■ 背景(外れ方の解剖・本命帯4,758R)
T7下位の奇襲(28.4%・カバー0%)は「1位が81%残り、下位(主にr5)が2-3着に
滑り込む」形(最頻r1-r2-r5)。T2まくられ轟沈(18.9%)は「r2-r3-r4世界」。
T4イン逃げはr2(=1号艇)-r1-r3。この形を警報連動で買い目に反映する。

■ 事前登録(結果を見る前に固定)
レース選択は確定済みの「5場×20-30%×低い順×日次予算cap4」(基準9行=102.5%)。
警報は各レース時点の過去分布(選別レースのみ・助走30R)のq67、優先はT2>T7>T4>無:
  T2警報: まくり分類器のP(まくり) ≥ q67
  T7警報: 下位2艇(3着内順位の5・6位)のP(3着内)最大値 ≥ q67 (bはその艇)
  T4条件: 予想1位が1号艇でない
構成(3連複は最大2本・残り3連単):
  AD1000: コア400円(複r1r2r3 200+単r1-r2-r3 200)+警報ブロック600円
    T2: 複r2r3r4 200+単r2-r3-r4 200+r3-r2-r4 200
    T7: 単r1-r2-b 200+r1-b-r2 200+r2-r1-b 200
    T4: 単r2-r1-r3 300+r2-r1-r4 300
    無: 単r3-r1-r2 300+r4-r1-r2 300
  AD2000: コア800円(複r1r2r3 300+単r1-r2-r3 300+複r2r3r4 200)
    +警報ブロック600円(T2時は複が2本埋まるため単r2-r3-r4 200+r3-r2-r4 200
    +r2-r4-r3 200)+差され600円(r3-r1-r2 300+r4-r1-r2 300)
  参考: H1000静的(複r1r2r3 200+複r2r3r4 100+単r1-r2-r3 100+r3-r1-r2 200
    +r4-r1-r2 200+r4-r2-r1 200) / A現行9行1,400円(基準)
判定: 105%超かつ8か月中5か月以上100%超で9月紙上追跡へ。
予報の有効判定: 警報オン群のタイプ発生率がオフ群の1.5倍以上。
見送りゲート: 基準9行をT2警報日に見送った場合のROI変化も測る(+3ptで候補)。
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
WARMUP = 30

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
train_all["is_top3"] = (train_all["arrival_order"] <= 3).astype(int)
ff = compute_form_features(conn)
eval_df = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2026-01-01'
""", conn)
eval_df = _encode(eval_df)
eval_df = eval_df.merge(ff, on=["race_id", "lane"], how="left")
eval_df = _attach_extra_features(eval_df, conn)
paymap = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-01-01'"):
    paymap[rid][(bt, comb)] = amt or 0
race_tech = dict(conn.execute(
    "SELECT race_id, winning_technique_number FROM races"))

# ---- まくり分類器の素材(全レース・ベクトル化) ----------------------------------
print("まくり素材構築中...", flush=True)
mframe = pd.read_sql_query("""
    SELECT r.race_id, r.date, r.race_no, r.venue_code, e.lane, e.reg_no,
           e.avg_st, e.racer_class, e.motor_2rate, res.arrival_order
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2025-01-01'
""", conn)
conn.close()
mframe["racer_class_ord"] = mframe["racer_class"].map(
    {"B2": 0, "B1": 1, "A2": 2, "A1": 3})
mframe = mframe.sort_values(["reg_no", "date", "race_no"]).reset_index(drop=True)
has_res = mframe["arrival_order"].notna()
mframe["_mak"] = ((mframe["arrival_order"] == 1)
                  & mframe["race_id"].map(race_tech).isin([3, 4])
                  ).astype(float).where(has_res)
mframe["p_makuri"] = mframe.groupby("reg_no", sort=False)["_mak"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())

l1 = mframe[mframe["lane"] == 1].set_index("race_id")
outer = mframe[mframe["lane"].between(3, 6)].groupby("race_id").agg(
    out_min_st=("avg_st", "min"), out_max_motor=("motor_2rate", "max"),
    out_max_class=("racer_class_ord", "max"), out_max_pmak=("p_makuri", "max"))
kado = mframe[mframe["lane"] == 4].set_index("race_id")
mdata = pd.DataFrame({
    "date": l1["date"], "venue_code": l1["venue_code"],
    "race_no": l1["race_no"],
    "l1_avg_st": l1["avg_st"], "l1_class": l1["racer_class_ord"],
    "l1_motor": l1["motor_2rate"],
}).join(outer).join(pd.DataFrame({
    "kado_pmak": kado["p_makuri"], "kado_st": kado["avg_st"]}))
mdata["st_edge"] = mdata["l1_avg_st"] - mdata["out_min_st"]
mdata["y"] = [1 if race_tech.get(rid) in (3, 4) else
              (0 if race_tech.get(rid) else np.nan) for rid in mdata.index]
MFEATS = ["l1_avg_st", "l1_class", "l1_motor", "out_min_st", "out_max_motor",
          "out_max_class", "out_max_pmak", "kado_pmak", "kado_st", "st_edge",
          "venue_code", "race_no"]


def train_lgb(train_df, feats, label, cats):
    ds = lgb.Dataset(train_df[feats], label=train_df[label],
                     categorical_feature=cats)
    return lgb.train({**PARAMS, "min_data_in_leaf": 50}, ds,
                     num_boost_round=200)


def train_v2(train_df, label):
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


def build_plans(l, b_lane, alarm):
    r1, r2, r3, r4 = l[:4]
    core1 = [("3連複", trio(r1, r2, r3), 200), ("3連単", f"{r1}-{r2}-{r3}", 200)]
    if alarm == "T2":
        blk = [("3連複", trio(r2, r3, r4), 200),
               ("3連単", f"{r2}-{r3}-{r4}", 200),
               ("3連単", f"{r3}-{r2}-{r4}", 200)]
    elif alarm == "T7":
        blk = [("3連単", f"{r1}-{r2}-{b_lane}", 200),
               ("3連単", f"{r1}-{b_lane}-{r2}", 200),
               ("3連単", f"{r2}-{r1}-{b_lane}", 200)]
    elif alarm == "T4":
        blk = [("3連単", f"{r2}-{r1}-{r3}", 300),
               ("3連単", f"{r2}-{r1}-{r4}", 300)]
    else:
        blk = [("3連単", f"{r3}-{r1}-{r2}", 300),
               ("3連単", f"{r4}-{r1}-{r2}", 300)]
    ad1000 = core1 + blk
    core2 = [("3連複", trio(r1, r2, r3), 300), ("3連単", f"{r1}-{r2}-{r3}", 300),
             ("3連複", trio(r2, r3, r4), 200)]
    blk2 = ([("3連単", f"{r2}-{r3}-{r4}", 200), ("3連単", f"{r3}-{r2}-{r4}", 200),
             ("3連単", f"{r2}-{r4}-{r3}", 200)] if alarm == "T2" else blk)
    ad2000 = core2 + blk2 + [("3連単", f"{r3}-{r1}-{r2}", 300),
                             ("3連単", f"{r4}-{r1}-{r2}", 300)]
    nine = [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
            ("3連複", trio(r1, r3, r4), 100), ("3連単", f"{r3}-{r1}-{r2}", 200),
            ("3連単", f"{r4}-{r1}-{r2}", 200), ("3連複", trio(r2, r3, r4), 100),
            ("3連単", f"{r3}-{r2}-{r1}", 100), ("3連単", f"{r4}-{r2}-{r1}", 300)]
    h1000 = [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r2, r3, r4), 100),
             ("3連単", f"{r1}-{r2}-{r3}", 100), ("3連単", f"{r3}-{r1}-{r2}", 200),
             ("3連単", f"{r4}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r2}-{r1}", 200)]
    return {"A 現行9行(基準)": nine, "H 静的1000": h1000,
            "AD1000 警報連動": ad1000, "AD2000 警報連動": ad2000}


def score(bets, pay, refund):
    merged = defaultdict(int)
    for bt, comb, y in bets:
        merged[(bt, comb)] += y
    st = rt = 0
    for (bt, comb), y in merged.items():
        sep = "-" if bt == "3連単" else "="
        members = {int(x) for x in comb.split(sep)}
        st += y
        rt += y if members & refund else pay.get((bt, comb), 0) * y // 100
    return st, rt


agg = defaultdict(lambda: [0, 0, 0])
alarm_n = defaultdict(int)
fc = defaultdict(lambda: [0, 0])       # 予報検証 {(信号, 群): [n, タイプ発生]}
gate = defaultdict(lambda: [0, 0])     # 見送りゲート {(条件): [st, rt]}
t2_hist, t7_hist = [], []

for m in EVAL_MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行 ×3モデル", flush=True)
    win_model = train_v2(tr_df, "is_winner")
    top3_model = train_v2(tr_df, "is_top3")
    mtr = mdata[(mdata["date"] < f"{m}-01") & mdata["y"].notna()]
    mak_model = train_lgb(mtr, MFEATS, "y", ["venue_code", "race_no"])
    md = ev.copy()
    md["p_win"] = win_model.predict(md[FEATURE_COLUMNS])
    md["p_top3"] = top3_model.predict(md[FEATURE_COLUMNS])
    mev = mdata[mdata["date"].str.startswith(m)]
    p_mak_map = dict(zip(mev.index, mak_model.predict(mev[MFEATS])))

    daily = defaultdict(lambda: {"konsen": 0, "pool": []})
    for rid, grp in md.groupby("race_id"):
        pay = paymap[rid]
        if not pay:
            continue
        gs = grp.sort_values("p_win", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        p1 = float(gs["p_win"].iloc[0])
        d = gs["date"].iloc[0]
        arr = {int(r["lane"]): r["arrival_order"] for _, r in grp.iterrows()
               if pd.notna(r["arrival_order"])}
        if len(arr) < 3:
            continue
        if p1 < 0.20 and len(lanes) >= 5:
            daily[d]["konsen"] += 1
            continue
        if (0.20 <= p1 < 0.30 and len(lanes) >= 6
                and int(gs["venue_code"].iloc[0]) in TARGET_VENUE_CODES):
            nonfin = {int(r["lane"]) for _, r in grp.iterrows()
                      if pd.isna(r["arrival_order"])}
            refund = {l for l in nonfin
                      if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
            gt3 = grp.sort_values("p_top3", ascending=False)
            t_lanes = [int(x) for x in gt3["lane"]]
            pt = [float(x) for x in gt3["p_top3"]]
            b_lane = t_lanes[4] if pt[4] >= pt[5] else t_lanes[5]
            t7sig = max(pt[4], pt[5])
            t2sig = p_mak_map.get(rid, np.nan)
            top3 = set(sorted(arr, key=arr.get)[:3])
            tech = race_tech.get(rid)
            t2_flag = lanes[0] not in top3 and tech in (3, 4)
            t7_flag = bool({t_lanes[4], t_lanes[5]} & top3)
            daily[d]["pool"].append(dict(
                p1=p1, lanes=lanes, pay=pay, refund=refund, b=b_lane,
                t2sig=t2sig, t7sig=t7sig, t2=t2_flag, t7=t7_flag))
    for d, info in daily.items():
        pool = sorted(info["pool"], key=lambda x: x["p1"])
        remaining = DAILY_BUDGET - KONSEN_UNIT * info["konsen"]
        take = min(CAP, max(0, remaining // HONMEI_UNIT))
        for x in pool[:take]:
            warm = len(t2_hist) >= WARMUP
            q2 = np.nanpercentile(t2_hist, 66.7) if warm else np.inf
            q7 = np.percentile(t7_hist, 66.7) if warm else np.inf
            if not np.isnan(x["t2sig"]):
                t2_hist.append(x["t2sig"])
            t7_hist.append(x["t7sig"])
            t2_on = (not np.isnan(x["t2sig"])) and x["t2sig"] >= q2
            t7_on = x["t7sig"] >= q7
            alarm = ("T2" if t2_on else "T7" if t7_on
                     else "T4" if x["lanes"][0] != 1 else "無")
            if not warm:
                continue
            alarm_n[alarm] += 1
            fc[("T2警報", "on" if t2_on else "off")][0] += 1
            fc[("T2警報", "on" if t2_on else "off")][1] += x["t2"]
            fc[("T7警報", "on" if t7_on else "off")][0] += 1
            fc[("T7警報", "on" if t7_on else "off")][1] += x["t7"]
            plans = build_plans(x["lanes"], x["b"], alarm)
            for arm, bets in plans.items():
                st, rt = score(bets, x["pay"], x["refund"])
                a = agg[(arm, m)]
                a[0] += st
                a[1] += rt
                a[2] += 1
            st, rt = score(plans["A 現行9行(基準)"], x["pay"], x["refund"])
            if not t2_on:
                gate[("9行・T2警報日を見送り",)][0] += st
                gate[("9行・T2警報日を見送り",)][1] += rt
            if not t7_on:
                gate[("9行・T7警報日を見送り",)][0] += st
                gate[("9行・T7警報日を見送り",)][1] += rt

print("\n===== タイプ連動構成 vs 現行(選別レース・助走後) =====")
for arm in ("A 現行9行(基準)", "H 静的1000", "AD1000 警報連動", "AD2000 警報連動"):
    tot = [0, 0, 0]
    ok = 0
    line = []
    for m in EVAL_MONTHS:
        a = agg[(arm, m)]
        for i in range(3):
            tot[i] += a[i]
        if a[0]:
            ok += a[1] / a[0] > 1
            line.append(f"{m[5:]}月{a[1] / a[0]:>5.0%}")
    st, rt, n = tot
    mark = "★" if st and rt / st > 1.05 and ok >= 5 else " "
    print(f"\n{mark}{arm}: {n}R 回収率{rt / st:.1%} 損益{rt - st:+,}円 "
          f"100%超の月{ok}/8")
    print("  " + " ".join(line))
print("\n警報の内訳: " + " ".join(f"{k}={v}R" for k, v in
                                  sorted(alarm_n.items())))
print("\n===== 予報の精度(警報オン群 vs オフ群のタイプ発生率) =====")
for sig in ("T2警報", "T7警報"):
    on = fc[(sig, "on")]
    off = fc[(sig, "off")]
    if on[0] and off[0]:
        ron, roff = on[1] / on[0], off[1] / off[0]
        lift = ron / roff if roff else float("inf")
        print(f"  {sig}: オン{on[0]}R 発生{ron:.1%} / オフ{off[0]}R 発生{roff:.1%} "
              f"リフト{lift:.2f}倍({'有効' if lift >= 1.5 else '無効'}・基準1.5)")
print("\n===== 見送りゲート(基準9行との比較) =====")
base = [sum(agg[('A 現行9行(基準)', m)][i] for m in EVAL_MONTHS)
        for i in range(2)]
print(f"  基準9行(全出動): 回収率{base[1] / base[0]:.1%}")
for k, (st, rt) in gate.items():
    print(f"  {k[0]}: 回収率{rt / st:.1%} ({(rt / st - base[1] / base[0]) * 100:+.1f}pt)")
print("\n(判定: 構成は105%超+5か月以上100%超で9月紙上へ。予報はリフト1.5倍、"
      "ゲートは+3ptで候補。同一データ設計につき即実弾はしない)")
