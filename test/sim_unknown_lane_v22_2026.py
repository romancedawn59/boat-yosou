# -*- coding: utf-8 -*-
"""1-B未知艇レーン & 1-C 2軸分類(v2.2)前倒しの2026-01〜08シミュレーション
(2026-08-31ケンさん指示「1-B/1-Cを202601から08でシミュレーションして」)

    py -X utf8 test/sim_unknown_lane_v22_2026.py

■ 1-B 未知艇レーン
超混戦帯(1位生値20%未満)のうち「特徴量欠けあり(新人等の未知艇を含む=
build_training_setで6艇そろわないレース)」だけを買った場合の⑬2,000円成績。
注意: 元の330.8%は2025-12〜2026-07の事後観察(verify_konsen_lane1_necessity.py)。
本シミュレーションの2026-01〜07は同じデータの再集計=確認ではなく分解
(月次安定性・少数的中への依存度)。真の新月は2026-08のみ。

■ 1-C 2軸分類(v2.2)前倒し = 堅め帯×P(沈没)の大穴一撃フラグ
堅め帯(1位生値50%以上)にZ1-2a分類器(verify_z1_2a_sink_classifier.pyと同じ
特徴量19本・リーク安全プロファイル)を月次walk-forwardで適用。
閾値は各月の学習データ内・時系列ホールドアウト(後ろ25%)で
目標精度17%を満たす最小閾値を固定(後出しなし)。フラグ月のレースに
保険複r2r3r4を100円。参考として上位K%作動点も表示。

■ 方法
月次walk-forward: 各月、その月より前の全データでv2を学習(⑬の月次検証と同じ型)。
2025-12はZ1-2a分類器の学習素材のみ(評価は2026-01〜08)。
"""
import sys
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

MONTHS_ALL = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
              "2026-05", "2026-06", "2026-07", "2026-08"]
EVAL_MONTHS = MONTHS_ALL[1:]

conn = db.connect(DB_PATH)
df = build_training_set(conn)
res_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= '2025-12-01' AND res.arrival_order IS NOT NULL"):
    res_all[rid][lane] = ao
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0

# 選手プロファイル(全履歴からshiftでリーク防止・Z1-2aと同一)
hist = pd.read_sql_query(
    """
    SELECT r.race_id, r.date, r.race_no, r.winning_technique_number AS tech,
           e.lane, e.reg_no, res.arrival_order, res.st_time, res.course
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    """, conn)
conn.close()

hist = hist.sort_values(["reg_no", "date", "race_no"]).reset_index(drop=True)
has_res = hist["arrival_order"].notna()
hist["_sink"] = (hist["arrival_order"] >= 4).astype(float).where(has_res)
hist["_makuri_win"] = ((hist["arrival_order"] == 1) & hist["tech"].isin([3, 4])
                       ).astype(float).where(has_res)
hist["_win_c35"] = ((hist["arrival_order"] == 1) & hist["course"].between(3, 5)
                    ).astype(float).where(has_res)
g = hist.groupby("reg_no", sort=False)
hist["p_makuri"] = g["_makuri_win"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
hist["p_win_c35"] = g["_win_c35"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
hist["p_sink"] = g["_sink"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
hist["p_st_sigma"] = g["st_time"].transform(
    lambda s: s.shift(1).rolling(20, min_periods=5).std())
prof = hist.set_index(["race_id", "lane"])[
    ["p_makuri", "p_win_c35", "p_sink", "p_st_sigma"]]


def trio_comb(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan13(lanes):
    r1, r2, r3, r4, r5 = lanes[:5]
    bets = [("3連単", f"{a}-{b}-{c}", 100)
            for a, b, c in permutations((r1, r2, r3))]
    bets += [("3連単", f"{a}-{b}-{c}", 100)
             for a, b, c in permutations((r1, r2, r4))]
    bets += [("3連単", f"{r3}-{r1}-{r2}", 300),
             ("3連単", f"{r4}-{r1}-{r2}", 300),
             ("3連複", trio_comb(r3, r4, r5), 200)]
    return bets


def score(bets, pay):
    merged = defaultdict(int)
    for bt, comb, y in bets:
        merged[(bt, comb)] += y
    st = sum(merged.values())
    rt = sum(pay.get(k, 0) * y // 100 for k, y in merged.items())
    return st, rt


# ---- 月次walk-forwardで超混戦(1-B)と堅め帯行(1-C素材)を収集 -------------------
konsen = defaultdict(lambda: [0, 0, 0, 0])  # {(month,層): [st, rt, n, hits]}
hit_log = []                                # 欠けありレーンの的中明細
katame_rows = []

for m in MONTHS_ALL:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, grp in month_df.groupby("race_id"):
        arr = res_all.get(rid, {})
        pay = payout_map[rid]
        if len(arr) < 3 or not pay:
            continue
        gs = grp.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in gs.iterrows()]
        if len(ranked) < 5:
            continue
        p1 = ranked[0]["prob"]
        lanes = [r["lane"] for r in ranked]

        if p1 < 0.20 and m != "2025-12":                    # --- 1-B 超混戦 ---
            stratum = "6艇フル" if len(ranked) >= 6 else "欠けあり"
            st, rt = score(plan13(lanes), pay)
            a = konsen[(m, stratum)]
            a[0] += st
            a[1] += rt
            a[2] += 1
            if rt:
                a[3] += 1
                if stratum == "欠けあり":
                    hit_log.append((m, rid, rt - st, rt))
            continue

        if p1 < 0.50:                                       # --- 1-C 堅め帯 ---
            continue
        fav = lanes[0]
        fav_row = gs.iloc[0]
        fav_arr = arr.get(fav)

        def pf(lane, col):
            try:
                v = prof.loc[(rid, lane), col]
                return float(v) if pd.notna(v) else np.nan
            except KeyError:
                return np.nan

        outer = grp[grp["lane"].between(3, 6)]
        atk_st = outer["avg_st"].min()
        atk_motor = outer["motor_2rate"].max()
        preds = [r["prob"] for r in ranked]
        kado = grp[grp["lane"] == 4]
        katame_rows.append({
            "race_id": rid, "date": str(fav_row["date"]), "month": m,
            "sink": 1 if (fav_arr is None or fav_arr >= 4) else 0,
            "lanes": lanes,
            "p1": p1, "gap12": p1 - preds[1], "gap23": preds[1] - preds[2],
            "fav_lane": fav, "fav_class": fav_row["racer_class_ord"],
            "fav_avg_st": fav_row["avg_st"],
            "fav_form_st": fav_row["form_last10_avg_st"],
            "fav_motor": fav_row["motor_2rate"],
            "fav_p_sink": pf(fav, "p_sink"),
            "fav_st_sigma": pf(fav, "p_st_sigma"),
            "st_edge": (fav_row["avg_st"] - atk_st) if pd.notna(atk_st)
                       else np.nan,
            "atk_motor_edge": (atk_motor - fav_row["motor_2rate"])
                              if pd.notna(atk_motor) else np.nan,
            "atk_class": outer["racer_class_ord"].max(),
            "atk_makuri": max((pf(int(l), "p_makuri") for l in outer["lane"]),
                              default=np.nan),
            "atk_c35": max((pf(int(l), "p_win_c35") for l in outer["lane"]),
                           default=np.nan),
            "kado_makuri": pf(4, "p_makuri") if len(kado) else np.nan,
            "kado_st": kado["avg_st"].iloc[0] if len(kado) else np.nan,
            "venue_code": int(fav_row["venue_code"]),
            "race_no": int(fav_row["race_no"]),
        })

# ---- 1-B レポート -------------------------------------------------------------
print("\n===== 1-B 未知艇レーン(超混戦×特徴量欠けあり・⑬2,000円) =====")
print(f"{'月':<9}{'欠けあり':>7}{'的中':>5}{'回収率':>9}{'損益':>10}"
      f"  |{'6艇フル':>7}{'回収率':>9}{'損益':>10}")
tot = {"欠けあり": [0, 0, 0, 0], "6艇フル": [0, 0, 0, 0]}
for m in EVAL_MONTHS:
    row = f"{m:<9}"
    for stratum in ("欠けあり", "6艇フル"):
        st, rt, n, h = konsen[(m, stratum)]
        for i in range(4):
            tot[stratum][i] += konsen[(m, stratum)][i]
        roi = rt / st if st else 0
        if stratum == "欠けあり":
            row += f"{n:>6}R{h:>5}{roi:>9.1%}{rt - st:>+9,}円  |"
        else:
            row += f"{n:>6}R{roi:>9.1%}{rt - st:>+9,}円"
    print(row)
for stratum in ("欠けあり", "6艇フル"):
    st, rt, n, h = tot[stratum]
    print(f"合計 {stratum:<6} {n:>4}R 的中{h:>3} 投資{st:>10,}円 "
          f"回収率{(rt / st if st else 0):>7.1%} 損益{rt - st:>+10,}円")

hit_log.sort(key=lambda x: -x[3])
print("\n欠けありレーンの的中明細(回収額降順・少数依存度の確認):")
st_all = tot["欠けあり"][0]
cum = 0
for m, rid, pl, rt in hit_log:
    cum += rt
    print(f"  {m} {rid}  回収{rt:>8,}円  (累積回収の"
          f"{cum / tot['欠けあり'][1]:.0%})" if tot["欠けあり"][1] else "")

# ---- 1-C レポート -------------------------------------------------------------
data = pd.DataFrame(katame_rows)
FEATS = ["p1", "gap12", "gap23", "fav_lane", "fav_class", "fav_avg_st",
         "fav_form_st", "fav_motor", "fav_p_sink", "fav_st_sigma",
         "st_edge", "atk_motor_edge", "atk_class", "atk_makuri", "atk_c35",
         "kado_makuri", "kado_st", "venue_code", "race_no"]
TARGET = 0.17


def train_clf(tr):
    ds = lgb.Dataset(tr[FEATS], label=tr["sink"],
                     categorical_feature=["venue_code", "race_no", "fav_lane"])
    return lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8,
         "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
         "seed": 7},
        ds, num_boost_round=300)


def precision_threshold(y, p, target, min_flags=10):
    order = np.argsort(-p)
    ys, ps = np.asarray(y)[order], np.asarray(p)[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where((prec >= target) & (np.arange(1, len(ys) + 1) >= min_flags))[0]
    return ps[ok[-1]] if len(ok) else None


print(f"\n===== 1-C 大穴一撃フラグ(堅め帯×Z1-2a・保険複r2r3r4 100円) =====")
print(f"堅め帯 標本: {len(data):,}R / 沈没率{data['sink'].mean():.1%}")
print(f"{'月':<9}{'堅めR':>6}{'フラグ':>5}{'沈没的中':>7}{'精度':>8}"
      f"{'複ROI':>9}{'損益':>9}")
agg = [0, 0, 0, 0, 0]  # n, flags, hits, st, rt
pooled = {"y": [], "p": [], "rid": [], "lanes": []}
for m in EVAL_MONTHS:
    tr = data[data["date"] < f"{m}-01"]
    te = data[data["month"] == m]
    if len(tr) < 100 or te.empty:
        print(f"{m:<9} 学習素材不足({len(tr)}行)でスキップ")
        continue
    tr_dates = sorted(tr["date"].unique())
    cal_start = tr_dates[int(len(tr_dates) * 0.75)]
    fit, cal = tr[tr["date"] < cal_start], tr[tr["date"] >= cal_start]
    if len(fit) < 50 or len(cal) < 30:
        print(f"{m:<9} 較正素材不足でスキップ")
        continue
    clf = train_clf(fit)
    p_cal = clf.predict(cal[FEATS])
    p_te = clf.predict(te[FEATS])
    pooled["y"] += list(te["sink"])
    pooled["p"] += list(p_te)
    pooled["rid"] += list(te["race_id"])
    pooled["lanes"] += list(te["lanes"])
    thr = precision_threshold(cal["sink"].values, p_cal, TARGET)
    if thr is None:
        print(f"{m:<9}{len(te):>5}R  較正側で精度17%に届く閾値なし")
        continue
    flags = te[p_te >= thr]
    st = rt = 0
    for _, r in flags.iterrows():
        comb = trio_comb(*r["lanes"][1:4])
        st += 100
        rt += payout_map[r["race_id"]].get(("3連複", comb), 0)
    hits = int(flags["sink"].sum())
    agg[0] += len(te)
    agg[1] += len(flags)
    agg[2] += hits
    agg[3] += st
    agg[4] += rt
    prec = hits / len(flags) if len(flags) else 0
    print(f"{m:<9}{len(te):>5}R{len(flags):>5}{hits:>7}{prec:>8.1%}"
          f"{(rt / st if st else 0):>9.1%}{rt - st:>+8,}円")

if agg[1]:
    print(f"合計: 評価{agg[0]:,}R フラグ{agg[1]}本 精度{agg[2] / agg[1]:.1%}"
          f"(必要17%) 複ROI{(agg[4] / agg[3] if agg[3] else 0):.1%}(基準150%) "
          f"損益{agg[4] - agg[3]:>+,}円")

print("\n上位K%作動点(2026-01〜08プール・参考):")
y = np.array(pooled["y"])
p = np.array(pooled["p"])
if len(y):
    order = np.argsort(-p)
    for pct in (2, 5, 10, 20):
        k = max(1, int(len(y) * pct / 100))
        top = order[:k]
        st = rt = 0
        for idx in top:
            comb = trio_comb(*pooled["lanes"][idx][1:4])
            st += 100
            rt += payout_map[pooled["rid"][idx]].get(("3連複", comb), 0)
        print(f"  top{pct:>2}%: 精度{y[top].mean():>6.1%} / "
              f"複ROI{rt / st:>6.1%} ({k}R)")
