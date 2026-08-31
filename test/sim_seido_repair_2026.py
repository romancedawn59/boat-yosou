# -*- coding: utf-8 -*-
"""②精度修理3案(2-A/2-B/2-C)の2026-01〜08シミュレーション
(2026-08-31ケンさん指示「2-A/2-B/2-Cを202601から08でシミュレーションして」)

    py -X utf8 test/sim_seido_repair_2026.py

■ 事前登録(結果を見る前に固定)
月次walk-forward(各月、その月より前の全データで学習)。評価2026-01〜08。
変種(全てv2と同じLightGBM設定・早期停止):
  V0  基準(現行37特徴量)
  VA  2-A: +KR指数(Elo・レース前値)+KRレース内順位+伸びギャップ(直近90日実測
      2連対率−番組表全国2連率)
  VB  2-B: +期内走数(1/1・7/1起点の当期出走数=印刷スタッツの信頼度代理)
      +通算2連対率(期に依存しない実力・expanding)
  VAB 2-A+2-B全部入り(9月の実装案)
評価指標:
  - log loss(確率品質・全艇)
  - 超混戦帯(各変種自身の順位でp1<0.20): 軸生存率(モデル1・2位が共に3着内)と
    ⑬2,000円回収率 — 8月の故障(12%)が直るかが主読点
  - 本命帯(0.20-0.35): 1位的中率・軸生存率
採用検討の基準: VA/VB/VABのどれかが「8月の超混戦軸生存率をV0比+10pt以上改善
かつ log lossがV0以下」。回収率は参考(小標本)。

■ 2-C Z1-2まくり予測
決まり手∈{まくり,まくり差し}を当てる分類器(LightGBM・特徴量は予測時点で
知り得るもののみ: 1号艇のST/級/モーター、外艇(3-6枠)のまくり率プロファイル・
ST差・モーター差、カド情報、場・R番号)。月次walk-forward。
  - AUC(全レース)
  - V0の超混戦をP(まくり)三分位で層別: 軸生存率・⑬回収率
  - ゲート案: P(まくり)上位1/3(閾値は学習側分布で固定)なら1抜き⑬、他は現行⑬。
    採用検討の基準: 現行⑬比+5pt以上かつ8か月中5か月以上で同等以上。
"""
import sys
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set

EVAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
               "2026-05", "2026-06", "2026-07", "2026-08"]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
df = build_training_set(conn)
res_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= '2026-01-01' AND res.arrival_order IS NOT NULL"):
    res_all[rid][lane] = ao
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-01-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
race_tech = dict(conn.execute(
    "SELECT race_id, winning_technique_number FROM races"))

# ---- KR指数(Elo・時系列順=リーク安全。レース前の値を特徴量にする) -------------
print("KR指数構築中...", flush=True)
races_meta = dict(conn.execute("SELECT race_id, date || '_' || printf('%02d', race_no) "
                               "FROM races"))
lane_racer = {}
for rid, lane, reg in conn.execute("SELECT race_id, lane, reg_no FROM entries"):
    lane_racer[(rid, lane)] = reg
finish_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    finish_all[rid][lane] = ao

R = defaultdict(lambda: 1500.0)
K = 3.0
kr_pre = {}
for rid in sorted(finish_all, key=lambda r: races_meta.get(r, "")):
    boats = [(lane, ao) for lane, ao in finish_all[rid].items()
             if (rid, lane) in lane_racer]
    regs = {lane: lane_racer[(rid, lane)] for lane, _ in boats}
    for lane, _ in boats:
        kr_pre[(rid, lane)] = R[regs[lane]]
    for i in range(len(boats)):
        for j in range(i + 1, len(boats)):
            li, ai = boats[i]
            lj, aj = boats[j]
            ri, rj = regs[li], regs[lj]
            e = 1 / (1 + 10 ** ((R[rj] - R[ri]) / 400))
            s = 1.0 if ai < aj else 0.0
            R[ri] += K * (s - e)
            R[rj] -= K * (s - e)
# 結果未確定レース(当日朝など)にも直前値を出せるようにdf側で埋める
df["kr"] = [kr_pre.get((r, l), np.nan)
            for r, l in zip(df["race_id"], df["lane"])]

# ---- 伸びギャップ・期内走数・通算2連対率(履歴からshiftでリーク防止) -----------
print("履歴プロファイル構築中...", flush=True)
hist = pd.read_sql_query(
    """
    SELECT r.race_id, r.date, r.race_no, e.lane, e.reg_no, res.arrival_order
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    """, conn)
conn.close()
hist = hist.sort_values(["reg_no", "date", "race_no"]).reset_index(drop=True)
hist["dt"] = pd.to_datetime(hist["date"])
has_res = hist["arrival_order"].notna()
hist["_top2"] = (hist["arrival_order"] <= 2).astype(float).where(has_res)


def roll90(s):
    r = s.rolling("90D", min_periods=5).mean()
    return r.shift(1)


g = hist.set_index("dt").groupby("reg_no", sort=False)
hist["hist90_top2"] = g["_top2"].transform(roll90).values
g2 = hist.groupby("reg_no", sort=False)
hist["career_top2"] = g2["_top2"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
period_start = hist["dt"].dt.year.astype(str) + np.where(
    hist["dt"].dt.month >= 7, "-07", "-01")
hist["_pkey"] = hist["reg_no"].astype(str) + period_start
hist["n_starts_period"] = hist.groupby("_pkey", sort=False).cumcount()
prof = hist.set_index(["race_id", "lane"])[
    ["hist90_top2", "career_top2", "n_starts_period"]]
idx = pd.MultiIndex.from_arrays([df["race_id"], df["lane"]])
joined = prof.reindex(idx)
df["hist90_top2"] = joined["hist90_top2"].values
df["career_top2"] = joined["career_top2"].values
df["n_starts_period"] = joined["n_starts_period"].values
df["nobi_gap"] = df["hist90_top2"] - df["national_2rate"] / 100.0
df["kr_rank"] = df.groupby("race_id")["kr"].rank(ascending=False)

VARIANTS = {
    "V0 基準": list(FEATURE_COLUMNS),
    "VA KR+伸び": list(FEATURE_COLUMNS) + ["kr", "kr_rank", "nobi_gap"],
    "VB 頑健化": list(FEATURE_COLUMNS) + ["n_starts_period", "career_top2"],
    "VAB 全部入り": list(FEATURE_COLUMNS) + ["kr", "kr_rank", "nobi_gap",
                                             "n_starts_period", "career_top2"],
}


def train_variant(train_df, feats):
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    ds = lgb.Dataset(tr[feats], label=tr["is_winner"],
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(va[feats], label=va["is_winner"], reference=ds)
    return lgb.train(PARAMS, ds, valid_sets=[vs], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


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
    return (sum(merged.values()),
            sum(pay.get(k, 0) * y // 100 for k, y in merged.items()))


# ---- 2-A/2-B: 月次walk-forward×4変種 ------------------------------------------
M = defaultdict(lambda: {"ll": 0.0, "nll": 0, "kn": 0, "kst": 0, "krt": 0,
                         "kax": 0, "hn": 0, "h1": 0, "hax": 0})
konsen_v0 = {}   # 2-C用: V0の超混戦レース {rid: {"lanes":…, "month":…}}

for m in EVAL_MONTHS:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)]
    if month_df.empty:
        continue
    for vname, feats in VARIANTS.items():
        print(f"{m} {vname} 学習中...", flush=True)
        booster = train_variant(train_df, feats)
        md = month_df.copy()
        md["pred"] = booster.predict(md[feats])
        eps = 1e-9
        p = md["pred"].clip(eps, 1 - eps)
        yv = md["is_winner"].astype(float)
        a = M[(vname, m)]
        a["ll"] += float(-(yv * np.log(p) + (1 - yv) * np.log(1 - p)).sum())
        a["nll"] += len(md)
        for rid, grp in md.groupby("race_id"):
            arr = res_all.get(rid, {})
            pay = payout_map[rid]
            if len(arr) < 3 or not pay:
                continue
            gs = grp.sort_values("pred", ascending=False)
            lanes = [int(x) for x in gs["lane"]]
            p1 = float(gs["pred"].iloc[0])
            if len(lanes) < 5:
                continue
            top3 = sorted(arr, key=arr.get)[:3]
            ax = lanes[0] in top3 and lanes[1] in top3
            if p1 < 0.20:
                st, rt = score(plan13(lanes), pay)
                a["kn"] += 1
                a["kst"] += st
                a["krt"] += rt
                a["kax"] += ax
                if vname == "V0 基準":
                    konsen_v0[rid] = {"lanes": lanes, "month": m}
            elif p1 < 0.35:
                a["hn"] += 1
                a["h1"] += arr.get(lanes[0]) == 1
                a["hax"] += ax

print("\n===== 2-A/2-B 注入アブレーション(月次walk-forward) =====")
print("― 超混戦帯(各変種自身の選別・⑬2,000円) ―")
hdr = f"{'月':<9}"
for vname in VARIANTS:
    hdr += f"{vname:<24}"
print(hdr + "   (各セル: R数/軸生存率/⑬回収率)")
for m in EVAL_MONTHS:
    row = f"{m:<9}"
    for vname in VARIANTS:
        a = M[(vname, m)]
        axr = a["kax"] / a["kn"] if a["kn"] else 0
        roi = a["krt"] / a["kst"] if a["kst"] else 0
        row += f"{a['kn']:>4}R {axr:>6.1%} {roi:>7.1%}   "
    print(row)
print("― 合計 ―")
for vname in VARIANTS:
    kn = sum(M[(vname, m)]["kn"] for m in EVAL_MONTHS)
    kax = sum(M[(vname, m)]["kax"] for m in EVAL_MONTHS)
    kst = sum(M[(vname, m)]["kst"] for m in EVAL_MONTHS)
    krt = sum(M[(vname, m)]["krt"] for m in EVAL_MONTHS)
    ll = sum(M[(vname, m)]["ll"] for m in EVAL_MONTHS)
    nll = sum(M[(vname, m)]["nll"] for m in EVAL_MONTHS)
    hn = sum(M[(vname, m)]["hn"] for m in EVAL_MONTHS)
    h1 = sum(M[(vname, m)]["h1"] for m in EVAL_MONTHS)
    hax = sum(M[(vname, m)]["hax"] for m in EVAL_MONTHS)
    print(f"  {vname:<12} logloss {ll / nll:.5f} | 超混戦{kn}R "
          f"軸生存{kax / kn:.1%} ⑬回収率{krt / kst:.1%} "
          f"損益{krt - kst:>+9,}円 | 本命帯{hn}R 1位的中{h1 / hn:.1%} "
          f"軸生存{hax / hn:.1%}")

# ---- 2-C: まくり予測(P(決まり手=まくり/まくり差し)) ---------------------------
print("\nまくり分類器の素材構築中...", flush=True)
mk = hist.copy()
mk["_makuri_win"] = ((mk["arrival_order"] == 1)
                     & mk["race_id"].map(race_tech).isin([3, 4])
                     ).astype(float).where(mk["arrival_order"].notna())
gm = mk.groupby("reg_no", sort=False)
mk["p_makuri"] = gm["_makuri_win"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
pmak = mk.set_index(["race_id", "lane"])["p_makuri"]

rows = []
for rid, grp in df[df["date"] >= "2025-01-01"].groupby("race_id"):
    tech = race_tech.get(rid)
    if tech is None:
        continue
    l1 = grp[grp["lane"] == 1]
    outer = grp[grp["lane"].between(3, 6)]
    if l1.empty or outer.empty:
        continue
    l1r = l1.iloc[0]

    def pm(lane):
        try:
            v = pmak.loc[(rid, lane)]
            v = v.iloc[0] if hasattr(v, "iloc") else v
            return float(v) if pd.notna(v) else np.nan
        except KeyError:
            return np.nan

    kado = grp[grp["lane"] == 4]
    rows.append({
        "race_id": rid, "date": l1r["date"],
        "y": 1 if tech in (3, 4) else 0,
        "l1_avg_st": l1r["avg_st"], "l1_class": l1r["racer_class_ord"],
        "l1_motor": l1r["motor_2rate"],
        "l1_form_st": l1r["form_last10_avg_st"],
        "out_min_st": outer["avg_st"].min(),
        "out_max_motor": outer["motor_2rate"].max(),
        "out_max_class": outer["racer_class_ord"].max(),
        "out_max_pmak": max((pm(int(l)) for l in outer["lane"]),
                            default=np.nan),
        "kado_pmak": pm(4) if len(kado) else np.nan,
        "kado_st": kado["avg_st"].iloc[0] if len(kado) else np.nan,
        "st_edge": l1r["avg_st"] - outer["avg_st"].min(),
        "venue_code": int(l1r["venue_code"]), "race_no": int(l1r["race_no"]),
    })
mdata = pd.DataFrame(rows)
MFEATS = [c for c in mdata.columns if c not in ("race_id", "date", "y")]

from sklearn.metrics import roc_auc_score

terc = defaultdict(lambda: [0, 0, 0, 0])   # {(月,層): [n, ax, st, rt]}
gate = defaultdict(lambda: [0, 0])          # {(月,アーム): [st, rt]}
aucs = []
for m in EVAL_MONTHS:
    tr = mdata[mdata["date"] < f"{m}-01"]
    te = mdata[mdata["date"].str.startswith(m)]
    if len(tr) < 500 or te.empty:
        continue
    ds = lgb.Dataset(tr[MFEATS], label=tr["y"],
                     categorical_feature=["venue_code", "race_no"])
    clf = lgb.train({"objective": "binary", "metric": "auc",
                     "learning_rate": 0.05, "num_leaves": 31,
                     "min_data_in_leaf": 50, "verbosity": -1, "seed": 7},
                    ds, num_boost_round=200)
    p_tr = clf.predict(tr[MFEATS])
    lo, hi = np.percentile(p_tr, [33.3, 66.7])
    p_te = clf.predict(te[MFEATS])
    try:
        aucs.append(roc_auc_score(te["y"], p_te))
    except ValueError:
        pass
    pmap = dict(zip(te["race_id"], p_te))
    for rid, info in konsen_v0.items():
        if info["month"] != m or rid not in pmap:
            continue
        lanes = info["lanes"]
        arr = res_all.get(rid, {})
        pay = payout_map[rid]
        top3 = sorted(arr, key=arr.get)[:3]
        ax = lanes[0] in top3 and lanes[1] in top3
        pv = pmap[rid]
        layer = "低" if pv < lo else ("中" if pv < hi else "高")
        st, rt = score(plan13(lanes), pay)
        t = terc[(m, layer)]
        t[0] += 1
        t[1] += ax
        t[2] += st
        t[3] += rt
        # ゲート案: P(まくり)高なら1抜き⑬(5艇残るときのみ)
        lanes_wo1 = [l for l in lanes if l != 1]
        gated = (plan13(lanes_wo1) if layer == "高" and len(lanes_wo1) >= 5
                 else plan13(lanes))
        gst, grt = score(gated, pay)
        ga = gate[(m, "ゲート")]
        ga[0] += gst
        ga[1] += grt
        gb = gate[(m, "現行⑬")]
        gb[0] += st
        gb[1] += rt

print("\n===== 2-C まくり予測(決まり手∈{まくり,まくり差し}) =====")
print(f"AUC(月次平均): {np.mean(aucs):.3f}")
print("V0超混戦のP(まくり)三分位層別(合計):")
for layer in ("低", "中", "高"):
    n = sum(terc[(m, layer)][0] for m in EVAL_MONTHS)
    ax = sum(terc[(m, layer)][1] for m in EVAL_MONTHS)
    st = sum(terc[(m, layer)][2] for m in EVAL_MONTHS)
    rt = sum(terc[(m, layer)][3] for m in EVAL_MONTHS)
    if n:
        print(f"  {layer}: {n:>4}R 軸生存{ax / n:>6.1%} "
              f"⑬回収率{rt / st:>7.1%} 損益{rt - st:>+9,}円")
print("ゲート案(P(まくり)高→1抜き⑬) vs 現行⑬:")
wins = 0
for m in EVAL_MONTHS:
    gs, gr = gate[(m, "ゲート")]
    bs, br = gate[(m, "現行⑬")]
    if not bs:
        continue
    groi, broi = gr / gs, br / bs
    wins += groi >= broi
    print(f"  {m}: ゲート{groi:>7.1%} vs 現行{broi:>7.1%}")
gs = sum(gate[(m, "ゲート")][0] for m in EVAL_MONTHS)
gr = sum(gate[(m, "ゲート")][1] for m in EVAL_MONTHS)
bs = sum(gate[(m, "現行⑬")][0] for m in EVAL_MONTHS)
br = sum(gate[(m, "現行⑬")][1] for m in EVAL_MONTHS)
if bs:
    print(f"  合計: ゲート{gr / gs:.1%}({gr - gs:+,}円) vs "
          f"現行{br / bs:.1%}({br - bs:+,}円) / ゲートが同等以上の月 {wins}/8")
print("\n(採用判定はいずれも事前登録基準に従う。小標本の回収率単独では動かない)")
