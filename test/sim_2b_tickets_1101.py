# -*- coding: utf-8 -*-
"""独自システム(案2b 条件付き連鎖)の着順確率で買った場合の払戻(2026-11-01ケンさん指示)

    py -X utf8 test/sim_2b_tickets_1101.py

買い方(100円/点・各方式の着順確率の上位から機械的に選ぶ):
  ① 3連単1点 = 3連単確率1位の組   ② 3連複1点 = 3連複確率1位の組(3連単6通りの合計)
  ③ 3連単5点 = 3連単確率の上位5組  ④ 3連複2点 = 3連複確率の上位2組
比べる方式: 案2b(独自) / M0(今の作り方を同条件で再現) / 本番配信(実際に配信した1着確率からの展開)
期間: 7/1〜10/31の全レース。7月・8月・9/1〜14は各月の前月末まで、9/15〜10/31は9/14までで学習(採点レースは学習に使わない)
事前宣言ではない記述的な集計。帯や堅さの区切りで良く見えるマスは後出しとして扱う
"""
import sys
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
import db  # noqa: E402
import lightgbm as lgb  # noqa: E402
from config import DB_PATH  # noqa: E402
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set  # noqa: E402
from predictors import LAMBDA2, LAMBDA3  # noqa: E402

BLOCKS = [("2026-07-01", "2026-07-31", "主検証"), ("2026-08-01", "2026-08-31", "主検証"), ("2026-09-01", "2026-09-14", "主検証"),
          ("2026-09-15", "2026-10-31", "参考(既見)"), ("2026-11-01", "2026-11-30", "前向き確認")]
LAMS = np.round(np.arange(0, 1.01, 0.1), 1)
rng = np.random.default_rng(20261101)
pd.set_option("display.width", 220)

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
df = build_training_set(conn)
hist = pd.read_sql_query("""
    SELECT r.date, r.race_id, e.reg_no, e.lane, res.course, res.arrival_order, res.st_time, r.winning_technique_number AS tech
    FROM entries e JOIN races r ON r.race_id = e.race_id
    JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane""", conn)
odds3 = defaultdict(dict)
for rid, cb, o in conn.execute("SELECT race_id, combination, odds FROM odds WHERE bet_type='3連単' AND odds>0 AND fetched_at!='final-backfill' AND race_id>='20260701'"):
    odds3[rid][cb] = o
conn.close()

# ---- 展開の特徴量(前日までの累積・sim_tenkai_position_models_1101.py と同一) ----
h = hist.dropna(subset=["reg_no"]).copy()
h["reg_no"] = h["reg_no"].astype(int)
h["start"] = h["arrival_order"].notna().astype(int)
h["win"] = (h["arrival_order"] == 1).astype(int)
for t, nm in ((2, "sashi"), (3, "makuri"), (4, "mz")):
    h[f"w_{nm}"] = ((h["arrival_order"] == 1) & (h["tech"] == t)).astype(int)
h["c1"] = (h["course"] == 1).astype(int)
h["c1_win"] = ((h["course"] == 1) & (h["arrival_order"] == 1)).astype(int)
h["c1_top3"] = ((h["course"] == 1) & (h["arrival_order"] <= 3)).astype(int)
h["mae"] = (h["course"].notna() & (h["course"] < h["lane"])).astype(int)
CNT = ["start", "win", "w_sashi", "w_makuri", "w_mz", "c1", "c1_win", "c1_top3", "mae"]
cum = h.groupby(["reg_no", "date"])[CNT].sum().groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0)
h = h.sort_values(["reg_no", "date", "race_id"])
h["st_std"] = h.groupby("reg_no")["st_time"].transform(lambda s: s.rolling(10, min_periods=3).std())
cum["st_std10"] = h.groupby(["reg_no", "date"])["st_std"].last().groupby(level=0).shift(1)
pri = h[h["date"] < "2026-07-01"][CNT].sum()   # 平滑化の事前値は主検証より前だけで決める
K = 20
sty = pd.DataFrame(index=cum.index)
for nm in ("w_sashi", "w_makuri", "w_mz", "mae"):
    sty[f"r_{nm}"] = (cum[nm] + K * pri[nm] / pri["start"]) / (cum["start"] + K)
sty["c1_win_rate"] = (cum["c1_win"] + K * pri["c1_win"] / pri["c1"]) / (cum["c1"] + K)
sty["c1_top3_rate"] = (cum["c1_top3"] + K * pri["c1_top3"] / pri["c1"]) / (cum["c1"] + K)
sty["st_std10"] = cum["st_std10"]
sty["n_hist"] = np.log1p(cum["start"])
STY = list(sty.columns)
df = df.merge(sty.reset_index(), on=["reg_no", "date"], how="left")
g = df.groupby("race_id")["arrival_order"].agg(["count", "nunique", "max"])
good = g[(g["count"] == 6) & (g["nunique"] == 6) & (g["max"] == 6)].index
df = df[df["race_id"].isin(good)].sort_values(["race_id", "lane"]).reset_index(drop=True)
df["pos"] = (df["arrival_order"] - 1).astype(int)
BASE = list(FEATURE_COLUMNS); FULL = BASE + STY
DIFF = ["national_win_rate", "kr", "avg_st", "motor_2rate", "racer_class_ord", "form_last10_avg_st", "r_w_makuri", "r_w_sashi"]
C2 = FULL + ["w_lane", "w_kr", "w_national_win_rate"] + [f"w_d_{x}" for x in DIFF]
C3 = C2 + ["s_lane", "s_kr"] + [f"s_d_{x}" for x in DIFF]
PB = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31}
PERM3 = np.array(list(permutations(range(6), 3)))
IDX3 = {tuple(p): i for i, p in enumerate(PERM3)}
MASK1 = [PERM3[:, 0] == a for a in range(6)]
MASK2 = [[(PERM3[:, 0] == a) & (PERM3[:, 1] == b) for b in range(6)] for a in range(6)]


def fit(d, cols, label):
    dates = np.sort(d["date"].unique()); cut = dates[int(len(dates) * .85)]
    a, b = d[d["date"] < cut], d[d["date"] >= cut]
    ds = lgb.Dataset(a[cols], label=a[label], categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(PB, ds, 1000, valid_sets=[lgb.Dataset(b[cols], label=b[label], reference=ds)],
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def add_cond(d, cond, pre):
    c = cond.rename(columns={"lane": f"{pre}lane", **{x: f"{pre}{x}" for x in DIFF}})
    out = d.merge(c, on="race_id")
    for x in DIFF:
        out[f"{pre}d_{x}"] = out[x] - out[f"{pre}{x}"]
    return out


def tri_harville(p):
    p2, p3 = p ** LAMBDA2, p ** LAMBDA3
    a, b, c = PERM3.T
    v = p[a] * p2[b] / (p2.sum() - p2[a]) * p3[c] / (p3.sum() - p3[a] - p3[b])
    return v / v.sum()


def ll3(tri, act):
    a, b, c = act
    pa = tri[MASK1[a]].sum(); pab = tri[MASK2[a][b]].sum(); pabc = tri[IDX3[(a, b, c)]]
    return np.array([-np.log(max(pa, 1e-9)), -np.log(max(pab / pa, 1e-9)), -np.log(max(pabc / pab, 1e-9))])


import glob
import json

pay = defaultdict(list)
conn = db.connect(DB_PATH)
for rid, bt, cb, a in conn.execute("SELECT race_id, bet_type, combination, amount_yen FROM payouts WHERE race_id>='20260701' AND bet_type IN ('3連単','3連複')"):
    pay[(rid, bt)].append((cb, a or 0))
conn.close()
prod = {}
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        if r.get("ranked") and len(r["ranked"]) == 6:
            prod[r["race_id"]] = np.array([dict((int(l), float(p)) for l, p in r["ranked"])[k] for k in range(1, 7)])

TICKET_BLOCKS = [("2026-07-01", "2026-07-31"), ("2026-08-01", "2026-08-31"), ("2026-09-01", "2026-09-14"), ("2026-09-15", "2026-10-31")]
TRIO_KEYS = sorted({"=".join(map(str, sorted((a + 1, b + 1, c + 1)))) for a, b, c in PERM3})
TRIO_IDX = np.array([TRIO_KEYS.index("=".join(map(str, sorted((a + 1, b + 1, c + 1))))) for a, b, c in PERM3])
TRI_KEYS = [f"{a + 1}-{b + 1}-{c + 1}" for a, b, c in PERM3]


def picks(tri):
    """着順確率(120通り) → 3連単上位5点・3連複上位2点"""
    trio = np.bincount(TRIO_IDX, weights=tri, minlength=20)
    t_order = np.argsort(-tri)[:5]; q_order = np.argsort(-trio)[:2]
    return [TRI_KEYS[i] for i in t_order], [TRIO_KEYS[i] for i in q_order], float(tri[t_order[0]]), float(trio[q_order[0]])


def ret(rid, bt, combos):
    return sum(a for cb, a in pay.get((rid, bt), []) if cb in combos)


recs = []
for start, end in TICKET_BLOCKS:
    ids = [r for r in df.loc[(df["date"] >= start) & (df["date"] <= end), "race_id"].unique() if (r, "3連単") in pay and (r, "3連複") in pay]
    tr = df[df["date"] < start]
    te = df[df["race_id"].isin(ids)].sort_values(["race_id", "lane"]).reset_index(drop=True)
    rids = te["race_id"].values[::6]; n = len(rids)
    print(f"[{start}〜{end}] 学習〜{tr['date'].max()} / 対象 {n:,}R 学習中...", flush=True)
    p_m0 = fit(tr, BASE, "is_winner").predict(te[BASE]).reshape(n, 6)
    p_full = fit(tr, FULL, "is_winner").predict(te[FULL]).reshape(n, 6)
    W = tr[tr["pos"] == 0][["race_id", "lane"] + DIFF]; S = tr[tr["pos"] == 1][["race_id", "lane"] + DIFF]
    T2 = add_cond(tr[tr["pos"] != 0], W, "w_"); T2["y"] = (T2["pos"] == 1).astype(int)
    m2 = fit(T2, C2, "y")
    T3 = add_cond(add_cond(tr[tr["pos"] > 1], W, "w_"), S, "s_"); T3["y"] = (T3["pos"] == 2).astype(int)
    m3 = fit(T3, C3, "y")
    del T2, T3
    ridx = {r: i for i, r in enumerate(rids)}
    P2 = np.zeros((n, 6, 6)); P3 = np.zeros((n, 6, 6, 6))
    for s in range(0, n, 800):
        chunk = te[te["race_id"].isin(rids[s:s + 800])]
        base_c = chunk[["race_id", "lane"] + DIFF]
        f2 = add_cond(chunk, base_c, "w_"); f2 = f2[f2["lane"] != f2["w_lane"]]
        for rid, a, b, v in zip(f2["race_id"], f2["w_lane"], f2["lane"], m2.predict(f2[C2])):
            P2[ridx[rid], a - 1, b - 1] = v
        f3 = add_cond(add_cond(chunk, base_c, "w_"), base_c, "s_")
        f3 = f3[(f3["lane"] != f3["w_lane"]) & (f3["lane"] != f3["s_lane"]) & (f3["w_lane"] != f3["s_lane"])]
        for rid, a, b, c, v in zip(f3["race_id"], f3["w_lane"], f3["s_lane"], f3["lane"], m3.predict(f3[C3])):
            P3[ridx[rid], a - 1, b - 1, c - 1] = v
    A_, B_, C_ = PERM3.T
    for i, rid in enumerate(rids):
        pf = p_full[i] / p_full[i].sum()
        p2 = P2[i] / P2[i].sum(axis=1, keepdims=True).clip(1e-12)
        p3 = P3[i] / P3[i].sum(axis=2, keepdims=True).clip(1e-12)
        t2b = pf[A_] * p2[A_, B_] * p3[A_, B_, C_]; t2b = t2b / t2b.sum()
        sysm = {"案2b(独自)": (t2b, pf.max()), "M0(今の作り方)": (tri_harville(p_m0[i] / p_m0[i].sum()), (p_m0[i] / p_m0[i].sum()).max())}
        if rid in prod:
            pp = prod[rid] / prod[rid].sum(); sysm["本番配信"] = (tri_harville(pp), pp.max())
        for name, (tri, p1) in sysm.items():
            t5, q2, ptop, qtop = picks(tri)
            recs.append(dict(rid=rid, month=rid[:6], system=name, p1=p1, ptop=ptop, qtop=qtop,
                             r_t1=ret(rid, "3連単", t5[:1]), r_q1=ret(rid, "3連複", q2[:1]),
                             r_t5=ret(rid, "3連単", t5), r_q2=ret(rid, "3連複", q2)))
    print(f"  完了", flush=True)

D = pd.DataFrame(recs)
D.to_csv(r"Y:\マイドライブ\boat\data_raw\tickets_2b_1101.csv", index=False)
TK = [("r_t1", 1, "①3連単1点"), ("r_q1", 1, "②3連複1点"), ("r_t5", 5, "③3連単5点"), ("r_q2", 2, "④3連複2点")]


def summ(d):
    out = {"R": len(d)}
    for col, k, lab in TK:
        v = d[col].values; tot = v.sum(); inv = len(d) * k * 100
        out[f"{lab} 的中率"] = (v > 0).mean(); out[f"{lab} 回収率"] = tot / inv
        out[f"{lab} 1本除く"] = (tot - v.max()) / inv
    return pd.Series(out)


def show(t, title):
    print(f"\n===== {title} =====")
    print(t.apply(lambda row: row.map("{:,.0f}".format) if row.name == "R" else row.map("{:.1%}".format), axis=1).to_string())


common = set(D[D["system"] == "本番配信"]["rid"])
Dc = D[D["rid"].isin(common)]
show(Dc.groupby("system").apply(summ).T, "全体(7/7〜10/31・3方式そろうレース)・100円/点")
for sysname in ("案2b(独自)", "M0(今の作り方)"):
    show(D[D["system"] == sysname].groupby("month").apply(summ).T, f"{sysname} 月別")
B2 = D[D["system"] == "案2b(独自)"].copy()
B2["帯"] = pd.cut(B2["p1"], [0, .15, .225, .30, .40, .50, 1], right=False)
show(B2.groupby("帯", observed=True).apply(summ).T, "案2b 1着確率の帯別")
B2["3単1位の確率"] = pd.cut(B2["ptop"], [0, .05, .08, .10, .15, .20, 1], right=False)
show(B2.groupby("3単1位の確率", observed=True).apply(summ).T, "案2b 3連単1位の組の確率別(堅さ)")
B2["3複1位の確率"] = pd.cut(B2["qtop"], [0, .15, .20, .25, .30, .40, 1], right=False)
show(B2.groupby("3複1位の確率", observed=True).apply(summ).T, "案2b 3連複1位の組の確率別(堅さ)")
print("\n[案2b 的中した払戻の大きい順(①〜④)]")
for col, k, lab in TK:
    v = B2[col].sort_values(ascending=False).values
    print(f"  {lab}: 上位5本 {list(map(int, v[:5]))} / 的中時の中央値 {int(np.median(v[v > 0]))}円")
