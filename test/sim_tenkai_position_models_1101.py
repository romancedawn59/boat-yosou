# -*- coding: utf-8 -*-
"""独自のレース展望(1〜6着予想)の作り直し: 案1 展開型 / 案2 着ごとの専用モデル(2026-11-01ケンさん指示)

    py -X utf8 test/sim_tenkai_position_models_1101.py

■ 比べる方式(すべて朝6:30に分かる情報だけ。学習=2026-09-14まで、採点=2026-09-15〜10-31の本番配信と同じレース)
  本番配信   : 実際に配信したr1〜r6(1着確率の順)と、その確率から機械的に作った着順分布
  M0         : 本番と同じ作り方を同条件で再現(1着モデル→Harville展開)。比較の土台
  M0s        : M0に「展開の特徴量」を足しただけ(材料の効果と仕組みの効果を分けるため)
  案2a       : 着順そのものを当てる専用モデル(1艇ごとに1〜6着の確率)→ 各着の確率表から1〜6着を割り当て
  案2b       : 9月の条件付き連鎖の作り直し(1着確率 → 1着艇を条件に2着 → 1・2着艇を条件に3着)
  案1        : 展開型。レースの決まり手(逃げ/差し/まくり/まくり差し/抜き・恵まれ)の確率を予測し、
               決まり手ごとに学習した着順モデルの結果を、その確率で混ぜる
  参考(上限) : 案1で決まり手が事前に当たっていたら(レース後にしか分からない=後出しの天井)
  参考(市場) : 締切前の3連単オッズから作った着順分布(オッズがあるレースだけ)
■ 展開の特徴量(選手ごと・前日までの全履歴から計算。当日の結果は使わない)
  まくり勝ち率/差し勝ち率/まくり差し勝ち率(出走あたり)、1コース時の1着率・3着内率、前付け率(進入が枠より内)、
  直近10走のST標準偏差(スタートの安定度)、履歴の走数
■ 物差し
  ① 着順分布の正確さ: log-loss(1着・1着を条件に2着・1・2着を条件に3着、合計)。小さいほど良い
  ② 1〜6着予想の当たり方: 着ごとの的中率、順位相関(1=完全一致)、3連単1点・3連複1点の的中率
■ 合格(運用候補)の基準(結果を見る前に固定)
  (a) 3連単合計のlog-lossがM0より小さく、差の95%区間が0を含まない
  (b) 順位相関がM0以上かつ本番配信以上
  (c) 2着・3着の的中率がM0を下回らない
  (d) 9月後半・10月の両方でlog-lossの改善の向きが同じ
"""
import glob
import json
import sys
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
import db  # noqa: E402
import lightgbm as lgb  # noqa: E402
from config import DB_PATH  # noqa: E402
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set  # noqa: E402
from predictors import LAMBDA2, LAMBDA3  # noqa: E402

SPLIT, END = "2026-09-15", "2026-10-31"
TECH5 = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}
TECH_NAMES = ["逃げ", "差し", "まくり", "まくり差し", "抜き・恵まれ"]
rng = np.random.default_rng(1101)
pd.set_option("display.width", 250)

print("データ準備中(本番の特徴量33本)...", flush=True)
conn = db.connect(DB_PATH)
df = build_training_set(conn)
hist = pd.read_sql_query("""
    SELECT r.date, r.race_id, e.reg_no, e.lane, res.course, res.arrival_order, res.st_time, r.winning_technique_number AS tech
    FROM entries e JOIN races r ON r.race_id = e.race_id
    JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane""", conn)
tech = dict(conn.execute("SELECT race_id, winning_technique_number FROM races WHERE winning_technique_number IS NOT NULL").fetchall())
odds3 = defaultdict(dict)
for rid, cb, o in conn.execute("SELECT race_id, combination, odds FROM odds WHERE bet_type='3連単' AND odds>0 AND fetched_at!='final-backfill' AND race_id>=?",
                               (SPLIT.replace("-", ""),)):
    odds3[rid][cb] = o
conn.close()

# ---------- 展開の特徴量(前日までの累積) ----------
print("展開の特徴量を計算中...", flush=True)
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
daily = h.groupby(["reg_no", "date"])[CNT].sum()
cum = daily.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0)
h = h.sort_values(["reg_no", "date", "race_id"])
h["st_std"] = h.groupby("reg_no")["st_time"].transform(lambda s: s.rolling(10, min_periods=3).std())
cum["st_std10"] = h.groupby(["reg_no", "date"])["st_std"].last().groupby(level=0).shift(1)
pri = h[h["date"] < SPLIT][CNT].sum()
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
df["tech5"] = df["race_id"].map(tech).map(TECH5)

prod = {}
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        if r.get("ranked") and len(r["ranked"]) == 6:
            prod[r["race_id"]] = {int(l): float(p) for l, p in r["ranked"]}
BASE = list(FEATURE_COLUMNS)
FULL = BASE + STY
tr = df[df["date"] < SPLIT].copy()
ev = df[(df["date"] >= SPLIT) & (df["date"] <= END) & df["race_id"].isin(prod)].copy().sort_values(["race_id", "lane"]).reset_index(drop=True)
RIDS = ev["race_id"].values[::6]
N = len(RIDS)
print(f"学習 {tr['race_id'].nunique():,}R(2025-07-15〜2026-09-14) / 採点 {N:,}R(2026-09-15〜10-31・本番配信と同じレース)", flush=True)

PB = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31}
PM6 = {"objective": "multiclass", "num_class": 6, "metric": "multi_logloss", "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50}
PK5 = {"objective": "multiclass", "num_class": 5, "metric": "multi_logloss", "verbosity": -1, "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100}


def fit(d, cols, label, params, cat=CATEGORICAL_FEATURES):
    dates = np.sort(d["date"].unique()); cut = dates[int(len(dates) * .85)]
    a, b = d[d["date"] < cut], d[d["date"] >= cut]
    ds = lgb.Dataset(a[cols], label=a[label], categorical_feature=cat)
    vs = lgb.Dataset(b[cols], label=b[label], reference=ds)
    return lgb.train(params, ds, 1000, valid_sets=[vs], callbacks=[lgb.early_stopping(30, verbose=False)])


def rownorm(p):
    p = p.reshape(-1, 6); return p / p.sum(axis=1, keepdims=True)


print("学習中: M0 / M0s(1着モデル)...", flush=True)
p_m0 = rownorm(fit(tr, BASE, "is_winner", PB).predict(ev[BASE]))
m_m0s = fit(tr, FULL, "is_winner", PB)
p_m0s = rownorm(m_m0s.predict(ev[FULL]))
print("学習中: 案2a(着順そのものの専用モデル)...", flush=True)
M2a = fit(tr, FULL, "pos", PM6).predict(ev[FULL]).reshape(N, 6, 6)
print("学習中: 案1(決まり手ごとの着順モデル×5)...", flush=True)
Mk = []
for k in range(5):
    sub = tr[tr["tech5"] == k]
    Mk.append(fit(sub, FULL, "pos", PM6).predict(ev[FULL]).reshape(N, 6, 6))
    print(f"  {TECH_NAMES[k]}: 学習{sub['race_id'].nunique():,}R", flush=True)

# ---------- 決まり手の予測(レース単位) ----------
RL = ["national_win_rate", "local_win_rate", "kr", "avg_st", "form_last10_avg_st", "motor_2rate", "racer_class_ord", "st_std10",
      "r_w_sashi", "r_w_makuri", "r_w_mz", "c1_win_rate", "c1_top3_rate", "r_mae", "form_lane_win_rate", "form_last10_win_rate"]


def race_frame(d):
    w = d.pivot(index="race_id", columns="lane", values=RL)
    w.columns = [f"{c}_{l}" for c, l in w.columns]
    w = w.join(d.groupby("race_id")[["venue_code", "race_no", "grade_ord", "date", "tech5"]].first())
    w["st_gap_in"] = w["avg_st_1"] - w[["avg_st_2", "avg_st_3", "avg_st_4"]].min(axis=1)
    w["kr_gap_in"] = w["kr_1"] - w[[f"kr_{l}" for l in range(2, 7)]].max(axis=1)
    w["nwr_gap_in"] = w["national_win_rate_1"] - w[[f"national_win_rate_{l}" for l in range(2, 7)]].max(axis=1)
    w["makuri_max_out"] = w[[f"r_w_makuri_{l}" for l in range(2, 7)]].max(axis=1)
    return w


print("学習中: 決まり手の予測モデル...", flush=True)
RT, RE = race_frame(tr), race_frame(ev)
assert list(RE.index) == list(RIDS)
KC = [c for c in RT.columns if c not in ("date", "tech5")]
RTk = RT[RT["tech5"].notna()].copy(); RTk["tech5"] = RTk["tech5"].astype(int)
kmodel = fit(RTk, KC, "tech5", PK5, cat=["venue_code", "race_no"])
PI = kmodel.predict(RE[KC])
base_rate = RTk["tech5"].value_counts(normalize=True).sort_index().values

# ---------- 案2b 条件付き連鎖 ----------
print("学習中: 案2b(条件付き連鎖)...", flush=True)
DIFF = ["national_win_rate", "kr", "avg_st", "motor_2rate", "racer_class_ord", "form_last10_avg_st", "r_w_makuri", "r_w_sashi"]


def add_cond(d, cond, pre):
    """d(候補艇)に条件艇condの枠と差分を付ける。condは race_id, lane, DIFF を持つ"""
    c = cond.rename(columns={"lane": f"{pre}lane", **{x: f"{pre}{x}" for x in DIFF}})
    out = d.merge(c, on="race_id")
    for x in DIFF:
        out[f"{pre}d_{x}"] = out[x] - out[f"{pre}{x}"]
    return out


C2 = FULL + ["w_lane", "w_kr", "w_national_win_rate"] + [f"w_d_{x}" for x in DIFF]
C3 = C2 + ["s_lane", "s_kr"] + [f"s_d_{x}" for x in DIFF]
W = tr[tr["pos"] == 0][["race_id", "lane"] + DIFF]; S = tr[tr["pos"] == 1][["race_id", "lane"] + DIFF]
T2 = add_cond(tr[tr["pos"] != 0], W, "w_"); T2["y"] = (T2["pos"] == 1).astype(int)
m2 = fit(T2, C2, "y", PB)
T3 = add_cond(add_cond(tr[tr["pos"] > 1], W, "w_"), S, "s_"); T3["y"] = (T3["pos"] == 2).astype(int)
m3 = fit(T3, C3, "y", PB)
del T2, T3
P2 = np.zeros((N, 6, 6)); P3 = np.zeros((N, 6, 6, 6))
ridx = {r: i for i, r in enumerate(RIDS)}
for s in range(0, N, 800):
    chunk = ev[ev["race_id"].isin(RIDS[s:s + 800])]
    base_c = chunk[["race_id", "lane"] + DIFF]
    f2 = add_cond(chunk, base_c, "w_"); f2 = f2[f2["lane"] != f2["w_lane"]]
    pr = m2.predict(f2[C2])
    for rid, a, b, v in zip(f2["race_id"], f2["w_lane"], f2["lane"], pr):
        P2[ridx[rid], a - 1, b - 1] = v
    f3 = add_cond(add_cond(chunk, base_c, "w_"), base_c, "s_")
    f3 = f3[(f3["lane"] != f3["w_lane"]) & (f3["lane"] != f3["s_lane"]) & (f3["w_lane"] != f3["s_lane"])]
    pr = m3.predict(f3[C3])
    for rid, a, b, c, v in zip(f3["race_id"], f3["w_lane"], f3["s_lane"], f3["lane"], pr):
        P3[ridx[rid], a - 1, b - 1, c - 1] = v
    print(f"  条件付き予測 {min(s + 800, N):,}/{N:,}R", flush=True)

# ---------- 着順分布と1〜6着予想 ----------
PERM3 = np.array(list(permutations(range(6), 3)))
PERM6 = np.array(list(permutations(range(6))))
IDX3 = {tuple(p): i for i, p in enumerate(PERM3)}
MASK1 = [PERM3[:, 0] == a for a in range(6)]
MASK2 = [[(PERM3[:, 0] == a) & (PERM3[:, 1] == b) for b in range(6)] for a in range(6)]


def sinkhorn(M, it=30):
    M = np.clip(M, 1e-6, None)
    for _ in range(it):
        M = M / M.sum(axis=1, keepdims=True); M = M / M.sum(axis=0, keepdims=True)
    return M


def tri_harville(p):
    p2, p3 = p ** LAMBDA2, p ** LAMBDA3
    a, b, c = PERM3.T
    v = p[a] * p2[b] / (p2.sum() - p2[a]) * p3[c] / (p3.sum() - p3[a] - p3[b])
    return v / v.sum()


def tri_matrix(M):
    a, b, c = PERM3.T
    c2, c3 = M[:, 1], M[:, 2]
    v = M[a, 0] * c2[b] / (c2.sum() - c2[a]) * c3[c] / (c3.sum() - c3[a] - c3[b])
    return v / v.sum()


def hungarian(M):
    r, c = linear_sum_assignment(-np.log(np.clip(M, 1e-9, None)))
    order = np.empty(6, dtype=int); order[c] = r
    return order


def tri_chain(pa, P2i, P3i):
    a, b, c = PERM3.T
    p2 = P2i / P2i.sum(axis=1, keepdims=True).clip(1e-12)
    p3 = P3i / P3i.sum(axis=2, keepdims=True).clip(1e-12)
    v = pa[a] * p2[a, b] * p3[a, b, c]
    return v / v.sum()


def order_chain(pa, P2i, P3i):
    a = int(np.argmax(pa)); b = int(np.argmax(np.where(np.arange(6) == a, -1, P2i[a])))
    rest3 = np.where(np.isin(np.arange(6), [a, b]), -1, P3i[a, b]); c = int(np.argmax(rest3))
    rest = [x for x in np.argsort(-pa) if x not in (a, b, c)]
    return np.array([a, b, c] + rest)


acts = ev["pos"].values.reshape(N, 6)            # [race, boat] = 着順-1
actual = np.argsort(acts, axis=1)                  # [race, 着] = 艇
tech_ev = RE["tech5"].values
METHODS = ["本番配信", "M0 本番の作り方を再現", "M0s M0+展開の特徴量", "案2a 着順の専用モデル", "案2b 条件付き連鎖", "案1 展開型", "参考 案1で決まり手が当たっていたら", "参考 市場(締切前オッズ)"]
rec = {m: [] for m in METHODS}
examples = []
for i, rid in enumerate(RIDS):
    act = actual[i]
    out = {}
    pp = np.array([prod[rid][l] for l in range(1, 7)]); pp = pp / pp.sum()
    out["本番配信"] = (tri_harville(pp), np.argsort(-pp))
    out["M0 本番の作り方を再現"] = (tri_harville(p_m0[i]), np.argsort(-p_m0[i]))
    out["M0s M0+展開の特徴量"] = (tri_harville(p_m0s[i]), np.argsort(-p_m0s[i]))
    M = sinkhorn(M2a[i]); out["案2a 着順の専用モデル"] = (tri_matrix(M), hungarian(M))
    out["案2b 条件付き連鎖"] = (tri_chain(p_m0s[i], P2[i], P3[i]), order_chain(p_m0s[i], P2[i], P3[i]))
    Ms = [sinkhorn(Mk[k][i]) for k in range(5)]
    tri1 = sum(PI[i, k] * tri_matrix(Ms[k]) for k in range(5))
    Mmix = sum(PI[i, k] * Ms[k] for k in range(5))
    out["案1 展開型"] = (tri1, hungarian(Mmix))
    if not np.isnan(tech_ev[i]):
        k = int(tech_ev[i]); out["参考 案1で決まり手が当たっていたら"] = (tri_matrix(Ms[k]), hungarian(Ms[k]))
    o = odds3.get(rid.replace("-", ""))
    if o and len(o) >= 108:
        q = np.array([1 / o.get(f"{a + 1}-{b + 1}-{c + 1}", 1e9) for a, b, c in PERM3]); q = q / q.sum()
        q = np.where(q <= 0, 1e-6, q); out["参考 市場(締切前オッズ)"] = (q / q.sum(), None)
    for m, (tri, order) in out.items():
        a, b, c = act[:3]
        pa = tri[MASK1[a]].sum(); pab = tri[MASK2[a][b]].sum(); pabc = tri[IDX3[(a, b, c)]]
        r_ = dict(rid=rid, month=rid[:6], ll1=-np.log(max(pa, 1e-9)), ll2=-np.log(max(pab / pa, 1e-9)), ll3=-np.log(max(pabc / pab, 1e-9)))
        r_["ll"] = r_["ll1"] + r_["ll2"] + r_["ll3"]
        if order is not None:
            rp = np.empty(6); rp[order] = np.arange(6); ra = np.empty(6); ra[act] = np.arange(6)
            r_["sp"] = 1 - ((rp - ra) ** 2).sum() / 35
            for j in range(6):
                r_[f"a{j + 1}"] = int(order[j] == act[j])
            r_["tri1"] = int(all(order[:3] == act[:3])); r_["trio1"] = int(set(order[:3]) == set(act[:3]))
        rec[m].append(r_)
    if rid[:7] == "2026103" and len(examples) < 2 and i % 23 == 0:
        examples.append((rid, PI[i], out["案1 展開型"][1], out["本番配信"][1], act))

R = {m: pd.DataFrame(v) for m, v in rec.items() if v}

# ---------- 報告 ----------
print(f"\n===== 決まり手の予測(採点{N:,}R) =====")
y = np.nan_to_num(tech_ev, nan=-1).astype(int); ok = y >= 0
ll_k = -np.log(np.clip(PI[ok, y[ok]], 1e-9, None)).mean(); ll_b = -np.log(base_rate[y[ok]]).mean()
print(f"log-loss 予測{ll_k:.4f} / 学習期間の出現率だけ{ll_b:.4f}(改善{ll_b - ll_k:+.4f}) / 最も確率の高い決まり手の的中率 {(PI[ok].argmax(1) == y[ok]).mean():.1%}(常に『逃げ』なら{(y[ok] == 0).mean():.1%})")
t = pd.DataFrame({"実際の割合": [(y[ok] == k).mean() for k in range(5)], "予測の平均": PI[ok].mean(0)}, index=TECH_NAMES)
print(t.to_string(formatters={c: "{:.1%}".format for c in t.columns}))
pm = PI[ok, 2] + PI[ok, 3]; am = np.isin(y[ok], [2, 3])
qs = pd.qcut(pm, 5, labels=["下位20%", "20〜40", "40〜60", "60〜80", "上位20%"])
print("[まくり系(まくり+まくり差し)の予測確率の5分位 → 実際のまくり系の割合]")
print(pd.DataFrame({"予測": pm, "実際": am, "q": qs}).groupby("q", observed=True)[["予測", "実際"]].mean().to_string(formatters={"予測": "{:.1%}".format, "実際": "{:.1%}".format}))

print("\n===== 着順分布の正確さ(log-loss・小さいほど良い)と1〜6着予想の当たり方(採点期間) =====")
rows = []
for m, d in R.items():
    row = dict(方式=m, R=len(d), 一着=d["ll1"].mean(), 二着=d["ll2"].mean(), 三着=d["ll3"].mean(), 合計=d["ll"].mean())
    if "sp" in d:
        row.update({**{f"{j}着的中": d[f"a{j}"].mean() for j in range(1, 7)}, "順位相関": d["sp"].mean(), "3連単1点": d["tri1"].mean(), "3連複1点": d["trio1"].mean()})
    rows.append(row)
T = pd.DataFrame(rows)
fm = {c: "{:.4f}".format for c in ("一着", "二着", "三着", "合計")}
fm.update({c: "{:.1%}".format for c in T.columns if c.endswith("的中") or c in ("3連単1点", "3連複1点")})
fm["順位相関"] = "{:.3f}".format
print(T.to_string(index=False, formatters=fm, na_rep="-"))


def boot(v, n=2000):
    v = np.asarray(v); idx = rng.integers(0, len(v), size=(n, len(v))); mm = v[idx].mean(1)
    return v.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


print("\n===== M0との差(レースごとの対の差・[ ]は95%区間)。log-lossは M0−方式(正なら方式が良い)、他は 方式−M0 =====")
base = R["M0 本番の作り方を再現"].set_index("rid")
rows = []
for m, d in R.items():
    if m in ("M0 本番の作り方を再現", "参考 市場(締切前オッズ)"):
        continue
    d = d.set_index("rid").loc[base.index.intersection(d.set_index("rid").index)]
    b = base.loc[d.index]
    row = {"方式": m}
    for col, sign, lab in (("ll", -1, "log-loss合計"), ("sp", 1, "順位相関"), ("a2", 1, "2着的中"), ("a3", 1, "3着的中")):
        if col not in d:
            continue
        mu, lo, hi = boot(sign * (d[col].values - b[col].values))
        mark = "◎" if lo > 0 else ("▲" if hi < 0 else "")
        row[lab] = f"{mu:+.4f}[{lo:+.4f},{hi:+.4f}]{mark}"
    for mon in ("202609", "202610"):
        dm = d[d["month"] == mon]; bm = b.loc[dm.index]
        row[f"log-loss改善 {mon[4:]}月"] = f"{(bm['ll'] - dm['ll']).mean():+.4f}"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\n===== 参考: 締切前オッズがあるレースだけで市場と比べる(log-loss合計) =====")
mk = R["参考 市場(締切前オッズ)"].set_index("rid")
for m, d in R.items():
    d = d.set_index("rid"); common = d.index.intersection(mk.index)
    print(f"  {m:<24} {len(common):>4}R 一着{d.loc[common, 'll1'].mean():.4f} 二着{d.loc[common, 'll2'].mean():.4f} 三着{d.loc[common, 'll3'].mean():.4f} 合計{d.loc[common, 'll'].mean():.4f}")

print("\n===== 合格判定(運用候補の基準 a〜d) =====")
prod_sp = R["本番配信"]["sp"].mean(); m0 = R["M0 本番の作り方を再現"]
for m in ("M0s M0+展開の特徴量", "案2a 着順の専用モデル", "案2b 条件付き連鎖", "案1 展開型"):
    d = R[m].set_index("rid").loc[m0["rid"]]; b = m0.set_index("rid")
    mu, lo, hi = boot(b["ll"].values - d["ll"].values)
    ca = lo > 0
    cb = d["sp"].mean() >= b["sp"].mean() and d["sp"].mean() >= prod_sp
    cc = d["a2"].mean() >= b["a2"].mean() and d["a3"].mean() >= b["a3"].mean()
    imp = [(b.loc[d.index[d["month"] == mon], "ll"] - d.loc[d["month"] == mon, "ll"]).mean() for mon in ("202609", "202610")]
    cd = (imp[0] > 0) == (imp[1] > 0) and imp[0] > 0
    print(f"  {m}: (a){'○' if ca else '×'} (b){'○' if cb else '×'} (c){'○' if cc else '×'} (d){'○' if cd else '×'} → {'運用候補' if all((ca, cb, cc, cd)) else '不合格'}")

print("\n===== 展望の出力例(10/31) =====")
for rid, pi, o1, op, act in examples:
    print(f"  {rid}: 決まり手予測 " + " ".join(f"{TECH_NAMES[k]}{pi[k]:.0%}" for k in range(5)))
    print(f"    案1の1〜6着予想 {'-'.join(str(x + 1) for x in o1)} / 本番配信 {'-'.join(str(x + 1) for x in op)} / 実際 {'-'.join(str(x + 1) for x in act)}")
