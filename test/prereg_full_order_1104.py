# -*- coding: utf-8 -*-
"""事前宣言(2026-11-04・結果を見る前に固定し、実行前にコミットする)
6艇の着順全体を学習に使うと、的中率と1〜6着の並びが良くなるか

    py -X utf8 test/prereg_full_order_1104.py

■ 背景(ケンさん): 1着だけ見ているから勝てない。6艇すべての着順をデータとして追う必要がある
■ 比べる方式(すべて朝6:30に分かる情報だけ。月ごとのwalk-forward: 7月・8月・9月・10/1〜11/3を、それぞれ前月末までで学習)
  M0                    = 今の作り方(1着モデル→Harville展開)。比較の土台
  ランキング学習        = 6艇の着順全体を正解にしたLightGBMのランキング学習(lambdarank)。点数をexp(点数/T)の強さにして
                          着順確率を作る(Plackett-Luce)。Tは学習期間の検証分で選ぶ
  案2b(6着まで連鎖)     = 1着確率→1着艇を条件に2着→1・2着艇を条件に3着→1〜3着艇を条件に4着→1〜4着艇を条件に5着(6着は残り)
  参考 案2b(3着まで連鎖) = 11/1の案2b。4着以下は1着確率の順
■ 問いと物差し
  問いA(ランキング学習)  : 単勝1点・2連複1点・3連複1点・3連単5点の的中率(各方式の着順確率で1位の組・上位5組)
  問いB(6着まで連鎖)     : 1〜6着の並びの順位相関(1=完全一致)。参考に着ごとの的中・完全一致
     ※ 6着まで延ばしても1〜3着の確率は案2bと同じなので、4つの的中率は問いBの対象にしない
  比較はレースごとの対の差(方式 − M0)、95%区間はブートストラップ2,000回
■ 合格の条件
  A: 4つの的中率のうち少なくとも1つでM0より高く差の95%区間が0より上、かつ4つとも有意に下がらない
  B: 順位相関がM0より高く差の95%区間が0より上
■ 注意
  案2b(3着まで連鎖)の4つの的中率は7/1〜10/31で既に見ている(参考扱い)。ランキング学習と6着までの連鎖は未採点
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


WF = [("2026-07-01", "2026-07-31"), ("2026-08-01", "2026-08-31"), ("2026-09-01", "2026-09-30"), ("2026-10-01", "2026-11-03")]
C4 = C3 + ["t_lane", "t_kr"] + [f"t_d_{x}" for x in DIFF]
C5 = C4 + ["f_lane", "f_kr"] + [f"f_d_{x}" for x in DIFF]
PR = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3], "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31,
      "min_data_in_leaf": 50, "lambdarank_truncation_level": 6}
A_, B_, C_ = PERM3.T
TEMPS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]


def fit_rank(d, cols):
    """6艇の着順全体を正解にしたランキング学習。点数→強さexp(点数/T)のTは学習期間の検証分で選ぶ"""
    d = d.sort_values(["race_id", "lane"])
    dates = np.sort(d["date"].unique()); cut = dates[int(len(dates) * .85)]
    a, b = d[d["date"] < cut], d[d["date"] >= cut]
    ds = lgb.Dataset(a[cols], label=5 - a["pos"].values, group=a.groupby("race_id", sort=False).size().values, categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(b[cols], label=5 - b["pos"].values, group=b.groupby("race_id", sort=False).size().values, reference=ds)
    m = lgb.train(PR, ds, 1000, valid_sets=[vs], callbacks=[lgb.early_stopping(30, verbose=False)])
    sb = m.predict(b[cols]).reshape(-1, 6); posb = b["pos"].values.reshape(-1, 6)
    orders = np.argsort(posb, axis=1)
    best = None
    for T in TEMPS:
        w = np.exp((sb - sb.max(1, keepdims=True)) / T)
        ll = 0.0
        for i in range(len(w)):
            Wt = w[i].sum()
            for k in range(3):
                l = orders[i, k]; ll -= np.log(w[i, l] / Wt); Wt -= w[i, l]
        if best is None or ll < best[0]:
            best = (ll, T)
    return m, best[1]


def tri_pl(w):
    Wt = w.sum(); v = w[A_] / Wt * w[B_] / (Wt - w[A_]) * w[C_] / (Wt - w[A_] - w[B_])
    return v / v.sum()


def cond_of(ch, col, base_c):
    """ch(race_id, 条件の艇の枠col) → add_cond 用の条件表(race_id, lane, DIFF)"""
    return ch[["race_id", col]].merge(base_c, left_on=["race_id", col], right_on=["race_id", "lane"]).drop(columns=[col])


def four_hits(tri, act):
    q2 = defaultdict(float); q3 = defaultdict(float)
    for j, (a, b, c) in enumerate(PERM3):
        q2[frozenset((a, b))] += tri[j]; q3[frozenset((a, b, c))] += tri[j]
    p1 = np.array([tri[PERM3[:, 0] == a].sum() for a in range(6)])
    top5 = set(np.argsort(-tri)[:5])
    return dict(tan=int(p1.argmax() == act[0]), q2=int(max(q2, key=q2.get) == frozenset(act[:2])),
                q3=int(max(q3, key=q3.get) == frozenset(act[:3])), t5=int(IDX3[tuple(act[:3])] in top5))


def order_metrics(order, act):
    rp = np.empty(6); rp[order] = np.arange(6); ra = np.empty(6); ra[act] = np.arange(6)
    out = {f"a{j + 1}": int(order[j] == act[j]) for j in range(6)}
    out["sp"] = 1 - ((rp - ra) ** 2).sum() / 35
    out["exact"] = int(all(order == act))
    return out


recs = []
for start, end in WF:
    tr = df[df["date"] < start]
    te = df[(df["date"] >= start) & (df["date"] <= end)].sort_values(["race_id", "lane"]).reset_index(drop=True)
    rids = te["race_id"].values[::6]; n = len(rids)
    print(f"[{start}〜{end}] 学習〜{tr['date'].max()}({tr['race_id'].nunique():,}R) / 採点 {n:,}R 学習中...", flush=True)
    p_m0 = fit(tr, BASE, "is_winner").predict(te[BASE]).reshape(n, 6)
    p_full = fit(tr, FULL, "is_winner").predict(te[FULL]).reshape(n, 6)
    Wc = tr[tr["pos"] == 0][["race_id", "lane"] + DIFF]; Sc = tr[tr["pos"] == 1][["race_id", "lane"] + DIFF]
    Tc = tr[tr["pos"] == 2][["race_id", "lane"] + DIFF]; Fc = tr[tr["pos"] == 3][["race_id", "lane"] + DIFF]
    T2 = add_cond(tr[tr["pos"] != 0], Wc, "w_"); T2["y"] = (T2["pos"] == 1).astype(int); m2 = fit(T2, C2, "y"); del T2
    T3 = add_cond(add_cond(tr[tr["pos"] > 1], Wc, "w_"), Sc, "s_"); T3["y"] = (T3["pos"] == 2).astype(int); m3 = fit(T3, C3, "y"); del T3
    T4 = add_cond(add_cond(add_cond(tr[tr["pos"] > 2], Wc, "w_"), Sc, "s_"), Tc, "t_"); T4["y"] = (T4["pos"] == 3).astype(int); m4 = fit(T4, C4, "y"); del T4
    T5 = add_cond(add_cond(add_cond(add_cond(tr[tr["pos"] > 3], Wc, "w_"), Sc, "s_"), Tc, "t_"), Fc, "f_"); T5["y"] = (T5["pos"] == 4).astype(int)
    m5 = fit(T5, C5, "y"); del T5
    mr, Tm = fit_rank(tr, FULL)
    s_r = mr.predict(te[FULL]).reshape(n, 6)
    print(f"  ランキング学習の温度 T={Tm}", flush=True)
    # 案2b: 2着・3着の条件付き確率(全組)
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
    # 案2bの連鎖で1〜3着を決め、4着・5着も条件付きモデルで順に決める(6着は残り)
    pf_n = p_full / p_full.sum(1, keepdims=True)
    ch_a = pf_n.argmax(1)
    ch_b = np.array([max((x for x in range(6) if x != ch_a[i]), key=lambda x: P2[i, ch_a[i], x]) for i in range(n)])
    ch_c = np.array([max((x for x in range(6) if x not in (ch_a[i], ch_b[i])), key=lambda x: P3[i, ch_a[i], ch_b[i], x]) for i in range(n)])
    base_c = te[["race_id", "lane"] + DIFF]
    ch = pd.DataFrame({"race_id": rids, "a": ch_a + 1, "b": ch_b + 1, "c": ch_c + 1})
    cand = te.merge(ch, on="race_id")
    cand = cand[(cand["lane"] != cand["a"]) & (cand["lane"] != cand["b"]) & (cand["lane"] != cand["c"])].drop(columns=["a", "b", "c"])
    f4 = add_cond(add_cond(add_cond(cand, cond_of(ch, "a", base_c), "w_"), cond_of(ch, "b", base_c), "s_"), cond_of(ch, "c", base_c), "t_")
    f4["p"] = m4.predict(f4[C4])
    ch_d = f4.loc[f4.groupby("race_id")["p"].idxmax(), ["race_id", "lane"]].set_index("race_id")["lane"].reindex(rids).values
    ch["d"] = ch_d
    cand5 = cand.merge(ch[["race_id", "d"]], on="race_id"); cand5 = cand5[cand5["lane"] != cand5["d"]].drop(columns=["d"])
    f5 = add_cond(add_cond(add_cond(add_cond(cand5, cond_of(ch, "a", base_c), "w_"), cond_of(ch, "b", base_c), "s_"), cond_of(ch, "c", base_c), "t_"),
                  cond_of(ch, "d", base_c), "f_")
    f5["p"] = m5.predict(f5[C5])
    ch_e = f5.loc[f5.groupby("race_id")["p"].idxmax(), ["race_id", "lane"]].set_index("race_id")["lane"].reindex(rids).values
    acts = np.argsort(te["pos"].values.reshape(n, 6), axis=1)
    for i, rid in enumerate(rids):
        act = acts[i]
        pm = p_m0[i] / p_m0[i].sum()
        p2 = P2[i] / P2[i].sum(axis=1, keepdims=True).clip(1e-12)
        p3 = P3[i] / P3[i].sum(axis=2, keepdims=True).clip(1e-12)
        t2b = pf_n[i][A_] * p2[A_, B_] * p3[A_, B_, C_]; t2b = t2b / t2b.sum()
        w = np.exp((s_r[i] - s_r[i].max()) / Tm)
        a, b, c, d, e = ch_a[i], ch_b[i], ch_c[i], ch_d[i] - 1, ch_e[i] - 1
        last = [x for x in range(6) if x not in (a, b, c, d, e)][0]
        order_2b3 = np.array([a, b, c] + [x for x in np.argsort(-pf_n[i]) if x not in (a, b, c)])
        order_2b6 = np.array([a, b, c, d, e, last])
        rows_i = [("M0 今の作り方", tri_harville(pm), np.argsort(-pm)),
                  ("案2b(3着まで連鎖)", t2b, order_2b3),
                  ("案2b(6着まで連鎖)", t2b, order_2b6),
                  ("ランキング学習(並び全体)", tri_pl(w), np.argsort(-s_r[i]))]
        for name, tri, order in rows_i:
            recs.append(dict(rid=rid, month=rid[:6], method=name, **four_hits(tri, act), **order_metrics(order, act)))
    print("  完了", flush=True)

R = pd.DataFrame(recs)
LAB = [("tan", "単勝1点"), ("q2", "2連複1点"), ("q3", "3連複1点"), ("t5", "3連単5点")]
print(f"\n対象 {R['rid'].nunique():,}R(7/1〜11/3・各月の前月末までで学習)")
print("\n===== 4つの的中率と1〜6着の並び =====")
t = R.groupby("method", sort=False).agg(**{lab: (k, "mean") for k, lab in LAB}, **{f"{j}着": (f"a{j}", "mean") for j in range(1, 7)},
                                         順位相関=("sp", "mean"), 完全一致=("exact", "mean"))
print(t.to_string(formatters={**{c: "{:.1%}".format for c in t.columns if c != "順位相関"}, "順位相関": "{:.3f}".format}))


def ci(v):
    v = np.asarray(v, dtype=float); idx = rng.integers(0, len(v), size=(2000, len(v))); mm = v[idx].mean(1)
    return v.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


def diff(method, col):
    a = R[R["method"] == method].set_index("rid")[col]; b = R[R["method"] == "M0 今の作り方"].set_index("rid")[col]
    return ci((a - b.loc[a.index]).values)


def fmt(x, pt=True):
    mu, lo, hi = x; k = 100 if pt else 1
    s = f"{mu * k:+.2f}{'pt' if pt else ''} [{lo * k:+.2f}, {hi * k:+.2f}]"
    return s + ("  ◎上がった" if lo > 0 else ("  ▲下がった" if hi < 0 else ""))


print("\n===== 問いA: ランキング学習 − M0(4つの的中率) =====")
resA = {}
for k, lab in LAB:
    resA[k] = diff("ランキング学習(並び全体)", k); print(f"  {lab}: {fmt(resA[k])}")
print(f"  参考 順位相関: {fmt(diff('ランキング学習(並び全体)', 'sp'), pt=False)}")
upA = any(v[1] > 0 for v in resA.values()); downA = any(v[2] < 0 for v in resA.values())
print(f"  判定A: {'合格' if upA and not downA else '不合格'}")
print("\n===== 問いB: 案2b(6着まで連鎖) − M0(1〜6着の並び) =====")
resB = diff("案2b(6着まで連鎖)", "sp")
print(f"  順位相関: {fmt(resB, pt=False)}")
for j in range(1, 7):
    print(f"  {j}着の的中: {fmt(diff('案2b(6着まで連鎖)', f'a{j}'))}")
print(f"  参考 案2b(3着まで連鎖)の順位相関: {fmt(diff('案2b(3着まで連鎖)', 'sp'), pt=False)}")
print(f"  判定B: {'合格' if resB[1] > 0 else '不合格'}")
print("\n===== 参考: 案2b(3着まで連鎖)− M0 の4つの的中率(7〜10月は既見) =====")
for k, lab in LAB:
    print(f"  {lab}: {fmt(diff('案2b(3着まで連鎖)', k))}")
print("\n===== 月別(M0との差) =====")
for m, g in R.groupby("month"):
    b = g[g["method"] == "M0 今の作り方"].set_index("rid")
    line = []
    for meth, cols in (("ランキング学習(並び全体)", ["tan", "q2", "q3", "t5"]), ("案2b(6着まで連鎖)", ["sp"])):
        a = g[g["method"] == meth].set_index("rid")
        line.append(meth[:6] + " " + " ".join(f"{c}{(a[c] - b.loc[a.index, c]).mean() * (100 if c != 'sp' else 1):+.2f}" for c in cols))
    print(f"  {m}: " + " | ".join(line))
