# -*- coding: utf-8 -*-
"""事前宣言(2026-11-01・結果を見る前に固定し、実行前にコミットする)
案2b(条件付き連鎖)の着順確率は、締切前オッズの市場にどこまで迫るか

    py -X utf8 test/prereg_2b_vs_market_1101.py

■ 問い
  3連単の着順確率を、今の作り方 M0(1着確率 → Harville展開)から 案2b(1着確率 → 1着艇を条件に2着 → 1・2着艇を条件に3着)に
  替えると、市場との精度の差はどれだけ縮むか。市場にモデルを混ぜる価値は増えるか。

■ データ
  主検証(未見) : 2026-07-01〜09-14 の締切前3連単オッズがあるレース。月ごとのwalk-forward
                 (7月は6/30まで、8月は7/31まで、9/1〜9/14は8/31までで学習)。
                 この期間で案2bと市場を比べたことは一度もない
  参考(既見)   : 2026-09-15〜10-31。11/1に797Rで 案2b 3.791 / 市場 3.767 / M0 3.863 を見た。判定には使わない
  前向き確認   : 2026-11-01〜11-30(10/31までで学習)。12/1以降に同じスクリプトを再実行して見る
■ 物差し
  3連単の着順確率のlog-loss = 1着 + 1着を条件に2着 + 1・2着を条件に3着。小さいほど正確。
  比較はレースごとの対の差、95%区間はブートストラップ2,000回
  市場確率 = 締切前3連単オッズの 1/オッズ を120通りで正規化(108通り以上そろうレースのみ)
■ 合格(採用候補)の条件: 主検証で次の3つをすべて満たす
  (i)   案2bのlog-lossがM0より小さく、差の95%区間が0を含まない
  (ii)  案2bと市場の差が、M0と市場の差の半分以下
  (iii) 混合 r = λ×モデル + (1−λ)×市場 で、λ(0〜1を0.1刻み)を7月で選び、8/1〜9/14で採点する。
        案2bの混合が市場単独より良く(95%区間が0を含まない)、かつその改善がM0の混合の改善より大きい
  付記: 案2bが市場単独より良い(市場−案2bの95%区間が0より上)なら「市場超え」と記録する(合格条件ではない)
  不合格なら案2bの着順確率は採用しない。前向き確認(11月)は合格した場合のみ昇格判断に使う。
■ 案2bの作り方(11/1の sim_tenkai_position_models_1101.py と同一)
  1着確率 = 本番の特徴量33本+展開の特徴量8本の1着モデル
  2着     = 1着艇を条件にした2着モデル(条件艇の枠・KR・勝率と、候補艇との差8項目)
  3着     = 1・2着艇を条件にした3着モデル(同上を2艇分)
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


rows = []   # レースごとの記録(区分・M0/案2b/市場の着順確率)
for start, end, kind in BLOCKS:
    test_ids = [r for r in df.loc[(df["date"] >= start) & (df["date"] <= end), "race_id"].unique() if len(odds3.get(r, {})) >= 108]
    if not test_ids:
        print(f"[{start}〜{end} {kind}] 締切前オッズのあるレースなし → 省略", flush=True)
        continue
    tr = df[df["date"] < start]
    te = df[df["race_id"].isin(test_ids)].sort_values(["race_id", "lane"]).reset_index(drop=True)
    rids = te["race_id"].values[::6]; n = len(rids)
    print(f"[{start}〜{end} {kind}] 学習〜{tr['date'].max()}({tr['race_id'].nunique():,}R) / 採点 {n:,}R 学習中...", flush=True)
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
    base_c = te[["race_id", "lane"] + DIFF]
    f2 = add_cond(te, base_c, "w_"); f2 = f2[f2["lane"] != f2["w_lane"]]
    for rid, a, b, v in zip(f2["race_id"], f2["w_lane"], f2["lane"], m2.predict(f2[C2])):
        P2[ridx[rid], a - 1, b - 1] = v
    f3 = add_cond(add_cond(te, base_c, "w_"), base_c, "s_")
    f3 = f3[(f3["lane"] != f3["w_lane"]) & (f3["lane"] != f3["s_lane"]) & (f3["w_lane"] != f3["s_lane"])]
    for rid, a, b, c, v in zip(f3["race_id"], f3["w_lane"], f3["s_lane"], f3["lane"], m3.predict(f3[C3])):
        P3[ridx[rid], a - 1, b - 1, c - 1] = v
    acts = np.argsort(te["pos"].values.reshape(n, 6), axis=1)
    A_, B_, C_ = PERM3.T
    for i, rid in enumerate(rids):
        pm0 = p_m0[i] / p_m0[i].sum(); pf = p_full[i] / p_full[i].sum()
        p2 = P2[i] / P2[i].sum(axis=1, keepdims=True).clip(1e-12)
        p3 = P3[i] / P3[i].sum(axis=2, keepdims=True).clip(1e-12)
        t2b = pf[A_] * p2[A_, B_] * p3[A_, B_, C_]; t2b = t2b / t2b.sum()
        o = odds3[rid]
        q = np.array([1 / o[f"{a + 1}-{b + 1}-{c + 1}"] if f"{a + 1}-{b + 1}-{c + 1}" in o else 0.0 for a, b, c in PERM3])
        q = q / q.sum(); q = np.where(q <= 0, 1e-6, q); q = q / q.sum()
        rows.append(dict(rid=rid, start=start, kind=kind, act=tuple(acts[i][:3]), m0=tri_harville(pm0), b2=t2b, mkt=q))
    print(f"  完了 {n:,}R", flush=True)

R = pd.DataFrame(rows)
for col in ("m0", "b2", "mkt"):
    L = np.vstack([ll3(t, a) for t, a in zip(R[col], R["act"])])
    R[f"{col}_1"], R[f"{col}_2"], R[f"{col}_3"] = L[:, 0], L[:, 1], L[:, 2]
    R[f"{col}_ll"] = L.sum(1)


def boot(v):
    v = np.asarray(v); idx = rng.integers(0, len(v), size=(2000, len(v))); mm = v[idx].mean(1)
    return v.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


def ci(v):
    mu, lo, hi = boot(v)
    return f"{mu:+.4f}[{lo:+.4f},{hi:+.4f}]" + ("◎" if lo > 0 else ("▲" if hi < 0 else ""))


print("\n===== 区分ごとのlog-loss(小さいほど正確)。1着 / 2着|1着 / 3着|1・2着 / 合計 =====")
tab = []
for (start, kind), d in R.groupby(["start", "kind"], sort=False):
    for col, lab in (("m0", "M0 今の作り方"), ("b2", "案2b 条件付き連鎖"), ("mkt", "市場 締切前オッズ")):
        tab.append(dict(区分=f"{start}〜 {kind}", R=len(d), 方式=lab, 一着=d[f"{col}_1"].mean(), 二着=d[f"{col}_2"].mean(), 三着=d[f"{col}_3"].mean(), 合計=d[f"{col}_ll"].mean()))
print(pd.DataFrame(tab).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

M = R[R["kind"] == "主検証"]
print(f"\n===== 主検証(7/1〜9/14・{len(M):,}R)の対の差(正なら左が悪い=右が良い) =====")
print(f"  M0 − 案2b        : {ci(M['m0_ll'] - M['b2_ll'])}   ← (i) 0より上なら案2bがM0より正確")
print(f"  M0 − 市場        : {ci(M['m0_ll'] - M['mkt_ll'])}   (M0と市場の差)")
print(f"  案2b − 市場      : {ci(M['b2_ll'] - M['mkt_ll'])}   (案2bと市場の差)")
print(f"  市場 − 案2b      : {ci(M['mkt_ll'] - M['b2_ll'])}   ← 付記: 0より上なら市場超え")
gap0 = (M["m0_ll"] - M["mkt_ll"]).mean(); gap2 = (M["b2_ll"] - M["mkt_ll"]).mean()
print(f"  市場との差の縮み: M0 {gap0:+.4f} → 案2b {gap2:+.4f}(縮んだ割合 {1 - gap2 / gap0:.0%})  ← (ii) 50%以上で合格")
for i, lab in ((1, "1着"), (2, "2着|1着"), (3, "3着|1・2着")):
    print(f"    {lab}: M0−市場 {(M[f'm0_{i}'] - M[f'mkt_{i}']).mean():+.4f} / 案2b−市場 {(M[f'b2_{i}'] - M[f'mkt_{i}']).mean():+.4f}")
print("  月別 M0−案2b / 案2b−市場: " + " | ".join(f"{s[5:7]}月{'前半' if s.endswith('09-01') else ''} {(d['m0_ll'] - d['b2_ll']).mean():+.4f} / {(d['b2_ll'] - d['mkt_ll']).mean():+.4f}"
                                            for s, d in M.groupby("start")))

print("\n===== (iii) 混合 r = λ×モデル + (1−λ)×市場(λは7月で選び、8/1〜9/14で採点) =====")
J = M[M["start"] == "2026-07-01"]; E = M[M["start"] != "2026-07-01"]


def blend_ll(d, col, lam):
    return np.array([ll3(lam * t + (1 - lam) * q, a).sum() for t, q, a in zip(d[col], d["mkt"], d["act"])])


res3 = {}
for col, lab in (("m0", "M0"), ("b2", "案2b")):
    curve = {lam: blend_ll(J, col, lam).mean() for lam in LAMS}
    lam = min(curve, key=curve.get)
    imp = E["mkt_ll"].values - blend_ll(E, col, lam)
    mu, lo, hi = boot(imp); res3[col] = (mu, lo)
    print(f"  {lab}: 7月で選んだλ={lam}(7月の改善 {curve[0.0] - curve[lam]:+.4f}) → 8/1〜9/14({len(E):,}R)で市場単独より {ci(imp)}")

ok1 = boot(M["m0_ll"] - M["b2_ll"])[1] > 0
ok2 = gap2 <= gap0 / 2
ok3 = res3["b2"][1] > 0 and res3["b2"][0] > res3["m0"][0]
beat = boot(M["mkt_ll"] - M["b2_ll"])[1] > 0
print(f"\n===== 判定 ===== (i){'○' if ok1 else '×'} (ii){'○' if ok2 else '×'} (iii){'○' if ok3 else '×'} → {'採用候補(11月の前向き確認へ)' if ok1 and ok2 and ok3 else '不合格(案2bの着順確率は採用しない)'}"
      f" / 付記 市場超え: {'あり' if beat else 'なし'}")
for kind in ("参考(既見)", "前向き確認"):
    d = R[R["kind"] == kind]
    if len(d):
        print(f"[{kind} {len(d):,}R] M0−案2b {ci(d['m0_ll'] - d['b2_ll'])} / 案2b−市場 {ci(d['b2_ll'] - d['mkt_ll'])}")
