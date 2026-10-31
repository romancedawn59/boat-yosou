# -*- coding: utf-8 -*-
"""独自システム(案2b)と市場の予想がずれたレースの結果と払戻(2026-11-01ケンさん指示)

    py -X utf8 test/analyze_2b_market_divergence_1101.py

対象: 締切前3連単オッズがある 7/1〜9/14(主・各月の前月末までで学習した未見の予測)と 9/15〜10/31(参考・9/14までで学習)
ずれの見方:
  A. 1着の見立て … 案2bの1着推し(1着確率最大の艇)と市場の1番人気(3連単オッズから出した1着確率最大の艇)が違うレース
  B. 組ごと      … 案2bの3連単上位20組を「案2bの確率÷市場の確率」で分け、実際の的中率・回収率(1組100円)を見る
  C. レース全体  … 120通りの確率分布の違い(全変動距離)の5分位ごとに、案2bと市場それぞれの上位買いの回収率
  D. 実例        … 1着推しが違い、確率差の大きいレースの結果と払戻
記述的な調査で事前宣言ではない。ここで良く見えた区切りを買い方に使う場合は、11月以降の未使用データで事前宣言して採点する
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


import pickle

VN = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
      13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}
DIV_BLOCKS = [("2026-07-01", "2026-07-31", "主"), ("2026-08-01", "2026-08-31", "主"), ("2026-09-01", "2026-09-14", "主"), ("2026-09-15", "2026-10-31", "参考")]
CACHE = r"Y:\マイドライブ\boat\data_raw\div_2b_market_1101.pkl"
pay = defaultdict(dict)
conn = db.connect(DB_PATH)
for rid, bt, cb, a in conn.execute("SELECT race_id, bet_type, combination, amount_yen FROM payouts WHERE race_id>='20260701' AND bet_type IN ('3連単','3連複','単勝')"):
    pay[rid][(bt, cb)] = a or 0
conn.close()
A_, B_, C_ = PERM3.T
TRI_KEYS = [f"{a + 1}-{b + 1}-{c + 1}" for a, b, c in PERM3]

rows = []
for start, end, kind in DIV_BLOCKS:
    ids = [r for r in df.loc[(df["date"] >= start) & (df["date"] <= end), "race_id"].unique() if len(odds3.get(r, {})) >= 108 and r in pay]
    tr = df[df["date"] < start]
    te = df[df["race_id"].isin(ids)].sort_values(["race_id", "lane"]).reset_index(drop=True)
    rids = te["race_id"].values[::6]; n = len(rids)
    print(f"[{start}〜{end}] 学習〜{tr['date'].max()} / 締切前オッズあり {n:,}R 学習中...", flush=True)
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
    for i, rid in enumerate(rids):
        pf = p_full[i] / p_full[i].sum()
        p2 = P2[i] / P2[i].sum(axis=1, keepdims=True).clip(1e-12)
        p3 = P3[i] / P3[i].sum(axis=2, keepdims=True).clip(1e-12)
        t = pf[A_] * p2[A_, B_] * p3[A_, B_, C_]; t = t / t.sum()
        o = odds3[rid]
        q = np.array([1 / o[k] if k in o else 0.0 for k in TRI_KEYS]); q = q / q.sum(); q = np.where(q <= 0, 1e-6, q); q = q / q.sum()
        rows.append(dict(rid=rid, kind=kind, t=t, q=q, act=tuple(int(x) for x in acts[i]), odds=np.array([o.get(k, np.nan) for k in TRI_KEYS])))
    print("  完了", flush=True)
pickle.dump(rows, open(CACHE, "wb"))

# ================= 分析 =================
recs, combos = [], []
for x in rows:
    t, q, act, rid = x["t"], x["q"], x["act"], x["rid"]
    w1t = np.array([t[PERM3[:, 0] == a].sum() for a in range(6)]); w1q = np.array([q[PERM3[:, 0] == a].sum() for a in range(6)])
    bt, bq = int(w1t.argmax()), int(w1q.argmax())
    win_combo = f"{act[0] + 1}-{act[1] + 1}-{act[2] + 1}"
    tri_pay = pay[rid].get(("3連単", win_combo), 0)
    t_top = np.argsort(-t); q_top = np.argsort(-q)
    t5 = [TRI_KEYS[i] for i in t_top[:5]]; q5 = [TRI_KEYS[i] for i in q_top[:5]]
    recs.append(dict(rid=rid, kind=x["kind"], venue=VN[int(rid[9:11])], rno=int(rid[12:14]),
                     tv=0.5 * np.abs(t - q).sum(), bt=bt + 1, bq=bq + 1, pt_bt=w1t[bt], pq_bt=w1q[bt], pq_bq=w1q[bq], pt_bq=w1t[bq],
                     winner=act[0] + 1, result=win_combo, tri_pay=tri_pay,
                     tan_bt=pay[rid].get(("単勝", str(bt + 1)), 0), tan_bq=pay[rid].get(("単勝", str(bq + 1)), 0),
                     t1=t5[0], q1=q5[0], hit_t1=int(t5[0] == win_combo), hit_q1=int(q5[0] == win_combo),
                     hit_t5=int(win_combo in t5), hit_q5=int(win_combo in q5)))
    for rank, j in enumerate(t_top[:20], 1):
        combos.append(dict(kind=x["kind"], ratio=t[j] / q[j], pt=t[j], pq=q[j], odds=x["odds"][j], hit=int(TRI_KEYS[j] == win_combo),
                           pay=tri_pay if TRI_KEYS[j] == win_combo else 0, trank=rank))
R = pd.DataFrame(recs); Cm = pd.DataFrame(combos)
print(f"\n対象 {len(R):,}R(主=7/1〜9/14の未見 {(R['kind'] == '主').sum():,}R / 参考=9/15〜10/31 {(R['kind'] == '参考').sum():,}R)")

print("\n===== A. 1着の見立てのずれ: 案2bの1着推しと市場の1番人気が違うレース =====")
for kind in ("主", "参考", None):
    d = R if kind is None else R[R["kind"] == kind]
    dd = d[d["bt"] != d["bq"]]; same = d[d["bt"] == d["bq"]]
    lab = "全体" if kind is None else kind
    print(f"[{lab}] ずれたレース {len(dd):,}/{len(d):,}R({len(dd) / len(d):.1%})")
    print(f"   1着: 案2bの推し {(dd['winner'] == dd['bt']).mean():.1%} / 市場の1番人気 {(dd['winner'] == dd['bq']).mean():.1%} / その他 {((dd['winner'] != dd['bt']) & (dd['winner'] != dd['bq'])).mean():.1%}")
    print(f"   見立て: 案2bの推しの1着確率 案2b{dd['pt_bt'].mean():.1%}・市場{dd['pq_bt'].mean():.1%} / 市場の1番人気の1着確率 案2b{dd['pt_bq'].mean():.1%}・市場{dd['pq_bq'].mean():.1%}")
    print(f"   単勝100円: 案2bの推し 回収率{dd['tan_bt'].sum() / (len(dd) * 100):.1%}(的中{(dd['tan_bt'] > 0).sum()}) / 市場の1番人気 {dd['tan_bq'].sum() / (len(dd) * 100):.1%}(的中{(dd['tan_bq'] > 0).sum()})")
    print(f"   3連単1点: 案2bの1位 的中{dd['hit_t1'].sum()}本 回収率{(dd['hit_t1'] * dd['tri_pay']).sum() / (len(dd) * 100):.1%} / 市場の1位 的中{dd['hit_q1'].sum()}本 {(dd['hit_q1'] * dd['tri_pay']).sum() / (len(dd) * 100):.1%}")
    print(f"   3連単5点: 案2b 的中{dd['hit_t5'].sum()}本 回収率{(dd['hit_t5'] * dd['tri_pay']).sum() / (len(dd) * 500):.1%} / 市場 的中{dd['hit_q5'].sum()}本 {(dd['hit_q5'] * dd['tri_pay']).sum() / (len(dd) * 500):.1%}"
          f" | 参考 ずれなしのレースで案2b 3連単5点 {(same['hit_t5'] * same['tri_pay']).sum() / (len(same) * 500):.1%}")
d = R[R["bt"] != R["bq"]].copy()
d["差"] = pd.cut(d["pt_bt"] - d["pq_bt"], [-1, .05, .10, .15, .20, 1], labels=["〜5pt", "5〜10pt", "10〜15pt", "15〜20pt", "20pt〜"])
print("\n[案2bの推しの1着確率 − 市場の見る同じ艇の1着確率(ずれたレース・全期間)]")
t = d.groupby("差", observed=True).apply(lambda g: pd.Series({"R": len(g), "案2bの推しの1着率": (g["winner"] == g["bt"]).mean(), "案2bの見立て": g["pt_bt"].mean(),
                                                           "市場の見立て": g["pq_bt"].mean(), "単勝回収率": g["tan_bt"].sum() / (len(g) * 100)}))
print(t.to_string(formatters={"R": "{:.0f}".format, **{c: "{:.1%}".format for c in ("案2bの推しの1着率", "案2bの見立て", "市場の見立て", "単勝回収率")}}))

print("\n===== B. 組ごとのずれ: 案2bの上位20組を『案2bの確率÷市場の確率』で分ける(1組100円) =====")
Cm["倍率"] = pd.cut(Cm["ratio"], [0, .5, .8, 1.25, 2, 3, 99], labels=["0.5倍未満", "0.5〜0.8", "0.8〜1.25", "1.25〜2倍", "2〜3倍", "3倍以上"])
for kind in ("主", "参考"):
    c = Cm[Cm["kind"] == kind]
    t = c.groupby("倍率", observed=True).apply(lambda g: pd.Series({"組数": len(g), "的中": g["hit"].sum(), "実際の的中率": g["hit"].mean(), "案2bの見立て": g["pt"].mean(),
                                                                   "市場の見立て": g["pq"].mean(), "オッズ中央値": g["odds"].median(), "回収率": g["pay"].sum() / (len(g) * 100)}))
    print(f"[{kind}]")
    print(t.to_string(formatters={"組数": "{:,.0f}".format, "的中": "{:.0f}".format, "実際の的中率": "{:.2%}".format, "案2bの見立て": "{:.2%}".format,
                                  "市場の見立て": "{:.2%}".format, "オッズ中央値": "{:.0f}倍".format, "回収率": "{:.1%}".format}))

print("\n===== C. レース全体のずれ(120通りの確率分布の違い)の5分位 =====")
R["ずれ"] = pd.qcut(R["tv"], 5, labels=["小", "やや小", "中", "やや大", "大"])
t = R.groupby("ずれ", observed=True).apply(lambda g: pd.Series({
    "R": len(g), "分布の違い": g["tv"].mean(), "1着推しが違う": (g["bt"] != g["bq"]).mean(),
    "案2b 3単1点": (g["hit_t1"] * g["tri_pay"]).sum() / (len(g) * 100), "市場 3単1点": (g["hit_q1"] * g["tri_pay"]).sum() / (len(g) * 100),
    "案2b 3単5点": (g["hit_t5"] * g["tri_pay"]).sum() / (len(g) * 500), "市場 3単5点": (g["hit_q5"] * g["tri_pay"]).sum() / (len(g) * 500),
    "3単払戻の中央値": g["tri_pay"].median()}))
print(t.to_string(formatters={"R": "{:.0f}".format, "分布の違い": "{:.3f}".format, "1着推しが違う": "{:.0%}".format, "3単払戻の中央値": "{:,.0f}円".format,
                              **{c: "{:.1%}".format for c in ("案2b 3単1点", "市場 3単1点", "案2b 3単5点", "市場 3単5点")}}))

print("\n===== D. ずれの大きかったレースの実例(1着推しが違い、確率差が大きい順に20件) =====")
d = R[R["bt"] != R["bq"]].copy(); d["gap"] = d["pt_bt"] - d["pq_bt"]
for _, x in d.sort_values("gap", ascending=False).head(20).iterrows():
    mark = "◎案2b的中" if x["winner"] == x["bt"] else ("○市場的中" if x["winner"] == x["bq"] else "×両方外れ")
    tan = x["tan_bt"] if x["winner"] == x["bt"] else (x["tan_bq"] if x["winner"] == x["bq"] else 0)
    print(f"  {x['rid'][:8]} {x['venue']}{x['rno']}R: 案2b {x['bt']}号艇1着{x['pt_bt']:.0%}(市場{x['pq_bt']:.0%}) / 市場 {x['bq']}号艇{x['pq_bq']:.0%} → 結果{x['result']} {mark}"
          f" 単勝{tan}円 3単{x['tri_pay']:,}円 | 案2b 3単1位{x['t1']}{'的中' if x['hit_t1'] else ''}")
print("\n[参考: ずれたレースで案2bの推しが勝った中の高配当トップ10]")
w = d[d["winner"] == d["bt"]].sort_values("tri_pay", ascending=False).head(10)
for _, x in w.iterrows():
    print(f"  {x['rid'][:8]} {x['venue']}{x['rno']}R: 案2b {x['bt']}号艇({x['pt_bt']:.0%}・市場{x['pq_bt']:.0%}) 結果{x['result']} 単勝{x['tan_bt']}円 3単{x['tri_pay']:,}円 案2b上位5{'的中' if x['hit_t5'] else '外れ'}")
