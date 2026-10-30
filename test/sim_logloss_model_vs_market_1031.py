# -*- coding: utf-8 -*-
"""市場確率 vs モデル確率の精度比較(2026-10-31ケンさん承認「着手して」)

    py -X utf8 test/sim_logloss_model_vs_market_1031.py

目的: 回収率ではなく確率の精度(log-loss)で「モデルが市場に勝っている升目」を探す。
      勝っている升目が無ければ、どんな券種・点数でも勝てない。勝っている着(1着/2着/3着)が券種を決める。

定義:
  モデル確率 p = 本番配信picks JSONの6艇1着確率 → trifecta_probs で3連単120通り(正規化)
  市場確率 q = 3連単オッズ(締切前の最終取得・final-backfill以外)の 1/オッズ を正規化
             (感度として確定オッズ odds_final でも計算)
  着ごとの確率 = 1着 P(a*)、1着固定の2着 P(b*|a*)、1・2着固定の3着 P(c*|a*,b*)。a*,b*,c*は実際の着順
  log-loss差 = 市場のlog-loss − モデルのlog-loss(正ならモデルが良い)。レースごとの対にした差の平均と、ブートストラップ95%区間
  混合検定 = r = λp + (1−λ)q(120通りで混合)。前半(7/7〜9/14)で升目ごとにλを選び、後半(9/15〜)で市場だけ(λ=0)より良くなるか
升目 = 1位勝率帯(<15/15〜22.5/22.5〜30/30〜40/40%〜) × 2着の絞り(2位と3位の1着確率差>=0.08)。補助に1号艇/外艇
合格(結果を見る前に固定): 後半で、前半で選んだλ>0 かつ log-loss改善の95%区間が0を含まない升目×着。
"""
import glob
import json
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
from predictors import trifecta_probs  # noqa: E402

SPLIT = "20260915"
BANDS = [("<15%", 0, .15), ("15〜22.5%", .15, .225), ("22.5〜30%", .225, .30), ("30〜40%", .30, .40), ("40%〜", .40, 1.01)]
LAMBDAS = np.round(np.arange(0, 1.01, 0.1), 1)
rng = np.random.default_rng(31)

print("データ準備中...", flush=True)
conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
odds15, oddsF = defaultdict(dict), defaultdict(dict)
for rid, cb, o in conn.execute("SELECT race_id, combination, odds FROM odds WHERE bet_type='3連単' AND odds>0 AND fetched_at!='final-backfill'"):
    odds15[rid][cb] = o
for rid, cb, o in conn.execute("SELECT race_id, combination, odds FROM odds_final WHERE bet_type='3連単' AND odds>0"):
    oddsF[rid][cb] = o
win = {}
for rid, cb in conn.execute("SELECT race_id, combination FROM payouts WHERE bet_type='3連単' AND race_id>='20260707'"):
    win.setdefault(rid, []).append(cb)
conn.close()
win = {r: v[0] for r, v in win.items() if len(v) == 1}


def band_of(p1):
    for name, lo, hi in BANDS:
        if lo <= p1 < hi:
            return name


def norm_from_odds(o, keys):
    if len(o) < len(keys) * 0.9:
        return None
    q = np.array([1 / o[k] if k in o and o[k] > 0 else 0.0 for k in keys])
    s = q.sum()
    if s <= 0:
        return None
    q = q / s
    q[q == 0] = 1e-6  # 売れていない目は極小
    return q / q.sum()


def components(prob, keys_arr, a, b, c):
    """120通り確率ベクトル → (P(a), P(b|a), P(c|a,b))"""
    m1 = keys_arr[:, 0] == a
    m2 = m1 & (keys_arr[:, 1] == b)
    m3 = m2 & (keys_arr[:, 2] == c)
    pa = prob[m1].sum(); pab = prob[m2].sum(); pabc = prob[m3].sum()
    return pa, pab / pa if pa > 0 else 1e-6, pabc / pab if pab > 0 else 1e-6


races = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid = r["race_id"]
        if not r.get("ranked") or len(r["ranked"]) < 6 or rid not in win or rid not in odds15:
            continue
        probs = {int(l): float(p) for l, p in r["ranked"]}
        tri = trifecta_probs(probs)
        keys = sorted(tri)
        keys_s = ["-".join(map(str, k)) for k in keys]
        p = np.array([tri[k] for k in keys]); p = p / p.sum()
        q = norm_from_odds(odds15[rid], keys_s)
        if q is None:
            continue
        qf = norm_from_odds(oddsF.get(rid, {}), keys_s)
        a, b, c = map(int, win[rid].split("-"))
        rk = r["ranked"]
        races.append(dict(rid=rid, period=("後半" if rid >= SPLIT else "前半"), band=band_of(rk[0][1]),
                          nikaku=("2着絞れる" if rk[1][1] - rk[2][1] >= .08 else "2着流動"), in1=("1号艇" if rk[0][0] == 1 else "外艇"),
                          p=p, q=q, qf=qf, keys=np.array(keys), abc=(a, b, c), p1=rk[0][1], r1=rk[0][0]))
print(f"対象 {len(races):,}R(本番確率×締切前オッズ×結果がそろうもの) 前半{sum(x['period'] == '前半' for x in races):,} 後半{sum(x['period'] == '後半' for x in races):,}"
      f" / 確定オッズもあるもの {sum(x['qf'] is not None for x in races):,}R")

# レースごとの log-loss(モデル・市場・確定市場) 着別
for x in races:
    a, b, c = x["abc"]
    for name, vec in (("m", x["p"]), ("q", x["q"]), ("f", x["qf"])):
        if vec is None:
            x[f"ll_{name}"] = None
            continue
        c1, c2, c3 = components(vec, x["keys"], a, b, c)
        x[f"ll_{name}"] = (-np.log(max(c1, 1e-6)), -np.log(max(c2, 1e-6)), -np.log(max(c3, 1e-6)))
df = pd.DataFrame([dict(rid=x["rid"], period=x["period"], band=x["band"], nikaku=x["nikaku"], in1=x["in1"], p1=x["p1"],
                        **{f"d{i + 1}": x["ll_q"][i] - x["ll_m"][i] for i in range(3)},
                        **{f"m{i + 1}": x["ll_m"][i] for i in range(3)}, **{f"q{i + 1}": x["ll_q"][i] for i in range(3)},
                        **({f"f{i + 1}": x["ll_f"][i] for i in range(3)} if x["ll_f"] else {}))
                   for x in races])
df["dall"] = df["d1"] + df["d2"] + df["d3"]
df["type"] = df["band"] + "×" + df["nikaku"]


def ci(v, n_boot=2000):
    v = np.asarray(v)
    if len(v) < 5:
        return np.nan, np.nan
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt(v):
    lo, hi = ci(v)
    mark = "◎" if lo > 0 else ("▲" if hi < 0 else "　")
    return f"{np.mean(v):+.4f}[{lo:+.3f},{hi:+.3f}]{mark}"


# ===== ① 全体・着別 =====
print("\n===== ① log-loss(小さいほど良い) 全体・着別・期間別。差=市場−モデル(正ならモデルが良い)、[ ]は95%区間、◎=モデル有意に良い ▲=市場有意に良い =====")
for period in ("前半", "後半", "全期間"):
    d = df if period == "全期間" else df[df["period"] == period]
    print(f"\n[{period} {len(d):,}R]  {'':6} {'モデル':>8} {'市場15分前':>8}  差")
    for i, lab in ((1, "1着"), (2, "2着|1着"), (3, "3着|1,2着")):
        print(f"  {lab:8} {d[f'm{i}'].mean():8.4f} {d[f'q{i}'].mean():8.4f}  {fmt(d[f'd{i}'])}")
    print(f"  {'3連単合計':8} {(d['m1'] + d['m2'] + d['m3']).mean():8.4f} {(d['q1'] + d['q2'] + d['q3']).mean():8.4f}  {fmt(d['dall'])}")
    if "f1" in d and d["f1"].notna().sum() > 100:
        dd = d[d["f1"].notna()]
        print(f"  (確定オッズ市場 {len(dd):,}R: 1着{dd['f1'].mean():.4f} 2着{dd['f2'].mean():.4f} 3着{dd['f3'].mean():.4f} / 同じRのモデル 1着{dd['m1'].mean():.4f} 2着{dd['m2'].mean():.4f} 3着{dd['m3'].mean():.4f} / 15分前市場 1着{dd['q1'].mean():.4f})")

# ===== ② 升目別 =====
print("\n===== ② 升目(1位勝率帯×2着の絞り)別の差=市場−モデル(後半)。前半は参考 =====")
rows = []
for ty, d in df.groupby("type"):
    for period in ("前半", "後半"):
        dd = d[d["period"] == period]
        if len(dd) < 20:
            continue
        rows.append(dict(升目=ty, 期間=period, R=len(dd), 差1着=fmt(dd["d1"]), 差2着=fmt(dd["d2"]), 差3着=fmt(dd["d3"]), 差合計=fmt(dd["dall"])))
print(pd.DataFrame(rows).to_string(index=False))
print("\n[補助] 1号艇/外艇 × 帯(後半)")
rows = []
for (b_, i_), d in df[df["period"] == "後半"].groupby(["band", "in1"]):
    if len(d) >= 20:
        rows.append(dict(帯=b_, 艇=i_, R=len(d), 差1着=fmt(d["d1"]), 差2着=fmt(d["d2"]), 差3着=fmt(d["d3"]), 差合計=fmt(d["dall"])))
print(pd.DataFrame(rows).to_string(index=False))

# ===== ③ 混合検定 =====
print("\n===== ③ 混合検定 r=λp+(1−λ)q: 前半で升目ごとにλを選び、後半で市場だけ(λ=0)との差を採点(正ならモデルを混ぜる価値あり) =====")


def ll_blend(xs, lam):
    out = np.zeros((len(xs), 3))
    for j, x in enumerate(xs):
        r = lam * x["p"] + (1 - lam) * x["q"]
        a, b, c = x["abc"]
        c1, c2, c3 = components(r, x["keys"], a, b, c)
        out[j] = (-np.log(max(c1, 1e-6)), -np.log(max(c2, 1e-6)), -np.log(max(c3, 1e-6)))
    return out


rows = []
groups = [("全体", races)] + [(ty, [x for x in races if x["band"] + "×" + x["nikaku"] == ty]) for ty in sorted(df["type"].unique())]
for name, xs in groups:
    A = [x for x in xs if x["period"] == "前半"]; B = [x for x in xs if x["period"] == "後半"]
    if len(A) < 50 or len(B) < 30:
        continue
    totA = {lam: ll_blend(A, lam).sum(axis=1).mean() for lam in LAMBDAS}
    lam_star = min(totA, key=totA.get)
    base = ll_blend(B, 0.0); mix = ll_blend(B, lam_star)
    d = base - mix  # 正ならモデル混合が良い
    ok = lam_star > 0 and ci(d.sum(axis=1))[0] > 0
    rows.append(dict(升目=name, 前半R=len(A), 前半λ=lam_star, 前半改善=totA[0.0] - totA[lam_star], 後半R=len(B),
                     後半改善合計=fmt(d.sum(axis=1)), 後半1着=fmt(d[:, 0]), 後半2着=fmt(d[:, 1]), 後半3着=fmt(d[:, 2]), 合格=("○" if ok else "×")))
print(pd.DataFrame(rows).to_string(index=False, formatters={"前半改善": "{:+.4f}".format}))

# ===== ④ 乖離の情報価値 =====
print("\n===== ④ モデル1位(r1)の1着確率: モデル−市場の差で分け、実際の1着率と比べる(全期間) =====")
rows = []
for x in races:
    a = x["r1"]; m1 = x["keys"][:, 0] == a
    rows.append(dict(period=x["period"], band=x["band"], pm=x["p"][m1].sum(), pq=x["q"][m1].sum(), hit=int(x["abc"][0] == a)))
g = pd.DataFrame(rows); g["diff"] = g["pm"] - g["pq"]
g["bin"] = pd.cut(g["diff"], [-1, -.10, -.05, -.02, .02, .05, .10, 1], labels=["<-10pt", "-10〜-5", "-5〜-2", "±2", "+2〜5", "+5〜10", ">+10pt"])
for period in ("前半", "後半"):
    t = g[g["period"] == period].groupby("bin", observed=True).agg(R=("hit", "size"), モデル=("pm", "mean"), 市場=("pq", "mean"), 実際=("hit", "mean"))
    print(f"\n[{period}]"); print(t.to_string(formatters={"モデル": "{:.1%}".format, "市場": "{:.1%}".format, "実際": "{:.1%}".format}))
print("\n[帯22.5〜30%だけ・全期間]")
t = g[g["band"] == "22.5〜30%"].groupby("bin", observed=True).agg(R=("hit", "size"), モデル=("pm", "mean"), 市場=("pq", "mean"), 実際=("hit", "mean"))
print(t.to_string(formatters={"モデル": "{:.1%}".format, "市場": "{:.1%}".format, "実際": "{:.1%}".format}))
