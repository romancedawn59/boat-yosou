# -*- coding: utf-8 -*-
"""目的をもって点数を絞る仕組みの探索(2026-10-28ケンさん依頼「全点買いは無し。目的をもって点数を絞る仕組みを探索」)

    py -X utf8 test/sim_point_selection_1028.py

材料(すべて買う前に見えるものだけ):
  確率  = 本番配信picks JSON(docs/data/picks_*.json)の6艇の1着確率 → predictors.trifecta_probs で120通り
  オッズ = odds(締切15分前・final-backfillを除く)の3連単/3連複
  払戻  = payouts(確定)。1点100円均等。
目的ごとの選び方(各K=1,2,3,5点):
  当てに行く   モデル確率の上位K
  期待値      モデル確率×オッズ(EV)の上位K / EV≥1.0・1.2のときだけ
  乖離        モデル確率÷市場想定確率の上位K(確率の下限0.5%・1%・2%)
  安い目を捨てる モデル確率の上位K、ただしオッズ10/20/30倍以上の目から
  r1負けの保険 頭がr2・r3の目でモデル確率の上位K
  消去してから  頭が1着確率ボトム2艇でない目から乖離の上位K
  中穴        市場人気11〜40位の目からモデル確率の上位K
  レース選別   市場1番人気≠モデル1位の目のレースだけ、モデル上位K / 一致するレースだけ
  3連複       モデル確率上位K / EV上位K
帯 = 1位勝率(全帯・〜22.5%・22.5〜30%・30〜50%・50%〜)。
合格(事前固定): 的中30本以上・回収率>100%・最大1本除いても>100%・黒字月が月数の過半・前半/後半の両方で>100%。
出力: data_raw/point_selection_1028.csv
"""
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
from predictors import trifecta_probs

OUT = r"Y:\マイドライブ\boat\data_raw\point_selection_1028.csv"
STAKE, MIN_HITS = 100, 30

print("データ準備中...", flush=True)
conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
odds3, oddsT = defaultdict(dict), defaultdict(dict)
for rid, bt, cb, o in conn.execute("SELECT race_id, bet_type, combination, odds FROM odds "
                                   "WHERE odds IS NOT NULL AND odds > 0 AND fetched_at != 'final-backfill'"):
    (odds3 if bt == "3連単" else oddsT)[rid][cb] = o
pay = defaultdict(dict)
for rid, bt, cb, a in conn.execute("SELECT race_id, bet_type, combination, amount_yen FROM payouts WHERE bet_type IN ('3連単','3連複')"):
    pay[rid][(bt, cb)] = a or 0
conn.close()

races = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid = r["race_id"]
        if not r.get("ranked") or len(r["ranked"]) < 6 or len(odds3.get(rid, {})) < 110 or rid not in pay:
            continue
        probs = {int(l): float(p) for l, p in r["ranked"]}
        tri = trifecta_probs(probs)
        z = sum(tri.values())
        tri = {"-".join(map(str, k)): v / z for k, v in tri.items()}
        o3 = odds3[rid]
        q = {cb: 1 / o for cb, o in o3.items()}
        zq = sum(q.values())
        q = {cb: v / zq for cb, v in q.items()}
        rank_mkt = {cb: i + 1 for i, cb in enumerate(sorted(o3, key=o3.get))}
        trio = defaultdict(float)
        for cb, v in tri.items():
            trio["=".join(sorted(cb.split("-")))] += v
        oT = oddsT.get(rid, {})
        ranked = [int(l) for l, _ in r["ranked"]]
        bottom2 = set(ranked[-2:])
        races.append(dict(rid=rid, month=rid[:6], p1=float(r["ranked"][0][1]), r=ranked, tri=tri, o3=o3, q=q,
                          rank_mkt=rank_mkt, trio=dict(trio), oT=oT, bottom2=bottom2, pay=pay[rid]))
print(f"対象レース {len(races):,}R(本番確率×15分前オッズ×払戻がそろうもの) 月別: "
      + " ".join(f"{m}:{n}" for m, n in sorted(pd.Series([x['month'] for x in races]).value_counts().items())), flush=True)


# ---------- 選び方 ----------
def top(d, k, key=None, cand=None):
    cand = list(d if cand is None else cand)
    return sorted(cand, key=(key or d.get), reverse=True)[:k]


def sel_model(x, k): return [("3連単", c) for c in top(x["tri"], k)]
def sel_ev(x, k, th=0.0):
    ev = {c: x["tri"][c] * x["o3"][c] for c in x["tri"] if c in x["o3"]}
    cand = [c for c in ev if ev[c] >= th]
    return [("3連単", c) for c in top(ev, k, cand=cand)]
def sel_ratio(x, k, floor):
    rt = {c: x["tri"][c] / x["q"][c] for c in x["tri"] if c in x["q"] and x["tri"][c] >= floor}
    return [("3連単", c) for c in top(rt, k)]
def sel_model_minodds(x, k, mo):
    cand = [c for c in x["tri"] if x["o3"].get(c, 0) >= mo]
    return [("3連単", c) for c in top(x["tri"], k, cand=cand)]
def sel_cover(x, k):
    heads = {str(x["r"][1]), str(x["r"][2])}
    cand = [c for c in x["tri"] if c.split("-")[0] in heads]
    return [("3連単", c) for c in top(x["tri"], k, cand=cand)]
def sel_elim_ratio(x, k, floor=0.01):
    rt = {c: x["tri"][c] / x["q"][c] for c in x["tri"]
          if c in x["q"] and x["tri"][c] >= floor and int(c.split("-")[0]) not in x["bottom2"]}
    return [("3連単", c) for c in top(rt, k)]
def sel_mid(x, k, lo=11, hi=40):
    cand = [c for c in x["tri"] if lo <= x["rank_mkt"].get(c, 999) <= hi]
    return [("3連単", c) for c in top(x["tri"], k, cand=cand)]
def sel_race_disagree(x, k, agree=False):
    fav = min(x["o3"], key=x["o3"].get)
    same = fav == top(x["tri"], 1)[0]
    return sel_model(x, k) if same == agree else []
def sel_trio(x, k): return [("3連複", c) for c in top(x["trio"], k)]
def sel_trio_ev(x, k):
    ev = {c: x["trio"][c] * x["oT"][c] for c in x["trio"] if c in x["oT"]}
    return [("3連複", c) for c in top(ev, k)]


SELECTORS = {
    "当てに行く: モデル上位": sel_model,
    "期待値: EV上位": sel_ev,
    "期待値: EV≥1.0の上位": lambda x, k: sel_ev(x, k, 1.0),
    "期待値: EV≥1.2の上位": lambda x, k: sel_ev(x, k, 1.2),
    "乖離: 比上位(p≥0.5%)": lambda x, k: sel_ratio(x, k, .005),
    "乖離: 比上位(p≥1%)": lambda x, k: sel_ratio(x, k, .01),
    "乖離: 比上位(p≥2%)": lambda x, k: sel_ratio(x, k, .02),
    "安い目を捨てる: 10倍以上": lambda x, k: sel_model_minodds(x, k, 10),
    "安い目を捨てる: 20倍以上": lambda x, k: sel_model_minodds(x, k, 20),
    "安い目を捨てる: 30倍以上": lambda x, k: sel_model_minodds(x, k, 30),
    "r1負けの保険: 頭r2/r3": sel_cover,
    "消去してから: 比上位(p≥1%)": sel_elim_ratio,
    "中穴: 人気11〜40位からモデル上位": sel_mid,
    "中穴: 人気20〜40位からモデル上位": lambda x, k: sel_mid(x, k, 20, 40),
    "レース選別: 市場1番人気≠モデル1位": sel_race_disagree,
    "レース選別: 市場1番人気=モデル1位": lambda x, k: sel_race_disagree(x, k, True),
    "3連複: モデル上位": sel_trio,
    "3連複: EV上位": sel_trio_ev,
}
KS = (1, 2, 3, 5)
BANDS = {"全帯": (0, 1.01), "〜22.5%": (0, .225), "22.5〜30%": (.225, .30), "30〜50%": (.30, .50), "50%〜": (.50, 1.01)}
mid_date = sorted(x["rid"][:8] for x in races)[len(races) // 2]

rows, detail = [], {}
for name, fn in SELECTORS.items():
    for k in KS:
        cost = np.zeros(len(races)); ret = np.zeros(len(races)); hit = np.zeros(len(races), dtype=bool)
        for i, x in enumerate(races):
            for bt, cb in fn(x, k):
                cost[i] += STAKE
                a = x["pay"].get((bt, cb), 0)
                if a:
                    ret[i] += a * STAKE / 100
                    hit[i] = True
        detail[(name, k)] = (cost, ret, hit)
        p1 = np.array([x["p1"] for x in races]); mon = np.array([x["month"] for x in races])
        half = np.array([x["rid"][:8] < mid_date for x in races])
        for band, (lo, hi) in BANDS.items():
            m = (p1 >= lo) & (p1 < hi) & (cost > 0)
            if m.sum() == 0:
                continue
            inv = cost[m].sum()
            mdf = pd.DataFrame({"m": mon[m], "c": cost[m], "r": ret[m]}).groupby("m").sum()
            mdf = mdf[mdf["c"] >= 100 * 50]
            h1 = m & half; h2 = m & ~half
            rows.append(dict(目的=name, K=k, 帯=band, R=int(m.sum()), 点数=round(cost[m].mean() / STAKE, 1), 的中R=int(hit[m].sum()),
                             的中率=hit[m].mean(), 回収率=ret[m].sum() / inv, 最大1本除く=(ret[m].sum() - ret[m].max()) / inv,
                             黒字月=int((mdf["r"] > mdf["c"]).sum()), 月数=len(mdf),
                             前半=ret[h1].sum() / max(cost[h1].sum(), 1), 後半=ret[h2].sum() / max(cost[h2].sum(), 1),
                             最大払戻=ret[m].max()))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
df["合格"] = (df["的中R"] >= MIN_HITS) & (df["回収率"] > 1) & (df["最大1本除く"] > 1) & (df["黒字月"] * 2 > df["月数"]) & (df["前半"] > 1) & (df["後半"] > 1)

pd.set_option("display.width", 250, "display.max_rows", 1000, "display.unicode.east_asian_width", True)
pct = "{:.1%}".format
fmt = {"的中率": pct, "回収率": pct, "最大1本除く": pct, "前半": pct, "後半": pct, "最大払戻": "{:,.0f}".format}
print(f"\n前半/後半の境目: {mid_date}")
print("\n===== 全帯: 目的×K(1点100円均等) =====")
print(df[df["帯"] == "全帯"].drop(columns=["帯", "合格"]).to_string(index=False, formatters=fmt))
print("\n===== 帯22.5〜30%: 目的×K =====")
print(df[df["帯"] == "22.5〜30%"].drop(columns=["帯", "合格"]).to_string(index=False, formatters=fmt))
print("\n===== 全マスの回収率上位25(的中30本以上) =====")
print(df[df["的中R"] >= MIN_HITS].sort_values("回収率", ascending=False).head(25).drop(columns=["合格"]).to_string(index=False, formatters=fmt))
print(f"\n===== 合格(的中{MIN_HITS}本以上・回収率>100%・最大1本除く>100%・黒字月が過半・前半後半とも>100%) =====")
print(f"マス総数 {len(df):,}")
ok = df[df["合格"]].copy()
if ok.empty:
    print("合格なし")
else:
    # 帯のシャッフル検定: 同じ選び方で、帯の所属だけを崩しても同じ回収率以上が出る確率(2,000回)
    rng = np.random.default_rng(7)
    p1_all = np.array([x["p1"] for x in races])
    pv = []
    for _, row in ok.iterrows():
        cost, ret, _ = detail[(row["目的"], row["K"])]
        lo, hi = BANDS[row["帯"]]
        m = (p1_all >= lo) & (p1_all < hi) & (cost > 0)
        pool = np.where(cost > 0)[0]
        real = ret[m].sum() / cost[m].sum()
        k = int(m.sum())
        sims = 0
        for _ in range(2000):
            idx = rng.choice(pool, k, replace=False)
            sims += ret[idx].sum() / cost[idx].sum() >= real
        pv.append(sims / 2000)
    ok["シャッフルp"] = pv
    print(ok.drop(columns=["合格"]).to_string(index=False, formatters={**fmt, "シャッフルp": "{:.3f}".format}))
    print(f"(合格 {len(ok)}マス / 総数 {len(df)}マス。5%水準なら偶然で約{len(df) * .05:.0f}マスが100%を超えうる)")

# ---------- 補助: モデルの3連単確率は当たっているか(確率の帯ごとの実際の的中率と市場) ----------
print("\n===== 補助: 3連単の目をモデル確率の帯で束ねたときの 実際の的中率 / モデル想定 / 市場想定 / 全部買った回収率 =====")
edges = [0, .002, .005, .01, .02, .03, .05, .08, .12, 1]
agg = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0])
for x in races:
    win = next((cb for (bt, cb) in x["pay"] if bt == "3連単"), None)
    for cb, p in x["tri"].items():
        if cb not in x["q"]:
            continue
        b = next(j for j in range(len(edges) - 1) if edges[j] <= p < edges[j + 1])
        a = agg[b]; a[0] += 1; a[1] += p; a[2] += x["q"][cb]
        if cb == win:
            a[3] += 1; a[4] += x["pay"][("3連単", cb)]
print(f"{'モデル確率':<14}{'目数':>9}{'実際':>8}{'モデル':>8}{'市場':>8}{'回収率':>8}")
for b in sorted(agg):
    n, sp, sq, h, r = agg[b]
    print(f"{edges[b]:>5.1%}〜{edges[b + 1]:<6.1%}{n:>9,}{h / n:>8.2%}{sp / n:>8.2%}{sq / n:>8.2%}{r / (n * 100):>8.1%}")
