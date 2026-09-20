# -*- coding: utf-8 -*-
"""② 確率の較正 → その出力で市場と比べる(2026-09-21ケンさん承認・r1システムv3.0)

    py -X utf8 test/verify_calibration_then_market.py

材料は本番配信picks JSONの確率だけ(2026-07-21〜)。
■ A 較正: 1艇ごとの「そのままの確率・6艇の合計・予想順位・枠」から実際の1着率を当てる
  補正器(小さなLightGBM)を作り、レース内で合計1に整える。
  事前登録の検証: 学習 〜2026-08-31 / 採点 2026-09-01〜。予想順位×合計の帯ごとに
  「出した確率と実際の1着率の差」を そのまま/比例配分/較正後 で比べる。
  合格基準: 採点期間の200艇以上のマスで、較正後の差が全て±3pt以内 かつ 平均のずれが比例配分より小さい。
■ B 3連単の確率: 較正後の確率から既存の式(predictors.trifecta_probs)で120通りを計算し、
  出した確率と実際の的中率を突き合わせる。
■ C 市場との比較: 日付ブロックの5分割で「そのレースを学習に使っていない較正確率」を全期間に付け、
  市場の想定確率 q=(1/オッズ)をレース内で合計1に整えたもの と全120通りで比べる。
  比 = 較正確率÷市場の想定確率 の帯ごとに、実際の的中数がどちらの見立てに近いかと、
  その帯の目を全部100円ずつ買った回収率を出す。
  合格基準: 比1.5以上の帯が 回収率100%超・最大1本除外後95%超・7〜8月と9月の両方で100%超。
"""
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
from predictors import trifecta_probs  # noqa: E402

PARAMS = {"objective": "binary", "verbosity": -1, "learning_rate": 0.03, "num_leaves": 7,
          "min_data_in_leaf": 150, "seed": 7}
FEATS = ["p_raw", "p_sum", "p_prop", "rank", "lane"]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
win = {rid: lane for rid, lane in conn.execute(
    "SELECT res.race_id, res.lane FROM results res JOIN races r ON r.race_id=res.race_id "
    "WHERE r.date>='2026-07-21' AND res.arrival_order=1")}
pay3 = {rid: (c, a or 0) for rid, c, a in conn.execute(
    "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p JOIN races r "
    "ON r.race_id=p.race_id WHERE r.date>='2026-07-21' AND p.bet_type='3連単'")}
odds = defaultdict(dict)
for tbl in ("odds_final", "odds"):       # 同じ目は締切前スナップショット(odds)を優先
    for rid, comb, o in conn.execute(
            f"SELECT race_id, combination, odds FROM {tbl} WHERE bet_type='3連単' "
            f"AND odds IS NOT NULL AND odds > 0 AND substr(race_id,1,8) >= '20260721'"):
        odds[rid][comb] = o
conn.close()

rows = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    d = os.path.basename(path)[6:16]
    if d < "2026-07-21":
        continue
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid = r["race_id"]
        if not r.get("ranked") or len(r["ranked"]) < 6 or rid not in win:
            continue
        s = sum(p for _l, p in r["ranked"])
        for i, (lane, p) in enumerate(r["ranked"]):
            rows.append((d, rid, lane, i + 1, p, s, p / s, int(win[rid] == lane)))
D = pd.DataFrame(rows, columns=["date", "race_id", "lane", "rank", "p_raw", "p_sum",
                                "p_prop", "win"])


def fit(tr):
    return lgb.train(PARAMS, lgb.Dataset(tr[FEATS], label=tr["win"]), num_boost_round=300)


def normalize(df, col):
    return df[col] / df.groupby("race_id")[col].transform("sum")


# ---------------- A 較正の検証(〜8/31で学習 → 9月で採点) ----------------
tr, te = D[D["date"] < "2026-09-01"], D[D["date"] >= "2026-09-01"].copy()
te["p_cal"] = fit(tr).predict(te[FEATS])
te["p_cal"] = normalize(te, "p_cal")
print(f"===== A 較正: 学習{tr['race_id'].nunique():,}R → 採点{te['race_id'].nunique():,}R =====")
te["sgrp"] = pd.cut(te["p_sum"], [0, .8, 1.0, 9], labels=["合計0.8未満", "合計0.8〜1.0", "合計1.0以上"])
te["rgrp"] = te["rank"].map(lambda k: f"r{k}" if k <= 3 else "r4〜r6")
print(f"{'マス':<22}{'艇数':>6}{'実際':>8}{'そのまま':>9}{'比例配分':>9}{'較正後':>8}  差(そのまま/比例/較正)")
errs = {"p_raw": [], "p_prop": [], "p_cal": []}
worst = 0.0
for (sg, rg), g in te.groupby(["sgrp", "rgrp"], observed=True):
    o = g["win"].mean()
    d = {c: (g[c].mean() - o) * 100 for c in errs}
    if len(g) >= 200:
        for c in errs:
            errs[c].append(abs(d[c]))
        worst = max(worst, abs(d["p_cal"]))
    print(f"{sg + ' ' + rg:<22}{len(g):>6,}{o:>8.1%}{g['p_raw'].mean():>9.1%}{g['p_prop'].mean():>9.1%}"
          f"{g['p_cal'].mean():>8.1%}  {d['p_raw']:>+5.1f} / {d['p_prop']:>+5.1f} / {d['p_cal']:>+5.1f} pt")
ll = {c: -np.mean(np.where(te["win"] == 1, np.log(te[c].clip(1e-6, 1 - 1e-6)),
                           np.log((1 - te[c]).clip(1e-6, 1 - 1e-6)))) for c in errs}
print("平均のずれ(200艇以上のマス): " + " / ".join(
    f"{n} {np.mean(errs[c]):.2f}pt" for n, c in (("そのまま", "p_raw"), ("比例配分", "p_prop"), ("較正後", "p_cal"))))
print("logloss: " + " / ".join(f"{n} {ll[c]:.4f}" for n, c in (("そのまま", "p_raw"), ("比例配分", "p_prop"), ("較正後", "p_cal"))))
okA = worst <= 3.0 and np.mean(errs["p_cal"]) < np.mean(errs["p_prop"])
print(f"判定A: {'合格' if okA else '不合格'}(較正後の最大のずれ {worst:.1f}pt)")

# ---------------- C用: 5分割で全期間に較正確率を付ける ----------------
days = sorted(D["date"].unique())
fold_of = {d: i * 5 // len(days) for i, d in enumerate(days)}
D["fold"] = D["date"].map(fold_of)
D["p_cal"] = 0.0
for k in range(5):
    m = fit(D[D["fold"] != k])
    D.loc[D["fold"] == k, "p_cal"] = m.predict(D.loc[D["fold"] == k, FEATS])
D["p_cal"] = normalize(D, "p_cal")

# ---------------- B 3連単の確率の確かさ / C 市場との比較 ----------------
tick = []   # (date, race_id, comb, p_model, p_prop_model, q_market, odds, hit, payout)
for rid, g in D.groupby("race_id"):
    o = odds.get(rid)
    if rid not in pay3 or not o or len(o) < 110:
        continue
    pm = trifecta_probs(dict(zip(g["lane"], g["p_cal"])))
    pp = trifecta_probs(dict(zip(g["lane"], g["p_prop"])))
    inv = {cb: 1 / x for cb, x in o.items()}
    tot = sum(inv.values())
    wc, amt = pay3[rid]
    date = g["date"].iloc[0]
    for (a, b, c), p in pm.items():
        cb = f"{a}-{b}-{c}"
        if cb in inv:
            tick.append((date, rid, p, pp[(a, b, c)], inv[cb] / tot, o[cb], int(cb == wc),
                         amt if cb == wc else 0))
T = pd.DataFrame(tick, columns=["date", "race_id", "p", "p_prop", "q", "odds", "hit", "pay"])
print(f"\n===== B 3連単の確率の確かさ: {T['race_id'].nunique():,}R × 120通り =====")
print(f"{'出した確率の帯':<16}{'目の数':>9}{'的中':>6}{'較正後の想定':>11}{'比例配分の想定':>13}{'市場の想定':>10}")
for lo, hi in ((0, .002), (.002, .005), (.005, .01), (.01, .02), (.02, .05), (.05, .10), (.10, 1)):
    g = T[(T["p"] >= lo) & (T["p"] < hi)]
    if len(g):
        print(f"{lo:>6.1%}〜{hi:>6.1%}{len(g):>10,}{int(g['hit'].sum()):>6}{g['p'].sum():>11.1f}"
              f"{g['p_prop'].sum():>13.1f}{g['q'].sum():>10.1f}")

print("\n===== C 市場との比較(比=較正確率÷市場の想定確率) =====")
T["ratio"] = T["p"] / T["q"]
print(f"{'比の帯':<12}{'目の数':>9}{'的中':>6}{'較正の想定':>10}{'市場の想定':>10}{'実際はどちら寄り':>14}"
      f"{'回収率':>8}{'最大1本除く':>10}{'7〜8月':>8}{'9月':>7}")
okC = False
for lo, hi in ((0, .5), (.5, .8), (.8, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 99)):
    g = T[(T["ratio"] >= lo) & (T["ratio"] < hi)]
    if not len(g):
        continue
    h, em, eq = g["hit"].sum(), g["p"].sum(), g["q"].sum()
    side = "較正" if abs(h - em) < abs(h - eq) else "市場"
    roi = g["pay"].sum() / (len(g) * 100)
    exb = (g["pay"].sum() - g["pay"].max()) / (len(g) * 100)
    a = g[g["date"] < "2026-09-01"]
    b = g[g["date"] >= "2026-09-01"]
    ra = a["pay"].sum() / max(1, len(a) * 100)
    rb = b["pay"].sum() / max(1, len(b) * 100)
    print(f"{lo:>4.2f}〜{hi:>5.2f}{len(g):>10,}{int(h):>6}{em:>10.1f}{eq:>10.1f}{side:>14}"
          f"{roi:>8.1%}{exb:>10.1%}{ra:>8.0%}{rb:>7.0%}")
    if lo >= 1.5 and roi > 1 and exb > 0.95 and ra > 1 and rb > 1:
        okC = True
print(f"判定C: {'合格' if okC else '不合格'}(比1.5以上の帯が回収率100%超・最大1本除外後95%超・両期間100%超)")
print(f"(参考) 全120通りを毎回買った回収率: {T['pay'].sum() / (len(T) * 100):.1%}")
