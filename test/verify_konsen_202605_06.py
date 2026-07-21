# -*- coding: utf-8 -*-
"""超混戦(全場×1位勝率20%未満)の2026年5〜6月固定シミュレーション

    py -X utf8 test/verify_konsen_202605_06.py

2026-05-01より前の全データで1回だけ学習し、5/1〜6/30を評価する
(検証⑦⑧⑨・verify_ken_v2_202605_06.pyと同じ枠組み)。
walk-forward 233日で採用したQ案(2026-07-21)が、独立した固定期間でも
再現するかを確認する頑健性チェック。

比較する構成(いずれも超混戦帯のみ・1レース1,000円):
  現行  3連複r1r2r3/r1r2r4/r1r3r4 + 3連単r3-r1-r2/r4-r1-r2 + C勝万舟
  K案   3連複を4艇ボックス化(r2r3r4を追加)しCは維持
  Q案   K案 + r3=r4=r5(深い波乱)を追加しC枠を廃止 ← 採用済み

walk-forwardでの結果(1,059R・本命帯含む): 除き239.8%→249.3%、的中48.1→53.1%、
DD-31,560→-29,450円。ここでは超混戦だけを取り出すため数字の水準は変わる。
"""
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import challengers as CH
import db
import predictors as P
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

EVAL_START, EVAL_END = "2026-05-01", "2026-06-30"
KONSEN_MAX = 0.20

conn = db.connect(DB_PATH)
df = build_training_set(conn)
actual = defaultdict(dict)
for rid, lane, order in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date BETWEEN ? AND ? AND res.arrival_order IS NOT NULL",
    (EVAL_START, EVAL_END)):
    actual[rid][order] = lane
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date BETWEEN ? AND ?",
    (EVAL_START, EVAL_END)):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

train_df = df[df["date"] < EVAL_START]
eval_df = df[(df["date"] >= EVAL_START) & (df["date"] <= EVAL_END)].copy()
print(f"学習 {len(train_df):,}行 / 評価 {len(eval_df):,}行(全24場)")
booster = train_fold(train_df)
eval_df["pred"] = booster.predict(eval_df[FEATURE_COLUMNS])

ctxs = []
for rid, g in eval_df.groupby("race_id"):
    res = actual[rid]
    if 1 not in res or 2 not in res or 3 not in res or not payout_map[rid]:
        continue
    g_sorted = g.sort_values("pred", ascending=False)
    ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
              for _, r in g_sorted.iterrows()]
    probs = P.normalize_probs(ranked)
    if len(probs) < 4:
        continue
    if ranked[0]["prob"] >= KONSEN_MAX:
        continue          # 超混戦だけを取り出す
    ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]),
                 "venue": int(g["venue_code"].iloc[0]),
                 "top": ranked[0]["prob"], "ranked": ranked, "probs": probs,
                 "top3": {res[1], res[2], res[3]}})
ctxs.sort(key=lambda c: (c["date"], c["rid"]))
days = sorted({c["date"] for c in ctxs})
print(f"超混戦レース {len(ctxs)}R / 出現日 {len(days)}日\n")


def _trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def compose(style, c):
    probs = c["probs"]
    if style == "現行":
        return P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(probs))
    if style == "Q案(採用)":
        return P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(probs),
                               konsen=True)
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    if style == "K案(ボックス)":
        plan = [
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 100, "検証済み"),
            ("3連複", _trio(r1, r3, r4), 100, "検証済み"),
            ("3連複", _trio(r2, r3, r4), 100, "軸外し"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
            ("3連単", f"{r4}-{r1}-{r2}", 200, "検証済み"),
        ]
        ex = {(bt, comb) for bt, comb, _y, _s in plan}
        for bt, comb, _p in P.picks_katsu(probs):
            if (bt, comb) not in ex:
                plan.append((bt, comb, 100, "勝万舟"))
                return plan
        plan[-1] = (plan[-1][0], plan[-1][1], plan[-1][2] + 100, plan[-1][3])
        return plan
    raise ValueError(style)


STYLES = ("現行", "K案(ボックス)", "Q案(採用)")

# r1が3着圏外に飛ぶ頻度(1軸構成が全滅する割合)
gone = sum(1 for c in ctxs if c["ranked"][0]["lane"] not in c["top3"])
print(f"1位予想が3着圏外に飛んだ: {gone}/{len(ctxs)}R ({gone/len(ctxs):.1%})\n")

print(f"=== 超混戦の5〜6月シミュレーション({len(ctxs)}R・{len(days)}日) ===")
print(f"{'構成':<14}{'的中率':>8}{'回収率':>9}{'最大1発除き':>11}{'損益':>12}"
      f"{'最大DD':>10}{'最長連敗':>8}")
results = {}
for style in STYLES:
    daily = defaultdict(float)
    tot = {"n": 0, "hits": 0, "stake": 0, "ret": 0, "max_ret": 0}
    monthly = defaultdict(lambda: {"stake": 0, "ret": 0, "n": 0, "hits": 0})
    for c in ctxs:
        plan = compose(style, c)
        pay = payout_map[c["rid"]]
        stake = sum(y for _, _, y, _ in plan)
        ret = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _ in plan)
        tot["n"] += 1
        tot["stake"] += stake
        tot["ret"] += ret
        tot["max_ret"] = max(tot["max_ret"], ret)
        tot["hits"] += 1 if ret else 0
        daily[c["date"]] += ret - stake
        m = monthly[c["date"][:7]]
        m["stake"] += stake
        m["ret"] += ret
        m["n"] += 1
        m["hits"] += 1 if ret else 0
    series = [daily[d] for d in days]
    roi = tot["ret"] / tot["stake"]
    roi_ex = (tot["ret"] - tot["max_ret"]) / tot["stake"]
    results[style] = {"roi": roi, "roi_ex": roi_ex, "monthly": monthly,
                      "pnl": tot["ret"] - tot["stake"]}
    print(f"{style:<14}{tot['hits']/tot['n']:>8.1%}{roi:>9.1%}{roi_ex:>11.1%}"
          f"{tot['ret']-tot['stake']:>+11,}円{CH.max_drawdown(series):>9,.0f}円"
          f"{CH.longest_losing_streak(series):>7}日")

print("\n--- 月別 ---")
print(f"{'構成':<14}{'月':<9}{'R数':>5}{'的中率':>8}{'回収率':>9}{'損益':>12}")
for style in STYLES:
    for month in sorted(results[style]["monthly"]):
        m = results[style]["monthly"][month]
        print(f"{style:<14}{month:<9}{m['n']:>5}{m['hits']/m['n']:>8.1%}"
              f"{m['ret']/m['stake']:>9.1%}{m['ret']-m['stake']:>+11,}円")

print("\n--- 現行との差 ---")
base = results["現行"]
for style in STYLES[1:]:
    r = results[style]
    print(f"{style}: 回収率{r['roi']-base['roi']:+.1f}pt "
          f"除き{r['roi_ex']-base['roi_ex']:+.1f}pt "
          f"損益{r['pnl']-base['pnl']:+,}円")
