# -*- coding: utf-8 -*-
"""超混戦帯のC枠の是非(2026-07-21ケンさん提案「C万舟狙いは必要?軸外しから選択?」)

    py -X utf8 test/verify_konsen_c_slot.py

背景: 前回検証(test/verify_axis_diversify.py)で、超混戦帯の3連複を4艇ボックス化
するK案が唯一3基準をクリアした(除き245.5%・的中52.0%・DD-29,450円)。
超混戦帯のC枠は単独217.3%と黒字だが、万舟圏の確率上位から選ぶため
r1(1位予想)を含む目も入る。1位予想は超混戦帯で24.9%が3着圏外に飛ぶので、
C枠もr1を含まない目に限定すれば、飛んだケースを3連複・3連単の両方でカバーできる。

【事前登録】判定基準(既存の構成検証と同一。ベースはK案ではなくA現行と比較):
  1. 最大1発除き回収率が現行(A)を上回る
  2. fold間で安定(現行の90%を下回るfoldがない)
  3. 最大ドローダウンが悪化しない

案(すべて超混戦帯のみ変更。本命帯は現行固定):
  K  3連複を4艇ボックス(前回の採用候補。C枠は現行のまま)
  N  K + C枠をr1を含まない万舟圏の目に限定
  O  K + C枠を廃止し、その100円を軸外し3連複(r2=r3=r4)へ回す
  P  現行構成のままC枠だけ軸外し限定(ボックス化しない=C枠単独の効果を見る)
"""
import sys
from collections import defaultdict
from itertools import groupby

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import challengers as CH
import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

HONMEI_MAX = 0.30
KONSEN_MAX = 0.20
CAP = 6

conn = db.connect(DB_PATH)
df = build_training_set(conn)
actual = defaultdict(dict)
for rid, lane, order in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,)):
    actual[rid][order] = lane
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]
n_days = len(dates)

ctxs = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        res = actual[rid]
        if 1 not in res or 2 not in res or 3 not in res or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]), "fold": i + 1,
                     "venue": int(g["venue_code"].iloc[0]),
                     "top": ranked[0]["prob"], "ranked": ranked, "probs": probs,
                     "top3": {res[1], res[2], res[3]}})
ctxs.sort(key=lambda c: (c["date"], c["rid"]))


def _trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def katsu_no_axis(probs, r1):
    """万舟圏(発生確率MANSHU_PROB_MAX以下)のうちr1を含まない目の確率上位5点"""
    tri = P.trifecta_probs(probs)
    cands = sorted(((k, p) for k, p in tri.items()
                    if p <= P.MANSHU_PROB_MAX and r1 not in k),
                   key=lambda x: -x[1])[:5]
    return [("3連単", f"{a}-{b}-{c}", p) for (a, b, c), p in cands]


def compose(style, c):
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    probs = c["probs"]

    def add_c(plan, picks):
        ex = {(bt, comb) for bt, comb, _y, _s in plan}
        for bt, comb, _p in picks:
            if (bt, comb) not in ex:
                plan.append((bt, comb, 100, "勝万舟"))
                return plan
        plan[-1] = (plan[-1][0], plan[-1][1], plan[-1][2] + 100, plan[-1][3])
        return plan

    box3f = [
        ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
        ("3連複", _trio(r1, r2, r4), 100, "検証済み"),
        ("3連複", _trio(r1, r3, r4), 100, "検証済み"),
        ("3連複", _trio(r2, r3, r4), 100, "軸外し"),
    ]
    fixed3t = [
        ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
        ("3連単", f"{r4}-{r1}-{r2}", 200, "検証済み"),
    ]

    if style == "現行":
        return P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(probs))

    if style == "K ボックス":
        return add_c(box3f + fixed3t, P.picks_katsu(probs))

    if style == "N ボックス+C軸外し":
        return add_c(box3f + fixed3t, katsu_no_axis(probs, r1))

    if style == "O ボックス+C廃止":
        # C枠の100円を軸外し3連複へ(r2=r3=r4を200円に)
        plan = [
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 100, "検証済み"),
            ("3連複", _trio(r1, r3, r4), 100, "検証済み"),
            ("3連複", _trio(r2, r3, r4), 200, "軸外し"),
        ] + fixed3t
        return plan

    if style == "P 現行+C軸外し":
        # ボックス化せず、C枠の選び方だけ変える(C枠単独の効果を見る)
        plan = [
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 200, "検証済み"),
            ("3連複", _trio(r1, r3, r4), 100, "検証済み"),
        ] + fixed3t
        return add_c(plan, katsu_no_axis(probs, r1))

    raise ValueError(style)


PLANS = [
    ("A 現行", "現行"),
    ("K ボックス", "K ボックス"),
    ("N ボックス+C軸外し", "N ボックス+C軸外し"),
    ("O ボックス+C廃止", "O ボックス+C廃止"),
    ("P 現行+C軸外し", "P 現行+C軸外し"),
]


def select_day(day_ctxs):
    hon = sorted((c for c in day_ctxs
                  if c["venue"] in TARGET_VENUE_CODES and c["top"] < HONMEI_MAX),
                 key=lambda c: c["top"])[:CAP]
    out = {}
    for c in day_ctxs:
        if c["top"] < KONSEN_MAX:
            out[c["rid"]] = "konsen"
    for c in hon:
        out[c["rid"]] = "honmei"
    return out


# 参考: 超混戦帯でC枠(現行/軸外し)がどれだけ当たるか
print("\n=== 超混戦帯のC枠の中身 ===")
kon = [c for c in ctxs if c["top"] < KONSEN_MAX]
cur_hit = axis_hit = cur_n = axis_n = 0
cur_ret = axis_ret = 0
for c in kon:
    r1 = c["ranked"][0]["lane"]
    pay = payout_map[c["rid"]]
    cur = P.picks_katsu(c["probs"])
    noax = katsu_no_axis(c["probs"], r1)
    if cur:
        cur_n += 1
        amt = pay.get((cur[0][0], cur[0][1]), 0)
        cur_ret += amt
        cur_hit += 1 if amt else 0
    if noax:
        axis_n += 1
        amt = pay.get((noax[0][0], noax[0][1]), 0)
        axis_ret += amt
        axis_hit += 1 if amt else 0
print(f"現行C枠  : 買えた{cur_n:>4}R 的中{cur_hit:>3}件 回収{cur_ret:>9,}円 "
      f"回収率{cur_ret/(cur_n*100) if cur_n else 0:>7.1%}")
print(f"軸外しC枠: 買えた{axis_n:>4}R 的中{axis_hit:>3}件 回収{axis_ret:>9,}円 "
      f"回収率{axis_ret/(axis_n*100) if axis_n else 0:>7.1%}")

print(f"\n=== 超混戦帯のC枠比較(walk-forward {n_days}日・本命帯は現行固定) ===")
print(f"{'案':<20}{'R数':>6}{'的中率':>8}{'回収率':>8}{'最大1発除き':>11}"
      f"{'損益':>12}{'最大DD':>10}{'最長連敗':>8}")
results = {}
for label, kon_style in PLANS:
    daily = {}
    tot = {"n": 0, "hits": 0, "stake": 0, "ret": 0, "max_ret": 0}
    per_fold = defaultdict(lambda: {"stake": 0, "ret": 0})
    for d, grp in groupby(ctxs, key=lambda c: c["date"]):
        day_list = list(grp)
        sel = select_day(day_list)
        pnl = 0
        for c in day_list:
            band = sel.get(c["rid"])
            if not band:
                continue
            plan = compose("現行" if band == "honmei" else kon_style, c)
            pay = payout_map[c["rid"]]
            stake = sum(y for _, _, y, _ in plan)
            ret = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _ in plan)
            tot["n"] += 1
            tot["stake"] += stake
            tot["ret"] += ret
            tot["max_ret"] = max(tot["max_ret"], ret)
            tot["hits"] += 1 if ret else 0
            per_fold[c["fold"]]["stake"] += stake
            per_fold[c["fold"]]["ret"] += ret
            pnl += ret - stake
        daily[d] = pnl
    series = [daily[d] for d in sorted(daily)]
    roi = tot["ret"] / tot["stake"]
    roi_ex = (tot["ret"] - tot["max_ret"]) / tot["stake"]
    dd = CH.max_drawdown(series)
    results[label] = {"roi": roi, "roi_ex": roi_ex, "dd": dd,
                      "fold": {f: v["ret"] / v["stake"] for f, v in per_fold.items()}}
    print(f"{label:<20}{tot['n']:>6,}{tot['hits']/tot['n']:>8.1%}{roi:>8.1%}"
          f"{roi_ex:>11.1%}{tot['ret']-tot['stake']:>+11,}円{dd:>9,.0f}円"
          f"{CH.longest_losing_streak(series):>7}日")

print("\n--- fold別回収率 ---")
print(f"{'案':<20}" + "".join(f"{'fold'+str(i+1):>10}" for i in range(N_FOLDS)))
for label, _ in PLANS:
    r = results[label]["fold"]
    print(f"{label:<20}" + "".join(f"{r.get(i+1, 0):>10.1%}" for i in range(N_FOLDS)))

print("\n--- 事前登録した判定基準に対する結果(基準=A現行) ---")
base = results["A 現行"]
for label, _ in PLANS[1:]:
    r = results[label]
    c1 = r["roi_ex"] > base["roi_ex"]
    c2 = all(r["fold"].get(f, 0) >= base["fold"].get(f, 0) * 0.9 for f in base["fold"])
    c3 = r["dd"] >= base["dd"]
    print(f"{label}: 除き回収率{'○' if c1 else '×'} fold安定{'○' if c2 else '×'} "
          f"DD{'○' if c3 else '×'} → {'採用候補' if (c1 and c2 and c3) else '不採用'}")
