# -*- coding: utf-8 -*-
"""C枠の置換と3連単の自由選択の検証(2026-07-21ケンさん提案)

    py -X utf8 test/verify_free_selection.py

提案:
  1. 本命帯の万舟狙い(⑥C勝万舟)は有効か。3連単予想に回した方がよいのでは
     → C枠は本命帯で90.6%の赤字(test/verify_slot_performance.py)。ただし
       昨夜検証したB案(C枠→⑤4位頭を300円)は1発除きで現行と誤差だった。
       今回は「確率上位の3連単に置き換える」= 当たりやすい目に回す案で、性質が違う
  2. 超混戦帯の④⑤(3位頭・4位頭の固定形)を自由選択にしてよいのでは
     → 超混戦帯で④540%・⑤823%と突出している固定形を、確率上位2点に置き換える。
       当たりやすくなる代わりに配当が下がる交換が得か損かを測る

【事前登録】判定基準(昨夜の帯別検証と同一。満たさなければ不採用):
  1. 最大1発除き回収率が現行(A)を上回る
  2. fold間で安定(現行の90%を下回るfoldがない)
  3. 最大ドローダウンが悪化しない
※構成の探索はこれまで10案すべて現行以下(構成9案+帯別4案)。
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
        if 1 not in actual[rid] or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]), "fold": i + 1,
                     "venue": int(g["venue_code"].iloc[0]),
                     "top": ranked[0]["prob"], "ranked": ranked, "probs": probs})
ctxs.sort(key=lambda c: (c["date"], c["rid"]))


def _trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def compose(style, c):
    """計1,000円・100円単位のプランを返す"""
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    probs = c["probs"]
    c_picks = P.picks_katsu(probs)
    base3f = [
        ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
        ("3連複", _trio(r1, r2, r4), 200, "検証済み"),
        ("3連複", _trio(r1, r3, r4), 100, "検証済み"),
    ]
    fixed3t = [
        ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
        ("3連単", f"{r4}-{r1}-{r2}", 200, "検証済み"),
    ]

    def top_trifecta(n, exclude):
        """3連単の発生確率上位n点(excludeと重複しないもの)"""
        out = []
        for (a, b, cc), _p in sorted(P.trifecta_probs(probs).items(),
                                     key=lambda x: -x[1]):
            comb = f"{a}-{b}-{cc}"
            if ("3連単", comb) in exclude:
                continue
            out.append(comb)
            if len(out) == n:
                break
        return out

    if style == "現行":
        return P.ken_portfolio("荒れ注意", c["ranked"], [], c_picks)

    if style == "C→3連単上位":
        # ①〜⑤は据え置き、C枠100円を「確率上位の3連単1点」に置き換える
        plan = base3f + fixed3t
        ex = {(bt, comb) for bt, comb, _y, _s in plan}
        top = top_trifecta(1, ex)
        if top:
            plan.append(("3連単", top[0], 100, "3連単上位"))
        else:
            plan[-1] = (plan[-1][0], plan[-1][1], plan[-1][2] + 100, plan[-1][3])
        return plan

    if style == "④⑤自由":
        # 3連複3点+Cは据え置き、④⑤を「確率上位の3連単2点」に置き換える
        plan = list(base3f)
        ex = {(bt, comb) for bt, comb, _y, _s in plan}
        tops = top_trifecta(2, ex)
        for comb in tops:
            plan.append(("3連単", comb, 200, "3連単上位"))
        while len(plan) < 5:  # 念のため(通常は起きない)
            plan.append(("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"))
        ex = {(bt, comb) for bt, comb, _y, _s in plan}
        for bt, comb, _p in c_picks:
            if (bt, comb) not in ex:
                plan.append((bt, comb, 100, "勝万舟"))
                break
        else:
            plan[-1] = (plan[-1][0], plan[-1][1], plan[-1][2] + 100, plan[-1][3])
        return plan

    raise ValueError(style)


# (ラベル, 本命帯style, 超混戦帯style)
PLANS = [
    ("A 現行", "現行", "現行"),
    ("E 本命C→3連単", "C→3連単上位", "現行"),
    ("F 超混戦④⑤自由", "現行", "④⑤自由"),
    ("G E+F", "C→3連単上位", "④⑤自由"),
    ("H 両帯とも自由", "④⑤自由", "④⑤自由"),
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


print(f"\n=== C枠置換と3連単自由選択の比較(walk-forward {n_days}日) ===")
print(f"{'案':<16}{'R数':>6}{'的中率':>8}{'回収率':>8}{'最大1発除き':>11}"
      f"{'損益':>12}{'最大DD':>10}{'最長連敗':>8}")
results = {}
for label, hon_style, kon_style in PLANS:
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
            plan = compose(hon_style if band == "honmei" else kon_style, c)
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
                      "pnl": tot["ret"] - tot["stake"],
                      "fold": {f: v["ret"] / v["stake"] for f, v in per_fold.items()}}
    print(f"{label:<16}{tot['n']:>6,}{tot['hits']/tot['n']:>8.1%}{roi:>8.1%}"
          f"{roi_ex:>11.1%}{tot['ret']-tot['stake']:>+11,}円{dd:>9,.0f}円"
          f"{CH.longest_losing_streak(series):>7}日")

print("\n--- fold別回収率 ---")
print(f"{'案':<16}" + "".join(f"{'fold'+str(i+1):>10}" for i in range(N_FOLDS)))
for label, *_ in PLANS:
    r = results[label]["fold"]
    print(f"{label:<16}" + "".join(f"{r.get(i+1, 0):>10.1%}" for i in range(N_FOLDS)))

print("\n--- 事前登録した判定基準に対する結果 ---")
base = results["A 現行"]
for label, *_ in PLANS[1:]:
    r = results[label]
    c1 = r["roi_ex"] > base["roi_ex"]
    c2 = all(r["fold"].get(f, 0) >= base["fold"].get(f, 0) * 0.9 for f in base["fold"])
    c3 = r["dd"] >= base["dd"]
    print(f"{label}: 除き回収率{'○' if c1 else '×'} fold安定{'○' if c2 else '×'} "
          f"DD{'○' if c3 else '×'} → {'採用候補' if (c1 and c2 and c3) else '不採用'}")
