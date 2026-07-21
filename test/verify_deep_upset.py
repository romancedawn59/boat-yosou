# -*- coding: utf-8 -*-
"""r3=r4=r5(1位・2位が揃って沈む展開)の検証
   (2026-07-21ケンさん提案「C万舟狙いはr3=r4=r5の展開シミュレーションで採用する?」)

    py -X utf8 test/verify_deep_upset.py

背景: 超混戦帯のC枠(万舟圏0.5%以下の3連単)は233日で的中3件しかなく、
価値を統計的に判断できない(test/verify_konsen_c_slot.py)。
提案は、その枠を「予測3位・4位・5位の3連複」=1位も2位も飛ぶ展開に充てる案。
超混戦帯は確率が平坦なのでr1r2が揃って崩れる展開はありえ、市場は上位人気を
軸に買うため配当も跳ねやすい。万舟を3連単でなく3連複で取りにいく形になる。

まず素の実力(単独の的中率・回収率)を現行C枠と比較し、そのうえで構成に組み込む。

【事前登録】判定基準(既存の構成検証と同一):
  1. 最大1発除き回収率が現行(A)を上回る
  2. fold間で安定(現行の90%を下回るfoldがない)
  3. 最大ドローダウンが悪化しない

案(すべて超混戦帯のみ変更。本命帯は現行固定):
  K 3連複4艇ボックス(前回の採用候補)
  Q K のC枠100円を r3=r4=r5 に置換
  R 現行構成のC枠100円を r3=r4=r5 に置換(ボックス化なし=この枠単独の効果)
  S K に r3=r4=r5 を足しC枠を廃止(軸外し2種類の布陣)
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
        # 母集団は他の構成検証と同一(4艇以上)にする。r5が無いレースでは
        # r3=r4=r5 を組めないのでC枠で代替する(compose側で処理)。
        # ここを5艇以上に絞ると母集団が変わり、他の検証と比較できなくなる
        # (2026-07-21に実際にやってしまい、K案の判定が逆転した)
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


# ===== 素の実力比較: 超混戦帯で各「保険枠」を1点100円で買い続けた場合 =====
print("\n=== 超混戦帯: 保険枠の素の実力(1点100円で買い続けた場合) ===")
kon = [c for c in ctxs if c["top"] < KONSEN_MAX]
print(f"対象 {len(kon)}レース")
print(f"{'枠':<22}{'買えた':>7}{'的中':>6}{'的中率':>8}{'平均払戻':>10}"
      f"{'回収':>11}{'回収率':>9}")


def measure_slot(label, picker):
    n = hits = ret = 0
    pays = []
    for c in kon:
        item = picker(c)
        if not item:
            continue
        bt, comb = item
        n += 1
        amt = payout_map[c["rid"]].get((bt, comb), 0)
        if amt:
            hits += 1
            ret += amt
            pays.append(amt)
    avg = sum(pays) / len(pays) if pays else 0
    print(f"{label:<22}{n:>7}{hits:>6}{hits/n if n else 0:>8.2%}{avg:>9,.0f}円"
          f"{ret:>10,}円{ret/(n*100) if n else 0:>9.1%}")
    return {"n": n, "hits": hits, "ret": ret}


def lanes_of(c):
    return [r["lane"] for r in c["ranked"]]


measure_slot("現行C枠(万舟圏3連単)", lambda c: (
    (P.picks_katsu(c["probs"])[0][0], P.picks_katsu(c["probs"])[0][1])
    if P.picks_katsu(c["probs"]) else None))
measure_slot("r2=r3=r4(軸外し)", lambda c: (
    "3連複", _trio(*lanes_of(c)[1:4])))
measure_slot("r3=r4=r5(提案・深い波乱)", lambda c: (
    ("3連複", _trio(*lanes_of(c)[2:5])) if len(lanes_of(c)) >= 5 else None))
measure_slot("参考 r1=r2=r3(本線)", lambda c: (
    "3連複", _trio(*lanes_of(c)[0:3])))


def compose(style, c):
    lanes = lanes_of(c)
    r1, r2, r3, r4 = lanes[:4]
    r5 = lanes[4] if len(lanes) >= 5 else None  # 4艇レースではNone
    probs = c["probs"]

    def add_c(plan):
        ex = {(bt, comb) for bt, comb, _y, _s in plan}
        for bt, comb, _p in P.picks_katsu(probs):
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
    cur3f = [
        ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
        ("3連複", _trio(r1, r2, r4), 200, "検証済み"),
        ("3連複", _trio(r1, r3, r4), 100, "検証済み"),
    ]

    if style == "現行":
        return P.ken_portfolio("荒れ注意", c["ranked"], [], P.picks_katsu(probs))
    if style == "K ボックス":
        return add_c(box3f + fixed3t)
    if style == "Q ボックス+345":
        if r5 is None:
            return add_c(box3f + fixed3t)   # 4艇レースはC枠で代替
        return box3f + fixed3t + [("3連複", _trio(r3, r4, r5), 100, "深い波乱")]
    if style == "R 現行+345":
        if r5 is None:
            return add_c(cur3f + fixed3t)
        return cur3f + fixed3t + [("3連複", _trio(r3, r4, r5), 100, "深い波乱")]
    if style == "S ボックス+345のみ":
        # C枠を廃止しr2r3r4とr3r4r5の二段構え(3連複r1r2r4を100→なし で調整)
        plan = [
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 100, "検証済み"),
            ("3連複", _trio(r2, r3, r4), 100, "軸外し"),
        ]
        if r5 is None:
            return add_c(plan + fixed3t)
        return plan + [("3連複", _trio(r3, r4, r5), 100, "深い波乱")] + fixed3t
    raise ValueError(style)


PLANS = [
    ("A 現行", "現行"),
    ("K ボックス", "K ボックス"),
    ("Q ボックス+345", "Q ボックス+345"),
    ("R 現行+345", "R 現行+345"),
    ("S ボックス+345(C無)", "S ボックス+345のみ"),
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


print(f"\n=== 構成比較(walk-forward {n_days}日・本命帯は現行固定) ===")
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
    results[label] = {"roi_ex": roi_ex, "dd": dd,
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
