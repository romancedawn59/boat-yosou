# -*- coding: utf-8 -*-
"""1軸集中の是非(2026-07-21ケンさん指摘「大混戦でも1軸だよね?」)

    py -X utf8 test/verify_axis_diversify.py

指摘: 現行V2構成はC枠以外の5点すべてがr1(モデルの1位予想)を含む実質1軸。
  ①r1=r2=r3 ②r1=r2=r4 ③r1=r3=r4 ④r3-r1-r2 ⑤r4-r1-r2
1位勝率20%未満の「誰も読めない」超混戦で、モデルの1位予想に全乗せしている。
r1が3着圏外に飛べば5点が同時に死ぬ。

※過去の「1抜き構成」検証(2026-07-19)は1号艇(枠番1)を抜く案で別物。
  今回はモデルの1位予想を抜く案=未検証。

【事前登録】判定基準(既存の構成検証と同一。満たさなければ不採用):
  1. 最大1発除き回収率が現行(A)を上回る
  2. fold間で安定(現行の90%を下回るfoldがない)
  3. 最大ドローダウンが悪化しない

案(いずれも超混戦帯のみ変更。本命帯は現行のまま):
  I ③r1=r3=r4 → r2=r3=r4(3連複の1点だけ軸を外す)
  J Iに加えて ⑤r4-r1-r2 → r4-r2-r3(3連単も1点r1を外す)
  K 3連複を上位4艇ボックス4点(r1軸3点+r1抜き1点)に増やし3連単は据え置き
参考として、超混戦帯でr1が実際に3着以内に入る率も測る。
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

# 参考: r1(1位予想)は実際どれだけ3着以内に残るか
print("\n=== 1位予想(r1)の3着内率 ===")
for label, pred in (("超混戦帯(20%未満)", lambda t: t < KONSEN_MAX),
                    ("本命帯(30%未満)", lambda t: t < HONMEI_MAX),
                    ("全レース", lambda t: True)):
    sel = [c for c in ctxs if pred(c["top"])]
    if not sel:
        continue
    in3 = sum(1 for c in sel if c["ranked"][0]["lane"] in c["top3"])
    win = sum(1 for c in sel if c["ranked"][0]["lane"] == max(
        c["top3"], key=lambda x: 0) and False)  # 未使用(1着率は下で別途)
    print(f"{label:<18} {len(sel):>6,}R  3着内 {in3/len(sel):>6.1%}"
          f"  (3着圏外 {1-in3/len(sel):>5.1%}=この時5点が同時に死ぬ)")


def _trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def compose(style, c):
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    c_picks = P.picks_katsu(c["probs"])

    def add_c(plan):
        ex = {(bt, comb) for bt, comb, _y, _s in plan}
        for bt, comb, _p in c_picks:
            if (bt, comb) not in ex:
                plan.append((bt, comb, 100, "勝万舟"))
                return plan
        plan[-1] = (plan[-1][0], plan[-1][1], plan[-1][2] + 100, plan[-1][3])
        return plan

    if style == "現行":
        return P.ken_portfolio("荒れ注意", c["ranked"], [], c_picks)

    if style == "I 3連複1点軸外し":
        return add_c([
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 200, "検証済み"),
            ("3連複", _trio(r2, r3, r4), 100, "軸外し"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
            ("3連単", f"{r4}-{r1}-{r2}", 200, "検証済み"),
        ])

    if style == "J 3連複+3連単軸外し":
        return add_c([
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 200, "検証済み"),
            ("3連複", _trio(r2, r3, r4), 100, "軸外し"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
            ("3連単", f"{r4}-{r2}-{r3}", 200, "軸外し"),
        ])

    if style == "K 4艇ボックス":
        return add_c([
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 100, "検証済み"),
            ("3連複", _trio(r1, r3, r4), 100, "検証済み"),
            ("3連複", _trio(r2, r3, r4), 100, "軸外し"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
            ("3連単", f"{r4}-{r1}-{r2}", 200, "検証済み"),
        ])
    raise ValueError(style)


# (ラベル, 本命帯style, 超混戦帯style)。本命帯はr1の3着圏外率が29.9%と
# 超混戦帯(24.9%)より高いため、ボックスを本命帯にも当てる案Lを追加した
PLANS = [
    ("A 現行", "現行", "現行"),
    ("I 3連複1点軸外し", "現行", "I 3連複1点軸外し"),
    ("J 3連複+3連単", "現行", "J 3連複+3連単軸外し"),
    ("K 超混戦のみボックス", "現行", "K 4艇ボックス"),
    ("L 両帯ボックス", "K 4艇ボックス", "K 4艇ボックス"),
    ("M 本命のみボックス", "K 4艇ボックス", "現行"),
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


print(f"\n=== 超混戦帯の軸分散(walk-forward {n_days}日・本命帯は現行固定) ===")
print(f"{'案':<18}{'R数':>6}{'的中率':>8}{'回収率':>8}{'最大1発除き':>11}"
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
    results[label] = {"roi_ex": roi_ex, "dd": dd,
                      "fold": {f: v["ret"] / v["stake"] for f, v in per_fold.items()}}
    print(f"{label:<18}{tot['n']:>6,}{tot['hits']/tot['n']:>8.1%}{roi:>8.1%}"
          f"{roi_ex:>11.1%}{tot['ret']-tot['stake']:>+11,}円{dd:>9,.0f}円"
          f"{CH.longest_losing_streak(series):>7}日")

print("\n--- fold別回収率 ---")
print(f"{'案':<18}" + "".join(f"{'fold'+str(i+1):>10}" for i in range(N_FOLDS)))
for label, *_ in PLANS:
    r = results[label]["fold"]
    print(f"{label:<18}" + "".join(f"{r.get(i+1, 0):>10.1%}" for i in range(N_FOLDS)))

print("\n--- 事前登録した判定基準に対する結果 ---")
base = results["A 現行"]
for label, *_ in PLANS[1:]:
    r = results[label]
    c1 = r["roi_ex"] > base["roi_ex"]
    c2 = all(r["fold"].get(f, 0) >= base["fold"].get(f, 0) * 0.9 for f in base["fold"])
    c3 = r["dd"] >= base["dd"]
    print(f"{label}: 除き回収率{'○' if c1 else '×'} fold安定{'○' if c2 else '×'} "
          f"DD{'○' if c3 else '×'} → {'採用候補' if (c1 and c2 and c3) else '不採用'}")

