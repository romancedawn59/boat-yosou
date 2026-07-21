# -*- coding: utf-8 -*-
"""帯別に買い方を変える検証(2026-07-21ケンさん指摘「本命と大荒れが同じ買い方はおかしい」)

    py -X utf8 test/verify_band_compositions.py

背景: 現行は本命(5場×30%未満×上位6)も超混戦(全場×20%未満)も同じV2構成6点。
だがtest/verify_slot_performance.pyで、帯により各点の効きが大きく違うと判明した。
  スロット別回収率(超混戦帯/本命帯): ①265.9/120.6 ②306.9/168.6 ③147.9/110.9
  ④540.0/156.1 ⑤823.3/230.3 ⑥C 217.3/90.6
超混戦帯は3連単穴目が突出、本命帯はC枠が唯一の赤字。

【事前登録】候補は今日のデータから素直に導かれる4案のみ(構成9案全滅の反省から
思いつきを列挙しない)。判定基準は以下3つを全て満たすこと。満たさなければ不採用:
  1. 最大1発除き回収率が現行(A)を上回る
  2. fold間で安定(現行を大きく下回るfoldがない)
  3. 最大ドローダウンが悪化しない
※構成⑪-Bで攻撃型(3連単厚め)は158.1%と現行180.3%に劣った実績あり。
  ただしあれは帯を分けない検証だった。

案:
  A(基準) 本命=現行6点        / 超混戦=現行6点
  B       本命=C枠を外し⑤300円 / 超混戦=現行6点
  C       本命=現行6点        / 超混戦=③を削り⑤300円
  D       本命=Bの配分        / 超混戦=Cの配分
いずれも1レース1,000円・100円単位を維持する。
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
    """styleに応じた買い目プラン(券種, 買い目, 金額, 出典)。計1,000円"""
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    c_picks = P.picks_katsu(c["probs"])
    if style == "現行":
        return P.ken_portfolio("荒れ注意", c["ranked"], [], c_picks)
    if style == "C抜き⑤厚":   # ①②③④は据え置き、C枠100円を⑤へ回す
        return [
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 200, "検証済み"),
            ("3連複", _trio(r1, r3, r4), 100, "検証済み"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
            ("3連単", f"{r4}-{r1}-{r2}", 300, "検証済み"),
        ]
    if style == "③抜き⑤厚":   # 3連複3点目を削り⑤へ回す(Cは残す)
        plan = [
            ("3連複", _trio(r1, r2, r3), 200, "検証済み"),
            ("3連複", _trio(r1, r2, r4), 200, "検証済み"),
            ("3連単", f"{r3}-{r1}-{r2}", 200, "検証済み"),
            ("3連単", f"{r4}-{r1}-{r2}", 300, "検証済み"),
        ]
        existing = {(bt, comb) for bt, comb, _y, _s in plan}
        for bt, comb, _p in c_picks:
            if (bt, comb) not in existing:
                plan.append((bt, comb, 100, "勝万舟"))
                break
        else:
            plan[-1] = (plan[-1][0], plan[-1][1], plan[-1][2] + 100, plan[-1][3])
        return plan
    raise ValueError(style)


# 案: (ラベル, 本命帯のstyle, 超混戦帯のstyle)
PLANS = [
    ("A 現行(共通)", "現行", "現行"),
    ("B 本命C抜き", "C抜き⑤厚", "現行"),
    ("C 超混戦③抜き", "現行", "③抜き⑤厚"),
    ("D 両方", "C抜き⑤厚", "③抜き⑤厚"),
]


def select_day(day_ctxs):
    """v2選別: 本命(5場×30%未満×上位6)+超混戦(全場×20%未満)。
    返り値 {rid: 'honmei'|'konsen'}(重複は本命扱い=購入1回)"""
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


print(f"\n=== 帯別構成の比較(walk-forward {n_days}日) ===")
print(f"{'案':<15}{'R数':>6}{'的中率':>8}{'回収率':>8}{'最大1発除き':>11}"
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
            style = hon_style if band == "honmei" else kon_style
            plan = compose(style, c)
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
    streak = CH.longest_losing_streak(series)
    results[label] = {"roi": roi, "roi_ex": roi_ex, "dd": dd,
                      "fold": {f: v["ret"] / v["stake"] for f, v in per_fold.items()}}
    print(f"{label:<15}{tot['n']:>6,}{tot['hits']/tot['n']:>8.1%}{roi:>8.1%}"
          f"{roi_ex:>11.1%}{tot['ret']-tot['stake']:>+11,}円{dd:>9,}円{streak:>7}日")

print("\n--- fold別回収率(安定性の確認) ---")
print(f"{'案':<15}" + "".join(f"{'fold'+str(i+1):>10}" for i in range(N_FOLDS)))
for label, *_ in PLANS:
    r = results[label]["fold"]
    print(f"{label:<15}" + "".join(f"{r.get(i+1, 0):>10.1%}" for i in range(N_FOLDS)))

print("\n--- 事前登録した判定基準に対する結果 ---")
base = results["A 現行(共通)"]
for label, *_ in PLANS[1:]:
    r = results[label]
    c1 = r["roi_ex"] > base["roi_ex"]
    c2 = all(r["fold"].get(f, 0) >= base["fold"].get(f, 0) * 0.9
             for f in base["fold"])
    c3 = r["dd"] >= base["dd"]  # DDは負の値。大きい(浅い)ほど良い
    verdict = "採用候補" if (c1 and c2 and c3) else "不採用"
    print(f"{label}: 除き回収率{'○' if c1 else '×'} "
          f"fold安定{'○' if c2 else '×'} DD{'○' if c3 else '×'} → {verdict}")
