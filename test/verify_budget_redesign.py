# -*- coding: utf-8 -*-
"""予算再設計シミュレーション(2026-08-04ケンさん相談)

    py -X utf8 test/verify_budget_redesign.py

案: 超混戦=全部⑬2,000円(優先) / 本命=cap4×1,200円 / 日次上限10,000円
  ①本命追加: 入替単2本(3位-2位-1位・4位-2位-1位)各100円
  ②本命追加: 4位-2位-1位 200円
比較: 現行(超混戦⑬2,000+本命cap6×1,000・上限なし)
月次学習8か月。日次処理: 超混戦を全部→残予算で本命(勝率低い順・最大cap)。
"""
import sys
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07"]

conn = db.connect(DB_PATH)
df = build_training_set(conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()


def plan13(ranked):
    plan = P.ken_portfolio("荒れ注意", ranked, [],
                           P.picks_katsu(P.normalize_probs(ranked)), konsen=True)
    return [(bt, cb, y) for bt, cb, y, _ in plan]


def honmei_plan(ranked, extra):
    plan = [(bt, cb, y) for bt, cb, y, _ in P.ken_portfolio(
        "荒れ注意", ranked, [], P.picks_katsu(P.normalize_probs(ranked)))]
    lanes = [r["lane"] for r in ranked]
    r1, r2, r3, r4 = lanes[:4]
    if extra == "①入替2本":
        plan += [("3連単", f"{r3}-{r2}-{r1}", 100), ("3連単", f"{r4}-{r2}-{r1}", 100)]
    elif extra == "②F入替200":
        plan += [("3連単", f"{r4}-{r2}-{r1}", 200)]
    elif extra == "③1400":
        plan += [("3連単", f"{r3}-{r2}-{r1}", 100), ("3連単", f"{r4}-{r2}-{r1}", 100),
                 ("3連単", f"{r4}-{r2}-{r1}", 200)]
    return plan


ARMS = ("現行(cap6×1000・上限なし)", "①cap4×1200(入替2本)", "②cap4×1200(F入替200)", "③cap4×1400(上限10200)")
agg = defaultdict(lambda: [0, 0])
day_spend = defaultdict(list)
trimmed_days = defaultdict(int)

for m in MONTHS:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    by_day = defaultdict(lambda: {"kon": [], "hon": []})
    for rid, g in month_df.groupby("race_id"):
        if not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        top = ranked[0]["prob"]
        d0 = g["date"].iloc[0]
        if top < 0.20:
            by_day[d0]["kon"].append((rid, ranked))
        elif top < 0.30 and int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES:
            by_day[d0]["hon"].append((rid, top, ranked))

    for d0, day in by_day.items():
        hon_sorted = sorted(day["hon"], key=lambda x: x[1])
        for arm in ARMS:
            spend = ret = 0
            # 超混戦は全部(全アーム共通)
            for rid, ranked in day["kon"]:
                bets = plan13(ranked)
                pay = payout_map[rid]
                spend += sum(y for _, _, y in bets)
                ret += sum(pay.get((bt, cb), 0) * y // 100 for bt, cb, y in bets)
            if arm.startswith("現行"):
                cap, unit, extra, limit = 6, 1000, None, None
            elif arm.startswith("①"):
                cap, unit, extra, limit = 4, 1200, "①入替2本", 10000
            elif arm.startswith("②"):
                cap, unit, extra, limit = 4, 1200, "②F入替200", 10000
            else:
                cap, unit, extra, limit = 4, 1400, "③1400", 10200
            taken = 0
            for rid, _top, ranked in hon_sorted:
                if taken >= cap:
                    break
                if limit and spend + unit > limit:
                    trimmed_days[arm] += 1
                    break
                bets = honmei_plan(ranked, extra)
                pay = payout_map[rid]
                spend += sum(y for _, _, y in bets)
                ret += sum(pay.get((bt, cb), 0) * y // 100 for bt, cb, y in bets)
                taken += 1
            a = agg[arm]
            a[0] += spend
            a[1] += ret
            day_spend[arm].append(spend)

print(f"\n===== 合計(2025-12〜2026-07) =====")
import statistics
for arm in ARMS:
    st, rt = agg[arm]
    ds = day_spend[arm]
    print(f"{arm:<24} 投資{st:>11,}円 回収率{rt/st:>7.1%} 損益{rt-st:>+11,}円")
    print(f"{'':>24} 日予算: 平均{statistics.mean(ds):,.0f}円 最大{max(ds):,}円 "
          f"1万円超の日{sum(1 for s in ds if s>10000)}日 本命を削った日{trimmed_days[arm]}日")
