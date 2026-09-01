# -*- coding: utf-8 -*-
"""本命選別の腕の指紋鑑定(2026-09-02・真白ゆい発見4の徹底調査)

    py -X utf8 test/sim_selection_skill_2026.py

■ 問い
実弾の本命は「5場×1位勝率20-30%×勝率が低い順に日次予算内で最大4R」で
選ばれる(predictors.select_shobusho)。この選別に腕はあるのか。
根拠だった検証⑰(cap4×低い順=162.3%)はリーク時代の数字なので測り直す。
実弾期間45日の実測では選別74R=85.8%(帯平均と同じ)・非選別14R=151.4%。

■ 方法(本番忠実・クリーン)
月次walk-forward 2026-01〜08・全艇ランク・返還処理あり。日ごとに:
konsen_n=全場の超混戦(⑬が組める<20%)数 → 残予算=10,200-2,000×konsen_n
→ take=min(4, 残予算//1,400)。プール=5場×20-30%帯。
アーム: ①低い順take(現行ルール) ②高い順take(逆) ③プール全部
④中央値順take(参考)。全て9行1,400円構成で採点。
■ 事前登録判定: ①が③(プール平均)を+5pt以上上回れば「選別に腕あり」。
①≈③なら「選別は無作為抽出と同じ=腕なし」、①<③なら「低い順は逆効果」。
"""
import sys
from collections import defaultdict
from itertools import permutations

import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import (FEATURE_COLUMNS, _ENTRY_COLS, _attach_extra_features,
                      _encode, build_training_set, compute_form_features)

EVAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
               "2026-05", "2026-06", "2026-07", "2026-08"]
DAILY_BUDGET, KONSEN_UNIT, HONMEI_UNIT, CAP = 10200, 2000, 1400, 4

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
eval_df = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2026-01-01'
""", conn)
eval_df = _encode(eval_df)
eval_df = eval_df.merge(compute_form_features(conn), on=["race_id", "lane"],
                        how="left")
eval_df = _attach_extra_features(eval_df, conn)
paymap = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-01-01'"):
    paymap[rid][(bt, comb)] = amt or 0
conn.close()


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan9(l):
    r1, r2, r3, r4 = l[:4]
    return [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
            ("3連複", trio(r1, r3, r4), 100), ("3連単", f"{r3}-{r1}-{r2}", 200),
            ("3連単", f"{r4}-{r1}-{r2}", 200), ("3連複", trio(r2, r3, r4), 100),
            ("3連単", f"{r3}-{r2}-{r1}", 100), ("3連単", f"{r4}-{r2}-{r1}", 300)]


def score(lanes, pay, refund):
    merged = defaultdict(int)
    for bt, comb, y in plan9(lanes):
        merged[(bt, comb)] += y
    st = rt = 0
    for (bt, comb), y in merged.items():
        sep = "-" if bt == "3連単" else "="
        members = {int(x) for x in comb.split(sep)}
        st += y
        rt += y if members & refund else pay.get((bt, comb), 0) * y // 100
    return st, rt


ARMS = ("①低い順(現行)", "②高い順(逆)", "③プール全部", "④中央値順")
agg = defaultdict(lambda: [0, 0, 0, 0])   # {(アーム, 月): [st, rt, n, hit1]}
for m in EVAL_MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行", flush=True)
    booster = train_fold(tr_df)
    md = ev.copy()
    md["pred"] = booster.predict(md[FEATURE_COLUMNS])
    daily = defaultdict(lambda: {"konsen": 0, "pool": []})
    for rid, grp in md.groupby("race_id"):
        pay = paymap[rid]
        if not pay:
            continue
        gs = grp.sort_values("pred", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        p1 = float(gs["pred"].iloc[0])
        d = gs["date"].iloc[0]
        arr = {int(r["lane"]): r["arrival_order"] for _, r in grp.iterrows()
               if pd.notna(r["arrival_order"])}
        if len(arr) < 3:
            continue
        if p1 < 0.20 and len(lanes) >= 5:
            daily[d]["konsen"] += 1
            continue
        if (0.20 <= p1 < 0.30 and len(lanes) >= 4
                and int(gs["venue_code"].iloc[0]) in TARGET_VENUE_CODES):
            nonfin = {int(r["lane"]) for _, r in grp.iterrows()
                      if pd.isna(r["arrival_order"])}
            refund = {l for l in nonfin
                      if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
            daily[d]["pool"].append(
                {"p1": p1, "lanes": lanes, "pay": pay, "refund": refund,
                 "win": arr.get(lanes[0]) == 1})
    for d, info in daily.items():
        pool = sorted(info["pool"], key=lambda x: x["p1"])
        remaining = DAILY_BUDGET - KONSEN_UNIT * info["konsen"]
        take = min(CAP, max(0, remaining // HONMEI_UNIT))
        mid = len(pool) // 2
        picks = {
            "①低い順(現行)": pool[:take],
            "②高い順(逆)": pool[::-1][:take],
            "③プール全部": pool,
            "④中央値順": sorted(pool, key=lambda x: abs(x["p1"] - 0.25))[:take],
        }
        for arm, rs in picks.items():
            for x in rs:
                st, rt = score(x["lanes"], x["pay"], x["refund"])
                a = agg[(arm, m)]
                a[0] += st
                a[1] += rt
                a[2] += 1
                a[3] += x["win"]

print("\n===== 本命選別の腕・クリーン8か月(5場×20-30%帯・9行1,400円) =====")
for arm in ARMS:
    tot = [0, 0, 0, 0]
    line = []
    for m in EVAL_MONTHS:
        a = agg[(arm, m)]
        for i in range(4):
            tot[i] += a[i]
        if a[0]:
            line.append(f"{m[5:]}月{a[1] / a[0]:>5.0%}")
    st, rt, n, h = tot
    print(f"\n{arm}: {n}R 1位的中{h / n:.1%} 回収率{rt / st:.1%} "
          f"損益{rt - st:+,}円")
    print("  " + " ".join(line))
print("\n(判定: ①-③が+5pt以上=腕あり / ①≈③=無作為と同じ / ①<③=低い順は逆効果)")
