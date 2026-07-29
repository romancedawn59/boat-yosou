# -*- coding: utf-8 -*-
"""伸び盛りコンボ最終テスト(2026-07-30ケンさん発案・事前登録)

    py -X utf8 test/verify_rising_slot.py

仮説: [r1]-[r2]-[X]型(Xを2着/3着席に配置、軸はモデル本命)で、
X=伸び盛り(gap>+10pt)のチケットは X=横ばい(|gap|<3pt)の同型を上回る。
Xはモデル順位3-5位に限定し、同順位で比較(ランク偏りの排除)。
これで不成立なら伸び盛りの出口は特徴量注入のみで確定(3回目の配置検証)。
"""
import sqlite3
import sys
import datetime as dt
from bisect import bisect_left
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

# --- 伸びギャップ下ごしらえ ---
raw = sqlite3.connect(DB_PATH)
races_date = dict(raw.execute("SELECT race_id, date FROM races"))
lane_racer = {}
printed = {}
for rid, lane, reg, n2 in raw.execute(
        "SELECT race_id, lane, reg_no, national_2rate FROM entries"):
    lane_racer[(rid, lane)] = reg
    printed[(rid, lane)] = n2
rows = []
for rid, lane, ao in raw.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    d = races_date.get(rid)
    reg = lane_racer.get((rid, lane))
    if d and reg:
        rows.append((d, reg, ao))
rows.sort()
hist = defaultdict(list)
for d, reg, ao in rows:
    hist[reg].append((d, 1 if ao <= 2 else 0))
raw.close()


def gap_of(rid, lane):
    reg = lane_racer.get((rid, lane))
    n2 = printed.get((rid, lane))
    d = races_date.get(rid)
    if reg is None or n2 is None or d is None:
        return None
    h = hist[reg]
    dates = [x[0] for x in h]
    hi = bisect_left(dates, d)
    d0 = (dt.date.fromisoformat(d) - dt.timedelta(days=90)).isoformat()
    lo = bisect_left(dates, d0)
    seg = h[lo:hi]
    if len(seg) < 12:
        return None
    return sum(t for _, t in seg) / len(seg) - n2 / 100.0


# --- walk-forward ---
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
        lanes = [int(r["lane"]) for _, r in g_sorted.iterrows()]
        if len(lanes) < 5:
            continue
        ctxs.append({"rid": rid, "top": float(g_sorted.iloc[0]["pred"]),
                     "lanes": lanes})


def trio_key(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


for scope_name, lo, hi in (("本命帯(20-35%)", 0.20, 0.35),
                           ("超混戦帯(<20%)", 0.0, 0.20)):
    sel = [c for c in ctxs if lo <= c["top"] < hi]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for c in sel:
        rid = c["rid"]
        r1, r2 = c["lanes"][:2]
        pay = payout_map[rid]
        for rank_idx in (2, 3, 4):          # モデル3-5位
            x = c["lanes"][rank_idx]
            g = gap_of(rid, x)
            if g is None:
                continue
            if g > 0.10:
                bucket = "★伸び盛り"
            elif -0.03 <= g < 0.03:
                bucket = "横ばい"
            else:
                continue
            tickets = [
                ("複r1r2X", "3連複", trio_key(r1, r2, x)),
                ("単r1-r2-X", "3連単", f"{r1}-{r2}-{x}"),
                ("単r1-X-r2", "3連単", f"{r1}-{x}-{r2}"),
            ]
            for tname, bt, comb in tickets:
                a = agg[tname][bucket]
                a[0] += 100
                got = pay.get((bt, comb), 0)
                a[1] += got
                if got:
                    a[2] += 1
    print(f"\n=== {scope_name}: {len(sel):,}R (Xはモデル3-5位限定) ===")
    print(f"{'チケット型':<12}{'X':<8}{'点数':>8}{'的中':>6}{'回収率':>8}")
    for tname in ("複r1r2X", "単r1-r2-X", "単r1-X-r2"):
        for bucket in ("★伸び盛り", "横ばい"):
            a = agg[tname][bucket]
            if not a[0]:
                continue
            print(f"{tname:<12}{bucket:<8}{a[0]//100:>8,}{a[2]:>6}"
                  f"{a[1]/a[0]:>8.1%}")
