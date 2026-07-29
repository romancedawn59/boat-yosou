# -*- coding: utf-8 -*-
"""2026年7月に限定した配分分析(2026-07-29ケンさん発案)

    py -X utf8 test/verify_july_allocation.py

問い: 7月のガミは「配分が悪かった」のか。
- 複vs単のバランス(どちらに金を置くべきだったか)
- 自信率帯ごとの的中数・的中率・回収率(どの自信帯が一番当たったか)
- 掛け金(200円線と100円線はそれぞれ働いたか)
- ガミレースの解剖(当たった線に幾ら置いていれば拾えたか)
スコープは実運用に合わせる: 本命=対象5場×1位20-35%、超混戦=全場×20%未満。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

CONF_BINS = [(0.000, 0.005, "0.5%未満"), (0.005, 0.010, "0.5-1%"),
             (0.010, 0.020, "1-2%"), (0.020, 0.050, "2-5%"),
             (0.050, 0.100, "5-10%"), (0.100, 0.200, "10-20%"),
             (0.200, 1.001, "20%以上")]

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
        d = str(g["date"].iloc[0])
        if not d.startswith("2026-07"):
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        vc = int(rid.split("_")[1])
        ctxs.append({"rid": rid, "date": d, "vc": vc,
                     "top": ranked[0]["prob"],
                     "probs": P.normalize_probs(ranked), "ranked": ranked})

print(f"\n2026年7月のレース(WF): {len(ctxs):,}R")


def plan_of(c, konsen):
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    r5 = lanes[4]
    tri = P.trifecta_probs(c["probs"])

    def tp(a, b, x):
        s = {a, b, x}
        return sum(p for k, p in tri.items() if set(k) == s)

    def key(a, b, x):
        y = sorted([a, b, x])
        return f"{y[0]}={y[1]}={y[2]}"

    if konsen:
        return [
            ("3連複", key(r1, r2, r3), 200, tp(r1, r2, r3)),
            ("3連複", key(r1, r2, r4), 100, tp(r1, r2, r4)),
            ("3連複", key(r1, r3, r4), 100, tp(r1, r3, r4)),
            ("3連複", key(r2, r3, r4), 100, tp(r2, r3, r4)),
            ("3連単", f"{r3}-{r1}-{r2}", 200, tri.get((r3, r1, r2), 0.0)),
            ("3連単", f"{r4}-{r1}-{r2}", 200, tri.get((r4, r1, r2), 0.0)),
            ("3連複", key(r3, r4, r5), 100, tp(r3, r4, r5)),
        ]
    plan = [
        ("3連複", key(r1, r2, r3), 200, tp(r1, r2, r3)),
        ("3連複", key(r1, r2, r4), 200, tp(r1, r2, r4)),
        ("3連複", key(r1, r3, r4), 100, tp(r1, r3, r4)),
        ("3連単", f"{r3}-{r1}-{r2}", 200, tri.get((r3, r1, r2), 0.0)),
        ("3連単", f"{r4}-{r1}-{r2}", 200, tri.get((r4, r1, r2), 0.0)),
    ]
    for bt, comb, p in P.picks_katsu(c["probs"]):
        if (bt, comb) not in {(b, cb) for b, cb, _y, _p in plan}:
            plan.append((bt, comb, 100, p))
            break
    return plan


def analyze(scope_name, sel, konsen):
    n = len(sel)
    print(f"\n{'='*66}\n=== {scope_name}: {n:,}R ===")
    by_bt = defaultdict(lambda: [0, 0, 0, 0])
    by_bin = defaultdict(lambda: [0, 0, 0, 0.0])
    by_stake = defaultdict(lambda: [0, 0, 0, 0])
    fm = gm = pl = 0
    gami_hits = []
    for c in sel:
        pay = payout_map[c["rid"]]
        stake = ret = 0
        hit_lines = []
        for bt, comb, yen, prob in plan_of(c, konsen):
            got = pay.get((bt, comb), 0) * yen // 100
            stake += yen
            ret += got
            b = by_bt[bt]
            b[0] += 1
            b[1] += yen
            b[2] += got
            if got:
                b[3] += 1
                hit_lines.append((bt, comb, yen, got, pay.get((bt, comb), 0)))
            lbl = next(l for lo, hi, l in CONF_BINS if lo <= prob < hi)
            k = by_bin[lbl]
            k[0] += 1
            k[1] += yen
            k[2] += got
            k[3] += prob
            if got:
                k.append(1) if False else None
            s = by_stake[yen]
            s[0] += 1
            s[1] += yen
            s[2] += got
            if got:
                s[3] += 1
        if ret == 0:
            fm += 1
        elif ret < stake:
            gm += 1
            gami_hits.extend(hit_lines)
        else:
            pl += 1

    print(f"レース内訳: 完全外れ{fm/n:.1%} / ガミ{gm/n:.1%} / プラス{pl/n:.1%}")
    print(f"\n--- 複vs単バランス(7月) ---")
    for bt, (cnt, st, rt, h) in sorted(by_bt.items()):
        print(f"  {bt}: {cnt:,}点 投資{st:,}円 回収{rt:,}円 "
              f"回収率{rt/st:.1%} 的中{h}本")
    print(f"\n--- 自信帯別(7月): どの帯が当たったか ---")
    print(f"  {'帯':<10}{'点数':>6}{'平均自信':>9}{'的中':>5}{'的中率':>8}{'回収率':>8}")
    hit_by_bin = defaultdict(int)
    for c in sel:
        pay = payout_map[c["rid"]]
        for bt, comb, yen, prob in plan_of(c, konsen):
            if pay.get((bt, comb), 0):
                lbl = next(l for lo, hi, l in CONF_BINS if lo <= prob < hi)
                hit_by_bin[lbl] += 1
    for lo, hi, lbl in CONF_BINS:
        k = by_bin.get(lbl)
        if not k or not k[0]:
            continue
        h = hit_by_bin[lbl]
        print(f"  {lbl:<10}{k[0]:>6,}{k[3]/k[0]:>9.2%}{h:>5}"
              f"{h/k[0]:>8.2%}{k[2]/k[1]:>8.1%}")
    print(f"\n--- 掛け金別(7月) ---")
    for yen, (cnt, st, rt, h) in sorted(by_stake.items()):
        print(f"  {yen}円線: {cnt:,}点 回収率{rt/st:.1%} 的中{h}本")
    if gami_hits:
        need = [1000 * 100 / p for _bt, _c, _y, _g, p in gami_hits if p]
        med = sorted(need)[len(need) // 2]
        from collections import Counter
        cc = Counter(bt for bt, *_ in gami_hits)
        print(f"\n--- ガミレース解剖({gm}R) ---")
        print(f"  ガミ時に当たっていた線: " +
              " ".join(f"{k}×{v}" for k, v in cc.items()))
        print(f"  その線で1,000円回収に必要だった賭け金の中央値: {med:,.0f}円")


hon = [c for c in ctxs if 0.20 <= c["top"] < 0.35 and c["vc"] in TARGET_VENUE_CODES]
kon = [c for c in ctxs if c["top"] < 0.20]
analyze("本命(5場×20-35%)", hon, konsen=False)
analyze("超混戦(全場×20%未満)", kon, konsen=True)
