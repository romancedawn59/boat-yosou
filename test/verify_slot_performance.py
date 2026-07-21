# -*- coding: utf-8 -*-
"""買い目1点ごとの成績と自信度の較正(2026-07-21ケンさん発案「自信ポイントを付ける」)

    py -X utf8 test/verify_slot_performance.py

問い: V2構成6点はそれぞれいくら稼いでいるのか。モデルの自信(確率)は当てになるのか。
構成の検証は9案すべて「まるごと比較」で行い全滅したが、6点を個別に分解したことはない。

測ること:
1. スロット別成績: 6点それぞれの的中率・平均配当・回収率(その点だけ買った場合)
2. 較正: 「自信X%」と付けた点が実際にX%当たるか(自信の信頼性)
   → 崩れていれば、モデル確率を根拠に金額配分している現構成の前提が揺らぐ
3. 自信帯別の回収率: 自信が高い点ほど儲かるのか

walk-forwardはbacktest.pyと同一fold。選別は超混戦帯(20%未満)と本命帯(30%未満)。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
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
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        ctxs.append({"rid": rid, "top": ranked[0]["prob"], "fold": i + 1,
                     "ranked": ranked, "probs": probs})

def slots_of(c):
    """V2構成6点を(ラベル, 券種, 買い目, 金額, モデル確率=自信)で返す"""
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    tri = P.trifecta_probs(c["probs"])

    def trio_p(a, b, x):
        s = {a, b, x}
        return sum(p for k, p in tri.items() if set(k) == s)

    def key(a, b, x):
        y = sorted([a, b, x])
        return f"{y[0]}={y[1]}={y[2]}"

    out = [
        ("①3連複 1=2=3", "3連複", key(r1, r2, r3), 200, trio_p(r1, r2, r3)),
        ("②3連複 1=2=4", "3連複", key(r1, r2, r4), 200, trio_p(r1, r2, r4)),
        ("③3連複 1=3=4", "3連複", key(r1, r3, r4), 100, trio_p(r1, r3, r4)),
        ("④3連単 3-1-2", "3連単", f"{r3}-{r1}-{r2}", 200, tri.get((r3, r1, r2), 0.0)),
        ("⑤3連単 4-1-2", "3連単", f"{r4}-{r1}-{r2}", 200, tri.get((r4, r1, r2), 0.0)),
    ]
    c_picks = P.picks_katsu(c["probs"])
    existing = {(bt, comb) for _l, bt, comb, _y, _p in out}
    for bt, comb, p in c_picks:
        if (bt, comb) not in existing:
            out.append(("⑥C勝万舟", bt, comb, 100, p))
            break
    return out

def report(scope_name, pred):
    sel = [c for c in ctxs if pred(c["top"])]
    print(f"\n=== {scope_name}({len(sel):,}レース) ===")
    agg = defaultdict(lambda: {"n": 0, "hits": 0, "stake": 0, "ret": 0,
                               "conf": 0.0, "pays": []})
    calib = defaultdict(lambda: {"n": 0, "hits": 0, "conf": 0.0,
                                 "stake": 0, "ret": 0})
    for c in sel:
        pay = payout_map[c["rid"]]
        for label, bt, comb, yen, prob in slots_of(c):
            amt = pay.get((bt, comb), 0)
            got = amt * yen // 100
            a = agg[label]
            a["n"] += 1
            a["stake"] += yen
            a["ret"] += got
            a["conf"] += prob
            if got:
                a["hits"] += 1
                a["pays"].append(got)
            b = next(lbl for lo, hi, lbl in CONF_BINS if lo <= prob < hi)
            k = calib[b]
            k["n"] += 1
            k["conf"] += prob
            k["stake"] += yen
            k["ret"] += got
            if got:
                k["hits"] += 1

    print("--- スロット別成績(その点だけ買った場合) ---")
    print(f"{'点':<15}{'平均自信':>8}{'的中率':>8}{'較正':>8}"
          f"{'平均払戻':>10}{'投資':>11}{'回収':>12}{'回収率':>9}")
    for label in ("①3連複 1=2=3", "②3連複 1=2=4", "③3連複 1=3=4",
                  "④3連単 3-1-2", "⑤3連単 4-1-2", "⑥C勝万舟"):
        a = agg.get(label)
        if not a or not a["n"]:
            continue
        conf = a["conf"] / a["n"]
        rate = a["hits"] / a["n"]
        avg = sum(a["pays"]) / len(a["pays"]) if a["pays"] else 0
        gap = rate - conf
        print(f"{label:<15}{conf:>8.2%}{rate:>8.2%}{gap:>+8.2%}"
              f"{avg:>9,.0f}円{a['stake']:>10,}円{a['ret']:>11,}円"
              f"{a['ret']/a['stake']:>9.1%}")
    tot_s = sum(a["stake"] for a in agg.values())
    tot_r = sum(a["ret"] for a in agg.values())
    print(f"{'合計(V2構成)':<15}{'':>8}{'':>8}{'':>8}{'':>10}"
          f"{tot_s:>10,}円{tot_r:>11,}円{tot_r/tot_s:>9.1%}")

    print("--- 較正: 「自信X%」と付けた点は実際にX%当たるか ---")
    print(f"{'自信帯':<10}{'点数':>8}{'平均自信':>9}{'実際の的中率':>12}"
          f"{'ズレ':>9}{'回収率':>9}")
    for _lo, _hi, lbl in CONF_BINS:
        k = calib.get(lbl)
        if not k or not k["n"]:
            continue
        conf = k["conf"] / k["n"]
        rate = k["hits"] / k["n"]
        print(f"{lbl:<10}{k['n']:>8,}{conf:>9.2%}{rate:>12.2%}"
              f"{rate-conf:>+9.2%}{k['ret']/k['stake']:>9.1%}")

report("超混戦帯(1位勝率20%未満)", lambda t: t < 0.20)
report("本命帯(1位勝率30%未満)", lambda t: t < 0.30)
