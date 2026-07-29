# -*- coding: utf-8 -*-
"""超混戦Q案構成(7点)のスロット分解と「ガミ」の構造分析

    py -X utf8 test/verify_konsen_slots_q.py

実運用で超混戦は10R中9的中(90%)なのに回収率59%=当たるのに拾えない。
Q案7点のどの点が「安い的中」でガミらせているのかをwalk-forward全期間で分解し、
レース単位のガミ率(的中したが投資割れ)と、配分変更の机上効果を出す。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

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
        if len(probs) < 5:
            continue
        if ranked[0]["prob"] >= 0.20:      # 超混戦帯のみ
            continue
        ctxs.append({"rid": rid, "ranked": ranked, "probs": probs})

print(f"\n超混戦帯(1位20%未満): {len(ctxs):,}レース")


def q_slots(c):
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4, r5 = lanes[:5]

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"

    return [
        ("A複 r1r2r3(200)", "3連複", trio(r1, r2, r3), 200),
        ("B複 r1r2r4(100)", "3連複", trio(r1, r2, r4), 100),
        ("C複 r1r3r4(100)", "3連複", trio(r1, r3, r4), 100),
        ("D複 軸外しr2r3r4(100)", "3連複", trio(r2, r3, r4), 100),
        ("E単 r3-r1-r2(200)", "3連単", f"{r3}-{r1}-{r2}", 200),
        ("F単 r4-r1-r2(200)", "3連単", f"{r4}-{r1}-{r2}", 200),
        ("G複 深い波乱r3r4r5(100)", "3連複", trio(r3, r4, r5), 100),
    ]


agg = defaultdict(lambda: {"n": 0, "hits": 0, "stake": 0, "ret": 0, "pays": []})
race_net = []
gami = full_miss = win = 0
gami_examples = defaultdict(int)   # ガミ時にどのスロットが当たっていたか

for c in ctxs:
    pay = payout_map[c["rid"]]
    total_stake = total_ret = 0
    hit_slots = []
    for label, bt, comb, yen in q_slots(c):
        amt = pay.get((bt, comb), 0)
        got = amt * yen // 100
        a = agg[label]
        a["n"] += 1
        a["stake"] += yen
        a["ret"] += got
        if got:
            a["hits"] += 1
            a["pays"].append(got)
            hit_slots.append(label)
        total_stake += yen
        total_ret += got
    race_net.append(total_ret - total_stake)
    if total_ret == 0:
        full_miss += 1
    elif total_ret < total_stake:
        gami += 1
        for s in hit_slots:
            gami_examples[s] += 1
    else:
        win += 1

n = len(ctxs)
print(f"\n--- レース単位の内訳(Q案7点・1,000円) ---")
print(f"  完全外れ: {full_miss:,}R ({full_miss/n:.1%})")
print(f"  ガミ(的中したが投資割れ): {gami:,}R ({gami/n:.1%})")
print(f"  プラス: {win:,}R ({win/n:.1%})")
print(f"  何か当たる率: {(gami+win)/n:.1%}")

print(f"\n--- スロット別成績 ---")
print(f"{'点':<22}{'的中率':>8}{'平均払戻':>10}{'投資':>11}{'回収':>12}{'回収率':>9}")
order = ["A複 r1r2r3(200)", "B複 r1r2r4(100)", "C複 r1r3r4(100)",
         "D複 軸外しr2r3r4(100)", "E単 r3-r1-r2(200)", "F単 r4-r1-r2(200)",
         "G複 深い波乱r3r4r5(100)"]
for label in order:
    a = agg[label]
    if not a["n"]:
        continue
    rate = a["hits"] / a["n"]
    avg = sum(a["pays"]) / len(a["pays"]) if a["pays"] else 0
    print(f"{label:<22}{rate:>8.2%}{avg:>9,.0f}円{a['stake']:>10,}円"
          f"{a['ret']:>11,}円{a['ret']/a['stake']:>9.1%}")
ts = sum(a["stake"] for a in agg.values())
tr = sum(a["ret"] for a in agg.values())
print(f"{'合計(Q案)':<22}{'':>8}{'':>10}{ts:>10,}円{tr:>11,}円{tr/ts:>9.1%}")

print(f"\n--- ガミ発生時、どのスロットの的中がガミらせているか ---")
for label in order:
    if gami_examples[label]:
        print(f"  {label}: ガミレースでの的中回数 {gami_examples[label]:,}")

# --- 配分変種のレース単位比較 ---
VARIANTS = {
    "現行Q案":       {"A": 200, "B": 100, "C": 100, "D": 100, "E": 200, "F": 200, "G": 100},
    "案1拾える複厚":  {"A": 300, "B": 200, "C": 0,   "D": 0,   "E": 200, "F": 200, "G": 100},
    "案2回収特化":    {"A": 200, "B": 200, "C": 0,   "D": 0,   "E": 200, "F": 400, "G": 0},
    "案3折衷":       {"A": 200, "B": 200, "C": 0,   "D": 100, "E": 200, "F": 200, "G": 100},
}
KEY = {"A複 r1r2r3(200)": "A", "B複 r1r2r4(100)": "B", "C複 r1r3r4(100)": "C",
       "D複 軸外しr2r3r4(100)": "D", "E単 r3-r1-r2(200)": "E",
       "F単 r4-r1-r2(200)": "F", "G複 深い波乱r3r4r5(100)": "G"}

print(f"\n--- 配分変種の比較(レース単位・1,000円) ---")
print(f"{'変種':<12}{'投資計':>10}{'回収率':>8}{'完全外れ':>8}{'ガミ':>7}{'プラス':>7}")
for vname, w in VARIANTS.items():
    st = rt = fm = gm = pl = 0
    for c in ctxs:
        pay = payout_map[c["rid"]]
        race_stake = race_ret = 0
        for label, bt, comb, _yen in q_slots(c):
            yen = w[KEY[label]]
            if not yen:
                continue
            amt = pay.get((bt, comb), 0)
            race_stake += yen
            race_ret += amt * yen // 100
        st += race_stake
        rt += race_ret
        if race_ret == 0:
            fm += 1
        elif race_ret < race_stake:
            gm += 1
        else:
            pl += 1
    print(f"{vname:<12}{st:>9,}円{rt/st:>8.1%}{fm/n:>8.1%}{gm/n:>7.1%}{pl/n:>7.1%}")
