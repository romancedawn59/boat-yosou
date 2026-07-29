# -*- coding: utf-8 -*-
"""超混戦: C・D複の廃止分を「A/Bトリオの発現率上位3連単」に回す案
(2026-07-29判断会中のケンさん発案・第3形態)

    py -X utf8 test/verify_konsen_ab_top_orders.py

■ 案の中身
A複{r1,r2,r3}・B複{r1,r2,r4}は当たりやすいが払戻が安くガミの主因。
そのトリオ内で発現確率(Harville+Benter)が最も高い並びを3連単で足し、
「複が当たる日」を単の厚みで黒字化できないか。

■ 事前登録(実行前に固定)
  変形ハ(2点版・計1,000円): A200 B100 E200 F200 G100
      + Aトリオ確率1位の単100 + Bトリオ確率1位の単100
  変形ニ(4点版・計1,200円): 同上 + A/Bトリオ確率2位の単 各100
      (1,000円予算を超えるため参考扱い。回収率で比較)
判定基準(変形ハ): 案1を回収率で上回り、ガミ率が案1+5pt以内、
最大1発除きでも案1超え。満たさなければ案1のまま。
参考表示(採否に使わない): 追加単スロット単体の回収率
(=「確率上位の並び」を市場が安売りしているかの直接測定)。
既存知見の予測: 確率上位の並び=市場の人気側であり安売りされていない
(7/21検証F「自由選択は市場と同じ買い方になり配当が伸びない」211.9%)。
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
races_month = {}
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
for rid, d in conn.execute(
        "SELECT race_id, date FROM races WHERE date >= ?", (TEST_START,)):
    races_month[rid] = d[:7]
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
        if len(ranked) < 5 or ranked[0]["prob"] >= 0.20:
            continue
        probs = P.normalize_probs(ranked)
        tri = P.trifecta_probs(probs)
        lanes = [r["lane"] for r in ranked]
        r1, r2, r3, r4, r5 = lanes[:5]

        def top_orders(members, k=2):
            cands = [(o, p) for o, p in tri.items() if set(o) == set(members)]
            cands.sort(key=lambda x: -x[1])
            return [f"{a}-{b}-{c}" for (a, b, c), _p in cands[:k]]

        ctxs.append({"rid": rid, "lanes": lanes, "month": races_month.get(rid, "?"),
                     "atop": top_orders((r1, r2, r3)), "btop": top_orders((r1, r2, r4))})

n = len(ctxs)
print(f"\n超混戦帯(1位20%未満): {n:,}レース")


def build_bets(c, variant):
    r1, r2, r3, r4, r5 = c["lanes"][:5]

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"
    base = {
        "A": ("3連複", trio(r1, r2, r3)), "B": ("3連複", trio(r1, r2, r4)),
        "C": ("3連複", trio(r1, r3, r4)), "D": ("3連複", trio(r2, r3, r4)),
        "E": ("3連単", f"{r3}-{r1}-{r2}"), "F": ("3連単", f"{r4}-{r1}-{r2}"),
        "G": ("3連複", trio(r3, r4, r5)),
        "Atop1": ("3連単", c["atop"][0]), "Btop1": ("3連単", c["btop"][0]),
        "Atop2": ("3連単", c["atop"][1] if len(c["atop"]) > 1 else c["atop"][0]),
        "Btop2": ("3連単", c["btop"][1] if len(c["btop"]) > 1 else c["btop"][0]),
    }
    return [(base[k], yen) for k, yen in variant.items() if yen]


VARIANTS = {
    "現行Q案": {"A": 200, "B": 100, "C": 100, "D": 100, "E": 200, "F": 200, "G": 100},
    "案1拾える複厚": {"A": 300, "B": 200, "E": 200, "F": 200, "G": 100},
    "変形ハ(単2点1000円)": {"A": 200, "B": 100, "Atop1": 100, "Btop1": 100,
                           "E": 200, "F": 200, "G": 100},
    "変形ニ(単4点1200円)": {"A": 200, "B": 100, "Atop1": 100, "Atop2": 100,
                           "Btop1": 100, "Btop2": 100, "E": 200, "F": 200, "G": 100},
}

print(f"\n--- 変種比較(レース単位) ---")
print(f"{'変種':<18}{'投資/R':>7}{'回収率':>8}{'完全外れ':>9}{'ガミ':>7}{'プラス':>7}{'最大1発除き':>11}")
summary = {}
for vname, w in VARIANTS.items():
    st = rt = fm = gm = pl = 0
    monthly = defaultdict(lambda: [0, 0])
    best_hit = 0
    for c in ctxs:
        pay = payout_map[c["rid"]]
        rs = rr = 0
        for (bt, comb), yen in build_bets(c, w):
            rs += yen
            rr += pay.get((bt, comb), 0) * yen // 100
        st += rs
        rt += rr
        m = monthly[c["month"]]
        m[0] += rs
        m[1] += rr
        best_hit = max(best_hit, rr)
        if rr == 0:
            fm += 1
        elif rr < rs:
            gm += 1
        else:
            pl += 1
    ex = (rt - best_hit) / st if st else 0
    summary[vname] = {"roi": rt / st, "gami": gm / n, "ex": ex,
                      "monthly": dict(monthly)}
    print(f"{vname:<18}{st//n:>6,}円{rt/st:>8.1%}{fm/n:>9.1%}{gm/n:>7.1%}"
          f"{pl/n:>7.1%}{ex:>11.1%}")

print(f"\n--- 月次(回収率) ---")
months = sorted({m for v in summary.values() for m in v["monthly"]})
print("月       " + "".join(f"{v:<16}" for v in VARIANTS))
for m in months:
    row = f"{m}  "
    for vname in VARIANTS:
        s, r = summary[vname]["monthly"].get(m, [0, 0])
        row += f"{(r/s if s else 0):>8.1%}        "
    print(row)

print(f"\n--- 参考(採否に使わない): 追加単スロット単体(各100円) ---")
for key, label in (("Atop1", "Aトリオ確率1位の単"), ("Atop2", "Aトリオ確率2位の単"),
                   ("Btop1", "Bトリオ確率1位の単"), ("Btop2", "Bトリオ確率2位の単")):
    st = rt = hits = 0
    pays = []
    for c in ctxs:
        pay = payout_map[c["rid"]]
        (bt, comb), _y = build_bets(c, {key: 100})[0]
        got = pay.get((bt, comb), 0)
        st += 100
        rt += got
        if got:
            hits += 1
            pays.append(got)
    avg = sum(pays) / hits if hits else 0
    print(f"  {label:<14} 的中{hits:>3}本({hits/n:.1%}) 回収率{rt/st:>7.1%} "
          f"平均払戻{avg:>8,.0f}円")

v, a1 = summary["変形ハ(単2点1000円)"], summary["案1拾える複厚"]
passed = (v["roi"] > a1["roi"] and v["gami"] <= a1["gami"] + 0.05
          and v["ex"] > a1["ex"])
print(f"\n===== 事前登録基準の判定(変形ハ) =====")
print(f"  回収率 {v['roi']:.1%} vs 案1 {a1['roi']:.1%} / ガミ {v['gami']:.1%} vs {a1['gami']:.1%} / "
      f"除き {v['ex']:.1%} vs {a1['ex']:.1%}")
print(f"  → {'案1超え・提出可' if passed else '基準未達 → 案1のまま'}")
