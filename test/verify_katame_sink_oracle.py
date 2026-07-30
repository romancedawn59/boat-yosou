# -*- coding: utf-8 -*-
"""堅めレースの「大穴の匂い」= 沈没オラクルの天井測定(2026-07-30夜・ケンさん発案)

    py -X utf8 test/verify_katame_sink_oracle.py

■ 問い
万舟の58%は堅めレース(1位50%以上)から出るが、帯ごと買うと75.5%で破産する。
「大穴の匂いがする堅めレース」だけ選べれば儲かるのか?
= §7-4の大穴一撃フラグ構想の天井測定。

■ 方法(本命帯の沈没解剖 verify_favorite_sinks_tenkai.py と同じ枠組み)
堅め帯(全場・1位生値50%以上)で「一番人気が4着以下に沈む」レースを
オラクル(後知恵)で選び、その中で買った場合の回収率=理論上の天井を測る。
実際には沈没は事前に分からない(Z1-2の課題)。

■ 事前登録(実行前に固定)
アーム(沈没オラクル条件・各100円):
  a 保険複r2r3r4(本命帯v2.1と同じ器)
  b C勝万舟1点(万舟圏の確率上位・現picks_katsuと同一)
  c 沈没時頻出形の3連単上位5点(パターンは沈没レースから集計=イン標本)
判定基準: いずれかが150%以上なら「Z1-2の適用範囲を堅め帯へ拡張する価値あり」
(本命帯の272.4%と同じ物差し)。150%未満なら大穴一撃フラグはこの帯で死に、
「堅めの大穴は匂いが分かっても獲れない」と結論する。
参考表示: 沈没率・決まり手分布・沈没時の配当分布。
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

KM = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "抜き", 6: "恵まれ"}

conn = db.connect(DB_PATH)
df = build_training_set(conn)
res_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,)):
    res_all[rid][lane] = ao
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
race_tech = dict(conn.execute(
    "SELECT race_id, winning_technique_number FROM races WHERE date >= ?",
    (TEST_START,)))
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
        arr = res_all.get(rid, {})
        if len(arr) < 3 or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5 or ranked[0]["prob"] < 0.50:
            continue
        top3 = sorted(arr, key=lambda l: arr[l])[:3]
        fav_arr = arr.get(ranked[0]["lane"])
        ctxs.append({"rid": rid, "ranked": ranked, "top3": top3,
                     "sink": fav_arr is None or fav_arr >= 4,
                     "tech": race_tech.get(rid)})

n = len(ctxs)
sinks = [c for c in ctxs if c["sink"]]
print(f"\n堅め帯(全場・1位50%以上): {n:,}R / 一番人気4着以下(沈没) "
      f"{len(sinks):,}R ({len(sinks)/n:.1%})")
t = Counter(c["tech"] for c in sinks if c["tech"])
print("沈没時の決まり手: " + " / ".join(
    f"{KM[k]}{v/len(sinks):.0%}" for k, v in t.most_common()))

# 沈没時の3連単配当分布
pays = []
for c in sinks:
    comb = "-".join(str(l) for l in c["top3"])
    amt = payout_map[c["rid"]].get(("3連単", comb), 0)
    if amt:
        pays.append(amt)
pays.sort()
if pays:
    print(f"沈没時の3連単配当: 中央値{pays[len(pays)//2]:,}円 / "
          f"万舟率{sum(1 for p in pays if p >= 10000)/len(pays):.0%}")

# 沈没時の頻出形(モデル順位空間)
pat = Counter()
for c in sinks:
    rank_of = {r["lane"]: i + 1 for i, r in enumerate(c["ranked"])}
    pat["-".join(f"r{rank_of[l]}" for l in c["top3"])] += 1
print("沈没時の頻出形: " + " / ".join(
    f"{k}({v/len(sinks):.0%})" for k, v in pat.most_common(5)))

print(f"\n=== 沈没オラクル(沈没レース{len(sinks):,}Rだけ買えた場合の天井) ===")
arms = {}
for c in sinks:
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    trio = sorted([r2, r3, r4])
    probs = P.normalize_probs(c["ranked"])
    katsu = P.picks_katsu(probs)
    bets = {
        "a 保険複r2r3r4": [("3連複", f"{trio[0]}={trio[1]}={trio[2]}", 100)],
        "b C勝万舟1点": ([(katsu[0][0], katsu[0][1], 100)] if katsu else []),
    }
    tops5 = [k for k, _ in pat.most_common(5)]
    o5 = []
    for key in tops5:
        rks = [int(x[1:]) for x in key.split("-")]
        try:
            o5.append(("3連単", "-".join(str(lanes[rk - 1]) for rk in rks), 100))
        except IndexError:
            pass
    bets["c 頻出形5点"] = o5
    pay = payout_map[c["rid"]]
    for name, bs in bets.items():
        a = arms.setdefault(name, [0, 0, 0])
        for bt, comb, y in bs:
            a[0] += y
            got = pay.get((bt, comb), 0) * y // 100
            a[1] += got
            if got:
                a[2] += 1

for name, (st, rt, hit) in arms.items():
    if st:
        print(f"  {name:<14} 的中{hit:>4}本 回収率{rt/st:>7.1%}")

best = max(rt / st for st, rt, _ in arms.values() if st)
print(f"\n===== 事前登録基準の判定 =====")
print(f"  最良アーム {best:.1%} → "
      f"{'150%以上: Z1-2の堅め帯拡張(大穴一撃フラグ)に価値あり' if best >= 1.5 else '150%未満: 堅めの大穴は匂いが分かっても獲れない → フラグ構想はこの帯で closed'}")
