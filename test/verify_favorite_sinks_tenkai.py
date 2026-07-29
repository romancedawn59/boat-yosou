# -*- coding: utf-8 -*-
"""本命帯: 一番人気(モデル1位)が4着以下に沈む展開の解剖と、その時の3連単の形
(2026-07-29判断会中のケンさん指示「入念なシミュレーション」= Z1-1.5)

    py -X utf8 test/verify_favorite_sinks_tenkai.py

■ 問い(ケンさんの設計)
1. 一番人気が4着以下になるのは「どんな展開」か
   (一番人気は1号艇とは限らない → 人気艇の枠・コース別に機構を分ける)
2. その展開のとき、3連単はどんな形になるのか(モデル順位空間で)
3. その形を狙い撃ちしたら(展開が読めた前提の天井)いくらになるのか
   → Z1-2(展開予測)を作る価値の見積もり

■ 方法
walk-forward予測(backtest.py同一fold)で本命帯(1位生値20〜35%)を再現し、
モデル1位艇の着順・レースの決まり手・勝者の進入コースで層別する。
進入コース・決まり手はレース後にしか分からない=ここでの利用は「解剖」であり
買いルールではない(買いに使うにはZ1-2の予測が必要)。

■ 事前登録
これは記述的解剖+オラクル天井の測定であり、マスの発見を即採用しない。
天井の判定基準: 「人気沈没が事前に分かる」オラクル条件で上位K点買いが
回収率150%を超える場合のみ「Z1-2を作る価値あり」と結論する
(C勝万舟オラクル検証(105%)と同じ物差し)。
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

KM = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "抜き", 6: "恵まれ"}

conn = db.connect(DB_PATH)
df = build_training_set(conn)
res_all = defaultdict(dict)     # rid -> lane -> (arrival, course)
for rid, lane, ao, course in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order, res.course FROM results res "
    "JOIN races r ON r.race_id = res.race_id WHERE r.date >= ?", (TEST_START,)):
    res_all[rid][lane] = (ao, course)
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
        rr = res_all.get(rid, {})
        arr = {lane: v[0] for lane, v in rr.items() if v[0]}
        if len(arr) < 3 or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        top_raw = ranked[0]["prob"]
        if not (0.20 <= top_raw < 0.35):
            continue
        lanes = [r["lane"] for r in ranked]
        rank_of = {lane: i + 1 for i, lane in enumerate(lanes)}   # lane -> モデル順位
        top3 = sorted(arr, key=lambda l: arr[l])[:3]
        fav = lanes[0]
        fav_arr = arr.get(fav)                                    # None=失格等
        fav_course = rr.get(fav, (None, None))[1]
        win_lane = top3[0]
        ctxs.append({
            "rid": rid, "lanes": lanes, "rank_of": rank_of,
            "fav": fav, "fav_arr": fav_arr, "fav_course": fav_course,
            "top3": top3, "tech": race_tech.get(rid),
            "win_course": rr.get(win_lane, (None, None))[1],
            "in5": int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES,
        })

n = len(ctxs)
sinks = [c for c in ctxs if c["fav_arr"] is None or c["fav_arr"] >= 4]
print(f"\n本命帯(全場・20〜35%): {n:,}R / 一番人気4着以下(沈没) {len(sinks):,}R "
      f"({len(sinks)/n:.1%})")
print(f"一番人気が1号艇のレース: {sum(1 for c in ctxs if c['fav']==1)/n:.1%}")

# ---- 1. どんな展開で沈むか --------------------------------------------------
print(f"\n=== 1. 一番人気が沈む展開(決まり手の分布) ===")
print(f"{'':<12}{'全レース':>10}{'沈没レース':>10}{'沈没率':>8}")
tech_all = Counter(c["tech"] for c in ctxs if c["tech"])
tech_sink = Counter(c["tech"] for c in sinks if c["tech"])
for k in sorted(tech_all):
    a, s = tech_all[k], tech_sink.get(k, 0)
    print(f"{KM[k]:<12}{a:>9,}R{s:>9,}R{s/a:>8.1%}")

print(f"\n--- 一番人気の枠×コース別の沈没率 ---")
for label, cond in (("人気=1号艇(イン)", lambda c: c["fav"] == 1),
                    ("人気=2-3号艇", lambda c: c["fav"] in (2, 3)),
                    ("人気=4-6号艇(外)", lambda c: c["fav"] >= 4)):
    grp = [c for c in ctxs if cond(c)]
    gs = [c for c in grp if c["fav_arr"] is None or c["fav_arr"] >= 4]
    if grp:
        t = Counter(c["tech"] for c in gs if c["tech"])
        top2 = "・".join(f"{KM[k]}{v/max(1,len(gs)):.0%}" for k, v in t.most_common(2))
        print(f"  {label:<16} {len(grp):>5,}R 沈没率{len(gs)/len(grp):>6.1%} "
              f"(沈没時の決まり手上位: {top2})")

# ---- 2. 沈没時の3連単の形 ---------------------------------------------------
print(f"\n=== 2. 沈没時の3連単の形(モデル順位空間・上位12パターン) ===")
pat = Counter()
pat_pay = defaultdict(list)
for c in sinks:
    key = "-".join(f"r{c['rank_of'][l]}" for l in c["top3"])
    pat[key] += 1
    comb = "-".join(str(l) for l in c["top3"])
    amt = payout_map[c["rid"]].get(("3連単", comb), 0)
    pat_pay[key].append(amt)
print(f"{'形(モデル順位)':<14}{'回数':>6}{'沈没内比率':>9}{'平均3連単配当':>12}")
for key, cnt in pat.most_common(12):
    avg = sum(pat_pay[key]) / len(pat_pay[key])
    print(f"{key:<14}{cnt:>6}{cnt/len(sinks):>9.1%}{avg:>11,.0f}円")
cover5 = sum(cnt for _k, cnt in pat.most_common(5))
print(f"上位5形のカバー率: {cover5/len(sinks):.1%} / "
      f"r6絡み: {sum(v for k,v in pat.items() if 'r6' in k)/len(sinks):.1%}")

print(f"\n--- 決まり手別の形の違い(ケンさんの仮説の検証) ---")
for k in (3, 2, 4):
    grp = [c for c in sinks if c["tech"] == k]
    if len(grp) < 30:
        continue
    p = Counter("-".join(f"r{c['rank_of'][l]}" for l in c["top3"]) for c in grp)
    top3p = " / ".join(f"{key}({cnt/len(grp):.0%})" for key, cnt in p.most_common(3))
    wc = Counter(c["win_course"] for c in grp if c["win_course"])
    wctop = "・".join(f"{c_}コース{v/len(grp):.0%}" for c_, v in wc.most_common(2))
    print(f"  {KM[k]:<8}({len(grp):,}R) 形上位: {top3p}")
    print(f"  {'':<8} 勝者コース: {wctop}")

# ---- 3. オラクル天井 --------------------------------------------------------
print(f"\n=== 3. オラクル天井(「沈没が事前に分かる」前提で沈没レースだけ買う) ===")
print(f"※実際には沈没は事前に分からない(Z1-2の課題)。これは上限の見積もり")
for K in (1, 3, 5):
    tops = [k for k, _ in pat.most_common(K)]
    st = rt = hit = 0
    for c in sinks:
        for key in tops:
            rks = [int(x[1:]) for x in key.split("-")]
            try:
                comb = "-".join(str(c["lanes"][rk - 1]) for rk in rks)
            except IndexError:
                continue
            st += 100
            got = payout_map[c["rid"]].get(("3連単", comb), 0)
            rt += got
            if got:
                hit += 1
    print(f"  上位{K}形を各100円: 的中{hit:>4}本 回収率{rt/st:>7.1%}"
          f"{'  ←判定対象' if K==5 else ''}")

# 比較: 同じオラクル条件で保険複r2r3r4を買ったら
st = rt = hit = 0
for c in sinks:
    trio_l = sorted(c["lanes"][1:4])
    comb = f"{trio_l[0]}={trio_l[1]}={trio_l[2]}"
    st += 100
    got = payout_map[c["rid"]].get(("3連複", comb), 0)
    rt += got
    if got:
        hit += 1
print(f"  (比較)同条件で保険複r2r3r4: 的中{hit:>4}本 回収率{rt/st:>7.1%}")

# オラクルなし(全レース買い)だと形は勝てるか
print(f"\n--- オラクルなし(本命帯全レースで機械買い)の場合 ---")
for K in (1, 5):
    tops = [k for k, _ in pat.most_common(K)]
    st = rt = 0
    for c in ctxs:
        for key in tops:
            rks = [int(x[1:]) for x in key.split("-")]
            try:
                comb = "-".join(str(c["lanes"][rk - 1]) for rk in rks)
            except IndexError:
                continue
            st += 100
            rt += payout_map[c["rid"]].get(("3連単", comb), 0)
    print(f"  上位{K}形を全{n:,}Rで各100円: 回収率{rt/st:>7.1%}")

verdict_roi = None
tops5 = [k for k, _ in pat.most_common(5)]
st = rt = 0
for c in sinks:
    for key in tops5:
        rks = [int(x[1:]) for x in key.split("-")]
        try:
            comb = "-".join(str(c["lanes"][rk - 1]) for rk in rks)
        except IndexError:
            continue
        st += 100
        rt += payout_map[c["rid"]].get(("3連単", comb), 0)
verdict_roi = rt / st if st else 0
print(f"\n===== 事前登録基準の判定 =====")
print(f"  オラクル×上位5形 = {verdict_roi:.1%} "
      f"{'>= 150% → Z1-2(展開予測)を作る価値あり' if verdict_roi >= 1.5 else '< 150% → 展開予測でもこの帯の狙い撃ちは天井が低い'}")
