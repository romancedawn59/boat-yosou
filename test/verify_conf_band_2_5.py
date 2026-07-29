# -*- coding: utf-8 -*-
"""自信2〜5%帯(想定15〜37倍)の分解検証(2026-07-29ケンさん「この数字化け物では?」)

    py -X utf8 test/verify_conf_band_2_5.py

較正表で2-5%帯が本命帯240.8%/超混戦帯736.3%と出た。
①再計算(独立に集計し直す) ②どのスロットが帯を構成しているか
③的中の中身(平均払戻・フェア比=ミスプライシングの倍率)
④実装可能な買い方「④⑤単を自信2-5%のときだけ買う」の成績
⑤勝ちの貢献分解: 月次・最大的中への依存度(上位N本除外)
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
        if len(ranked) < 5:
            continue
        ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]),
                     "top": ranked[0]["prob"],
                     "probs": P.normalize_probs(ranked),
                     "ranked": ranked})


def v2_slots(c):
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]
    tri = P.trifecta_probs(c["probs"])

    def tp(a, b, x):
        s = {a, b, x}
        return sum(p for k, p in tri.items() if set(k) == s)

    def key(a, b, x):
        y = sorted([a, b, x])
        return f"{y[0]}={y[1]}={y[2]}"

    out = [
        ("①複123", "3連複", key(r1, r2, r3), 200, tp(r1, r2, r3)),
        ("②複124", "3連複", key(r1, r2, r4), 200, tp(r1, r2, r4)),
        ("③複134", "3連複", key(r1, r3, r4), 100, tp(r1, r3, r4)),
        ("④単312", "3連単", f"{r3}-{r1}-{r2}", 200, tri.get((r3, r1, r2), 0.0)),
        ("⑤単412", "3連単", f"{r4}-{r1}-{r2}", 200, tri.get((r4, r1, r2), 0.0)),
    ]
    for bt, comb, p in P.picks_katsu(c["probs"]):
        if (bt, comb) not in {(b, cb) for _l, b, cb, _y, _p in out}:
            out.append(("⑥C万舟", bt, comb, 100, p))
            break
    return out


for scope_name, lo, hi in (("本命帯(20-35%)", 0.20, 0.35),
                           ("超混戦帯(<20%)", 0.0, 0.20)):
    sel = [c for c in ctxs if lo <= c["top"] < hi]
    print(f"\n{'='*70}\n=== {scope_name}: {len(sel):,}R ===")

    # ①② 2-5%帯の再集計とスロット構成
    band = []   # (slot, bt, comb, yen, prob, got, date, rid)
    for c in sel:
        pay = payout_map[c["rid"]]
        for label, bt, comb, yen, prob in v2_slots(c):
            if 0.02 <= prob < 0.05:
                got = pay.get((bt, comb), 0) * yen // 100
                band.append((label, bt, yen, prob, got, c["date"], c["rid"]))
    n = len(band)
    st = sum(b[2] for b in band)
    rt = sum(b[4] for b in band)
    hits = [b for b in band if b[4] > 0]
    print(f"① 再計算: {n:,}点 投資{st:,}円 回収{rt:,}円 回収率{rt/st:.1%} "
          f"的中{len(hits)}本({len(hits)/n:.2%})")
    print("② スロット構成:")
    comp = defaultdict(lambda: [0, 0, 0, 0])
    for label, bt, yen, prob, got, d, rid in band:
        s = comp[label]
        s[0] += 1
        s[1] += yen
        s[2] += got
        if got:
            s[3] += 1
    for label, (cnt, stake, ret, h) in sorted(comp.items()):
        print(f"   {label}: {cnt:,}点 的中{h} 回収率{ret/stake:.1%}")

    # ③ ミスプライシング倍率
    if hits:
        fair = [0.75 / b[3] for b in hits]                 # フェア想定倍率
        act = [b[4] / b[2] * 1 for b in hits]              # 実際の倍率(=払戻/賭金)
        avg_fair = sum(fair) / len(fair)
        avg_act = sum(act) / len(act)
        print(f"③ 的中{len(hits)}本の平均: フェア想定{avg_fair:.0f}倍 → "
              f"実際{avg_act:.0f}倍 (市場の過小評価倍率 {avg_act/avg_fair:.2f}x)")

    # ⑤ 依存度と月次
    pays = sorted((b[4] for b in hits), reverse=True)
    if pays and rt:
        for k in (1, 3, 5):
            if len(pays) > k:
                r_ex = (rt - sum(pays[:k])) / st
                print(f"⑤ 上位{k}本除外後の回収率: {r_ex:.1%}")
        monthly = defaultdict(lambda: [0, 0])
        for label, bt, yen, prob, got, d, rid in band:
            monthly[d[:7]][0] += yen
            monthly[d[:7]][1] += got
        cells = [f"{m[-2:]}月{v[1]/v[0]:.0%}" for m, v in sorted(monthly.items())]
        print("   月次: " + "  ".join(cells))

    # ④ 実装可能な買い方: ④⑤単を自信2-5%のときだけ200円
    st2 = rt2 = n2 = h2 = 0
    monthly2 = defaultdict(lambda: [0, 0])
    for c in sel:
        pay = payout_map[c["rid"]]
        for label, bt, comb, yen, prob in v2_slots(c):
            if label in ("④単312", "⑤単412") and 0.02 <= prob < 0.05:
                got = pay.get((bt, comb), 0) * yen // 100
                st2 += yen
                rt2 += got
                n2 += 1
                if got:
                    h2 += 1
                monthly2[c["date"][:7]][0] += yen
                monthly2[c["date"][:7]][1] += got
    if st2:
        print(f"④ 実装形「④⑤単×自信2-5%のみ」: {n2:,}点 投資{st2:,}円 "
              f"回収{rt2:,}円 回収率{rt2/st2:.1%} 的中{h2}本")
        cells = [f"{m[-2:]}月{v[1]/v[0]:.0%}" for m, v in sorted(monthly2.items())
                 if v[0]]
        print("   月次: " + "  ".join(cells))
