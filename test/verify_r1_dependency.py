# -*- coding: utf-8 -*-
"""r1依存の検証(2026-07-29ケンさん発案)+超混戦配分変種の月次安定性

    py -X utf8 test/verify_r1_dependency.py

問い(ケンさん):
「この買い方はr1(≒1号艇/1番人気)がちょいミスして2-3着に残る前提。
 r1が圏外に飛んだら話にならない。①本命レースでr1がちゃんと入っていたか、
 ②飛んだとき何が来ていたか(大穴の精度を高められる形か)、
 ③その結果は人気側(織り込み済み)か大穴かを調べたい」

人気の代理指標は3連単の払戻金額(=確定オッズ×100円)を使う。
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
        ctxs.append({"rid": rid, "date": str(g["date"].iloc[0]),
                     "top": ranked[0]["prob"], "ranked": ranked, "probs": probs})

PAY_BANDS = [("〜20倍(人気側=織り込み済)", 0, 2000),
             ("20〜55倍(中穴)", 2000, 5500),
             ("55〜150倍(中大穴)", 5500, 15000),
             ("150倍〜(大穴)", 15000, 10**9)]


def analyze_r1(scope_name, lo, hi):
    sel = [c for c in ctxs if lo <= c["top"] < hi]
    n = len(sel)
    if not n:
        return
    r1_in = r1_win = r1_second = r1_third = r1_out = 0
    r1_is_lane1 = 0
    miss_trio_pattern = defaultdict(int)
    miss_pay = defaultdict(int)
    in_pay = defaultdict(int)
    for c in sel:
        res = actual[c["rid"]]
        top3 = {res.get(1), res.get(2), res.get(3)}
        lanes = [r["lane"] for r in c["ranked"]]
        r1, r2, r3, r4, r5 = lanes[:5]
        r6 = lanes[5] if len(lanes) >= 6 else None
        if r1 == 1:
            r1_is_lane1 += 1
        san = payout_map[c["rid"]].get(
            ("3連単", f"{res.get(1)}-{res.get(2)}-{res.get(3)}"), 0)
        band = next(lbl for lbl, blo, bhi in PAY_BANDS if blo <= san < bhi)
        if r1 in top3:
            r1_in += 1
            if res.get(1) == r1:
                r1_win += 1
            elif res.get(2) == r1:
                r1_second += 1
            else:
                r1_third += 1
            in_pay[band] += 1
        else:
            r1_out += 1
            miss_pay[band] += 1
            others = top3
            named = {r2, r3, r4, r5}
            inter = len(others & named)
            if others <= {r2, r3, r4}:
                key = "r2r3r4(軸外しDで拾える)"
            elif others <= {r3, r4, r5}:
                key = "r3r4r5(深い波乱Gで拾える)"
            elif others <= named:
                key = "r2〜r5の他の組(現構成外)"
            elif r6 in others and inter == 2:
                key = "r6絡み(モデル最下位が入着)"
            else:
                key = "その他"
            miss_trio_pattern[key] += 1

    print(f"\n=== {scope_name}({n:,}R) ===")
    print(f"  モデル1位がそのまま1号艇: {r1_is_lane1/n:.1%}")
    print(f"  r1が3着以内: {r1_in/n:.1%} (内訳: 1着{r1_win/n:.1%} "
          f"2着{r1_second/n:.1%} 3着{r1_third/n:.1%})")
    print(f"  r1圏外(構成のC以外全滅): {r1_out/n:.1%}")
    print(f"  --- r1圏外時({r1_out}R)、何が来たか ---")
    for k, v in sorted(miss_trio_pattern.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}R ({v/max(1,r1_out):.1%})")
    print(f"  --- 結果の人気帯(3連単配当=人気の代理) ---")
    print(f"  {'帯':<24}{'r1圏内':>10}{'r1圏外':>10}")
    for lbl, _, _ in PAY_BANDS:
        i_, o_ = in_pay[lbl], miss_pay[lbl]
        print(f"  {lbl:<24}{i_/max(1,r1_in):>10.1%}{o_/max(1,r1_out):>10.1%}")


analyze_r1("本命帯(1位勝率20-35%)", 0.20, 0.35)
analyze_r1("超混戦帯(1位勝率20%未満)", 0.0, 0.20)

# --- 超混戦Q案変種の月次安定性 ---
print("\n\n=== 超混戦Q案変種の月次安定性 ===")
VARIANTS = {
    "現行Q案": {"A": 200, "B": 100, "C": 100, "D": 100, "E": 200, "F": 200, "G": 100},
    "案1拾える複厚": {"A": 300, "B": 200, "C": 0, "D": 0, "E": 200, "F": 200, "G": 100},
}


def q_slots(c):
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4, r5 = lanes[:5]

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"

    return [
        ("A", "3連複", trio(r1, r2, r3)), ("B", "3連複", trio(r1, r2, r4)),
        ("C", "3連複", trio(r1, r3, r4)), ("D", "3連複", trio(r2, r3, r4)),
        ("E", "3連単", f"{r3}-{r1}-{r2}"), ("F", "3連単", f"{r4}-{r1}-{r2}"),
        ("G", "3連複", trio(r3, r4, r5)),
    ]


konsen = [c for c in ctxs if c["top"] < 0.20]
months = sorted({c["date"][:7] for c in konsen})
print(f"{'月':<10}" + "".join(f"{v:>16}" for v in VARIANTS))
for m in months:
    sel = [c for c in konsen if c["date"][:7] == m]
    cells = []
    for vname, w in VARIANTS.items():
        st = rt = 0
        for c in sel:
            pay = payout_map[c["rid"]]
            for key, bt, comb in q_slots(c):
                yen = w[key]
                if yen:
                    st += yen
                    rt += pay.get((bt, comb), 0) * yen // 100
        cells.append(f"{rt/st:>10.1%}({len(sel):>3}R)" if st else "—")
    print(f"{m:<10}" + "".join(f"{x:>16}" for x in cells))
