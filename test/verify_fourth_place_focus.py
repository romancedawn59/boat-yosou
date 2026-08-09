# -*- coding: utf-8 -*-
"""検証⑳: 「4着まで重要視」の仕組み変更の判定+4着モデルの候補精査(2026-08-09ケンさん指示)

    py -X utf8 test/verify_fourth_place_focus.py

■ 背景(ケンさんの指示)
超接戦(⑬)や本命1,400円(⑰)の買い目は上位4艇(◎○▲△)+⑬は5艇目まで使う。
現在の◎○▲△は「1着率の順」で決めており、丸亀2Rの三宅型(勝ち切らないが絡む)が
買い目から漏れる。→「4着までの絡み力」で席次を決める仕組みに変更できるか。

■ 事前登録(結果を見る前に固定)
- 追加学習: 2着/3着/4着の専用モデル3本(特徴量28本・ラベルのみ変更)
- 席次候補(スロット割当のみ変更・構成/金額/選別は完全に不変):
  A) 現行: 1着率の降順で◎○▲△(基準)
  B) R4a: ◎=1着率1位固定、○▲△(⑬は5艇目も)=残りから「絡みスコア
     (P2着+P3着+P4着)」の降順
  C) R4b: ◎○=1着率1-2位固定(差され頭の攻撃力を保持)、▲△以下=絡みスコア順
- バックテスト: 月次学習8か月(2025-12〜2026-07)。帯=本番同様
  (超混戦=全場×1位20%未満に⑬2,000円 / 本命=5場×20〜30%に⑰1,400円)
- 採用基準(帯ごと): 回収率が現行+5pt以上 かつ 8か月中5か月以上で現行以上
  かつ 最低月悪化なし。満たした場合のみ本番実装に進む。それ以外は現状維持
- 併記(相談パート): ①算出可否 ②上位4艇セットの的中率(現行vs絡みスコア)
  ③各案の的中時平均払戻 ④追加学習・推論の実測負荷
"""
import sys
import time
from collections import defaultdict
from itertools import permutations

import lightgbm as lgb

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set

MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07"]

conn = db.connect(DB_PATH)
df = build_training_set(conn)
res_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id WHERE r.date >= '2025-12-01'"):
    res_all[rid][lane] = ao
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

for pos, col in ((2, "is_second"), (3, "is_third"), (4, "is_fourth")):
    df[col] = (df["arrival_order"] == pos).astype(int)


def train_pos(train_df, label):
    ds = lgb.Dataset(train_df[FEATURE_COLUMNS], label=train_df[label],
                     categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(
        {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
         "min_data_in_leaf": 50, "feature_fraction": 0.9, "verbosity": -1,
         "seed": 7},
        ds, num_boost_round=400)


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan17(l):
    r1, r2, r3, r4 = l[:4]
    return [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
            ("3連複", trio(r1, r3, r4), 100),
            ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200),
            ("3連複", trio(r2, r3, r4), 100),
            ("3連単", f"{r3}-{r2}-{r1}", 100), ("3連単", f"{r4}-{r2}-{r1}", 100),
            ("3連単", f"{r4}-{r2}-{r1}", 200)]


def plan13(l):
    r1, r2, r3, r4, r5 = l[:5]
    p = [("3連単", f"{a}-{b}-{c}", 100)
         for mset in ((r1, r2, r3), (r1, r2, r4)) for a, b, c in permutations(mset)]
    return p + [("3連単", f"{r3}-{r1}-{r2}", 300),
                ("3連単", f"{r4}-{r1}-{r2}", 300),
                ("3連複", trio(r3, r4, r5), 200)]


ARMS = ("A 現行(1着率順)", "B R4a(◎固定+絡み順)", "C R4b(◎○固定+絡み順)")
agg = defaultdict(lambda: [0, 0])       # {(band, arm, month): [st, rt]}
hitpay = defaultdict(lambda: [0, 0])    # {(band, arm): [hits, ret_when_hit]}
top4_match = defaultdict(lambda: [0, 0, 0])   # {arm: [完全一致, 一致艇数計, n]}
train_sec = 0.0
n234 = 0

for m in MONTHS:
    tr = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習4本...", flush=True)
    booster = train_fold(tr)
    t0 = time.perf_counter()
    b2 = train_pos(tr, "is_second")
    b3 = train_pos(tr, "is_third")
    b4 = train_pos(tr, "is_fourth")
    train_sec += time.perf_counter() - t0
    month_df["p1"] = booster.predict(month_df[FEATURE_COLUMNS])
    month_df["sc"] = (b2.predict(month_df[FEATURE_COLUMNS])
                      + b3.predict(month_df[FEATURE_COLUMNS])
                      + b4.predict(month_df[FEATURE_COLUMNS]))
    for rid, g in month_df.groupby("race_id"):
        pay = payout_map[rid]
        gs = g.sort_values("p1", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        if len(lanes) < 5:
            continue
        p1 = float(gs["p1"].iloc[0])
        venue = int(g["venue_code"].iloc[0])
        sc = dict(zip(g["lane"].astype(int), g["sc"]))
        orders = {
            "A 現行(1着率順)": lanes,
            "B R4a(◎固定+絡み順)":
                [lanes[0]] + sorted(lanes[1:], key=lambda l: -sc[l]),
            "C R4b(◎○固定+絡み順)":
                lanes[:2] + sorted(lanes[2:], key=lambda l: -sc[l]),
        }
        # 相談②: 上位4艇セットの的中(全レース・着順確定分)
        arr = {l: a for l, a in res_all.get(rid, {}).items() if a}
        if len(arr) >= 4:
            actual4 = set(sorted(arr, key=lambda l: arr[l])[:4])
            n234 += 1
            for arm, o in orders.items():
                inter = len(actual4 & set(o[:4]))
                t = top4_match[arm]
                t[0] += inter == 4
                t[1] += inter
                t[2] += 1
        if not pay:
            continue
        band = ("超混戦" if p1 < 0.20
                else "本命" if p1 < 0.30 and venue in TARGET_VENUE_CODES
                else None)
        if band is None:
            continue
        make = plan13 if band == "超混戦" else plan17
        for arm, o in orders.items():
            merged = defaultdict(int)
            for bt, comb, y in make(o):
                merged[(bt, comb)] += y
            st = sum(merged.values())
            rt = sum(pay.get(k, 0) * y // 100 for k, y in merged.items())
            a = agg[(band, arm, m)]
            a[0] += st
            a[1] += rt
            h = hitpay[(band, arm)]
            if rt:
                h[0] += 1
                h[1] += rt

print("\n===== ①算出可否 =====")
print("可能(2着/3着/4着モデルの追加学習のみ・本番コード変更なしで席次を再計算できる)")

print("\n===== ②上位4艇セットの的中率(全レース) =====")
for arm in ARMS:
    t = top4_match[arm]
    print(f"  {arm:<22} 4艇完全一致{t[0]/t[2]:>6.1%} 平均一致{t[1]/t[2]:.2f}艇")

for band in ("超混戦", "本命"):
    print(f"\n===== 帯別バックテスト: {band} =====")
    print(f"{'月':<9}" + "".join(f"{a[:12]:<15}" for a in ARMS))
    for m in MONTHS:
        row = f"{m:<9}"
        for arm in ARMS:
            st, rt = agg[(band, arm, m)]
            row += f"{(rt/st if st else 0):>12.1%}   "
        print(row)
    base_roi = None
    for arm in ARMS:
        st = sum(agg[(band, arm, m)][0] for m in MONTHS)
        rt = sum(agg[(band, arm, m)][1] for m in MONTHS)
        months_ok = [m for m in MONTHS if agg[(band, arm, m)][0]]
        low = min(agg[(band, arm, m)][1] / agg[(band, arm, m)][0]
                  for m in months_ok)
        wins = sum(
            1 for m in months_ok
            if (agg[(band, arm, m)][1] / agg[(band, arm, m)][0])
            >= (agg[(band, ARMS[0], m)][1] / agg[(band, ARMS[0], m)][0]))
        h, hr = hitpay[(band, arm)]
        roi = rt / st if st else 0
        if arm == ARMS[0]:
            base_roi = roi
        print(f"  {arm:<22} 回収率{roi:>7.1%} 損益{rt-st:>+10,}円 "
              f"最低月{low:>6.1%} 現行以上の月{wins}/8 "
              f"的中時平均払戻{(hr/h if h else 0):>8,.0f}円")
    print(f"  → 事前登録基準(+5pt/過半月/最低月維持)の判定は上記から機械的に判定")

print("\n===== ④演算負荷 =====")
print(f"追加学習(3モデル×8か月): {train_sec:.0f}秒(月次再学習+{train_sec/8:.0f}秒/月)")
print("推論は⑲実測で1日約0.01秒・トークン消費ゼロ → 開催全レース適用は負荷面で問題なし")
