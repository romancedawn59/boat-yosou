# -*- coding: utf-8 -*-
"""検証㉑: 3着固定フォーメーション([◎,○]-[◎,○]-3着固定)の追加セル評価(2026-08-09ケンさん発案)

    py -X utf8 test/verify_third_fixed_formation.py

■ ケンさんの案
「1着予想=2着予想-3着予想」= ◎○の1-2着入れ替え2点に、3着席を固定した3連単。
3着席に「勝ち切らないが絡む」艇(丸亀2R三宅型)を置ければ、⑲で見つけた
2着/3着専用モデルの本命帯シグナルの換金口になり得る。

■ 事前登録(結果を見る前に固定)
- 追加セル(現行プランに+2点×100円=+200円。既存の構成・金額・選別は不変):
  X1: 3連単 ◎-○-(3着席) と ○-◎-(3着席)。3着席=3着専用モデルの
      スコアが最大の艇(◎○を除く)
  X2(対照): 同形で3着席=現行▲(1着率3位)。X1がX2に勝てなければ
      「専用モデルの選抜」に価値はなく形自体の問題になる
- 帯: 本命(5場×20〜30%・⑰に追加)と超混戦(全場×20%未満・⑬に追加)
- 月次学習8か月(2025-12〜2026-07)。3着モデルは⑲⑳と同一の学習方法
- 判定: X1の限界効率(追加200円だけのROI)が150%以上 かつ X2を上回る場合のみ
  「9/1候補・8月前向き紙上並走」へ。即採用はしない(同一データ3切り目のため)。
  それ以外は見送りを正直に報告
"""
import sys
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
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

df["is_third"] = (df["arrival_order"] == 3).astype(int)


def train_third(train_df):
    ds = lgb.Dataset(train_df[FEATURE_COLUMNS], label=train_df["is_third"],
                     categorical_feature=CATEGORICAL_FEATURES)
    return lgb.train(
        {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
         "min_data_in_leaf": 50, "feature_fraction": 0.9, "verbosity": -1,
         "seed": 7},
        ds, num_boost_round=400)


ARMS = ("X1 3着=専用モデル選抜", "X2 3着=現行▲(対照)")
# {(band, arm): [追加st, 追加rt, 追加的中, 3着席が▲と別の艇だった回数]}
agg = defaultdict(lambda: [0, 0, 0, 0])
base_agg = defaultdict(lambda: [0, 0])   # {(band): 現行プランの[st, rt]}(参考)
n_band = defaultdict(int)


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
         for ms in ((r1, r2, r3), (r1, r2, r4)) for a, b, c in permutations(ms)]
    return p + [("3連単", f"{r3}-{r1}-{r2}", 300),
                ("3連単", f"{r4}-{r1}-{r2}", 300),
                ("3連複", trio(r3, r4, r5), 200)]


for m in MONTHS:
    tr = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習...", flush=True)
    b_win = train_fold(tr)
    b_3rd = train_third(tr)
    month_df["p1"] = b_win.predict(month_df[FEATURE_COLUMNS])
    month_df["p3m"] = b_3rd.predict(month_df[FEATURE_COLUMNS])
    for rid, g in month_df.groupby("race_id"):
        pay = payout_map[rid]
        if not pay:
            continue
        gs = g.sort_values("p1", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        if len(lanes) < 5:
            continue
        p1 = float(gs["p1"].iloc[0])
        venue = int(g["venue_code"].iloc[0])
        band = ("超混戦" if p1 < 0.20
                else "本命" if p1 < 0.30 and venue in TARGET_VENUE_CODES
                else None)
        if band is None:
            continue
        n_band[band] += 1
        r1, r2 = lanes[0], lanes[1]
        p3m = dict(zip(g["lane"].astype(int), g["p3m"]))
        third_model = max((l for l in lanes if l not in (r1, r2)),
                          key=lambda l: p3m[l])
        third_rank = lanes[2]
        # 現行プラン(参考のベース)
        make = plan13 if band == "超混戦" else plan17
        merged = defaultdict(int)
        for bt, comb, y in make(lanes):
            merged[(bt, comb)] += y
        b = base_agg[band]
        b[0] += sum(merged.values())
        b[1] += sum(pay.get(k, 0) * y // 100 for k, y in merged.items())
        # 追加セル(既存プランと重複する目は「追加の100円」として加算=傾斜扱い)
        for arm, third in ((ARMS[0], third_model), (ARMS[1], third_rank)):
            a = agg[(band, arm)]
            hit = False
            for h1, h2 in ((r1, r2), (r2, r1)):
                comb = f"{h1}-{h2}-{third}"
                a[0] += 100
                got = pay.get(("3連単", comb), 0)
                a[1] += got
                hit = hit or got > 0
            a[2] += 1 if hit else 0
            a[3] += 1 if third != third_rank else 0

print("\n===== 検証㉑: 3着固定フォーメーション(追加2点×100円の限界効率) =====")
for band in ("本命", "超混戦"):
    if n_band[band] == 0:
        continue
    bs, br = base_agg[band]
    print(f"\n[{band}] {n_band[band]:,}R (参考: 現行プラン回収率{br/bs:.1%})")
    for arm in ARMS:
        st, rt, hits, diff = agg[(band, arm)]
        avg = rt / hits if hits else 0
        note = (f" 3着席が▲と別の艇: {diff/n_band[band]:.0%}"
                if arm == ARMS[0] else "")
        print(f"  {arm:<18} 追加{st:,}円 回収{rt:,}円 限界効率{rt/st:>7.1%} "
              f"的中率{hits/n_band[band]:>5.1%} 的中時平均{avg:>8,.0f}円{note}")

print("\n===== 事前登録判定 =====")
for band in ("本命", "超混戦"):
    x1 = agg[(band, ARMS[0])]
    x2 = agg[(band, ARMS[1])]
    roi1 = x1[1] / x1[0] if x1[0] else 0
    roi2 = x2[1] / x2[0] if x2[0] else 0
    ok = roi1 >= 1.5 and roi1 > roi2
    print(f"  {band}: X1={roi1:.1%} X2={roi2:.1%} → "
          + ("基準クリア → 9/1候補・8月前向き紙上並走へ" if ok
             else "基準未達 → 見送り"))
