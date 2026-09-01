# -*- coding: utf-8 -*-
"""消し予報マップの買い方3種(2026-09-02ケンさん指示・真白ゆい案の実装)

    py -X utf8 test/sim_kesi_map_2026.py

■ 背景(発見の経緯)
超混戦213Rは「予想5・6位が3着以内に絡んだか」で階段状に分かれる:
両方圏外81R(軸生存61.7%・中央値2,620円) / 5位のみ74R / 6位のみ44R /
両方絡む14R(軸生存0%・万舟64%)。これをレース前に予報して買い分ける。
消し予報 = 3着内モデルの下位2艇(専用順位5・6位)のP(3着内)。

■ 事前登録(結果を見る前に固定)
- 順位は専用順位(3着内モデル)。ゲート閾値は「その時点までに見た超混戦の
  過去分布」の33/67パーセンタイル(先読みなし・30R未満の助走期間は全戦略見送り)
- 戦略1 消し自信・狭く厚く: s=下位2艇のP(3着内)の和がq33以下の日のみ
  2連複 t1=t2 500円 + 3連複 t1t2t3 300円 + t1t2t4 200円 (計1,000円)
- 戦略2 滑り込み穴: 下位2艇のP(3着内)の最大値がq67以上の日のみ、その艇bを
  3着に置く: 3連複 t1t2b 400円 + 3連単 t1-t2-b 300円 + t2-t1-b 300円 (計1,000円)
- 戦略3 混沌回避⑬: 専用順位⑬2,000円。ただしs≥q67(両方怪しい)または
  1位生値<17%(深い混沌)の日は見送り
- 参考基準: 現行順位⑬を全レース(2,000円)
- 注意: 階段の発見と同一期間での検証=設計確認。採否は9月の前方データで判定
■ 方法: 月次walk-forward(2025-12は助走)・全entriesランク・返還処理あり
"""
import sys
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

MONTHS_ALL = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
              "2026-05", "2026-06", "2026-07", "2026-08"]
EVAL_MONTHS = MONTHS_ALL[1:]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}
WARMUP = 30

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
train_all["is_top3"] = (train_all["arrival_order"] <= 3).astype(int)
eval_df = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2025-12-01'
""", conn)
eval_df = _encode(eval_df)
eval_df = eval_df.merge(compute_form_features(conn), on=["race_id", "lane"],
                        how="left")
eval_df = _attach_extra_features(eval_df, conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()


def train_lgb(train_df, label):
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    ds = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr[label],
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(va[FEATURE_COLUMNS], label=va[label], reference=ds)
    return lgb.train(PARAMS, ds, valid_sets=[vs], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def two(a, b):
    return f"{min(a, b)}={max(a, b)}"


def plan13(lanes):
    r1, r2, r3, r4, r5 = lanes[:5]
    bets = [("3連単", f"{a}-{b}-{c}", 100)
            for a, b, c in permutations((r1, r2, r3))]
    bets += [("3連単", f"{a}-{b}-{c}", 100)
             for a, b, c in permutations((r1, r2, r4))]
    bets += [("3連単", f"{r3}-{r1}-{r2}", 300),
             ("3連単", f"{r4}-{r1}-{r2}", 300),
             ("3連複", trio(r3, r4, r5), 200)]
    return bets


def score(bets, pay, refund):
    merged = defaultdict(int)
    for bt, comb, y in bets:
        merged[(bt, comb)] += y
    st = rt = 0
    for (bt, comb), y in merged.items():
        sep = "-" if bt in ("3連単", "2連単") else "="
        members = {int(x) for x in comb.split(sep)}
        st += y
        rt += y if members & refund else pay.get((bt, comb), 0) * y // 100
    return st, rt


ARMS = ("戦略1 消し自信", "戦略2 滑り込み穴", "戦略3 混沌回避⑬", "基準 現行⑬")
agg = defaultdict(lambda: [0, 0, 0, 0])   # {(戦略, 月): [st, rt, n, hit]}
s_hist, bmax_hist = [], []

for m in MONTHS_ALL:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行 ×2モデル", flush=True)
    win_model = train_lgb(tr_df, "is_winner")
    top3_model = train_lgb(tr_df, "is_top3")
    md = ev.copy()
    md["p_win"] = win_model.predict(md[FEATURE_COLUMNS])
    md["p_top3"] = top3_model.predict(md[FEATURE_COLUMNS])
    for rid in sorted(md["race_id"].unique()):
        grp = md[md["race_id"] == rid]
        pay = payout_map[rid]
        if not pay:
            continue
        gw = grp.sort_values("p_win", ascending=False)
        p1 = float(gw["p_win"].iloc[0])
        if p1 >= 0.20 or len(gw) < 5:
            continue
        arr = {int(r["lane"]): r["arrival_order"] for _, r in grp.iterrows()
               if pd.notna(r["arrival_order"])}
        if len(arr) < 3:
            continue
        nonfin = {int(r["lane"]) for _, r in grp.iterrows()
                  if pd.isna(r["arrival_order"])}
        refund = {l for l in nonfin
                  if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
        gt = grp.sort_values("p_top3", ascending=False)
        t = [int(x) for x in gt["lane"]]
        pt = [float(x) for x in gt["p_top3"]]
        lanes_w = [int(x) for x in gw["lane"]]

        s = pt[4] + pt[5] if len(pt) >= 6 else pt[4]
        bmax_lane = (t[5] if len(pt) >= 6 and pt[5] > pt[4] else t[4])
        bmax = max(pt[4:6]) if len(pt) >= 6 else pt[4]
        warm = len(s_hist) >= WARMUP
        q33, q67 = (np.percentile(s_hist, [33.3, 66.7]) if warm else (0, 0))
        b67 = np.percentile(bmax_hist, 66.7) if warm else 0
        s_hist.append(s)
        bmax_hist.append(bmax)
        if m == "2025-12" or not warm:
            continue

        plans = {"基準 現行⑬": plan13(lanes_w)}
        if s <= q33:
            plans["戦略1 消し自信"] = [
                ("2連複", two(t[0], t[1]), 500),
                ("3連複", trio(t[0], t[1], t[2]), 300),
                ("3連複", trio(t[0], t[1], t[3]), 200)]
        if bmax >= b67:
            b = bmax_lane
            plans["戦略2 滑り込み穴"] = [
                ("3連複", trio(t[0], t[1], b), 400),
                ("3連単", f"{t[0]}-{t[1]}-{b}", 300),
                ("3連単", f"{t[1]}-{t[0]}-{b}", 300)]
        if s < q67 and p1 >= 0.17:
            plans["戦略3 混沌回避⑬"] = plan13(t)
        for name, bets in plans.items():
            st, rt = score(bets, pay, refund)
            a = agg[(name, m)]
            a[0] += st
            a[1] += rt
            a[2] += 1
            a[3] += rt > 0

print("\n===== 消し予報マップの買い方3種(2026-01〜08・全24場) =====")
for name in ARMS:
    tot = [0, 0, 0, 0]
    line = []
    for m in EVAL_MONTHS:
        a = agg[(name, m)]
        for i in range(4):
            tot[i] += a[i]
        if a[0]:
            line.append(f"{m[5:]}月{a[1] / a[0]:>5.0%}")
    st, rt, n, hit = tot
    if not st:
        continue
    print(f"\n{name}: 出動{n}R 的中{hit}R({hit / n:.0%}) "
          f"回収率{rt / st:.1%} 損益{rt - st:+,}円")
    print("  " + " ".join(line))
print("\n(注意: 階段発見と同一期間の設計確認。採否判定は9月の前方データで行う)")
