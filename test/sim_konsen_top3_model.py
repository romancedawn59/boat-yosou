# -*- coding: utf-8 -*-
"""発案A: 超混戦タブ専用「3着内モデル」による順位予想の再設計
(2026-09-01・ケンさん「超接戦タブがついたレース用の順位予想を改めて考えられそう?」)

    py -X utf8 test/sim_konsen_top3_model.py

■ 着眼
現行v2は「1着確率」で学習しており、超混戦の順位別3連対率は上位3位が団子
(60.6/62.0/59.2%)。超混戦で必要なのは「3着以内に絡む艇」の見極めなので、
ラベルを3着以内に替えた専用モデルで、タブ付きレースの中の順位だけを
並べ替える。タブ付け(超混戦判定)自体は現行モデルのまま(本番の選別は不変)。

■ 方法(本番忠実・リーク修正済み)
月次walk-forward 2026-01〜08。各月、勝率モデル(タブ付け用)と3着内モデル
(並べ替え用)をそれぞれ学習(v2.2の33特徴量・同一パラメータ)。
全entries(非完走艇含む)でランクし、勝率モデルのp1<0.20を超混戦とする。
順位付け3案: (i)現行=勝率順(基準) (ii)3着内確率順 (iii)ハイブリッド
(1位は勝率順の1位・2位以下は残りを3着内確率順)。

■ 評価(各順位付けについて同一レース集合で)
- 順位別3連対率(団子が割れるか) / 軸生存率(1・2位が共に3着内)
- 表彰台⊂上位4艇率(4艇BOXの器) / 参考: ⑬回収率・4艇3連複BOX回収率(返還処理)

■ 事前登録(結果を見る前に固定)
合格 = 軸生存率が基準比+5pt以上 かつ 順位別3連対率が1位>2位>3位の単調。
合格なら9月の紙上に「超混戦専用順位」を並走表示する相談へ。
不合格なら数字を正直に報告し、発案C(EVフィルタ)へ進む。
"""
import sys
from collections import defaultdict
from itertools import combinations, permutations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

EVAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
               "2026-05", "2026-06", "2026-07", "2026-08"]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}

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
    WHERE r.date >= '2026-01-01'
""", conn)
eval_df = _encode(eval_df)
eval_df = eval_df.merge(compute_form_features(conn), on=["race_id", "lane"],
                        how="left")
eval_df = _attach_extra_features(eval_df, conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-01-01'"):
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


def trio_comb(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan13(lanes):
    r1, r2, r3, r4, r5 = lanes[:5]
    bets = [("3連単", f"{a}-{b}-{c}", 100)
            for a, b, c in permutations((r1, r2, r3))]
    bets += [("3連単", f"{a}-{b}-{c}", 100)
             for a, b, c in permutations((r1, r2, r4))]
    bets += [("3連単", f"{r3}-{r1}-{r2}", 300),
             ("3連単", f"{r4}-{r1}-{r2}", 300),
             ("3連複", trio_comb(r3, r4, r5), 200)]
    return bets


def score_refund(bets, pay, refund):
    merged = defaultdict(int)
    for bt, comb, y in bets:
        merged[(bt, comb)] += y
    st = rt = 0
    for (bt, comb), y in merged.items():
        sep = "-" if bt == "3連単" else "="
        members = {int(x) for x in comb.split(sep)}
        st += y
        rt += y if members & refund else pay.get((bt, comb), 0) * y // 100
    return st, rt


ORDERINGS = ("i 現行(勝率順)", "ii 3着内順", "iii ハイブリッド")
rank_top3 = {o: defaultdict(lambda: [0, 0]) for o in ORDERINGS}
stats = {o: defaultdict(float) for o in ORDERINGS}
for m in EVAL_MONTHS:
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
    for rid, grp in md.groupby("race_id"):
        pay = payout_map[rid]
        if not pay:
            continue
        gw = grp.sort_values("p_win", ascending=False)
        lanes_w = [int(x) for x in gw["lane"]]
        p1 = float(gw["p_win"].iloc[0])
        if p1 >= 0.20 or len(lanes_w) < 5:
            continue
        arr = {int(r["lane"]): r["arrival_order"] for _, r in grp.iterrows()
               if pd.notna(r["arrival_order"])}
        if len(arr) < 3:
            continue
        nonfin = set(int(r["lane"]) for _, r in grp.iterrows()
                     if pd.isna(r["arrival_order"]))
        refund = {l for l in nonfin
                  if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
        top3 = set(sorted(arr, key=arr.get)[:3])

        gt = grp.sort_values("p_top3", ascending=False)
        lanes_t = [int(x) for x in gt["lane"]]
        lanes_h = [lanes_w[0]] + [l for l in lanes_t if l != lanes_w[0]]
        for oname, lanes in zip(ORDERINGS, (lanes_w, lanes_t, lanes_h)):
            for k, lane in enumerate(lanes, 1):
                rank_top3[oname][k][0] += 1
                rank_top3[oname][k][1] += lane in top3
            s = stats[oname]
            s["n"] += 1
            s["axis"] += lanes[0] in top3 and lanes[1] in top3
            s["in4"] += top3 <= set(lanes[:4])
            st, rt = score_refund(plan13(lanes), pay, refund)
            s["kst"] += st
            s["krt"] += rt
            box4 = [("3連複", trio_comb(*t), 100)
                    for t in combinations(lanes[:4], 3)]
            st, rt = score_refund(box4, pay, refund)
            s["bst"] += st
            s["brt"] += rt

print("\n===== 発案A: 超混戦タブ専用の順位予想(3案対決) =====")
print("― 順位別3連対率 ―")
hdr = f"{'順位':<5}"
for o in ORDERINGS:
    hdr += f"{o:<18}"
print(hdr)
for k in range(1, 7):
    row = f"{k}位   "
    for o in ORDERINGS:
        n, t = rank_top3[o][k]
        row += f"{(t / n if n else 0):>8.1%}          "
    print(row)
print("― 総合 ―")
for o in ORDERINGS:
    s = stats[o]
    n = s["n"]
    print(f"{o}: {n:.0f}R 軸生存{s['axis'] / n:.1%} 表彰台⊂上位4艇{s['in4'] / n:.1%} "
          f"⑬回収率{s['krt'] / s['kst']:.1%} 4艇複BOX{s['brt'] / s['bst']:.1%}")

b = stats[ORDERINGS[0]]
print("\n===== 事前登録判定 =====")
for o in ORDERINGS[1:]:
    s = stats[o]
    d = s["axis"] / s["n"] - b["axis"] / b["n"]
    r = rank_top3[o]
    mono = (r[1][1] / r[1][0] > r[2][1] / r[2][0] > r[3][1] / r[3][0])
    ok = d >= 0.05 and mono
    print(f"{o}: 軸生存差{d:+.1%}(基準+5pt) 単調性{'○' if mono else '×'} → "
          f"{'合格: 9月紙上並走の相談へ' if ok else '不合格'}")
