# -*- coding: utf-8 -*-
"""②新視点スクリーニング: 未開拓の券種市場(ワイド・複勝・2連系・単勝)を帯別に一斉検証
(2026-08-31ケンさん指示「まったく違う視点・買い方で利益が出そうなものがないか再調査」)

    py -X utf8 test/sim_new_markets_2026.py

■ 背景
これまでの検証は3連単・3連複に集中していた。DBにはワイド(拡連複)18.5万件・
複勝12.4万件・2連単/2連複/単勝が眠っており、券種市場としては未開拓。
希釈の法則(1隻の情報は1隻に純粋な券種で最も効く)と「市場は並びの結合構造に雑」
の2法則から、単艇系(単勝・複勝)と2艇系(ワイド・2連)は独立に検証する価値がある。

■ 方法(本番忠実・リーク修正済み)
- 学習: build_training_set(v2.2の33特徴量)
- 評価: 全entries(非完走艇含む)でランク付け(sim_clean_rank_2026.pyと同一)
- 返還近似: 非完走かつST記録なしの艇を含む目は投資返還、ST記録ありは没
- 月次walk-forward 2026-01〜08・全24場・各アーム1点100円

■ 事前登録(結果を見る前に固定)
- アーム(モデル順位r1..r4使用): 単勝r1 / 複勝r1 / 複勝r3 / 複勝r4 /
  ワイドr1=r2 / ワイドr1=r3 / ワイドr3=r4 / 2連複r1=r2 / 2連単r1-r2 / 2連単r2-r1
- 帯: 堅め(p1≥0.50) / 標準(0.35-0.50) / 本命(0.20-0.35) / 超混戦(<0.20)
- 判定: 8か月合計で回収率100%超のセルを「候補」、110%超を「有望」とし、
  有望セルは月次安定性(8か月中5か月以上100%超)を確認したうえで
  9月の紙上追跡に昇格させる。100%以下のセルはその場で棄却。
  多重比較(10アーム×4帯=40セル)を自覚し、単月の突出は無視する。
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _encode, _attach_extra_features, build_training_set,
                      compute_form_features)

EVAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
               "2026-05", "2026-06", "2026-07", "2026-08"]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}
BANDS = ("堅め", "標準", "本命", "超混戦")

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
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


def train_variant(train_df):
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    ds = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr["is_winner"],
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(va[FEATURE_COLUMNS], label=va["is_winner"], reference=ds)
    return lgb.train(PARAMS, ds, valid_sets=[vs], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def arms(r):
    r1, r2, r3, r4 = r[:4]
    def two(a, b):
        return f"{min(a, b)}={max(a, b)}"
    return {
        "単勝r1": ("単勝", str(r1)),
        "複勝r1": ("複勝", str(r1)),
        "複勝r3": ("複勝", str(r3)),
        "複勝r4": ("複勝", str(r4)),
        "ワイドr1=r2": ("拡連複", two(r1, r2)),
        "ワイドr1=r3": ("拡連複", two(r1, r3)),
        "ワイドr3=r4": ("拡連複", two(r3, r4)),
        "2連複r1=r2": ("2連複", two(r1, r2)),
        "2連単r1-r2": ("2連単", f"{r1}-{r2}"),
        "2連単r2-r1": ("2連単", f"{r2}-{r1}"),
    }


agg = defaultdict(lambda: [0, 0, 0])          # {(帯, アーム): [st, rt, hit]}
monthly = defaultdict(lambda: [0, 0])          # {(帯, アーム, 月): [st, rt]}
for m in EVAL_MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行", flush=True)
    booster = train_variant(tr_df)
    md = ev.copy()
    md["pred"] = booster.predict(md[FEATURE_COLUMNS])
    for rid, grp in md.groupby("race_id"):
        pay = payout_map[rid]
        if not pay:
            continue
        gs = grp.sort_values("pred", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        if len(lanes) < 4:
            continue
        arr_n = grp["arrival_order"].notna().sum()
        if arr_n < 3:
            continue
        nonfin = {int(r["lane"]) for _, r in grp.iterrows()
                  if pd.isna(r["arrival_order"])}
        refund = {l for l in nonfin
                  if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
        p1 = float(gs["pred"].iloc[0])
        band = ("堅め" if p1 >= 0.50 else "標準" if p1 >= 0.35
                else "本命" if p1 >= 0.20 else "超混戦")
        for name, (bt, comb) in arms(lanes).items():
            sep = "-" if bt in ("2連単",) else "="
            members = ({int(comb)} if bt in ("単勝", "複勝")
                       else {int(x) for x in comb.split(sep)})
            a = agg[(band, name)]
            mo = monthly[(band, name, m)]
            a[0] += 100
            mo[0] += 100
            if members & refund:
                a[1] += 100
                mo[1] += 100
                continue
            got = pay.get((bt, comb), 0)
            a[1] += got
            mo[1] += got
            if got:
                a[2] += 1

print("\n===== 券種市場スクリーニング(全24場・各100円・返還処理あり) =====")
for band in BANDS:
    n_races = agg[(band, "単勝r1")][0] // 100
    print(f"\n― {band}帯({n_races:,}R) ―")
    cells = sorted(((name, agg[(band, name)]) for name in arms([1, 2, 3, 4])),
                   key=lambda kv: -(kv[1][1] / kv[1][0] if kv[1][0] else 0))
    for name, (st, rt, hit) in cells:
        if not st:
            continue
        roi = rt / st
        stable = sum(
            1 for mm in EVAL_MONTHS
            if monthly[(band, name, mm)][0]
            and monthly[(band, name, mm)][1] / monthly[(band, name, mm)][0] > 1)
        tag = ("★有望" if roi > 1.10 else "☆候補" if roi > 1.00 else "  ")
        print(f"  {tag} {name:<12} 回収率{roi:>7.1%} 的中率{hit / (st // 100):>6.1%} "
              f"損益{rt - st:>+9,}円 100%超の月{stable}/8")

print("\n(事前登録: ★有望かつ100%超が5か月以上のセルだけ9月紙上追跡へ。"
      "40セルの多重比較につき単発の突出は採用しない)")
