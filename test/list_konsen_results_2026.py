# -*- coding: utf-8 -*-
"""2026-01〜08の超混戦全レース一覧(着順・3連単払戻・人気順)を書き出す

    py -X utf8 test/list_konsen_results_2026.py

超混戦の定義は本番忠実版(sim_clean_rank_2026.py以降)と同一:
月次walk-forward・v2.2の33特徴量・全entries(非完走艇含む)でランクし
1位生値<0.20。人気順は朝オッズ(oddsテーブル・2026-05以降のみ収集)による
3連単の当選組の人気位。1〜4月はオッズ未収集のため「-」。
出力: コンソール表 + CSV(scratchpad)。
"""
import csv
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH, TARGET_VENUE_CODES
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

EVAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
               "2026-05", "2026-06", "2026-07", "2026-08"]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}
VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川",
         6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国",
         11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀",
         16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松",
         21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}
OUT_CSV = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
           r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad"
           r"\konsen_results_2026.csv")

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
pay3t = {}
for rid, comb, amt in conn.execute(
    "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id "
    "WHERE r.date >= '2026-01-01' AND p.bet_type = '3連単'"):
    pay3t[rid] = (comb, amt or 0)
odds3t = defaultdict(dict)
for rid, comb, o in conn.execute(
        "SELECT race_id, combination, odds FROM odds "
        "WHERE bet_type = '3連単' AND odds IS NOT NULL"):
    odds3t[rid][comb] = o
conn.close()


def train_month(train_df):
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    ds = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr["is_winner"],
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(va[FEATURE_COLUMNS], label=va["is_winner"], reference=ds)
    return lgb.train(PARAMS, ds, valid_sets=[vs], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


rows = []
for m in EVAL_MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行", flush=True)
    booster = train_month(tr_df)
    md = ev.copy()
    md["pred"] = booster.predict(md[FEATURE_COLUMNS])
    for rid, grp in md.groupby("race_id"):
        if rid not in pay3t:
            continue
        gs = grp.sort_values("pred", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        p1 = float(gs["pred"].iloc[0])
        if p1 >= 0.20 or len(lanes) < 5:
            continue
        if grp["arrival_order"].notna().sum() < 3:
            continue
        comb, amt = pay3t[rid]
        odds = odds3t.get(rid)
        if odds and comb in odds:
            rank = sorted(odds.values()).index(odds[comb]) + 1
            ninki = str(rank)
        else:
            ninki = "-"
        vc = int(gs["venue_code"].iloc[0])
        rows.append({
            "予想(1位→6位)": "-".join(str(l) for l in lanes),
            "日付": gs["date"].iloc[0],
            "場": VENUE.get(vc, str(vc)),
            "R": int(gs["race_no"].iloc[0]),
            "結果(3連単)": comb,
            "払戻(100円)": amt,
            "人気順": ninki,
            "モデル1位勝率": f"{p1:.1%}",
            "対象5場": "○" if vc in TARGET_VENUE_CODES else "",
        })

rows.sort(key=lambda r: (r["日付"], r["場"], r["R"]))
with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nCSV書き出し: {OUT_CSV} ({len(rows)}レース)")

print(f"\n{'予想(1位→6位)':<14}{'日付':<11}{'場':<5}{'R':>3} {'結果':<10}"
      f"{'払戻':>9}{'人気':>4}{'1位勝率':>8} 5場")
for r in rows:
    print(f"{r['予想(1位→6位)']:<14}{r['日付']:<11}{r['場']:<5}{r['R']:>3} "
          f"{r['結果(3連単)']:<10}{r['払戻(100円)']:>8,}円{r['人気順']:>4}"
          f"{r['モデル1位勝率']:>8} {r['対象5場']}")

amts = sorted(r["払戻(100円)"] for r in rows)
n = len(amts)
ge55 = sum(1 for a in amts if a >= 5500)
print(f"\n集計: {n}R / 払戻中央値{amts[n // 2]:,}円 / 55倍以上{ge55}R"
      f"({ge55 / n:.0%}) / 万舟{sum(1 for a in amts if a >= 10000)}R")
ranked_known = [int(r["人気順"]) for r in rows if r["人気順"] != "-"]
if ranked_known:
    import statistics
    print(f"人気順が分かる{len(ranked_known)}R: 中央値{statistics.median(ranked_known):.0f}番人気 / "
          f"10番人気以内{sum(1 for x in ranked_known if x <= 10)}R / "
          f"50番人気以降{sum(1 for x in ranked_known if x >= 50)}R")
