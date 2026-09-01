# -*- coding: utf-8 -*-
"""本命帯の月毎まとめ(2026-01〜08・本番忠実版)+レース別CSV
(2026-09-02ケンさん指示。増額判断の材料を兼ねる)

    py -X utf8 test/list_honmei_monthly_2026.py

本命帯 = 全艇ランク(リーク修正済み)でモデル1位の勝率が0.20以上0.35未満。
月次walk-forward・v2.2の33特徴量。月毎に全24場と対象5場を分けて集計:
R数 / 1位的中率 / 軸生存率(1・2位が共に3着内) / 3連単払戻中央値 /
現行9行1,400円構成の紙上回収率(返還処理あり)。
レース別データ(日付・場・R・予想・結果・払戻・人気順・1位勝率・対象5場)は
CSVに書き出す(人気順は朝オッズ収集済みの2026-05以降の一部のみ)。
"""
import csv
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import (FEATURE_COLUMNS, _ENTRY_COLS, _attach_extra_features,
                      _encode, build_training_set, compute_form_features)

EVAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
               "2026-05", "2026-06", "2026-07", "2026-08"]
VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川",
         6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国",
         11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀",
         16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松",
         21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}
OUT_CSV = (r"C:\Users\roman\AppData\Local\Temp\claude\Y---------boat"
           r"\f2ba1165-daaf-4c5f-9402-057ab6a60d05\scratchpad"
           r"\honmei_results_2026.csv")

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
paymap = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-01-01'"):
    paymap[rid][(bt, comb)] = amt or 0
    if bt == "3連単":
        pay3t[rid] = (comb, amt or 0)
odds3t = defaultdict(dict)
for rid, comb, o in conn.execute(
        "SELECT race_id, combination, odds FROM odds "
        "WHERE bet_type = '3連単' AND odds IS NOT NULL"):
    odds3t[rid][comb] = o
conn.close()


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def honmei_plan(lanes):
    r1, r2, r3, r4 = lanes[:4]
    return [
        ("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
        ("3連複", trio(r1, r3, r4), 100),
        ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200),
        ("3連複", trio(r2, r3, r4), 100),
        ("3連単", f"{r3}-{r2}-{r1}", 100), ("3連単", f"{r4}-{r2}-{r1}", 300),
    ]


def score(lanes, pay, refund):
    merged = defaultdict(int)
    for bt, comb, y in honmei_plan(lanes):
        merged[(bt, comb)] += y
    st = rt = 0
    for (bt, comb), y in merged.items():
        sep = "-" if bt == "3連単" else "="
        members = {int(x) for x in comb.split(sep)}
        st += y
        rt += y if members & refund else pay.get((bt, comb), 0) * y // 100
    return st, rt


rows = []
agg = defaultdict(lambda: defaultdict(float))   # {(月, スコープ): 指標}
for m in EVAL_MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行", flush=True)
    booster = train_fold(tr_df)
    md = ev.copy()
    md["pred"] = booster.predict(md[FEATURE_COLUMNS])
    for rid, grp in md.groupby("race_id"):
        if rid not in pay3t:
            continue
        gs = grp.sort_values("pred", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        p1 = float(gs["pred"].iloc[0])
        if not (0.20 <= p1 < 0.35) or len(lanes) < 4:
            continue
        arr = {int(r["lane"]): r["arrival_order"] for _, r in grp.iterrows()
               if pd.notna(r["arrival_order"])}
        if len(arr) < 3:
            continue
        nonfin = {int(r["lane"]) for _, r in grp.iterrows()
                  if pd.isna(r["arrival_order"])}
        refund = {l for l in nonfin
                  if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
        comb, amt = pay3t[rid]
        odds = odds3t.get(rid)
        ninki = (str(sorted(odds.values()).index(odds[comb]) + 1)
                 if odds and comb in odds else "-")
        vc = int(gs["venue_code"].iloc[0])
        in5 = vc in TARGET_VENUE_CODES
        top3 = sorted(arr, key=arr.get)[:3]
        st, rt = score(lanes, paymap[rid], refund)
        rows.append({
            "日付": gs["date"].iloc[0], "場": VENUE.get(vc, str(vc)),
            "R": int(gs["race_no"].iloc[0]),
            "予想(1位→6位)": "-".join(str(l) for l in lanes),
            "結果(3連単)": comb, "払戻(100円)": amt, "人気順": ninki,
            "モデル1位勝率": f"{p1:.1%}", "対象5場": "○" if in5 else "",
        })
        for scope in (["全場", "5場"] if in5 else ["全場"]):
            a = agg[(m, scope)]
            a["n"] += 1
            a["hit1"] += arr.get(lanes[0]) == 1
            a["axis"] += lanes[0] in top3 and lanes[1] in top3
            a["st"] += st
            a["rt"] += rt
            if "amts" not in a:
                a["amts"] = []
            a["amts"].append(amt)

rows.sort(key=lambda r: (r["日付"], r["場"], r["R"]))
with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nCSV書き出し: {OUT_CSV} ({len(rows)}レース)")

for scope in ("5場", "全場"):
    print(f"\n===== 本命帯 月毎まとめ({scope}) =====")
    print(f"{'月':<9}{'R数':>5}{'1位的中':>8}{'軸生存':>8}{'払戻中央値':>10}"
          f"{'9行構成回収率':>11}{'損益':>11}")
    tot = defaultdict(float)
    tot_amts = []
    for m in EVAL_MONTHS:
        a = agg[(m, scope)]
        if not a.get("n"):
            continue
        amts = sorted(a["amts"])
        med = amts[len(amts) // 2]
        n = a["n"]
        print(f"{m:<9}{n:>4.0f}R{a['hit1'] / n:>8.1%}{a['axis'] / n:>8.1%}"
              f"{med:>9,}円{a['rt'] / a['st']:>10.1%}"
              f"{a['rt'] - a['st']:>+10,.0f}円")
        for k in ("n", "hit1", "axis", "st", "rt"):
            tot[k] += a[k]
        tot_amts += amts
    if tot["n"]:
        tot_amts.sort()
        print(f"{'合計':<8}{tot['n']:>4.0f}R{tot['hit1'] / tot['n']:>8.1%}"
              f"{tot['axis'] / tot['n']:>8.1%}"
              f"{tot_amts[len(tot_amts) // 2]:>9,}円"
              f"{tot['rt'] / tot['st']:>10.1%}"
              f"{tot['rt'] - tot['st']:>+10,.0f}円")
print("\n(9行構成=現行の本命1,400円。紙上・返還処理あり。実配信は帯内から"
      "選別・上限ありのためこの数字は帯全体の成績)")
