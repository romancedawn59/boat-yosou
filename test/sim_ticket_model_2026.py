# -*- coding: utf-8 -*-
"""案A: 買い目そのものを学習する(チケット単位モデル)
(2026-09-21ケンさん指示「今までから逸脱した全く新たな買い目予想アプローチ」)

    py -X utf8 test/sim_ticket_model_2026.py

■ 発想
従来は「艇ごとの1着確率→順位→固定の型」。ここでは順序を逆にし、
「予想i位-j位-k位の3連単の券」を1行として、当たり外れ(と払戻)を直接学習する。
艇の確率・3着内確率・枠・級・モーター・枠の位置関係を券の特徴量にし、
オッズは一切見ない(朝に買える)。

■ 方法(本番忠実・事前登録)
- 月次walk-forward。2025-09〜2026-08の各月、その月より前の全データで
  1着モデルと3着内モデルを学習し、全entries(非完走含む)に確率を付ける
- 対象レース: 全24場×1位勝率20〜30%×6艇。券は予想1〜5位の順列60通り
- 券モデル: 評価月mより前の月の券だけで学習(助走4か月以上)。評価は2026-01〜08
  M1 的中確率の分類器 / M1EV=的中確率×その並びの過去平均配当(学習期間内・縮小推定)
  M2 払戻額(100円あたり・上限3万円でクリップ)の回帰
- 買い方: 各レースでスコア上位6点を各100円(600円/R)。返還対象艇を含む券は返還
- 比較: 独立近似(Harville)の確率上位6点(=従来方式で同じ点数を買った場合)
■ 合格基準: 全24場で回収率105%超 かつ 最大1本除外後も100%超 かつ
  8か月中5か月以上100%超。満たさなければ不合格として記録する。
"""
import sys
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH, TARGET_VENUE_CODES
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

MONTHS = ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02",
          "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
EVAL_MONTHS = MONTHS[4:]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}
TICKETS = list(permutations(range(5), 3))      # 予想1〜5位の順列60通り
TOPK = 6

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
    WHERE r.date >= '2025-09-01' AND r.date < '2026-09-01'
""", conn)
eval_df = _encode(eval_df)
eval_df = eval_df.merge(compute_form_features(conn), on=["race_id", "lane"],
                        how="left")
eval_df = _attach_extra_features(eval_df, conn)
pay3 = {}
for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id "
        "WHERE r.date >= '2025-09-01' AND p.bet_type = '3連単'"):
    pay3[rid] = (comb, amt or 0)
conn.close()


def train_v2(train_df, label):
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    ds = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr[label],
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(va[FEATURE_COLUMNS], label=va[label], reference=ds)
    return lgb.train(PARAMS, ds, valid_sets=[vs], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


# ---- 月次WFで艇の確率を付け、券の行を作る ------------------------------------
ticket_rows = []
for m in MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行 ×2モデル", flush=True)
    win_model = train_v2(tr_df, "is_winner")
    top3_model = train_v2(tr_df, "is_top3")
    md = ev.copy()
    md["p_win"] = win_model.predict(md[FEATURE_COLUMNS])
    md["p_top3"] = top3_model.predict(md[FEATURE_COLUMNS])
    p1 = md.groupby("race_id")["p_win"].max()
    band = set(p1[(p1 >= 0.20) & (p1 < 0.30)].index)
    for rid, grp in md[md["race_id"].isin(band)].groupby("race_id"):
        if rid not in pay3 or len(grp) < 6:
            continue
        if grp["arrival_order"].notna().sum() < 3:
            continue
        gs = grp.sort_values("p_win", ascending=False).reset_index(drop=True)
        lanes = [int(x) for x in gs["lane"]]
        pw = [float(x) for x in gs["p_win"]]
        pt = [float(x) for x in gs["p_top3"]]
        cls = [float(x) if pd.notna(x) else np.nan for x in gs["racer_class_ord"]]
        mot = [float(x) if pd.notna(x) else np.nan for x in gs["motor_2rate"]]
        tot = sum(pw)
        nw = [x / tot for x in pw]
        refund = {lanes[i] for i in range(6)
                  if pd.isna(gs["arrival_order"].iloc[i])
                  and pd.isna(gs["st_time"].iloc[i])}
        win_comb, win_amt = pay3[rid]
        vc = int(gs["venue_code"].iloc[0])
        in5 = vc in TARGET_VENUE_CODES
        for (i, j, k) in TICKETS:
            a, b, c = lanes[i], lanes[j], lanes[k]
            comb = f"{a}-{b}-{c}"
            if {a, b, c} & refund:
                ret, hit = 100, 0
            else:
                hit = int(comb == win_comb)
                ret = win_amt if hit else 0
            d1, d2 = 1 - nw[i], 1 - nw[i] - nw[j]
            harv = nw[i] * (nw[j] / d1) * (nw[k] / d2) if d1 > 0 and d2 > 0 else 0
            ticket_rows.append((
                m, rid, in5, i + 1, j + 1, k + 1, pw[i], pw[j], pw[k],
                pt[i], pt[j], pt[k], a, b, c, cls[i], cls[j], cls[k],
                mot[i], mot[j], mot[k], int(b == a + 1), int(b == a - 1),
                int(c == a + 1 or c == b + 1), pw[0], pw[0] - pw[1], harv,
                vc, hit, ret))

COLS = ["month", "race_id", "in5", "ri", "rj", "rk", "pw_i", "pw_j", "pw_k",
        "pt_i", "pt_j", "pt_k", "lane_i", "lane_j", "lane_k", "cls_i", "cls_j",
        "cls_k", "mot_i", "mot_j", "mot_k", "j_out_nb", "j_in_nb", "k_mark",
        "p1", "gap12", "harv", "venue_code", "hit", "ret"]
T = pd.DataFrame(ticket_rows, columns=COLS)
FEATS = [c for c in COLS if c not in ("month", "race_id", "in5", "hit", "ret")]
print(f"\n券の行数: {len(T):,} / レース数: {T['race_id'].nunique():,} "
      f"/ 的中率(1券あたり): {T['hit'].mean():.2%}", flush=True)

# ---- 券モデルのwalk-forward ----------------------------------------------------
picks = defaultdict(list)     # {アーム: [(month, in5, ret)]} 1券ごと
for m in EVAL_MONTHS:
    tr = T[T["month"] < m]
    te = T[T["month"] == m]
    if tr["month"].nunique() < 4 or te.empty:
        continue
    print(f"{m}: 券モデル学習 {len(tr):,}行", flush=True)
    clf = lgb.train({"objective": "binary", "verbosity": -1,
                     "learning_rate": 0.05, "num_leaves": 31,
                     "min_data_in_leaf": 200, "seed": 7},
                    lgb.Dataset(tr[FEATS], label=tr["hit"],
                                categorical_feature=["venue_code"]),
                    num_boost_round=300)
    reg = lgb.train({"objective": "regression", "verbosity": -1,
                     "learning_rate": 0.05, "num_leaves": 31,
                     "min_data_in_leaf": 200, "seed": 7},
                    lgb.Dataset(tr[FEATS], label=tr["ret"].clip(upper=30000),
                                categorical_feature=["venue_code"]),
                    num_boost_round=300)
    hits = tr[tr["hit"] == 1]
    overall = hits["ret"].mean()
    g = hits.groupby(["ri", "rj", "rk"])["ret"].agg(["sum", "count"])
    avgpay = {k: (v["sum"] + overall * 10) / (v["count"] + 10)
              for k, v in g.iterrows()}
    te = te.copy()
    te["s_prob"] = clf.predict(te[FEATS])
    te["s_ev"] = [p * avgpay.get((a, b, c), overall) for p, a, b, c in
                  zip(te["s_prob"], te["ri"], te["rj"], te["rk"])]
    te["s_reg"] = reg.predict(te[FEATS])
    te["s_harv"] = te["harv"]
    for arm, col in (("M1 的中確率上位6", "s_prob"), ("M1EV 確率×過去平均配当 上位6", "s_ev"),
                     ("M2 払戻回帰 上位6", "s_reg"), ("比較: 独立近似の確率上位6", "s_harv")):
        top = te.sort_values(col, ascending=False).groupby("race_id").head(TOPK)
        for in5, ret in zip(top["in5"], top["ret"]):
            picks[arm].append((m, in5, ret))

print("\n===== 案A 券モデル(2026-01〜08・各レース上位6点×100円) =====")
for scope, flt in (("全24場", lambda x: True), ("対象5場", lambda x: x)):
    print(f"\n― {scope} ―")
    for arm, rows in picks.items():
        sub = [(mm, ret) for mm, in5, ret in rows if flt(in5)]
        if not sub:
            continue
        st = len(sub) * 100
        rt = sum(r for _, r in sub)
        best = max(r for _, r in sub)
        mon = defaultdict(lambda: [0, 0])
        for mm, r in sub:
            mon[mm][0] += 100
            mon[mm][1] += r
        ok = sum(1 for v in mon.values() if v[1] > v[0])
        hitn = sum(1 for _, r in sub if r > 100)
        print(f"{arm}: {st // (TOPK * 100)}R 的中{hitn}回 回収率{rt / st:.1%} "
              f"損益{rt - st:+,}円 最大1本除く{(rt - best) / (st - 100):.1%} "
              f"100%超の月{ok}/{len(mon)}")
        print("   " + " ".join(f"{mm[5:]}月{v[1] / v[0]:.0%}"
                               for mm, v in sorted(mon.items())))
print("\n(合格基準: 全24場で105%超・最大1本除外後100%超・5か月以上100%超)")
