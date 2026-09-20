# -*- coding: utf-8 -*-
"""ドンピシャ1位(予想1-2-3位そのまま)1点500円で勝てる場の洗い出し・保有全記録版
(2026-09-21ケンさん指示。「現行5場」の区別なし・全24場同列・全帯)

    py -X utf8 test/sim_donpisha_all_records_0921.py

■ 方法(本番忠実)
DBは2025-07-15〜。月次walk-forward: 2025-10〜2026-08の各月を、その月より前の
全データで学習した1着モデルで予測(全entriesに確率・非完走艇も順位に含める)。
2026-09は本番配信picks JSON。予想1-2-3位の3連単を1点500円。
返還対象艇(着順なし・STなし)を含む券は500円返還。
■ 出力
① 場別(全帯): 発生確率・的中時払戻・期待値・3期間(2025-10〜2026-01 / 02〜05 / 06〜09)
② 場×1位勝率の帯 の期待値
③ 抽出: 期待値500円超 かつ 3期間中2期間以上黒字 かつ 最大1本除外後も500円超
"""
import glob
import json
import math
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from config import DB_PATH, VENUE_NAMES
from features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, _ENTRY_COLS,
                      _attach_extra_features, _encode, build_training_set,
                      compute_form_features)

MONTHS = ["2025-10", "2025-11", "2025-12"] + [f"2026-{m:02d}" for m in range(1, 9)]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}
YEN = 500
WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
PERIODS = [("25/10〜26/01", "2025-10", "2026-01"), ("26/02〜05", "2026-02", "2026-05"),
           ("26/06〜09", "2026-06", "2026-09")]
BANDS = [("〜22.5%", 0, .225), ("22.5〜30%", .225, .30), ("30〜40%", .30, .40),
         ("40〜55%", .40, .55), ("55%〜", .55, 1.01)]

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)
eval_df = pd.read_sql_query(f"""
    SELECT r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
           {_ENTRY_COLS}, res.arrival_order, res.st_time
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    WHERE r.date >= '2025-10-01' AND r.date < '2026-09-01'
""", conn)
eval_df = _encode(eval_df)
eval_df = eval_df.merge(compute_form_features(conn), on=["race_id", "lane"], how="left")
eval_df = _attach_extra_features(eval_df, conn)
pay3 = {}
for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "WHERE p.bet_type = '3連単'"):
    pay3[rid] = (comb, amt or 0)
sept_arr = defaultdict(dict)
for rid, lane, ao, st in conn.execute(
        "SELECT e.race_id, e.lane, res.arrival_order, res.st_time FROM entries e "
        "JOIN races r ON r.race_id=e.race_id "
        "LEFT JOIN results res ON res.race_id=e.race_id AND res.lane=e.lane "
        "WHERE r.date >= '2026-09-01'"):
    sept_arr[rid][lane] = (ao, st)
conn.close()

rows = []   # (month, venue, p1, 払戻円(500円買い)) 返還は500
dump = []   # 全レースの予想順位を保存(他の買い方の試算に再利用)
for m in MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"].sort_values("date")
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    print(f"{m}: 学習{len(tr_df):,}行", flush=True)
    cutoff = tr_df["date"].iloc[int(len(tr_df) * 0.9)]
    a, b = tr_df[tr_df["date"] < cutoff], tr_df[tr_df["date"] >= cutoff]
    ds = lgb.Dataset(a[FEATURE_COLUMNS], label=a["is_winner"],
                     categorical_feature=CATEGORICAL_FEATURES)
    model = lgb.train(PARAMS, ds, num_boost_round=500,
                      valid_sets=[lgb.Dataset(b[FEATURE_COLUMNS], label=b["is_winner"],
                                              reference=ds)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    md = ev.copy()
    md["p"] = model.predict(md[FEATURE_COLUMNS])
    for rid, g in md.groupby("race_id"):
        if rid not in pay3 or len(g) < 6 or g["arrival_order"].notna().sum() < 3:
            continue
        gs = g.sort_values("p", ascending=False)
        top = [int(x) for x in gs["lane"].iloc[:3]]
        refund = {int(l) for l, ao, st in zip(g["lane"], g["arrival_order"], g["st_time"])
                  if pd.isna(ao) and pd.isna(st)}
        comb, amt = pay3[rid]
        if set(top) & refund:
            got = YEN
        else:
            got = amt * YEN // 100 if comb == "-".join(map(str, top)) else 0
        rows.append((m, int(g["venue_code"].iloc[0]), float(gs["p"].iloc[0]), got))
        dump.append((rid, m, int(g["venue_code"].iloc[0]), float(gs["p"].iloc[0]),
                     "-".join(str(int(x)) for x in gs["lane"]),
                     "-".join(map(str, sorted(refund)))))

for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-09-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        rid = r["race_id"]
        a = sept_arr.get(rid)
        if not r.get("ranked") or len(r["ranked"]) < 6 or rid not in pay3 or not a \
                or sum(1 for v in a.values() if v[0] is not None) < 3:
            continue
        top = [x[0] for x in r["ranked"][:3]]
        refund = {l for l, (ao, st) in a.items() if ao is None and st is None}
        comb, amt = pay3[rid]
        if set(top) & refund:
            got = YEN
        else:
            got = amt * YEN // 100 if comb == "-".join(map(str, top)) else 0
        rows.append(("2026-09", r["venue_code"], r["ranked"][0][1], got))
        dump.append((rid, "2026-09", r["venue_code"], r["ranked"][0][1],
                     "-".join(str(x[0]) for x in r["ranked"]),
                     "-".join(map(str, sorted(refund)))))

pd.DataFrame(dump, columns=["race_id", "month", "venue", "p1", "pred", "refund"]).to_csv(
    WF_LEDGER, index=False, encoding="utf-8-sig")
D = pd.DataFrame(rows, columns=["month", "venue", "p1", "got"])
D["hit"] = D["got"] > YEN
print(f"\n対象: {len(D):,}R(2025-10〜2026-09・全24場・全帯)")


def wilson(h, n, z=1.645):
    p = h / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return max(0, (c - w) / d), (c + w) / d


def table(sub, title):
    print(f"\n===== {title} =====")
    print(f"{'場':<6}{'R数':>6}{'的中':>5}{'発生確率':>8}{'(90%幅)':>15}{'的中時払戻':>10}"
          f"{'期待値':>8}{'通算損益':>12}{'最大1本除く':>10}"
          + "".join(f"{p[0]:>13}" for p in PERIODS) + f"{'黒字期間':>8}")
    out = []
    for v, g in sub.groupby("venue"):
        n, h = len(g), int(g["hit"].sum())
        evv = g["got"].sum() / n
        exb = (g["got"].sum() - g["got"].max()) / n
        per = []
        for _nm, a, b in PERIODS:
            t = g[(g["month"] >= a) & (g["month"] <= b)]
            per.append(t["got"].sum() / (len(t) * YEN) if len(t) else float("nan"))
        out.append((evv, v, n, h, exb, per))
    picked = []
    for evv, v, n, h, exb, per in sorted(out, reverse=True):
        l, u = wilson(h, n)
        avg = sub[(sub["venue"] == v) & sub["hit"]]["got"].mean() if h else 0
        ok = sum(1 for x in per if x == x and x > 1)
        print(f"{VENUE_NAMES[v]:<6}{n:>6,}{h:>5}{h / n:>8.1%}  ({l:>4.1%}〜{u:>5.1%})"
              f"{avg:>9,.0f}円{evv:>7,.0f}円{(evv - YEN) * n:>+11,.0f}円{exb:>9,.0f}円"
              + "".join(f"{x:>13.0%}" for x in per) + f"{ok:>6}/3")
        if evv > YEN and ok >= 2 and exb > YEN:
            picked.append(VENUE_NAMES[v])
    n = len(sub)
    print(f"{'全場計':<6}{n:>6,}{int(sub['hit'].sum()):>5}{sub['hit'].mean():>8.1%}"
          f"{'':>17}{sub[sub['hit']]['got'].mean():>9,.0f}円{sub['got'].sum() / n:>7,.0f}円"
          f"{sub['got'].sum() - n * YEN:>+11,.0f}円")
    print("  抽出(期待値500円超・2期間以上黒字・最大1本除外後も500円超): "
          + ("、".join(picked) if picked else "該当なし"))


table(D, "① 場別・全帯")
for name, lo, hi in BANDS:
    table(D[(D["p1"] >= lo) & (D["p1"] < hi)], f"② 1位勝率 {name}")
