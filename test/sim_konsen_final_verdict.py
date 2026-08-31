# -*- coding: utf-8 -*-
"""超混戦は本当に勝てるのか——クリーンな土台での最終判定材料
(2026-08-31深夜・ケンさんの問い「超混戦は本当に勝てるの?」)

    py -X utf8 test/sim_konsen_final_verdict.py

■ 問い(2点に還元)
(a) システムの心臓とされた「差され税」(E/F差され単が適正の2.48倍で安売り)は、
    リーク修正後のクリーンな土台でも実在するか
(b) 唯一生き残った「5場限定135.3%(+40,280円・57R)」は実力か、1発の偶然か

■ 方法(本番忠実・sim_clean_rank_2026.pyと同一の土台)
全entries(非完走艇含む)でランク付け・返還処理あり・月次walk-forward
2026-01〜08・v2.2の33特徴量。超混戦帯(p1<0.20)のみ対象。
- ⑬を線別(BOX12並び/E/F差され/G複)に分解して全場・5場で採点
- E/F差され税の実測: 各レースのE/F想定確率(独立近似)から公正配当(0.75/p)を
  出し、実際の3連単配当との比(実配当/公正配当)の中央値を測る。
  1.0超=市場が安売り(税は実在)、1.0以下=税は幻
- 5場の的中明細を回収額降順で列挙し、最大の1発を除いた回収率も出す

■ 事前登録(結果を見る前に固定)
- 差され税: E/F価格比の中央値が1.5倍以上なら「実在」、1.0-1.5は「弱い」、
  1.0未満は「幻」と結論
- 5場: 最大的中1本を除いて100%を割るなら「1発依存=実力と認めない」
- 両方が否定された場合、「超混戦は現構成では勝てない帯」と結論し、
  9月の紙上でもE/F系の代替仮説がない限り10月の実弾復帰は勧告しない
"""
import sys
from collections import defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
import predictors as P
from config import DB_PATH, TARGET_VENUE_CODES
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


def trio_comb(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan13_lines(lanes):
    r = lanes
    lines = []
    for tag, members in (("A", (0, 1, 2)), ("B", (0, 1, 3))):
        for pm in permutations(members):
            role = f"{tag} {'-'.join(str(i + 1) for i in pm)}"
            lines.append((role, "3連単",
                          f"{r[pm[0]]}-{r[pm[1]]}-{r[pm[2]]}", 100))
    lines.append(("E差され+300", "3連単", f"{r[2]}-{r[0]}-{r[1]}", 300))
    lines.append(("F差され+300", "3連単", f"{r[3]}-{r[0]}-{r[1]}", 300))
    lines.append(("G複", "3連複", trio_comb(r[2], r[3], r[4]), 200))
    return lines


line_agg = defaultdict(lambda: [0, 0, 0])       # 全場 {役割: [st, rt, hit]}
m5 = defaultdict(lambda: [0, 0])                # 5場 {月: [st, rt]}
hits5 = []                                      # 5場的中明細
ratios = []                                     # E/F 実配当/公正配当
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
        pay = payout_map[rid]
        if not pay:
            continue
        gs = grp.sort_values("pred", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        p1 = float(gs["pred"].iloc[0])
        if p1 >= 0.20 or len(lanes) < 5:
            continue
        if grp["arrival_order"].notna().sum() < 3:
            continue
        nonfin = {int(r["lane"]) for _, r in grp.iterrows()
                  if pd.isna(r["arrival_order"])}
        refund = {l for l in nonfin
                  if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in gs.iterrows()]
        probs = P.normalize_probs(ranked)
        in5 = int(gs["venue_code"].iloc[0]) in TARGET_VENUE_CODES

        race_st = race_rt = 0
        for role, bt, comb, y in plan13_lines(lanes):
            sep = "-" if bt == "3連単" else "="
            members = {int(x) for x in comb.split(sep)}
            a = line_agg[role]
            a[0] += y
            race_st += y
            if members & refund:
                a[1] += y
                race_rt += y
                continue
            got = pay.get((bt, comb), 0) * y // 100
            a[1] += got
            race_rt += got
            if got:
                a[2] += 1
        if in5:
            m5[m][0] += race_st
            m5[m][1] += race_rt
            if race_rt > race_st:
                hits5.append((m, rid, race_rt))

        # 差され税の実測(E/F): 実配当 / 公正配当(0.75/p)
        for idx in (2, 3):
            a3, b3, c3 = lanes[idx], lanes[0], lanes[1]
            pa = probs.get(a3, 0)
            pb = probs.get(b3, 0)
            pc = probs.get(c3, 0)
            d1, d2 = 1 - pa, 1 - pa - pb
            if d1 <= 0 or d2 <= 0:
                continue
            p_line = pa * (pb / d1) * (pc / d2)
            if p_line <= 0:
                continue
            amt = pay.get(("3連単", f"{a3}-{b3}-{c3}"), 0)
            if amt and not ({a3, b3, c3} & nonfin):
                fair = 0.75 / p_line * 100
                ratios.append(amt / fair)

print("\n===== (a) 差され税はクリーンな土台でも実在するか =====")
r = np.array(ratios)
print(f"E/F的中{len(r)}本の 実配当/公正配当: 中央値{np.median(r):.2f}倍 "
      f"平均{r.mean():.2f}倍 1.0超の割合{(r > 1).mean():.0%}")
med = np.median(r)
print("判定: " + ("実在(1.5倍以上)" if med >= 1.5
                 else "弱い(1.0-1.5倍)" if med >= 1.0 else "幻(1.0未満)"))

print("\n― ⑬線別(全場・クリーン) ―")
for role in sorted(line_agg, key=lambda k: line_agg[k][1] / max(1, line_agg[k][0])):
    st, rt, h = line_agg[role]
    print(f"  {role:<12} 回収率{rt / st:>7.1%} 的中{h:>3} 損益{rt - st:>+9,}円")

print("\n===== (b) 5場135%は実力か1発か =====")
tot_st = sum(v[0] for v in m5.values())
tot_rt = sum(v[1] for v in m5.values())
for m in EVAL_MONTHS:
    st, rt = m5[m]
    if st:
        print(f"  {m}: {st // 2000:>3}R 回収率{rt / st:>7.1%} 損益{rt - st:>+9,}円")
print(f"  合計: 回収率{tot_rt / tot_st:.1%} 損益{tot_rt - tot_st:+,}円")
hits5.sort(key=lambda x: -x[2])
print("  的中明細(上位8):")
for m, rid, rt in hits5[:8]:
    print(f"    {m} {rid} 回収{rt:,}円")
if hits5:
    top = hits5[0][2]
    roi_wo = (tot_rt - top) / (tot_st - 2000)
    print(f"  最大の1本({top:,}円)を除くと: 回収率{roi_wo:.1%}")
    print("  判定: " + ("1発依存=実力と認めない(除外後100%未満)"
                       if roi_wo < 1.0 else "1発依存ではない(除外後も100%以上)"))
