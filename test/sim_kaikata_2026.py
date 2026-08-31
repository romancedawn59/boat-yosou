# -*- coding: utf-8 -*-
"""③買い方3案(3-A/3-B/3-C)の2026-01〜08シミュレーション
(2026-08-31ケンさん指示「3-A/3-B/3-Cを202601から08でシミュレーションして」)

    py -X utf8 test/sim_kaikata_2026.py

■ 事前登録(結果を見る前に固定)
月次walk-forward(各月、その月より前の全データでv2を学習)。
2025-12は分類器・線成績の助走月(評価は2026-01〜08)。

3-A 名前のある税の棚卸し:
  ⑬の線を役割単位(BOX12並び・E/F差され追加・G複)、本命1,400円を9行単位で
  8か月分解(記述統計)。政策テスト=「観測30R以上かつ累積回収率50%未満の
  BOX線をその月から外し、浮いた金をE/Fへ折半(100円単位)」をwalk-forwardで
  適用し現行⑬と比較。E/F/Gは名前のある税(差され税・全滅保険)なので外さない。
  採用検討基準: 現行⑬比+5pt以上かつ8か月中5か月以上で同等以上。

3-B 夢税カット(線レベル):
  ⑬のBOX線のうちモデル想定確率(独立近似trifecta)0.5%未満=夢税帯
  (実測回収20-74%)を捨て、浮いた金をE/Fへ折半。同基準で⑬と比較。

3-C 保険複ダイヤル(P(沈没)連動):
  本命帯(0.20≤p1<0.35)にZ1-2a型P(沈没)分類器(月次WF・閾値=学習側三分位)。
  高: 保険複100→300(入替厚300→100から充当) / 低: 保険複0・単r3-r1-r2に+100 /
  中: 現行のまま。総額1,400円不変。現行固定と比較。
  採用検討基準: 現行比+3pt以上かつ高リスク層で保険複的中率がベース比リフト1.2倍以上。
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
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

MONTHS_ALL = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
              "2026-05", "2026-06", "2026-07", "2026-08"]
EVAL_MONTHS = MONTHS_ALL[1:]

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
df = build_training_set(conn)
res_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= '2025-12-01' AND res.arrival_order IS NOT NULL"):
    res_all[rid][lane] = ao
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
race_tech = dict(conn.execute(
    "SELECT race_id, winning_technique_number FROM races"))

# 選手プロファイル(3-C分類器用・shiftでリーク防止)
print("履歴プロファイル構築中...", flush=True)
hist = pd.read_sql_query(
    """
    SELECT r.race_id, r.date, r.race_no, e.lane, e.reg_no,
           res.arrival_order, res.st_time
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    """, conn)
conn.close()
hist = hist.sort_values(["reg_no", "date", "race_no"]).reset_index(drop=True)
has_res = hist["arrival_order"].notna()
hist["_sink"] = (hist["arrival_order"] >= 4).astype(float).where(has_res)
hist["_makuri_win"] = ((hist["arrival_order"] == 1)
                       & hist["race_id"].map(race_tech).isin([3, 4])
                       ).astype(float).where(has_res)
g = hist.groupby("reg_no", sort=False)
hist["p_makuri"] = g["_makuri_win"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
hist["p_sink"] = g["_sink"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
hist["p_st_sigma"] = g["st_time"].transform(
    lambda s: s.shift(1).rolling(20, min_periods=5).std())
prof = hist.set_index(["race_id", "lane"])[["p_makuri", "p_sink", "p_st_sigma"]]


def trio_comb(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def plan13_lines(lanes):
    """⑬を役割ラベル付き線に分解。[(役割, 券種, 組み合わせ, 円)]"""
    r = lanes
    lines = []
    for tag, members in (("A", (0, 1, 2)), ("B", (0, 1, 3))):
        for pm in permutations(members):
            role = f"{tag} {'-'.join(str(i + 1) for i in pm)}"
            comb = f"{r[pm[0]]}-{r[pm[1]]}-{r[pm[2]]}"
            lines.append((role, "3連単", comb, 100))
    lines.append(("E差され+300", "3連単", f"{r[2]}-{r[0]}-{r[1]}", 300))
    lines.append(("F差され+300", "3連単", f"{r[3]}-{r[0]}-{r[1]}", 300))
    lines.append(("G複", "3連複", trio_comb(r[2], r[3], r[4]), 200))
    return lines


def honmei_lines(lanes):
    r1, r2, r3, r4 = lanes[:4]
    return [
        ("複r1r2r3", "3連複", trio_comb(r1, r2, r3), 200),
        ("複r1r2r4", "3連複", trio_comb(r1, r2, r4), 200),
        ("複r1r3r4", "3連複", trio_comb(r1, r3, r4), 100),
        ("単r3-r1-r2", "3連単", f"{r3}-{r1}-{r2}", 200),
        ("単r4-r1-r2", "3連単", f"{r4}-{r1}-{r2}", 200),
        ("保険複r2r3r4", "3連複", trio_comb(r2, r3, r4), 100),
        ("入替r3-r2-r1", "3連単", f"{r3}-{r2}-{r1}", 100),
        ("入替r4-r2-r1", "3連単", f"{r4}-{r2}-{r1}", 300),
    ]


def score_lines(lines, pay):
    merged = defaultdict(int)
    for _role, bt, comb, y in lines:
        merged[(bt, comb)] += y
    return (sum(merged.values()),
            sum(pay.get(k, 0) * y // 100 for k, y in merged.items()))


def tri_prob(probs, a, b, c):
    pa, pb, pc = probs.get(a, 0), probs.get(b, 0), probs.get(c, 0)
    d1, d2 = 1 - pa, 1 - pa - pb
    if d1 <= 0 or d2 <= 0:
        return 0.0
    return pa * (pb / d1) * (pc / d2)


# ---- 月次walk-forward本体 ------------------------------------------------------
line13 = defaultdict(lambda: [0, 0, 0])    # {役割: [st, rt, hits]} 8か月分解
lineH = defaultdict(lambda: [0, 0, 0])
cum13 = defaultdict(lambda: [0, 0])        # 3-A政策用の累積(助走月含む・決定は前月まで)
arms3a = defaultdict(lambda: [0, 0])       # {(月, アーム): [st, rt]}
arms3b = defaultdict(lambda: [0, 0])
cut_counter = defaultdict(int)             # 3-Bで捨てられた線の役割分布
dropped_log = {}                           # 3-Aで各月外していた線
honmei_rows = []                           # 3-C分類器素材

for m in MONTHS_ALL:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])

    # 3-Aの当月ドロップ集合(前月までの累積で決定・当月データは使わない)
    dropped = {role for role, (st, rt) in cum13.items()
               if role[0] in "AB" and st >= 30 * 100 and rt / st < 0.50}
    dropped_log[m] = sorted(dropped)

    month_line13 = defaultdict(lambda: [0, 0])
    for rid, grp in month_df.groupby("race_id"):
        arr = res_all.get(rid, {})
        pay = payout_map[rid]
        if len(arr) < 3 or not pay:
            continue
        gs = grp.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in gs.iterrows()]
        lanes = [r["lane"] for r in ranked]
        p1 = ranked[0]["prob"]

        if p1 < 0.20 and len(lanes) >= 5:                    # --- 超混戦 ---
            base = plan13_lines(lanes)
            for role, bt, comb, y in base:
                got = pay.get((bt, comb), 0) * y // 100
                month_line13[role][0] += y
                month_line13[role][1] += got
                if m != "2025-12":
                    a = line13[role]
                    a[0] += y
                    a[1] += got
                    a[2] += got > 0
            if m == "2025-12":
                continue
            # 3-A政策アーム: ドロップ線の金をE/Fへ折半
            freed = sum(y for role, _b, _c, y in base if role in dropped)
            kept = [l for l in base if l[0] not in dropped]
            add_e = freed // 2 // 100 * 100
            add_f = freed - add_e
            pol = kept + [("E補強", "3連単", f"{lanes[2]}-{lanes[0]}-{lanes[1]}",
                           add_e)] if add_e else list(kept)
            if add_f:
                pol.append(("F補強", "3連単",
                            f"{lanes[3]}-{lanes[0]}-{lanes[1]}", add_f))
            st, rt = score_lines(base, pay)
            arms3a[(m, "現行⑬")][0] += st
            arms3a[(m, "現行⑬")][1] += rt
            pst, prt = score_lines(pol, pay)
            arms3a[(m, "棚卸し⑬")][0] += pst
            arms3a[(m, "棚卸し⑬")][1] += prt
            # 3-B夢税カット: BOX線で想定確率<0.5%を捨てE/Fへ折半
            probs = P.normalize_probs(ranked)
            cut = []
            keep = []
            for role, bt, comb, y in base:
                if role[0] in "AB":
                    a3, b3, c3 = (int(x) for x in comb.split("-"))
                    if tri_prob(probs, a3, b3, c3) < 0.005:
                        cut.append((role, y))
                        cut_counter[role] += 1
                        continue
                keep.append((role, bt, comb, y))
            freed = sum(y for _r, y in cut)
            add_e = freed // 2 // 100 * 100
            add_f = freed - add_e
            if add_e:
                keep.append(("E補強", "3連単",
                             f"{lanes[2]}-{lanes[0]}-{lanes[1]}", add_e))
            if add_f:
                keep.append(("F補強", "3連単",
                             f"{lanes[3]}-{lanes[0]}-{lanes[1]}", add_f))
            arms3b[(m, "現行⑬")][0] += st
            arms3b[(m, "現行⑬")][1] += rt
            bst, brt = score_lines(keep, pay)
            arms3b[(m, "夢税カット⑬")][0] += bst
            arms3b[(m, "夢税カット⑬")][1] += brt
            continue

        if 0.20 <= p1 < 0.35 and len(lanes) >= 4:            # --- 本命帯 ---
            base = honmei_lines(lanes)
            if m != "2025-12":
                for role, bt, comb, y in base:
                    got = pay.get((bt, comb), 0) * y // 100
                    a = lineH[role]
                    a[0] += y
                    a[1] += got
                    a[2] += got > 0
            fav = lanes[0]
            fav_row = gs.iloc[0]
            fav_arr = arr.get(fav)

            def pf(lane, col):
                try:
                    v = prof.loc[(rid, lane), col]
                    v = v.iloc[0] if hasattr(v, "iloc") else v
                    return float(v) if pd.notna(v) else np.nan
                except KeyError:
                    return np.nan

            outer = grp[grp["lane"].between(3, 6)]
            preds = [r["prob"] for r in ranked]
            honmei_rows.append({
                "race_id": rid, "date": str(fav_row["date"]), "month": m,
                "sink": 1 if (fav_arr is None or fav_arr >= 4) else 0,
                "lanes": lanes,
                "p1": p1, "gap12": p1 - preds[1],
                "fav_lane": fav, "fav_class": fav_row["racer_class_ord"],
                "fav_avg_st": fav_row["avg_st"],
                "fav_motor": fav_row["motor_2rate"],
                "fav_p_sink": pf(fav, "p_sink"),
                "fav_st_sigma": pf(fav, "p_st_sigma"),
                "st_edge": fav_row["avg_st"] - outer["avg_st"].min(),
                "atk_makuri": max((pf(int(l), "p_makuri")
                                   for l in outer["lane"]), default=np.nan),
                "kado_makuri": pf(4, "p_makuri"),
                "venue_code": int(fav_row["venue_code"]),
                "race_no": int(fav_row["race_no"]),
            })

    # 月末に累積へ反映(次月の決定に使う)
    for role, (st, rt) in month_line13.items():
        cum13[role][0] += st
        cum13[role][1] += rt

# ---- 3-A レポート --------------------------------------------------------------
print("\n===== 3-A 名前のある税の棚卸し(2026-01〜08) =====")
print("― ⑬の線別成績(役割ごと・8か月) ―")
for role in sorted(line13, key=lambda r: line13[r][1] / max(1, line13[r][0])):
    st, rt, h = line13[role]
    print(f"  {role:<12} 投資{st:>9,}円 回収率{rt / st:>7.1%} "
          f"的中{h:>3} 損益{rt - st:>+9,}円")
print("― 本命帯の行別成績(8か月) ―")
for role in sorted(lineH, key=lambda r: lineH[r][1] / max(1, lineH[r][0])):
    st, rt, h = lineH[role]
    print(f"  {role:<12} 投資{st:>9,}円 回収率{rt / st:>7.1%} "
          f"的中{h:>4} 損益{rt - st:>+9,}円")
print("― 政策アーム(累積50%未満のBOX線を翌月から外しE/Fへ) ―")
wins = 0
for m in EVAL_MONTHS:
    bs, br = arms3a[(m, "現行⑬")]
    ps, pr = arms3a[(m, "棚卸し⑬")]
    if not bs:
        continue
    wins += (pr / ps if ps else 0) >= br / bs
    d = dropped_log.get(m, [])
    print(f"  {m}: 現行{br / bs:>7.1%} vs 棚卸し{(pr / ps if ps else 0):>7.1%}"
          f"  外した線: {','.join(d) if d else 'なし'}")
bs = sum(arms3a[(m, "現行⑬")][0] for m in EVAL_MONTHS)
br = sum(arms3a[(m, "現行⑬")][1] for m in EVAL_MONTHS)
ps = sum(arms3a[(m, "棚卸し⑬")][0] for m in EVAL_MONTHS)
pr = sum(arms3a[(m, "棚卸し⑬")][1] for m in EVAL_MONTHS)
print(f"  合計: 現行{br / bs:.1%}({br - bs:+,}円) vs "
      f"棚卸し{pr / ps:.1%}({pr - ps:+,}円) / 同等以上の月{wins}/8")

# ---- 3-B レポート --------------------------------------------------------------
print("\n===== 3-B 夢税カット(想定確率0.5%未満のBOX線を捨てE/Fへ) =====")
wins = 0
for m in EVAL_MONTHS:
    bs, br = arms3b[(m, "現行⑬")]
    cs, cr = arms3b[(m, "夢税カット⑬")]
    if not bs:
        continue
    wins += (cr / cs if cs else 0) >= br / bs
    print(f"  {m}: 現行{br / bs:>7.1%} vs 夢税カット{(cr / cs if cs else 0):>7.1%}")
bs = sum(arms3b[(m, "現行⑬")][0] for m in EVAL_MONTHS)
br = sum(arms3b[(m, "現行⑬")][1] for m in EVAL_MONTHS)
cs = sum(arms3b[(m, "夢税カット⑬")][0] for m in EVAL_MONTHS)
cr = sum(arms3b[(m, "夢税カット⑬")][1] for m in EVAL_MONTHS)
print(f"  合計: 現行{br / bs:.1%}({br - bs:+,}円) vs "
      f"夢税カット{cr / cs:.1%}({cr - cs:+,}円) / 同等以上の月{wins}/8")
top_cut = sorted(cut_counter.items(), key=lambda kv: -kv[1])[:6]
print("  捨てられやすい線: " + " / ".join(f"{r}({c}回)" for r, c in top_cut))

# ---- 3-C レポート --------------------------------------------------------------
print("\n===== 3-C 保険複ダイヤル(本命帯×P(沈没)三分位) =====")
hd = pd.DataFrame(honmei_rows)
HFEATS = [c for c in hd.columns
          if c not in ("race_id", "date", "month", "sink", "lanes")]
dial = defaultdict(lambda: [0, 0])
layer_stat = defaultdict(lambda: [0, 0, 0])   # {層: [n, 沈没, 保険複的中]}
for m in EVAL_MONTHS:
    tr = hd[hd["date"] < f"{m}-01"]
    te = hd[hd["month"] == m]
    if len(tr) < 300 or te.empty:
        print(f"  {m}: 学習素材不足({len(tr)}行)でスキップ")
        continue
    ds = lgb.Dataset(tr[HFEATS], label=tr["sink"],
                     categorical_feature=["venue_code", "race_no", "fav_lane"])
    clf = lgb.train({"objective": "binary", "metric": "auc",
                     "learning_rate": 0.05, "num_leaves": 31,
                     "min_data_in_leaf": 50, "verbosity": -1, "seed": 7},
                    ds, num_boost_round=200)
    p_tr = clf.predict(tr[HFEATS])
    lo, hi = np.percentile(p_tr, [33.3, 66.7])
    p_te = clf.predict(te[HFEATS])
    for (_, row), pv in zip(te.iterrows(), p_te):
        lanes = row["lanes"]
        pay = payout_map[row["race_id"]]
        base = honmei_lines(lanes)
        layer = "低" if pv < lo else ("中" if pv < hi else "高")
        mod = []
        for role, bt, comb, y in base:
            if layer == "高":
                if role == "保険複r2r3r4":
                    y = 300
                elif role == "入替r4-r2-r1":
                    y = 100
            elif layer == "低":
                if role == "保険複r2r3r4":
                    y = 0
                elif role == "単r3-r1-r2":
                    y = 300
            if y:
                mod.append((role, bt, comb, y))
        st, rt = score_lines(base, pay)
        dial["現行固定"][0] += st
        dial["現行固定"][1] += rt
        mst, mrt = score_lines(mod, pay)
        dial["ダイヤル"][0] += mst
        dial["ダイヤル"][1] += mrt
        ls = layer_stat[layer]
        ls[0] += 1
        ls[1] += row["sink"]
        trio = trio_comb(*lanes[1:4])
        ls[2] += payout_map[row["race_id"]].get(("3連複", trio), 0) > 0

base_sink = hd[hd["month"] != "2025-12"]["sink"].mean()
print(f"  本命帯{len(hd):,}R 沈没率{base_sink:.1%}")
for layer in ("低", "中", "高"):
    n, s, h = layer_stat[layer]
    if n:
        print(f"  {layer}: {n:>5}R 沈没率{s / n:>6.1%} 保険複的中率{h / n:>6.1%}")
for arm, (st, rt) in dial.items():
    print(f"  {arm:<6} 投資{st:>11,}円 回収率{rt / st:>7.1%} 損益{rt - st:>+10,}円")
print("\n(採用判定はいずれも事前登録基準に従う。小標本の回収率単独では動かない)")
