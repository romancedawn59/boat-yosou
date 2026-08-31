# -*- coding: utf-8 -*-
"""本番忠実版シミュレーション: 全艇ランク+返還処理で⑬とVAB注入を再検証
(2026-08-31晩・リーク発覚を受けた緊急再検証)

    py -X utf8 test/sim_clean_rank_2026.py

■ 発覚したリーク(このスクリプトが直すもの)
従来の超混戦バックテストは build_training_set(=resultsとINNER JOIN)の行で
レース内順位を作っていた。非完走艇(F・欠場・事故=着順NULL)は行ごと消えるため、
(1)「特徴量欠け」層別の正体は非完走レースの後知恵選別だった(1-Bの315%は無効)
(2) 事故で強い艇が消えると残りの最大確率が下がり、事故レースほど超混戦帯に
    吸い込まれる(帯判定のリーク)
(3) 買い目も「消えた艇を避けた」構成になっていた(構成のリーク)
本番のpredict.pyは全6艇の番組表でランクするので、この歪みは本番に存在しない。

■ 方法(本番忠実)
- 学習: build_training_set(完走艇のみ・従来通り=正当)
- 評価: 全entries(非完走艇含む)に特徴量を付けて予測し、全艇でランク付け
- 返還処理: 非完走艇のうちST記録なし(F/L/欠場の近似)を含む買い目は投資額返還、
  ST記録あり(スタート後の事故)を含む買い目は没(実際の返還規定の近似)
- 月次walk-forward 2026-01〜08。変種はV0(現行37特徴量)とVAB(+KR指数・KR順位・
  伸びギャップ・期内走数・通算2連対率)の2本。
- 超混戦(p1<0.20)は⑬2,000円、本命帯(0.20≤p1<0.35)は現行9行1,400円。
  全24場と実運用5場の両方を集計。軸生存率は完走艇の3着内で判定。

■ 事前登録(結果を見る前に固定)
- ⑬の真の回収率が全24場・8か月で100%を下回る場合、「超混戦バックテスト優位は
  主にリーク由来」と結論し、紙上降格の継続を勧告する
- VAB採用基準は従来と同じ: 8月の軸生存率がV0比+10pt以上かつloglossがV0以下
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
                      _encode, build_training_set, compute_form_features)

EVAL_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04",
               "2026-05", "2026-06", "2026-07", "2026-08"]
PARAMS = {"objective": "binary", "metric": "auc", "verbosity": -1,
          "learning_rate": 0.05, "num_leaves": 31}

print("データ準備中...", flush=True)
conn = db.connect(DB_PATH)
train_all = build_training_set(conn)

# 評価用: 全entries(非完走含む)+特徴量
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

payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2026-01-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
race_tech = dict(conn.execute(
    "SELECT race_id, winning_technique_number FROM races"))

# ---- KR指数(全entriesにレース前の値を付与) -------------------------------------
print("KR指数構築中...", flush=True)
races_key = dict(conn.execute(
    "SELECT race_id, date || '_' || printf('%02d', race_no) FROM races"))
entry_lanes = defaultdict(list)
lane_racer = {}
for rid, lane, reg in conn.execute("SELECT race_id, lane, reg_no FROM entries"):
    lane_racer[(rid, lane)] = reg
    entry_lanes[rid].append(lane)
finish_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    finish_all[rid][lane] = ao

R = defaultdict(lambda: 1500.0)
K = 3.0
kr_pre = {}
for rid in sorted(entry_lanes, key=lambda r: races_key.get(r, "")):
    for lane in entry_lanes[rid]:
        kr_pre[(rid, lane)] = R[lane_racer[(rid, lane)]]
    boats = [(lane, ao) for lane, ao in finish_all.get(rid, {}).items()
             if (rid, lane) in lane_racer]
    for i in range(len(boats)):
        for j in range(i + 1, len(boats)):
            li, ai = boats[i]
            lj, aj = boats[j]
            ri, rj = lane_racer[(rid, li)], lane_racer[(rid, lj)]
            e = 1 / (1 + 10 ** ((R[rj] - R[ri]) / 400))
            s = 1.0 if ai < aj else 0.0
            R[ri] += K * (s - e)
            R[rj] -= K * (s - e)

# ---- 伸びギャップ・期内走数・通算2連対率(shiftでリーク防止) --------------------
print("履歴プロファイル構築中...", flush=True)
hist = pd.read_sql_query("""
    SELECT r.race_id, r.date, r.race_no, e.lane, e.reg_no, res.arrival_order
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
""", conn)
conn.close()
hist = hist.sort_values(["reg_no", "date", "race_no"]).reset_index(drop=True)
hist["dt"] = pd.to_datetime(hist["date"])
has_res = hist["arrival_order"].notna()
hist["_top2"] = (hist["arrival_order"] <= 2).astype(float).where(has_res)
g = hist.set_index("dt").groupby("reg_no", sort=False)
hist["hist90_top2"] = g["_top2"].transform(
    lambda s: s.rolling("90D", min_periods=5).mean().shift(1)).values
g2 = hist.groupby("reg_no", sort=False)
hist["career_top2"] = g2["_top2"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
pstart = hist["dt"].dt.year.astype(str) + np.where(
    hist["dt"].dt.month >= 7, "-07", "-01")
hist["_pkey"] = hist["reg_no"].astype(str) + pstart
hist["n_starts_period"] = hist.groupby("_pkey", sort=False).cumcount()
prof = hist.set_index(["race_id", "lane"])[
    ["hist90_top2", "career_top2", "n_starts_period"]]


def attach_extras(df):
    df = df.copy()
    df["kr"] = [kr_pre.get((r, l), np.nan)
                for r, l in zip(df["race_id"], df["lane"])]
    idx = pd.MultiIndex.from_arrays([df["race_id"], df["lane"]])
    j = prof.reindex(idx)
    df["hist90_top2"] = j["hist90_top2"].values
    df["career_top2"] = j["career_top2"].values
    df["n_starts_period"] = j["n_starts_period"].values
    df["nobi_gap"] = df["hist90_top2"] - df["national_2rate"] / 100.0
    df["kr_rank"] = df.groupby("race_id")["kr"].rank(ascending=False)
    return df


train_all = attach_extras(train_all)
eval_df = attach_extras(eval_df)

VARIANTS = {
    "V0 現行": list(FEATURE_COLUMNS),
    "VAB 注入": list(FEATURE_COLUMNS) + ["kr", "kr_rank", "nobi_gap",
                                          "n_starts_period", "career_top2"],
}


def train_variant(train_df, feats):
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr = train_df[train_df["date"] < cutoff]
    va = train_df[train_df["date"] >= cutoff]
    ds = lgb.Dataset(tr[feats], label=tr["is_winner"],
                     categorical_feature=CATEGORICAL_FEATURES)
    vs = lgb.Dataset(va[feats], label=va["is_winner"], reference=ds)
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


def honmei_plan(lanes):
    r1, r2, r3, r4 = lanes[:4]
    return [
        ("3連複", trio_comb(r1, r2, r3), 200),
        ("3連複", trio_comb(r1, r2, r4), 200),
        ("3連複", trio_comb(r1, r3, r4), 100),
        ("3連単", f"{r3}-{r1}-{r2}", 200),
        ("3連単", f"{r4}-{r1}-{r2}", 200),
        ("3連複", trio_comb(r2, r3, r4), 100),
        ("3連単", f"{r3}-{r2}-{r1}", 100),
        ("3連単", f"{r4}-{r2}-{r1}", 300),
    ]


def score_with_refund(bets, pay, refund_lanes, dead_lanes):
    """返還近似: refund_lanes(F/L/欠場近似)を含む目は投資返還。
    dead_lanes(スタート後の事故)を含む目は没。それ以外は通常採点"""
    merged = defaultdict(int)
    for bt, comb, y in bets:
        merged[(bt, comb)] += y
    st = rt = 0
    for (bt, comb), y in merged.items():
        sep = "-" if bt == "3連単" else "="
        members = {int(x) for x in comb.split(sep)}
        st += y
        if members & refund_lanes:
            rt += y                      # 返還=実質ノーカウント
        else:
            rt += pay.get((bt, comb), 0) * y // 100
    return st, rt


# ---- 月次walk-forward ----------------------------------------------------------
M = defaultdict(lambda: defaultdict(float))
for m in EVAL_MONTHS:
    tr_df = train_all[train_all["date"] < f"{m}-01"]
    ev = eval_df[eval_df["date"].str.startswith(m)]
    if ev.empty:
        continue
    for vname, feats in VARIANTS.items():
        print(f"{m} {vname} 学習中...", flush=True)
        booster = train_variant(tr_df, feats)
        md = ev.copy()
        md["pred"] = booster.predict(md[feats])
        # logloss(完走艇=ラベル確定行のみで測る)
        lab = md[md["arrival_order"].notna()]
        p = lab["pred"].clip(1e-9, 1 - 1e-9)
        yv = (lab["arrival_order"] == 1).astype(float)
        a = M[(vname, m)]
        a["ll"] += float(-(yv * np.log(p) + (1 - yv) * np.log(1 - p)).sum())
        a["nll"] += len(lab)
        for rid, grp in md.groupby("race_id"):
            pay = payout_map[rid]
            if not pay:
                continue
            gs = grp.sort_values("pred", ascending=False)
            lanes = [int(x) for x in gs["lane"]]
            p1 = float(gs["pred"].iloc[0])
            if len(lanes) < 5:
                continue
            arr = {int(r["lane"]): r["arrival_order"] for _, r in grp.iterrows()
                   if pd.notna(r["arrival_order"])}
            if len(arr) < 3:
                continue
            nonfin = {int(r["lane"]) for _, r in grp.iterrows()
                      if pd.isna(r["arrival_order"])}
            refund = {l for l in nonfin
                      if pd.isna(grp[grp["lane"] == l]["st_time"].iloc[0])}
            dead = nonfin - refund
            top3 = sorted(arr, key=arr.get)[:3]
            ax = lanes[0] in top3 and lanes[1] in top3
            in5 = int(gs["venue_code"].iloc[0]) in TARGET_VENUE_CODES
            if p1 < 0.20:
                st, rt = score_with_refund(plan13(lanes), pay, refund, dead)
                a["kn"] += 1
                a["kst"] += st
                a["krt"] += rt
                a["kax"] += ax
                a["kdirty"] += bool(nonfin)
                if in5:
                    a["kn5"] += 1
                    a["kst5"] += st
                    a["krt5"] += rt
            elif p1 < 0.35:
                st, rt = score_with_refund(honmei_plan(lanes), pay, refund, dead)
                a["hn"] += 1
                a["hst"] += st
                a["hrt"] += rt
                a["hax"] += ax
                a["h1"] += arr.get(lanes[0]) == 1

print("\n===== 本番忠実版(全艇ランク+返還処理) =====")
print("― 超混戦帯・⑬2,000円(全24場) ―")
print(f"{'月':<9}" + "".join(f"{v:<30}" for v in VARIANTS)
      + "(各セル: R数/軸生存/回収率/事故R率)")
for m in EVAL_MONTHS:
    row = f"{m:<9}"
    for vname in VARIANTS:
        a = M[(vname, m)]
        kn = a["kn"]
        axr = a["kax"] / kn if kn else 0
        roi = a["krt"] / a["kst"] if a["kst"] else 0
        dr = a["kdirty"] / kn if kn else 0
        row += f"{kn:>4.0f}R {axr:>6.1%} {roi:>7.1%} {dr:>5.1%}   "
    print(row)
print("― 合計 ―")
for vname in VARIANTS:
    kn = sum(M[(vname, m)]["kn"] for m in EVAL_MONTHS)
    kax = sum(M[(vname, m)]["kax"] for m in EVAL_MONTHS)
    kst = sum(M[(vname, m)]["kst"] for m in EVAL_MONTHS)
    krt = sum(M[(vname, m)]["krt"] for m in EVAL_MONTHS)
    k5n = sum(M[(vname, m)]["kn5"] for m in EVAL_MONTHS)
    k5st = sum(M[(vname, m)]["kst5"] for m in EVAL_MONTHS)
    k5rt = sum(M[(vname, m)]["krt5"] for m in EVAL_MONTHS)
    ll = sum(M[(vname, m)]["ll"] for m in EVAL_MONTHS)
    nll = sum(M[(vname, m)]["nll"] for m in EVAL_MONTHS)
    hn = sum(M[(vname, m)]["hn"] for m in EVAL_MONTHS)
    hst = sum(M[(vname, m)]["hst"] for m in EVAL_MONTHS)
    hrt = sum(M[(vname, m)]["hrt"] for m in EVAL_MONTHS)
    h1 = sum(M[(vname, m)]["h1"] for m in EVAL_MONTHS)
    hax = sum(M[(vname, m)]["hax"] for m in EVAL_MONTHS)
    print(f"{vname}: logloss {ll / nll:.5f}")
    print(f"  超混戦(全場) {kn:.0f}R 軸生存{kax / kn:.1%} "
          f"⑬回収率{krt / kst:.1%} 損益{krt - kst:>+10,.0f}円")
    print(f"  超混戦(5場)  {k5n:.0f}R ⑬回収率"
          f"{(k5rt / k5st if k5st else 0):.1%} 損益{k5rt - k5st:>+10,.0f}円")
    print(f"  本命帯(全場) {hn:.0f}R 1位的中{h1 / hn:.1%} 軸生存{hax / hn:.1%} "
          f"回収率{hrt / hst:.1%} 損益{hrt - hst:>+10,.0f}円")

# 8月のVAB判定(事前登録基準)
a0, av = M[("V0 現行", "2026-08")], M[("VAB 注入", "2026-08")]
if a0["kn"] and av["kn"]:
    d = av["kax"] / av["kn"] - a0["kax"] / a0["kn"]
    ll0 = sum(M[("V0 現行", m)]["ll"] for m in EVAL_MONTHS) / \
        sum(M[("V0 現行", m)]["nll"] for m in EVAL_MONTHS)
    llv = sum(M[("VAB 注入", m)]["ll"] for m in EVAL_MONTHS) / \
        sum(M[("VAB 注入", m)]["nll"] for m in EVAL_MONTHS)
    print(f"\nVAB判定: 8月軸生存差{d:+.1%}(基準+10pt) / "
          f"logloss {llv:.5f} vs {ll0:.5f}({'改善' if llv <= ll0 else '悪化'})")
print("(事前登録: ⑬全場8か月が100%未満なら『超混戦優位は主にリーク由来』と結論)")
