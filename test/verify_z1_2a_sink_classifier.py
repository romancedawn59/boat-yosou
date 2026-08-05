# -*- coding: utf-8 -*-
"""Z1-2a: P(沈没)分類器の先行ベンチマーク(2026-08-05着手・ケンさん号令)

    py -X utf8 test/verify_z1_2a_sink_classifier.py

■ 位置づけ(z1-tenkai-research正本)
沈没=モデル1位艇が4着以下。オラクル天井は測定済み:
本命帯(20-35%)沈没率29.5%・オラクル×保険複272.4% → 必要精度37%
堅め帯(50%+)沈没率10.6%・オラクル×保険複598.2% → 必要精度17%
Z1-2aはLightGBM分類器の先行ベンチマーク。後続Z1-2b(モンテカルロ)が
これに勝てたら機構理解が本物、という物差しを先に立てる。

■ 事前登録(結果を見る前に固定)
- 帯の再現・沈没定義はZ1-1.5(verify_favorite_sinks_tenkai.py)と同一
  (walk-forward 5fold・全24場)。分類器の学習は各foldで「それ以前のfold」のみ
  =fold1は学習専用で評価から除外(fold2-5で判定)
- 閾値は後出しで選ばない: 各foldの学習データ上で目標精度(本命37%/堅め17%)を
  達成する最小閾値を決め、そのままテストに適用する
- 判定基準(帯ごと):
  (1) テスト精度が必要精度(37%/17%)以上
  (2) フラグ立ちレースの保険複r2r3r4(100円)回収率が150%以上
  両方を満たした帯は「ゲート実装の相談」に進む。満たさない場合は
  数字を正直に報告し、特徴量強化かZ1-2bへ(即採用はしない)
- 分類器の特徴量は「予測時点で知り得る情報」のみ(進入・決まり手・展示は使わない。
  選手のまくり率等はすべて当該レースより前の履歴から算出)
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import lightgbm as lgb
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

conn = db.connect(DB_PATH)
df = build_training_set(conn)

# ---- レース結果・払戻・選手履歴(リーク安全なプロファイル計算用) ----------------
res_all = defaultdict(dict)
for rid, lane, ao in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id WHERE r.date >= ?", (TEST_START,)):
    res_all[rid][lane] = ao
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0

# 全期間の出走履歴(プロファイルは履歴全体から、shiftでリーク防止)
hist = pd.read_sql_query(
    """
    SELECT r.race_id, r.date, r.race_no, r.winning_technique_number AS tech,
           e.lane, e.reg_no, res.arrival_order, res.st_time, res.course
    FROM entries e
    JOIN races r ON r.race_id = e.race_id
    LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
    """, conn)
conn.close()

hist = hist.sort_values(["reg_no", "date", "race_no"]).reset_index(drop=True)
has_res = hist["arrival_order"].notna()
hist["_start"] = has_res.astype(float)
hist["_sink"] = (hist["arrival_order"] >= 4).astype(float).where(has_res)
hist["_makuri_win"] = ((hist["arrival_order"] == 1) & hist["tech"].isin([3, 4])
                       ).astype(float).where(has_res)
hist["_win_c35"] = ((hist["arrival_order"] == 1) & hist["course"].between(3, 5)
                    ).astype(float).where(has_res)
g = hist.groupby("reg_no", sort=False)
# 通算まくり系勝率(出走あたり)・外コース勝率・沈没率・ST安定度(直近20走σ)
hist["p_makuri"] = g["_makuri_win"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
hist["p_win_c35"] = g["_win_c35"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
hist["p_sink"] = g["_sink"].transform(
    lambda s: s.shift(1).expanding(min_periods=10).mean())
hist["p_st_sigma"] = g["st_time"].transform(
    lambda s: s.shift(1).rolling(20, min_periods=5).std())
prof = hist.set_index(["race_id", "lane"])[
    ["p_makuri", "p_win_c35", "p_sink", "p_st_sigma"]]

# ---- v2のwalk-forwardで帯とモデル順位を再現(Z1-1.5と同一) --------------------
test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

rows = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_v2 = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} v2学習中...", flush=True)
    booster = train_fold(train_v2)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, grp in fold_df.groupby("race_id"):
        arr = {l: a for l, a in res_all.get(rid, {}).items() if a}
        if len(arr) < 3 or not payout_map[rid]:
            continue
        gs = grp.sort_values("pred", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        preds = [float(x) for x in gs["pred"]]
        if len(lanes) < 5:
            continue
        p1 = preds[0]
        band = ("honmei" if 0.20 <= p1 < 0.35
                else "katame" if p1 >= 0.50 else None)
        if band is None:
            continue
        fav = lanes[0]
        fav_row = gs.iloc[0]
        fav_arr = arr.get(fav)
        sink = 1 if (fav_arr is None or fav_arr >= 4) else 0

        def pf(lane, col):
            try:
                v = prof.loc[(rid, lane), col]
                return float(v) if pd.notna(v) else np.nan
            except KeyError:
                return np.nan

        outer = grp[grp["lane"].between(3, 6)]
        atk_st = outer["avg_st"].min()
        atk_motor = outer["motor_2rate"].max()
        atk_class = outer["racer_class_ord"].max()
        atk_makuri = max((pf(int(l), "p_makuri") for l in outer["lane"]),
                         default=np.nan)
        atk_c35 = max((pf(int(l), "p_win_c35") for l in outer["lane"]),
                      default=np.nan)
        kado = grp[grp["lane"] == 4]
        rows.append({
            "race_id": rid, "date": fav_row["date"], "fold": i + 1,
            "band": band, "sink": sink,
            "lanes": lanes,
            # --- 分類器特徴量 ---
            "p1": p1, "gap12": p1 - preds[1], "gap23": preds[1] - preds[2],
            "fav_lane": fav, "fav_class": fav_row["racer_class_ord"],
            "fav_avg_st": fav_row["avg_st"],
            "fav_form_st": fav_row["form_last10_avg_st"],
            "fav_motor": fav_row["motor_2rate"],
            "fav_p_sink": pf(fav, "p_sink"),
            "fav_st_sigma": pf(fav, "p_st_sigma"),
            "st_edge": (fav_row["avg_st"] - atk_st) if pd.notna(atk_st) else np.nan,
            "atk_motor_edge": (atk_motor - fav_row["motor_2rate"])
                              if pd.notna(atk_motor) else np.nan,
            "atk_class": atk_class,
            "atk_makuri": atk_makuri, "atk_c35": atk_c35,
            "kado_makuri": pf(4, "p_makuri") if len(kado) else np.nan,
            "kado_st": kado["avg_st"].iloc[0] if len(kado) else np.nan,
            "venue_code": int(fav_row["venue_code"]),
            "race_no": int(fav_row["race_no"]),
        })

data = pd.DataFrame(rows)
FEATS = ["p1", "gap12", "gap23", "fav_lane", "fav_class", "fav_avg_st",
         "fav_form_st", "fav_motor", "fav_p_sink", "fav_st_sigma",
         "st_edge", "atk_motor_edge", "atk_class", "atk_makuri", "atk_c35",
         "kado_makuri", "kado_st", "venue_code", "race_no"]
TARGETS = {"honmei": 0.37, "katame": 0.17}
print(f"\n標本: {len(data):,}R (本命帯{(data['band']=='honmei').sum():,}"
      f" / 堅め帯{(data['band']=='katame').sum():,})")
for band in ("honmei", "katame"):
    d = data[data["band"] == band]
    print(f"  {band}: 沈没率{d['sink'].mean():.1%}")

# ---- Z1-2a分類器: fold2-5をwalk-forwardで評価 --------------------------------
def train_clf(tr):
    ds = lgb.Dataset(tr[FEATS], label=tr["sink"],
                     categorical_feature=["venue_code", "race_no", "fav_lane"])
    return lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8,
         "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
         "seed": 7},
        ds, num_boost_round=300)


def precision_threshold(y, p, target):
    """精度がtargetに達する最小閾値(学習データ上で決める)。届かなければNone"""
    order = np.argsort(-p)
    ys, ps = np.asarray(y)[order], np.asarray(p)[order]
    cum_hit = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    prec = cum_hit / k
    ok = np.where((prec >= target) & (k >= 20))[0]   # 最低20本は立てる
    if len(ok) == 0:
        return None
    return ps[ok[-1]]                                 # 最も広く取れる点


agg = {b: {"n": 0, "flag": 0, "hit": 0, "st": 0, "rt": 0, "auc": []}
       for b in TARGETS}
pooled = {b: {"y": [], "p": [], "rid": [], "lanes": []} for b in TARGETS}
for fold in range(2, N_FOLDS + 1):
    tr = data[data["fold"] < fold]
    te = data[data["fold"] == fold]
    if tr.empty or te.empty:
        continue
    # 閾値較正は学習データ内の時系列ホールドアウトで行う(v0は学習データ自身の
    # 予測で較正したため過学習した精度曲線に基づく甘い閾値になった。2026-08-05修正)
    tr_dates = sorted(tr["date"].unique())
    cal_start = tr_dates[int(len(tr_dates) * 0.75)]
    fit, cal = tr[tr["date"] < cal_start], tr[tr["date"] >= cal_start]
    clf = train_clf(fit)
    p_cal = clf.predict(cal[FEATS])
    p_te = clf.predict(te[FEATS])
    for band, target in TARGETS.items():
        calb, teb = cal["band"] == band, te["band"] == band
        if teb.sum() == 0:
            continue
        try:
            from sklearn.metrics import roc_auc_score
            agg[band]["auc"].append(
                roc_auc_score(te.loc[teb, "sink"], p_te[teb.values]))
        except Exception:
            pass
        pooled[band]["y"] += list(te.loc[teb, "sink"])
        pooled[band]["p"] += list(p_te[teb.values])
        pooled[band]["rid"] += list(te.loc[teb, "race_id"])
        pooled[band]["lanes"] += list(te.loc[teb, "lanes"])
        thr = precision_threshold(cal.loc[calb, "sink"], p_cal[calb.values],
                                  target)
        if thr is None:
            continue
        flags = te[teb & (p_te >= thr)]
        a = agg[band]
        a["n"] += int(teb.sum())
        a["flag"] += len(flags)
        a["hit"] += int(flags["sink"].sum())
        for _, r in flags.iterrows():
            trio = sorted(r["lanes"][1:4])
            comb = f"{trio[0]}={trio[1]}={trio[2]}"
            a["st"] += 100
            a["rt"] += payout_map[r["race_id"]].get(("3連複", comb), 0)

print("\n===== Z1-2a 判定(fold2-5・閾値は学習側で固定) =====")
for band, target in TARGETS.items():
    a = agg[band]
    name = "本命帯(20-35%)" if band == "honmei" else "堅め帯(50%+)"
    auc = np.mean(a["auc"]) if a["auc"] else float("nan")
    if a["flag"] == 0:
        print(f"{name}: 学習側で精度{target:.0%}に届く閾値なし(AUC {auc:.3f})")
        continue
    prec = a["hit"] / a["flag"]
    roi = a["rt"] / a["st"] if a["st"] else 0
    base = data[(data["band"] == band) & (data["fold"] >= 2)]["sink"].mean()
    print(f"{name}: 評価{a['n']:,}R AUC {auc:.3f} / フラグ{a['flag']}本 "
          f"精度{prec:.1%}(ベース{base:.1%}・必要{target:.0%}) / "
          f"保険複ROI {roi:.1%}(基準150%)")
    ok = prec >= target and roi >= 1.5
    print(f"  → {'両基準クリア: ゲート実装の相談へ' if ok else '基準未達: 正直に報告し特徴量強化/Z1-2bへ'}")

# 参考: 到達可能な作動点の全体像(プール済みテスト予測の上位K%precision)
print("\n上位K%作動点(テスト・fold2-5プール・参考):")
for band in TARGETS:
    y = np.array(pooled[band]["y"])
    p = np.array(pooled[band]["p"])
    if len(y) == 0:
        continue
    order = np.argsort(-p)
    line = []
    for pct in (2, 5, 10, 20):
        k = max(1, int(len(y) * pct / 100))
        top = order[:k]
        prec = y[top].mean()
        st = rt = 0
        for idx in top:
            lanes = pooled[band]["lanes"][idx]
            trio = sorted(lanes[1:4])
            comb = f"{trio[0]}={trio[1]}={trio[2]}"
            st += 100
            rt += payout_map[pooled[band]["rid"][idx]].get(("3連複", comb), 0)
        line.append(f"top{pct}%: 精度{prec:.1%}/複ROI{rt / st:.0%}({k}R)")
    name = "本命帯" if band == "honmei" else "堅め帯"
    print(f"  {name}: " + " | ".join(line))

clf_full = train_clf(data)
imp = sorted(zip(FEATS, clf_full.feature_importance("gain")),
             key=lambda x: -x[1])
print("\n特徴量重要度(gain上位10):")
tot = sum(v for _f, v in imp)
for f, v in imp[:10]:
    print(f"  {f:<16}{v / tot:>6.1%}")
