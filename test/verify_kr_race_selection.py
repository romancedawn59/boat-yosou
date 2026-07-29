# -*- coding: utf-8 -*-
"""KR指数(Elo)はレース選別に連携できるか(2026-07-29深夜・ケンさん発案)

    py -X utf8 test/verify_kr_race_selection.py

■ 理屈(伸び盛りレース選別と同型)
検証⑬でKRのモデル注入は不合格(1着の見極めは市場も鋭く換金されない)。
残る出口は市場側: ラベル税の実測(レース内KR1位がB級なら単勝回収88.5%、
A1なら76.1%=+12.4pt)は「市場が級別の看板で値付けし、実力(KR)との乖離を
見逃す」ことを示す。ならば「KR1位がB級のレース」は市場の値付けが甘い
レースであり、選別マーカーになりうる(verify_rising_race_selection.pyと同じ構図)。

■ 事前登録(実行前に固定)
旗: レース内KR1位(バーンイン後のElo・更新前値=リークなし)の艇の級がB1/B2
分割: 本命(5場20-30%cap6・保険複入り構成)/超混戦(全場20%未満・案1構成)
判定: いずれかの帯で旗ありが旗なしを回収率+15pt以上上回れば8月紙上追跡へ。
今夜の選別変更はしない。参考表示(採否に使わない): KR1位が非A1の広い旗。
"""
import importlib.util
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import sqlite3

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

# KR指数(Elo)は検証⑬のビルダーを再利用(時系列逐次更新・更新前値=WF安全)
spec = importlib.util.spec_from_file_location(
    "abl", r"Y:\マイドライブ\boat\test\verify_v2_features_ablation.py")
abl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abl)

print("KR指数を構築中...", flush=True)
extra = abl.build_extra_features()          # (race_id, lane, kr_rating, ...)
kr = {(r.race_id, int(r.lane)): float(r.kr_rating)
      for r in extra.itertuples() if r.kr_rating == r.kr_rating}

raw = sqlite3.connect(DB_PATH)
klass = {(rid, lane): kl for rid, lane, kl in raw.execute(
    "SELECT race_id, lane, racer_class FROM entries")}
raw.close()


def kr_top_class(rid, lanes):
    """レース内KR1位の艇の級(KR未整備の艇がいれば判定しない)"""
    vals = [(kr.get((rid, l)), l) for l in lanes]
    if any(v is None for v, _ in vals):
        return None
    top_lane = max(vals)[1]
    return klass.get((rid, top_lane))


conn = db.connect(DB_PATH)
df = build_training_set(conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,)):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

test_df = df[df["date"] >= TEST_START]
dates = sorted(test_df["date"].unique())
fold_size = len(dates) // N_FOLDS
boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

konsen_ctx, honmei_ctx = [], []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        if not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        top = ranked[0]["prob"]
        c = {"rid": rid, "date": g["date"].iloc[0], "top": top, "ranked": ranked}
        if top < 0.20:
            konsen_ctx.append(c)
        elif (top < 0.30
              and int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES):
            honmei_ctx.append(c)

by_day = defaultdict(list)
for c in honmei_ctx:
    by_day[c["date"]].append(c)
honmei_sel = []
for d, cs in by_day.items():
    cs.sort(key=lambda c: c["top"])
    honmei_sel.extend(cs[:6])

for band, ctxs, konsen in (("超混戦帯(全場・案1構成)", konsen_ctx, True),
                           ("本命帯(5場cap6・保険複入り構成)", honmei_sel, False)):
    # {旗: [投資, 回収, R数]} 旗: "B級"(主判定) / "A2以下"(参考) / "A1"(なし側)
    agg = defaultdict(lambda: [0, 0, 0])
    for c in ctxs:
        lanes = [r["lane"] for r in c["ranked"]]
        kc = kr_top_class(c["rid"], lanes)
        if kc is None:
            continue
        plan = P.ken_portfolio("荒れ注意", c["ranked"], [],
                               P.picks_katsu(P.normalize_probs(c["ranked"])),
                               konsen=konsen)
        if not plan:
            continue
        pay = payout_map[c["rid"]]
        st = sum(y for _, _, y, _ in plan)
        rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _s in plan)
        keys = ["旗あり(KR1位=B級)"] if kc in ("B1", "B2") else ["旗なし(KR1位=A級)"]
        if kc != "A1":
            keys.append("参考: KR1位が非A1")
        for k in keys:
            a = agg[k]
            a[0] += st
            a[1] += rt
            a[2] += 1
    print(f"\n=== {band} ===")
    for k in ("旗あり(KR1位=B級)", "旗なし(KR1位=A級)", "参考: KR1位が非A1"):
        st, rt, n = agg[k]
        if st:
            print(f"  {k:<18} {n:>5,}R 回収率{rt/st:>7.1%}")
    a1 = agg["旗あり(KR1位=B級)"]
    a0 = agg["旗なし(KR1位=A級)"]
    if a1[0] and a0[0]:
        lift = a1[1] / a1[0] - a0[1] / a0[0]
        print(f"  リフト: {lift*100:+.1f}pt "
              f"{'→ +15pt基準クリア・8月紙上追跡へ' if lift >= 0.15 else '→ 基準未達'}")
