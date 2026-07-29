# -*- coding: utf-8 -*-
"""伸び盛り艇の存在は勝負所「選定」の理由になるか(2026-07-29判断会・議題C派生)

    py -X utf8 test/verify_rising_race_selection.py

■ 問い(ケンさん)
伸び盛りマーク(直近90日実測2連対率−番組表2連率>+10pt)を表示に留めず、
本命・超混戦レースの選定理由の一つにできるか。

■ 理屈の整理(事前登録)
- モデル注入は⑬で不合格(モデルはform列から自力で差分を作れる=新情報でない)
- しかし伸び盛りの本質は「市場のアンカー(番組表勝率・級別)の古さ」=市場側の歪み。
  買い目の回収率は(モデルの精度−市場の精度)で決まるため、
  「市場が古い値付けをしている艇がいるレース」は配当が構造的に甘い可能性がある
  →レース選定のマーカーとしては理論上成立する(⑬とは別ルート)
- 反対材料: E/F単×伸び盛り頭の複利は超混戦で棄却済み(verify_ef_rising)。
  本命帯112.7%vs84.8%は事後観察(未検証)

■ 事前登録
分割: モデル上位4艇に伸び盛り艇(gap>+0.10、90日窓12走以上)が1艇でもいるか
プラン: 今夜採用したv2.1構成(超混戦=案1 / 本命=保険複入りV2構成)
判定: いずれかの帯で「あり」が「なし」を回収率+15pt以上上回れば
「8月の紙上選別チャレンジャーに追加し9/1に判定」。今夜の選定ロジック変更はしない
(採用は前向き検証を通ってから。今夜は追跡する価値の有無だけを決める)。
"""
import datetime as dt
import sys
from bisect import bisect_left
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import sqlite3

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

GAP_MIN = 0.10
WINDOW = 90
MIN_RACES = 12

# --- 伸びギャップの下ごしらえ(verify_ef_rising.pyと同一ロジック) ---
raw = sqlite3.connect(DB_PATH)
races_date = dict(raw.execute("SELECT race_id, date FROM races"))
lane_racer, printed = {}, {}
for rid, lane, reg, n2 in raw.execute(
        "SELECT race_id, lane, reg_no, national_2rate FROM entries"):
    lane_racer[(rid, lane)] = reg
    printed[(rid, lane)] = n2
hist = defaultdict(list)
rows = []
for rid, lane, ao in raw.execute(
        "SELECT race_id, lane, arrival_order FROM results "
        "WHERE arrival_order IS NOT NULL"):
    d = races_date.get(rid)
    reg = lane_racer.get((rid, lane))
    if d and reg:
        rows.append((d, reg, ao))
rows.sort()
for d, reg, ao in rows:
    hist[reg].append((d, 1 if ao <= 2 else 0))
hist_dates = {reg: [x[0] for x in v] for reg, v in hist.items()}
raw.close()


def gap_of(rid, lane):
    reg = lane_racer.get((rid, lane))
    n2 = printed.get((rid, lane))
    d = races_date.get(rid)
    if reg is None or n2 is None or d is None or reg not in hist:
        return None
    dates = hist_dates[reg]
    hi = bisect_left(dates, d)
    d0 = (dt.date.fromisoformat(d) - dt.timedelta(days=WINDOW)).isoformat()
    lo = bisect_left(dates, d0)
    seg = hist[reg][lo:hi]
    if len(seg) < MIN_RACES:
        return None
    return sum(t for _, t in seg) / len(seg) - n2 / 100.0


# --- walk-forward ---
conn = db.connect(DB_PATH)
df = build_training_set(conn)
actual = defaultdict(dict)
for rid, lane, order in conn.execute(
    "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
    "JOIN races r ON r.race_id = res.race_id "
    "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,)):
    actual[rid][order] = lane
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
        if 1 not in actual[rid] or not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5:
            continue
        top_raw = ranked[0]["prob"]
        c = {"rid": rid, "date": g["date"].iloc[0], "top": top_raw,
             "ranked": ranked}
        if top_raw < 0.20:
            konsen_ctx.append(c)
        elif (top_raw < 0.30
              and int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES):
            honmei_ctx.append(c)

# 本命は日毎cap6を再現
by_day = defaultdict(list)
for c in honmei_ctx:
    by_day[c["date"]].append(c)
honmei_sel = []
for d, cs in by_day.items():
    cs.sort(key=lambda c: c["top"])
    honmei_sel.extend(cs[:6])


def plan_of(c, konsen):
    probs = P.normalize_probs(c["ranked"])
    return P.ken_portfolio("荒れ注意", c["ranked"], [],
                           P.picks_katsu(probs), konsen=konsen)


def has_rising(c):
    for r in c["ranked"][:4]:
        g = gap_of(c["rid"], r["lane"])
        if g is not None and g > GAP_MIN:
            return True
    return False


for band, ctxs, konsen in (("超混戦帯(全場・案1構成)", konsen_ctx, True),
                           ("本命帯(5場cap6・保険複入り構成)", honmei_sel, False)):
    split = {True: [0, 0, 0, 0], False: [0, 0, 0, 0]}  # [投資,回収,R数,的中R]
    for c in ctxs:
        plan = plan_of(c, konsen)
        if not plan:
            continue
        pay = payout_map[c["rid"]]
        rs = sum(y for _, _, y, _ in plan)
        rr = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _ in plan)
        s = split[has_rising(c)]
        s[0] += rs
        s[1] += rr
        s[2] += 1
        s[3] += 1 if rr else 0
    print(f"\n=== {band} ===")
    for flag, label in ((True, "伸び盛り艇あり(上位4艇内)"), (False, "なし")):
        st, rt, nr, hits = split[flag]
        if st:
            print(f"  {label:<22} {nr:>5,}R({nr/(split[True][2]+split[False][2]):.0%}) "
                  f"回収率{rt/st:>7.1%}  何か当たる率{hits/nr:.1%}")
    stT, rtT = split[True][0], split[True][1]
    stF, rtF = split[False][0], split[False][1]
    if stT and stF:
        lift = rtT / stT - rtF / stF
        print(f"  リフト: {lift*100:+.1f}pt "
              f"{'→ +15pt基準クリア・8月紙上チャレンジャー化' if lift >= 0.15 else '→ +15pt未満・選定理由には昇格させない'}")
