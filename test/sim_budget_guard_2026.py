# -*- coding: utf-8 -*-
"""④予算上限3案(4-A/4-B/4-C)の2026-01〜08シミュレーション
(2026-08-31ケンさん指示「4-A/4-B/4-Cを202601から08でシミュレーションして」)

    py -X utf8 test/sim_budget_guard_2026.py

■ スコープ
実運用と同じ5場(config.TARGET_VENUE_CODES)に限定。④は資金管理の話であり、
閾値(-15,000円等)は実弾スケールの数字のため全24場では意味が変わる。
月次walk-forward(各月、その月より前の全データでv2学習)。2025-12は
ラダー窓・ROI窓の助走(評価は2026-01〜08)。構成は現行のまま:
超混戦=⑬2,000円 / 本命帯(0.20≤p1<0.35)=9行1,400円。

■ 事前登録(結果を見る前に固定)
4-A 段階昇格ラダー(超混戦のみ):
  直近20Rの軸生存率(モデル1・2位が共に3着内、当該レースより前)で当月内も逐次更新:
  ≥40%→2,000円 / ≥35%→1,000円 / ≥30%→500円 / <30%→0円(紙上)。
  助走20R未満は2,000円。金額は⑬の線形縮尺(100円単位の丸めは実装時課題として注記)。
4-B 帯別サーキットブレーカー:
  帯ごとに月内累積損益が-15,000円以下になったら当月の残りを紙上化。
  超混戦・本命帯それぞれ独立。参考として-10,000円版も表示(感度・採否根拠にしない)。
4-C エッジ比例の上限:
  本命帯=検証済みエッジありとして1,400円固定(変更なし)。
  超混戦=直近60Rの⑬回収率で当月内も逐次更新:
  ≥150%→×1.0 / ≥100%→×0.5 / ≥70%→×0.25 / <70%→×0。助走60R未満は×1.0。
判定基準: ベースライン(固定額)比で「総損益を改善 または 最悪月の損失を半減」
したアームを採用検討へ。的中率は変わらない(買う/買わないの制御のみ)。
"""
import sys
from collections import defaultdict, deque
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

MONTHS_ALL = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
              "2026-05", "2026-06", "2026-07", "2026-08"]
EVAL_MONTHS = MONTHS_ALL[1:]
STOP_LOSS = -15000
STOP_LOSS_REF = -10000

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
conn.close()


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


def score(bets, pay):
    merged = defaultdict(int)
    for bt, comb, y in bets:
        merged[(bt, comb)] += y
    return (sum(merged.values()),
            sum(pay.get(k, 0) * y // 100 for k, y in merged.items()))


# ---- 月次walk-forwardで5場のレース別損益ストリームを作る -----------------------
stream_k = []   # 超混戦: (month, date, rid, stake, ret, axis)
stream_h = []   # 本命帯: (month, date, rid, stake, ret)
for m in MONTHS_ALL:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[(df["date"].str.startswith(m))
                  & (df["venue_code"].isin(TARGET_VENUE_CODES))].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, grp in month_df.groupby("race_id"):
        arr = res_all.get(rid, {})
        pay = payout_map[rid]
        if len(arr) < 3 or not pay:
            continue
        gs = grp.sort_values("pred", ascending=False)
        lanes = [int(x) for x in gs["lane"]]
        p1 = float(gs["pred"].iloc[0])
        date = str(gs["date"].iloc[0])
        if p1 < 0.20 and len(lanes) >= 5:
            st, rt = score(plan13(lanes), pay)
            top3 = sorted(arr, key=arr.get)[:3]
            axis = lanes[0] in top3 and lanes[1] in top3
            stream_k.append((m, date, rid, st, rt, axis))
        elif 0.20 <= p1 < 0.35 and len(lanes) >= 4:
            st, rt = score(honmei_plan(lanes), pay)
            stream_h.append((m, date, rid, st, rt))

stream_k.sort(key=lambda x: (x[1], x[2]))
stream_h.sort(key=lambda x: (x[1], x[2]))
print(f"\n5場ストリーム: 超混戦{len(stream_k)}R / 本命帯{len(stream_h)}R "
      f"(うち助走2025-12: 超混戦"
      f"{sum(1 for s in stream_k if s[0] == '2025-12')}R)")

# ---- ベースライン(固定額) ------------------------------------------------------
base_k = defaultdict(lambda: [0, 0])
base_h = defaultdict(lambda: [0, 0])
for m, _d, _r, st, rt, _ax in stream_k:
    if m != "2025-12":
        base_k[m][0] += st
        base_k[m][1] += rt
for m, _d, _r, st, rt in stream_h:
    if m != "2025-12":
        base_h[m][0] += st
        base_h[m][1] += rt

# ---- 4-A 段階昇格ラダー(超混戦・逐次) ------------------------------------------
window = deque(maxlen=20)
ladder = defaultdict(lambda: [0, 0])
level_hist = defaultdict(lambda: defaultdict(int))
for m, _d, _r, st, rt, ax in stream_k:
    if len(window) < 20:
        level = 2000
    else:
        rate = sum(window) / len(window)
        level = (2000 if rate >= 0.40 else 1000 if rate >= 0.35
                 else 500 if rate >= 0.30 else 0)
    if m != "2025-12":
        f = level / 2000
        ladder[m][0] += round(st * f)
        ladder[m][1] += round(rt * f)
        level_hist[m][level] += 1
    window.append(ax)

# ---- 4-B サーキットブレーカー(帯別・月内停止) ----------------------------------
def breaker(stream, stop):
    out = defaultdict(lambda: [0, 0, None])   # {月: [st, rt, 停止日]}
    cum = {}
    stopped = {}
    for row in stream:
        m = row[0]
        if m == "2025-12":
            continue
        st, rt = row[3], row[4]
        if stopped.get(m):
            continue
        out[m][0] += st
        out[m][1] += rt
        cum[m] = cum.get(m, 0) + rt - st
        if cum[m] <= stop:
            stopped[m] = True
            out[m][2] = row[1]
    return out


brk_k = breaker(stream_k, STOP_LOSS)
brk_h = breaker(stream_h, STOP_LOSS)
brk_k_ref = breaker(stream_k, STOP_LOSS_REF)
brk_h_ref = breaker(stream_h, STOP_LOSS_REF)

# ---- 4-C エッジ比例(超混戦・直近60R回収率で逐次) -------------------------------
roi_win = deque(maxlen=60)
edge = defaultdict(lambda: [0, 0])
mult_hist = defaultdict(lambda: defaultdict(int))
for m, _d, _r, st, rt, _ax in stream_k:
    if len(roi_win) < 60:
        mult = 1.0
    else:
        s = sum(x[0] for x in roi_win)
        r = sum(x[1] for x in roi_win)
        roi = r / s if s else 0
        mult = 1.0 if roi >= 1.5 else 0.5 if roi >= 1.0 else \
            0.25 if roi >= 0.7 else 0.0
    if m != "2025-12":
        edge[m][0] += round(st * mult)
        edge[m][1] += round(rt * mult)
        mult_hist[m][mult] += 1
    roi_win.append((st, rt))

# ---- レポート ------------------------------------------------------------------
print("\n===== 超混戦(5場・⑬) 月別比較 =====")
print(f"{'月':<9}{'固定2000':>16}{'4-Aラダー':>16}{'4-Bブレーカー':>16}"
      f"{'4-Cエッジ比例':>16}")
for m in EVAL_MONTHS:
    bs, br = base_k[m]
    if not bs:
        continue
    ls, lr = ladder[m]
    ks, kr = brk_k[m][0], brk_k[m][1]
    es, er = edge[m]
    print(f"{m:<9}{br - bs:>+13,}円{lr - ls:>+13,}円{kr - ks:>+13,}円"
          f"{er - es:>+13,}円"
          + (f"  <停止{brk_k[m][2][5:]}>" if brk_k[m][2] else ""))
for name, agg_ in (("固定2000", base_k), ("4-Aラダー", ladder),
                   ("4-Cエッジ比例", edge)):
    st = sum(agg_[m][0] for m in EVAL_MONTHS)
    rt = sum(agg_[m][1] for m in EVAL_MONTHS)
    worst = min((agg_[m][1] - agg_[m][0] for m in EVAL_MONTHS if agg_[m][0]),
                default=0)
    print(f"  {name:<10} 投資{st:>10,}円 回収率{(rt / st if st else 0):>7.1%} "
          f"損益{rt - st:>+10,}円 最悪月{worst:>+9,}円")
st = sum(brk_k[m][0] for m in EVAL_MONTHS)
rt = sum(brk_k[m][1] for m in EVAL_MONTHS)
worst = min((brk_k[m][1] - brk_k[m][0] for m in EVAL_MONTHS if brk_k[m][0]),
            default=0)
n_stop = sum(1 for m in EVAL_MONTHS if brk_k[m][2])
print(f"  4-Bブレーカー 投資{st:>10,}円 回収率{(rt / st if st else 0):>7.1%} "
      f"損益{rt - st:>+10,}円 最悪月{worst:>+9,}円 発動{n_stop}か月")
st = sum(brk_k_ref[m][0] for m in EVAL_MONTHS)
rt = sum(brk_k_ref[m][1] for m in EVAL_MONTHS)
print(f"  (参考-10,000円版: 損益{rt - st:>+,}円 "
      f"発動{sum(1 for m in EVAL_MONTHS if brk_k_ref[m][2])}か月)")

print("\n4-Aの月別レベル分布(2000/1000/500/0の適用R数):")
for m in EVAL_MONTHS:
    lh = level_hist[m]
    if lh:
        print(f"  {m}: " + " ".join(f"{lv}円×{lh[lv]}" for lv in
                                     (2000, 1000, 500, 0) if lh[lv]))
print("4-Cの月別倍率分布:")
for m in EVAL_MONTHS:
    mh = mult_hist[m]
    if mh:
        print(f"  {m}: " + " ".join(f"×{mu}→{mh[mu]}R" for mu in
                                     (1.0, 0.5, 0.25, 0.0) if mh[mu]))

print("\n===== 本命帯(5場・1,400円) 月別比較 =====")
print(f"{'月':<9}{'固定1400':>16}{'4-Bブレーカー':>16}")
for m in EVAL_MONTHS:
    bs, br = base_h[m]
    if not bs:
        continue
    print(f"{m:<9}{br - bs:>+13,}円{brk_h[m][1] - brk_h[m][0]:>+13,}円"
          + (f"  <停止{brk_h[m][2][5:]}>" if brk_h[m][2] else ""))
bs = sum(base_h[m][0] for m in EVAL_MONTHS)
br = sum(base_h[m][1] for m in EVAL_MONTHS)
ks = sum(brk_h[m][0] for m in EVAL_MONTHS)
kr = sum(brk_h[m][1] for m in EVAL_MONTHS)
worst_b = min((base_h[m][1] - base_h[m][0] for m in EVAL_MONTHS
               if base_h[m][0]), default=0)
worst_k = min((brk_h[m][1] - brk_h[m][0] for m in EVAL_MONTHS
               if brk_h[m][0]), default=0)
print(f"  固定1400     損益{br - bs:>+10,}円 最悪月{worst_b:>+9,}円")
print(f"  4-Bブレーカー 損益{kr - ks:>+10,}円 最悪月{worst_k:>+9,}円 "
      f"発動{sum(1 for m in EVAL_MONTHS if brk_h[m][2])}か月")
st = sum(brk_h_ref[m][0] for m in EVAL_MONTHS)
rt = sum(brk_h_ref[m][1] for m in EVAL_MONTHS)
print(f"  (参考-10,000円版: 損益{rt - st:>+,}円 "
      f"発動{sum(1 for m in EVAL_MONTHS if brk_h_ref[m][2])}か月)")
print("\n(採用判定は事前登録基準: 総損益改善または最悪月の損失半減)")
