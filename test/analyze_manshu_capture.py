# -*- coding: utf-8 -*-
"""万舟の所在と捕捉率の探索(walk-forward・全24場)

    py -X utf8 test/analyze_manshu_capture.py

問い(2026-07-21ケンさん発案): 「なぜ他場の万舟を捉えられなかったのか」。
7/21単日では万舟23件中うちが捉えたのは3件、しかも荒れ注意と判定できていたのは22%
だった。単日では偶然と区別できないため、過去データでバイアスなしに測る。

測ること(探索フェーズ。判定はしない):
- 万舟(決着3連単の払戻が1万円以上)は、モデルのどの確信度帯で出ているか
- その帯を仮に全部買っていたら回収率はどうなるか(万舟を追う価値があるか)
- C勝万舟(展開確率で選ぶ万舟圏5点・現在は購入していない枠)の的中はどこに集中するか

注意: 万舟を「捉える」こと自体は目的ではない。目的は回収率。万舟捕捉率が上がっても
回収率が下がる例は検証⑥⑨⑩⑫で繰り返し確認されている。ここで得るのは仮説であって
結論ではない。仮説は事前登録のうえ紙上で検証すること。

walk-forwardはbacktest.pyと同一fold(学習期間に未来を含めない)。
"""
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

MANSHU_MIN = 10_000
BANDS = [(0.00, 0.20, "〜20%"), (0.20, 0.25, "20-25%"), (0.25, 0.30, "25-30%"),
         (0.30, 0.35, "30-35%"), (0.35, 0.50, "35-50%"), (0.50, 1.01, "50%〜")]

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
n_days = len(dates)

recs = []
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        res = actual[rid]
        if 1 not in res or 2 not in res or 3 not in res:
            continue
        pay = payout_map[rid]
        if not pay:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        probs = P.normalize_probs(ranked)
        if len(probs) < 4:
            continue
        finish = f"{res[1]}-{res[2]}-{res[3]}"
        santan = pay.get(("3連単", finish), 0)
        if santan <= 0:
            continue
        s = sorted([res[1], res[2], res[3]])
        trio_key = f"{s[0]}={s[1]}={s[2]}"
        # 全レースに荒れ注意構成を当てた場合の収支(帯ごとの素の実力を測るため)
        plan = P.ken_portfolio("荒れ注意", ranked, [], P.picks_katsu(probs))
        stake = sum(y for _, _, y, _ in plan)
        ret = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _ in plan)
        c_picks = P.picks_katsu(probs)
        # C勝万舟は通常5点(確率が平坦なレースでは0点になりうる)。
        # 投資額は必ず実際の点数×100で数える(1点分で割ると回収率が5倍に化ける)
        recs.append({
            "c_n": len(c_picks),
            "date": str(g["date"].iloc[0]),
            "venue": int(g["venue_code"].iloc[0]),
            "top": ranked[0]["prob"],
            "santan": santan,
            "manshu": santan >= MANSHU_MIN,
            "stake": stake, "ret": ret,
            "hit3t": ("3連単", finish) in {(bt, c) for bt, c, _y, _s in plan},
            "hit3f": ("3連複", trio_key) in {(bt, c) for bt, c, _y, _s in plan},
            "c_hit": any(bt == "3連単" and c == finish for bt, c, _p in c_picks),
        })

print(f"\n対象レース {len(recs):,}件 / {n_days}日 / 全24場\n")

def band_of(top):
    return next(lbl for lo, hi, lbl in BANDS if lo <= top < hi)

print("=== 万舟(3連単1万円以上)はどの確信度帯で出ているか ===")
print(f"{'帯':<9}{'R数':>7}{'万舟数':>7}{'万舟率':>7}{'万舟の占有':>9}"
      f"{'ken捕捉':>8}{'C的中':>7}{'帯の回収率':>9}")
tot_manshu = sum(1 for r in recs if r["manshu"])
for lo, hi, lbl in BANDS:
    rs = [r for r in recs if lo <= r["top"] < hi]
    if not rs:
        continue
    m = [r for r in rs if r["manshu"]]
    cap = sum(1 for r in m if r["hit3t"] or r["hit3f"])
    chit = sum(1 for r in m if r["c_hit"])
    stake = sum(r["stake"] for r in rs)
    ret = sum(r["ret"] for r in rs)
    print(f"{lbl:<9}{len(rs):>7,}{len(m):>7,}{len(m)/len(rs):>7.1%}"
          f"{len(m)/tot_manshu:>9.1%}{cap/len(m) if m else 0:>8.1%}"
          f"{chit/len(m) if m else 0:>7.1%}{ret/stake if stake else 0:>9.1%}")

print(f"\n万舟 合計 {tot_manshu:,}件({tot_manshu/len(recs):.1%})")

print("\n=== 場スコープ別 ===")
for label, pred in (("対象5場", lambda v: v in TARGET_VENUE_CODES),
                    ("他19場", lambda v: v not in TARGET_VENUE_CODES)):
    rs = [r for r in recs if pred(r["venue"])]
    m = [r for r in rs if r["manshu"]]
    cap = sum(1 for r in m if r["hit3t"] or r["hit3f"])
    chit = sum(1 for r in m if r["c_hit"])
    print(f"{label}: {len(rs):,}R 万舟{len(m):,}件({len(m)/len(rs):.1%}) "
          f"ken捕捉{cap/len(m) if m else 0:.1%} C的中{chit/len(m) if m else 0:.1%}")

print("\n=== C勝万舟(現在は購入していない枠)の実力 ===")
c_hits = [r for r in recs if r["c_hit"]]
c_stake_all = sum(r["c_n"] * 100 for r in recs)
c_ret_all = sum(r["santan"] for r in c_hits)
print(f"的中 {len(c_hits):,}件 / 全{len(recs):,}R ({len(c_hits)/len(recs):.2%})")
if c_hits:
    avg = sum(r["santan"] for r in c_hits) / len(c_hits)
    manshu_rate = sum(1 for r in c_hits if r["manshu"]) / len(c_hits)
    print(f"的中時の平均払戻 {avg:,.0f}円 / うち万舟 {manshu_rate:.1%}")
    print(f"C案を全レースで買い続けた場合(実点数×100円): "
          f"投資{c_stake_all:,}円 回収{c_ret_all:,}円 "
          f"回収率{c_ret_all/c_stake_all:.1%}")
    print("帯別:")
    for lo, hi, lbl in BANDS:
        rs = [r for r in recs if lo <= r["top"] < hi]
        if not rs:
            continue
        ch = [r for r in rs if r["c_hit"]]
        st = sum(r["c_n"] * 100 for r in rs)
        rt = sum(r["santan"] for r in ch)
        print(f"  {lbl:<9} 的中{len(ch):>4}件 / {len(rs):>6,}R "
              f"({len(ch)/len(rs):>5.2%}) 回収率{rt/st if st else 0:>7.1%}")
