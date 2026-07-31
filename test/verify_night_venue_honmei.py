# -*- coding: utf-8 -*-
"""ナイター場を本命レイヤーの6場目に追加できるか(2026-07-31ケンさん発案)

    py -X utf8 test/verify_night_venue_honmei.py

■ 背景
現5場のうちナイターは若松のみ。18時以降に買える本命枠を増やしたい(生活動線)。
全場一括拡張は棄却済み(他19場・除き96.3%=希釈)だが、場を選んだ追加は未検証。
超混戦は既に全場対象なのでこの検証は本命帯(v2.1構成)のみ。

■ 事前登録(実行前に固定)
対象: ナイター6場(桐生1・蒲郡7・住之江12・丸亀15・下関19・大村24)+
     比較用に現5場(江戸川3・平和島4・常滑8・尼崎13・若松20)
帯: 1位生値20〜30%(本命帯)・walk-forward 5fold・v2.1構成(保険複入り)1,000円
基準: 回収率100%以上 かつ 最大1発除き90%以上 かつ 150R以上
→ 満たした場のみ「6場目候補・8月紙上観測→9/1判定」。満たさなければ見送り
(場別切りは7場の多重比較なので、クリアしても即採用はしない)。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

NIGHT = {1: "桐生", 7: "蒲郡", 12: "住之江", 15: "丸亀", 19: "下関", 24: "大村"}
CURRENT = {3: "江戸川", 4: "平和島", 8: "常滑", 13: "尼崎", 20: "若松"}

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

agg = defaultdict(lambda: [0, 0, 0, 0, 0])   # {venue: [st, rt, n, hit, best]}
for i in range(N_FOLDS):
    f_start, f_end = boundaries[i], boundaries[i + 1]
    train_df = df[df["date"] < f_start]
    fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
    print(f"fold{i+1} 学習中...", flush=True)
    booster = train_fold(train_df)
    fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])
    for rid, g in fold_df.groupby("race_id"):
        vc = int(g["venue_code"].iloc[0])
        if vc not in NIGHT and vc not in CURRENT:
            continue
        if not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5 or not (0.20 <= ranked[0]["prob"] < 0.30):
            continue
        plan = P.ken_portfolio("荒れ注意", ranked, [],
                               P.picks_katsu(P.normalize_probs(ranked)))
        pay = payout_map[rid]
        st = sum(y for _, _, y, _ in plan)
        rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _s in plan)
        a = agg[vc]
        a[0] += st
        a[1] += rt
        a[2] += 1
        a[3] += 1 if rt else 0
        a[4] = max(a[4], rt)

print(f"\n===== 本命帯(20〜30%)・v2.1構成・場別(walk-forward) =====")
print(f"{'場':<8}{'R数':>5}{'何か当たる率':>10}{'回収率':>8}{'最大1発除き':>10}{'損益':>11}{'判定':>16}")
for group, names in (("【ナイター候補】", NIGHT), ("【現5場(参考)】", CURRENT)):
    print(group)
    for vc, name in names.items():
        st, rt, n, hit, best = agg[vc]
        if not st:
            continue
        ex = (rt - best) / st
        ok = (rt / st >= 1.00 and ex >= 0.90 and n >= 150)
        verdict = "○ 候補(8月紙上へ)" if ok and vc in NIGHT else (
            "—" if vc in CURRENT else "× 見送り")
        print(f"  {name:<6}{n:>5,}{hit/n:>10.1%}{rt/st:>8.1%}{ex:>10.1%}"
              f"{rt-st:>+10,}円{verdict:>16}")
