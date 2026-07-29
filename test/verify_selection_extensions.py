# -*- coding: utf-8 -*-
"""本命判断システムの拡張候補①③の初回検証(2026-07-29夜・ケンさん発案の9月検討事項)

    py -X utf8 test/verify_selection_extensions.py

■ ケンさんの3つの問いと扱い
① 何号艇が一番人気か(枠の形)で本命の成績は変わるか
   → 本命選別再現(5場・20〜30%・日cap6)を人気の枠番3群で層別
② 「一番予想が決めにくいレース」は勝負所になるか
   → 検証済みのため再実行しない: 挑戦者β1(1-2位差)121.9%・β2(エントロピー)84.0%
     はチャンピオン(1位勝率)169.1%に大敗(test/verify_challengers.py・7/29再確認)
③ 「1着は堅いが2・3着が荒れる」レースは勝負所になるか
   → 堅め帯(1位生値50%以上・5場)で事前登録の2アームを測る:
     (a) 1位頭・2-3着全流し20点(各100円)
     (b) 1位頭・相手はモデル4-6位のみ6点(各100円)=「頭堅い×相手荒れ」の純形
     参考: 現行の堅め表示プラン(購入対象外)
   合格ライン(事前固定): 回収率105%以上のアームのみ「本命拡張の候補」として
   8月紙上追跡に進める。未満なら棄却(堅め帯機械買い75.5%・1番人気6倍未満53%の
   既存知見どおり「市場が正しい」ことの確認になる)
"""
import sys
from collections import defaultdict
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

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

honmei_ctx, katame_ctx = [], []
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
        if int(g["venue_code"].iloc[0]) not in TARGET_VENUE_CODES:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 6:
            continue
        top = ranked[0]["prob"]
        c = {"rid": rid, "date": g["date"].iloc[0], "top": top, "ranked": ranked}
        if 0.20 <= top < 0.30:
            honmei_ctx.append(c)
        elif top >= 0.50:
            katame_ctx.append(c)

# 本命は日毎cap6を再現
by_day = defaultdict(list)
for c in honmei_ctx:
    by_day[c["date"]].append(c)
honmei_sel = []
for d, cs in by_day.items():
    cs.sort(key=lambda c: c["top"])
    honmei_sel.extend(cs[:6])

# ---- ① 人気の枠番で層別(v2.1プラン) --------------------------------------
print(f"\n=== ① 本命(5場・20〜30%・日cap6={len(honmei_sel):,}R)を人気の枠番で層別 ===")
groups = {"人気=1号艇": [], "人気=2-3号艇": [], "人気=4-6号艇": []}
for c in honmei_sel:
    fav = c["ranked"][0]["lane"]
    key = ("人気=1号艇" if fav == 1 else
           "人気=2-3号艇" if fav in (2, 3) else "人気=4-6号艇")
    groups[key].append(c)
for key, cs in groups.items():
    st = rt = hit = 0
    for c in cs:
        plan = P.ken_portfolio("荒れ注意", c["ranked"], [],
                               P.picks_katsu(P.normalize_probs(c["ranked"])))
        pay = payout_map[c["rid"]]
        s = sum(y for _, _, y, _ in plan)
        r = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y, _s in plan)
        st += s
        rt += r
        hit += 1 if r else 0
    if st:
        print(f"  {key:<12} {len(cs):>4,}R({len(cs)/len(honmei_sel):.0%}) "
              f"何か当たる率{hit/len(cs):>6.1%} 回収率{rt/st:>7.1%} 損益{rt-st:+,}円")

# ---- ③ 堅め帯の「頭固定・2-3着荒れ」 --------------------------------------
n = len(katame_ctx)
print(f"\n=== ③ 堅め帯(5場・1位生値50%以上={n:,}R) ===")
win_top = sum(1 for c in katame_ctx
              if payout_map[c["rid"]].get(
                  ("単勝", str(c["ranked"][0]["lane"])), None) is not None
              or True)  # 1着率は3連単キーから判定する(下で計算)


def top1_won(c):
    pay = payout_map[c["rid"]]
    fav = c["ranked"][0]["lane"]
    for (bt, comb) in pay:
        if bt == "3連単":
            return comb.startswith(f"{fav}-")
    return False


n_won = sum(1 for c in katame_ctx if top1_won(c))
print(f"  モデル1位の1着率: {n_won/n:.1%}(=「1着は堅い」の実測)")

arms = {
    "(a)1位頭・全流し20点": lambda lanes: [
        ("3連単", f"{lanes[0]}-{x}-{y}", 100)
        for x, y in permutations(lanes[1:6], 2)],
    "(b)1位頭・相手4-6位のみ6点": lambda lanes: [
        ("3連単", f"{lanes[0]}-{x}-{y}", 100)
        for x, y in permutations(lanes[3:6], 2)],
    "(参考)現行の堅め表示プラン": None,
}
for name, fn in arms.items():
    st = rt = hit = 0
    for c in katame_ctx:
        lanes = [r["lane"] for r in c["ranked"]]
        if fn is None:
            probs = P.normalize_probs(c["ranked"])
            plan = [(bt, comb, y) for bt, comb, y, _s in
                    P.ken_portfolio("堅め", c["ranked"], P.picks_yamada(probs),
                                    P.picks_katsu(probs))]
        else:
            plan = fn(lanes)
        pay = payout_map[c["rid"]]
        s = sum(y for _, _, y in plan)
        r = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y in plan)
        st += s
        rt += r
        hit += 1 if r else 0
    print(f"  {name:<22} 何か当たる率{hit/n:>6.1%} 回収率{rt/st:>7.1%} "
          f"損益{rt-st:+,}円(投資{st:,}円)")

print(f"\n===== 事前登録基準の判定(③) =====")
print(f"  合格ライン105%以上のアームのみ8月紙上追跡へ。①は層別の記述(選別変更なし)")
