# -*- coding: utf-8 -*-
"""本命帯: C勝万舟枠を「オーソドックス1着が消えた世界の本線」に置き換える案
(2026-07-29判断会中のケンさん発案・議題B派生)

    py -X utf8 test/verify_honmei_c_orthodox_off.py

■ 案の中身
本命帯のC枠(万舟圏の確率上位1点100円・回収率74.8%で唯一の赤字スロット)を、
「モデル1位(オーソドックス展開の1着)が消えた世界で最も素直な並び」=
3連単 r2-r3-r4 の1点100円に置き換える。
r1圏外は本命帯の29.4%で発生し、その72.3%はtop3⊆{r2..r5}
(test/verify_r1_dependency.py)。r1圏外日は大穴日(55倍超53.4%)なので
素直な繰り上がり並びでも配当が乗る可能性がある。

■ 既存知見(正直に)
- 1抜き構成検証(2026-07-19): 手動1抜きは「モデルが自動で1抜き化するため
  市場織り込み済み・配当が伸びない」で3案とも不採用
- 保険ブロック検証(verify_r1_insurance): r2r3r4複1点はEV中立(約71% vs C枠74.8%)
- 超混戦帯の参考値: r2-r3-r4素直単 233.8%(ただし12マスからの事後選択・未検証)

■ 事前登録(実行前に固定)
スコープ: 実運用と同じ本命選別を再現(5場・1位生値30%未満・日毎に低い順cap6)。
アーム(この3つだけ):
  現行     = V2構成6点(C枠100円)
  保険複   = C→3連複r2=r3=r4 100円(議題Bの選択肢(b)・対照)
  1着消し単 = C→3連単r2-r3-r4 100円(今回の新案)
判定基準: 「1着消し単」がレース単位回収率で現行を上回り、かつ最大1発除きでも
上回る場合のみ議題Bの有力候補として提出。現行±2pt以内ならEV中立=好みメニュー入り。
それ未満なら棄却。
参考表示(採否に使わない): 各Cスロット単体の回収率・月次。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

HONMEI_MAX = 0.30
CAP = 6

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

ctxs = []
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
        if int(g["venue_code"].iloc[0]) not in TARGET_VENUE_CODES:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        top_raw = ranked[0]["prob"]
        if len(ranked) < 5 or not (0.20 <= top_raw < HONMEI_MAX):
            continue
        ctxs.append({"rid": rid, "date": g["date"].iloc[0], "top": top_raw,
                     "ranked": ranked})

# 実運用の本命選別を再現: 日毎に1位勝率が低い順へcap6
by_day = defaultdict(list)
for c in ctxs:
    by_day[c["date"]].append(c)
sel = []
for d, cs in by_day.items():
    cs.sort(key=lambda c: c["top"])
    sel.extend(cs[:CAP])
sel.sort(key=lambda c: c["date"])
n = len(sel)
print(f"\n本命選別再現(5場・20〜30%・日cap{CAP}): {n:,}レース")


def v2_plan(c, c_slot):
    """V2構成6点。c_slot: 'katsu'|'hoken'|'orthodox_off'"""
    lanes = [r["lane"] for r in c["ranked"]]
    r1, r2, r3, r4 = lanes[:4]

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"
    plan = [
        ("3連複", trio(r1, r2, r3), 200),
        ("3連複", trio(r1, r2, r4), 200),
        ("3連複", trio(r1, r3, r4), 100),
        ("3連単", f"{r3}-{r1}-{r2}", 200),
        ("3連単", f"{r4}-{r1}-{r2}", 200),
    ]
    if c_slot == "katsu":
        probs = P.normalize_probs(c["ranked"])
        existing = {(bt, comb) for bt, comb, _y in plan}
        for bt, comb, _p in P.picks_katsu(probs):
            if (bt, comb) not in existing:
                plan.append((bt, comb, 100))
                break
    elif c_slot == "hoken":
        plan.append(("3連複", trio(r2, r3, r4), 100))
    elif c_slot == "orthodox_off":
        plan.append(("3連単", f"{r2}-{r3}-{r4}", 100))
    return plan


ARMS = {"現行(C勝万舟)": "katsu", "保険複r2r3r4": "hoken",
        "1着消し単r2-r3-r4": "orthodox_off"}

print(f"\n--- レース単位比較 ---")
print(f"{'アーム':<18}{'回収率':>8}{'ガミ':>7}{'プラス':>7}{'最大1発除き':>11}")
summary = {}
slot_stat = {}
for aname, mode in ARMS.items():
    st = rt = gm = pl = 0
    monthly = defaultdict(lambda: [0, 0])
    best_hit = 0
    s_st = s_rt = s_hit = 0
    for c in sel:
        pay = payout_map[c["rid"]]
        rs = rr = 0
        plan = v2_plan(c, mode)
        for bt, comb, yen in plan:
            rs += yen
            got = pay.get((bt, comb), 0) * yen // 100
            rr += got
            is_c_slot = (yen == 100 and (bt, comb) not in
                         {(b2, c2) for b2, c2, y2 in plan[:5]})
        # Cスロット単体(6点目)
        if len(plan) >= 6:
            bt, comb, yen = plan[5]
            got = pay.get((bt, comb), 0) * yen // 100
            s_st += yen
            s_rt += got
            if got:
                s_hit += 1
        st += rs
        rt += rr
        m = monthly[c["date"][:7]]
        m[0] += rs
        m[1] += rr
        best_hit = max(best_hit, rr)
        if rr == 0:
            pass
        elif rr < rs:
            gm += 1
        else:
            pl += 1
    ex = (rt - best_hit) / st if st else 0
    summary[aname] = {"roi": rt / st, "ex": ex, "monthly": dict(monthly)}
    slot_stat[aname] = (s_st, s_rt, s_hit)
    print(f"{aname:<18}{rt/st:>8.1%}{gm/n:>7.1%}{pl/n:>7.1%}{ex:>11.1%}")

print(f"\n--- Cスロット単体(6点目・各100円) ---")
for aname in ARMS:
    s_st, s_rt, s_hit = slot_stat[aname]
    if s_st:
        print(f"  {aname:<18} 的中{s_hit:>3}本({s_hit/(s_st//100):.2%}) "
              f"回収率{s_rt/s_st:>7.1%}")

print(f"\n--- 月次(回収率) ---")
months = sorted({m for v in summary.values() for m in v["monthly"]})
print("月       " + "".join(f"{a:<18}" for a in ARMS))
for m in months:
    row = f"{m}  "
    for aname in ARMS:
        s, r = summary[aname]["monthly"].get(m, [0, 0])
        row += f"{(r/s if s else 0):>9.1%}         "
    print(row)

cur, new = summary["現行(C勝万舟)"], summary["1着消し単r2-r3-r4"]
print(f"\n===== 事前登録基準の判定 =====")
if new["roi"] > cur["roi"] and new["ex"] > cur["ex"]:
    verdict = "現行超え → 議題Bの有力候補として提出"
elif abs(new["roi"] - cur["roi"]) <= 0.02:
    verdict = "EV中立(±2pt以内) → 好みメニュー入り"
else:
    verdict = "現行未満 → 棄却"
print(f"  1着消し単 {new['roi']:.1%}(除き{new['ex']:.1%}) vs "
      f"現行 {cur['roi']:.1%}(除き{cur['ex']:.1%}) → {verdict}")
