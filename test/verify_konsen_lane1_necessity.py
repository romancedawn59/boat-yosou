# -*- coding: utf-8 -*-
"""超混戦に1号艇は本当に必要か(2026-08-05ケンさん発案・事前登録)

    py -X utf8 test/verify_konsen_lane1_necessity.py

■ 問い
超混戦(全場×1位生値20%未満)の買い目=⑬構成はモデル順位(r1..r5)で組むが、
そこに1号艇が入るのは妥当か。「1号艇が絡む目」は価値の源泉か、それとも
惰性で買っている死に目か。

■ 事前登録(結果を見る前に固定)
- 標本: 月次学習8か月(2025-12〜2026-07)walk-forward・超混戦帯(既存⑬検証と同一)
- 記述統計: 超混戦での1号艇の1着率/3着内率(全レース比)、
  1号艇のモデル順位分布、当たり目への1号艇関与率
- アーム(全て⑬と同ロジック・同レース):
  A) 現行⑬2,000円(基準)
  B) 1抜き⑬: 順位から艇1を除いて詰めた上位5艇で⑬を構成(同額2,000円)
  C) ⑬の「1号艇を含む目」のみ(減額)= 1号艇関与分の損益分解
  D) ⑬の「1号艇を含まない目」のみ(減額)= 非関与分の損益分解
- 判定基準: B(1抜き)を採用するのは
  「回収率が⑬比+5pt以上 かつ 8か月中5か月以上で⑬以上 かつ 最低月が⑬以上」
  をすべて満たす場合のみ。それ以外は現状維持(C/Dは分解の理解用で採否対象外)
- 先行検証との関係: 2026-07-19「1抜き構成3案」の棄却はv2全体(本命帯中心)。
  超混戦帯限定の1抜きは今回が初検証。結果がどうであれ蒸し返さないため記録する
"""
import sys
from collections import Counter, defaultdict
from itertools import permutations

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07"]

conn = db.connect(DB_PATH)
df = build_training_set(conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()


def trio(a, b, c):
    s = sorted([a, b, c])
    return f"{s[0]}={s[1]}={s[2]}"


def box(members, yen):
    return [("3連単", f"{a}-{b}-{c}", yen) for a, b, c in permutations(members)]


def plan13(lanes):
    """⑬「BOX+差され傾斜」2,000円。lanes=モデル順の5艇以上"""
    r1, r2, r3, r4, r5 = lanes[:5]
    return (box((r1, r2, r3), 100) + box((r1, r2, r4), 100)
            + [("3連単", f"{r3}-{r1}-{r2}", 300),
               ("3連単", f"{r4}-{r1}-{r2}", 300),
               ("3連複", trio(r3, r4, r5), 200)])


def involves_lane1(bt, comb):
    sep = "-" if bt == "3連単" else "="
    return 1 in {int(x) for x in comb.split(sep)}


def score(bets, pay):
    merged = defaultdict(int)
    for bt, comb, y in bets:
        merged[(bt, comb)] += y
    st = sum(merged.values())
    rt = sum(pay.get(k, 0) * y // 100 for k, y in merged.items())
    return st, rt


ARMS = ("A 現行⑬(2,000円)", "B 1抜き⑬(2,000円)",
        "C 1号艇を含む目のみ", "D 1号艇を含まない目のみ")
agg = defaultdict(lambda: [0, 0])          # {(arm, month): [st, rt]}
sub_diff = defaultdict(lambda: [0, 0])     # A/Bが異なるレースでの直接対決
sub_full = defaultdict(lambda: [0, 0, 0])  # 特徴量6艇フルか否かの層別(A基準)
n = n_all = 0
lane1_win = lane1_top3 = 0                 # 超混戦での1号艇成績
lane1_win_all = lane1_top3_all = 0         # 全レース(同期間)比較用
rank_pos = Counter()                       # 1号艇のモデル順位(1始まり)
hit_with1 = hit_total = 0                  # 当たり目(3連単)への1号艇関与

for m in MONTHS:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, g in month_df.groupby("race_id"):
        pay = payout_map[rid]
        if not pay:
            continue
        win = next((c for (bt, c) in pay if bt == "3連単"), None)
        if win is None:
            continue
        win_lanes = [int(x) for x in win.split("-")]
        n_all += 1
        lane1_win_all += win_lanes[0] == 1
        lane1_top3_all += 1 in win_lanes

        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        # 母集団は本番同様「⑬が組める5艇以上」(既存⑬検証と同一の499R)。
        # 初稿の6艇縛りは標本を292Rに縮め、利益の集中する
        # 「特徴量欠け(新人等)を含むレース」を落としていた(2026-08-05修正)
        if len(ranked) < 5 or ranked[0]["prob"] >= 0.20:
            continue
        n += 1
        lanes = [r["lane"] for r in ranked]
        lane1_win += win_lanes[0] == 1
        lane1_top3 += 1 in win_lanes
        if 1 in lanes:
            rank_pos[lanes.index(1) + 1] += 1

        base = plan13(lanes)
        lanes_wo1 = [l for l in lanes if l != 1]
        # Bは政策として評価: 1抜きで5艇残るなら1抜き⑬、残らなければ現行⑬のまま
        b_diff = len(lanes_wo1) >= 5 and 1 in lanes
        arms = {
            "A 現行⑬(2,000円)": base,
            "B 1抜き⑬(2,000円)": plan13(lanes_wo1) if b_diff else base,
            "C 1号艇を含む目のみ": [b for b in base if involves_lane1(b[0], b[1])],
            "D 1号艇を含まない目のみ": [b for b in base if not involves_lane1(b[0], b[1])],
        }
        st_a, rt_a = score(base, pay)
        if rt_a > 0:
            hit_total += 1
            hit_with1 += 1 in win_lanes
        for arm, bets in arms.items():
            st, rt = score(bets, pay)
            a = agg[(arm, m)]
            a[0] += st
            a[1] += rt
        # 参考分解: A/Bが実際に異なるレース(=1抜きが構成できるレース)での直接対決と、
        # 特徴量が6艇そろうか(揃わない=新人等の未知艇を含む)での層別
        if b_diff:
            for arm in ("A 現行⑬(2,000円)", "B 1抜き⑬(2,000円)"):
                st, rt = score(arms[arm], pay)
                a = sub_diff[arm]
                a[0] += st
                a[1] += rt
        key = "6艇フル" if len(ranked) >= 6 else "特徴量欠けあり"
        a = sub_full[key]
        a[0] += st_a
        a[1] += rt_a
        a[2] += 1

print(f"\n===== 記述統計 =====")
print(f"対象期間全レース: {n_all:,}R / 1号艇1着率 {lane1_win_all / n_all:.1%} "
      f"/ 3着内率 {lane1_top3_all / n_all:.1%}")
print(f"超混戦帯: {n:,}R / 1号艇1着率 {lane1_win / n:.1%} "
      f"/ 3着内率 {lane1_top3 / n:.1%}")
print("1号艇のモデル順位分布(超混戦):",
      {f"{k}位": f"{v / n:.1%}" for k, v in sorted(rank_pos.items())})
print(f"現行⑬の的中{hit_total}Rのうち、決まり目に1号艇が絡んだのは{hit_with1}R "
      f"({hit_with1 / hit_total:.1%})" if hit_total else "⑬的中なし")

print(f"\n{'月':<9}" + "".join(f"{a[:10]:<13}" for a in ARMS))
for m in MONTHS:
    row = f"{m:<9}"
    for arm in ARMS:
        st, rt = agg[(arm, m)]
        row += f"{(rt / st if st else 0):>10.1%}   "
    print(row)

print("\n===== 合計 =====")
for arm in ARMS:
    st = sum(agg[(arm, m)][0] for m in MONTHS)
    rt = sum(agg[(arm, m)][1] for m in MONTHS)
    months_ok = [m for m in MONTHS if agg[(arm, m)][0]]
    lows = min(agg[(arm, m)][1] / agg[(arm, m)][0] for m in months_ok)
    print(f"  {arm:<16} 投資{st:>11,}円 回収率{rt / st:>7.1%} "
          f"損益{rt - st:>+11,}円 最低月{lows:>7.1%}")

print("\n===== 参考分解 =====")
for arm, (st, rt) in sub_diff.items():
    print(f"  [1抜きが構成できたレースのみ] {arm:<16} 投資{st:>10,}円 "
          f"回収率{(rt / st if st else 0):>7.1%} 損益{rt - st:>+10,}円")
for key, (st, rt, cnt) in sub_full.items():
    print(f"  [層別・現行⑬] {key:<8} {cnt:>4}R 投資{st:>10,}円 "
          f"回収率{(rt / st if st else 0):>7.1%} 損益{rt - st:>+10,}円")

# 事前登録の判定(B vs A)
tot = {arm: [sum(agg[(arm, m)][i] for m in MONTHS) for i in (0, 1)]
       for arm in ARMS}
roi = {arm: rt / st for arm, (st, rt) in tot.items()}
b_beats = sum(
    1 for m in MONTHS
    if agg[("B 1抜き⑬(2,000円)", m)][0]
    and (agg[("B 1抜き⑬(2,000円)", m)][1] / agg[("B 1抜き⑬(2,000円)", m)][0])
    >= (agg[("A 現行⑬(2,000円)", m)][1] / agg[("A 現行⑬(2,000円)", m)][0]))
low = {arm: min(agg[(arm, m)][1] / agg[(arm, m)][0]
                for m in MONTHS if agg[(arm, m)][0]) for arm in ARMS}
ok = (roi["B 1抜き⑬(2,000円)"] - roi["A 現行⑬(2,000円)"] >= 0.05
      and b_beats >= 5
      and low["B 1抜き⑬(2,000円)"] >= low["A 現行⑬(2,000円)"])
print(f"\n===== 事前登録判定 =====")
print(f"B-A回収率差 {roi['B 1抜き⑬(2,000円)'] - roi['A 現行⑬(2,000円)']:+.1%} "
      f"(基準+5pt) / Bが上回った月 {b_beats}/8 (基準5以上) / "
      f"最低月 B{low['B 1抜き⑬(2,000円)']:.1%} vs A{low['A 現行⑬(2,000円)']:.1%}")
print("判定: " + ("B(1抜き)採用基準を満たす → 要相談" if ok
                  else "基準を満たさず → 現状維持(⑬のまま)"))
