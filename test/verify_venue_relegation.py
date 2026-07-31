# -*- coding: utf-8 -*-
"""本命レイヤーの「神6入れ替え戦」シミュレーション(2026-07-31ケンさん発案)

    py -X utf8 test/verify_venue_relegation.py

■ 仕組み
毎月初に「その時点までのwalk-forward成績」で場を選定し、選ばれた6場だけを
翌月の本命レイヤー(20〜30%帯・日cap6・v2.1構成)として運用したらどうなったかを測る。
選定に未来の成績は一切使わない(入れ替え戦そのものを後知恵なしで検証)。

■ パターン(事前固定)
  A 現行固定5場(基準・入れ替えなし)
  B ケンさん案: 通算回収率順・30R以上で資格・5pt以内の競りはレース数多い方
  C 直近90日回収率順・30R以上
  D 通算の最大1発除き回収率順・30R以上(まぐれ1発場を弾く)
  E 通算回収率100%超(30R以上)のみ・最大6場(足りなければ少数)
統計蓄積: 2025-12〜 / 入れ替え戦の採点: 2026-02〜2026-07(資格30Rが貯まってから)
判定: 合計損益・回収率で基準Aと比較。場の入れ替わり回数(安定性)も報告。
注意: 月次の場別成績はノイズが大きいことが判明済み(場別×帯別は50-170Rで大きく揺れる)。
これは「場の実力は翌月も持続するか」の検証であり、持続しなければ入れ替え戦は
高値掴みの機械になる。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

VN = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
      7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
      13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
      19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}
CURRENT5 = [3, 4, 8, 13, 20]
MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07"]
EVAL_FROM = "2026-02"

conn = db.connect(DB_PATH)
df = build_training_set(conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

# ---- 全24場の本命帯レースをwalk-forwardで蓄積 -------------------------------
records = []       # {date, month, venue, top, stake, ret}
for m in MONTHS:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, g in month_df.groupby("race_id"):
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
        records.append({"date": g["date"].iloc[0], "month": m,
                        "venue": int(g["venue_code"].iloc[0]),
                        "top": ranked[0]["prob"], "stake": st, "ret": rt})

# ---- 選定ルール -------------------------------------------------------------
def venue_stats(past):
    s = defaultdict(lambda: [0, 0, 0, 0])   # v: [st, rt, n, best]
    for r in past:
        a = s[r["venue"]]
        a[0] += r["stake"]
        a[1] += r["ret"]
        a[2] += 1
        a[3] = max(a[3], r["ret"])
    return s


def pick(past, rule, month):
    s = venue_stats(past)
    if rule == "A":
        return CURRENT5
    if rule == "C":
        import datetime as dt
        d0 = (dt.date.fromisoformat(f"{month}-01")
              - dt.timedelta(days=90)).isoformat()
        s = venue_stats([r for r in past if r["date"] >= d0])
    q = {v: a for v, a in s.items() if a[2] >= 30 and a[0] > 0}
    if rule in ("B", "C"):
        key = lambda v: (-round(q[v][1] / q[v][0] / 0.05), -q[v][2])
        return sorted(q, key=key)[:6]
    if rule == "D":
        key = lambda v: -((q[v][1] - q[v][3]) / q[v][0])
        return sorted(q, key=key)[:6]
    if rule == "E":
        win = [v for v in q if q[v][1] / q[v][0] > 1.00]
        return sorted(win, key=lambda v: -q[v][1] / q[v][0])[:6]
    return []


RULES = {"A 現行固定5場": "A", "B ケンさん案(通算+30R)": "B",
         "C 直近90日フォーム": "C", "D 1発除き重視": "D", "E 黒字場のみ": "E"}
eval_months = [m for m in MONTHS if m >= EVAL_FROM]
result = defaultdict(lambda: [0, 0, 0, 0])
rosters = defaultdict(dict)
churn = defaultdict(int)

for name, rule in RULES.items():
    prev_roster = None
    for m in eval_months:
        past = [r for r in records if r["month"] < m]
        roster = pick(past, rule, m)
        rosters[name][m] = roster
        if prev_roster is not None:
            churn[name] += len(set(roster) - set(prev_roster))
        prev_roster = roster
        # 当月の運用: roster場の帯レースを日毎cap6(1位勝率低い順)
        by_day = defaultdict(list)
        for r in records:
            if r["month"] == m and r["venue"] in roster:
                by_day[r["date"]].append(r)
        for d, rs in by_day.items():
            rs.sort(key=lambda r: r["top"])
            for r in rs[:6]:
                a = result[name]
                a[0] += r["stake"]
                a[1] += r["ret"]
                a[2] += 1
                a[3] += 1 if r["ret"] else 0

print(f"\n===== 入れ替え戦の成績(採点: {EVAL_FROM}〜2026-07・日cap6) =====")
print(f"{'パターン':<18}{'R数':>5}{'的中率':>7}{'回収率':>8}{'損益':>11}{'入替回数':>7}")
for name in RULES:
    st, rt, n, h = result[name]
    if st:
        print(f"{name:<18}{n:>5,}{h/n:>7.1%}{rt/st:>8.1%}{rt-st:>+10,}円{churn[name]:>6}")

print(f"\n===== 各月の「神6」(パターン別) =====")
for name in RULES:
    if name.startswith("A"):
        continue
    print(f"--- {name} ---")
    for m in eval_months:
        names = "・".join(VN[v] for v in rosters[name][m])
        print(f"  {m}: {names}")
