# -*- coding: utf-8 -*-
"""「5場 vs 全場の70pt差」は本物か・今も開いているかの再測定(2026-07-31ケンさん発案)

    py -X utf8 test/verify_scope_gap_recheck.py

■ 背景
7/18の全場拡張検証では 5場176.2% vs 他19場106.4%(除き96.3%)と大差だった。
ケンさんの疑問「同じとは言わないが70ptも開く????」は統計的に正当:
- 5場は当初ユーザー希望+戸田除外で決めた面もあり、以後の検証は同じデータを見てきた
- 場別×帯別の数字は切り方で大きく揺れることが今週2回実証された(場別表・入れ替え戦)

■ 方法(事前固定)
月次学習(本番同方式)で2025-12〜2026-07の8か月、本命帯(1位生値20〜30%)を
3スコープで比較: 現5場 / 他19場 / 全24場。
それぞれ「日cap6(実運用形)」と「capなし(帯の素の力)」の両方、v2.1構成1,000円。
月別も出し、差が期間依存かどうかを見る。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
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

records = []
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

VN = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
      7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
      13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
      19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}
HALF = "2026-04"   # 前半: 2025-12〜2026-03 / 後半: 2026-04〜07

# ---- 各場単独の成績(束ねない・ケンさん指定) --------------------------------
print(f"\n===== 24場それぞれ単独(本命帯・capなし・v2.1構成) =====")
print(f"{'順':>3} {'場':<6}{'R数':>5}{'当たる率':>8}{'回収率':>8}{'1発除き':>8}"
      f"{'損益':>10}{'前半':>8}{'後半':>8}")
stats = {}
for v in VN:
    rs = [r for r in records if r["venue"] == v]
    if not rs:
        continue
    st = sum(r["stake"] for r in rs)
    rt = sum(r["ret"] for r in rs)
    best = max(r["ret"] for r in rs)
    h1 = [r for r in rs if r["month"] < HALF]
    h2 = [r for r in rs if r["month"] >= HALF]
    roi1 = (sum(r["ret"] for r in h1) / sum(r["stake"] for r in h1)
            if h1 else float("nan"))
    roi2 = (sum(r["ret"] for r in h2) / sum(r["stake"] for r in h2)
            if h2 else float("nan"))
    stats[v] = {"n": len(rs), "roi": rt / st, "ex": (rt - best) / st,
                "pnl": rt - st, "roi1": roi1, "roi2": roi2,
                "hit": sum(1 for r in rs if r["ret"]) / len(rs)}
rank1 = {v: i for i, v in enumerate(
    sorted(stats, key=lambda v: -stats[v]["roi1"]))}
rank2 = {v: i for i, v in enumerate(
    sorted(stats, key=lambda v: -stats[v]["roi2"]))}
for i, v in enumerate(sorted(stats, key=lambda v: -stats[v]["roi"]), 1):
    s = stats[v]
    star = "★" if v in TARGET_VENUE_CODES else " "
    print(f"{i:>3} {star}{VN[v]:<5}{s['n']:>5}{s['hit']:>8.1%}{s['roi']:>8.1%}"
          f"{s['ex']:>8.1%}{s['pnl']:>+9,}円{s['roi1']:>8.1%}{s['roi2']:>8.1%}")

# 前後半の順位相関(場の実力は持続するか)
import statistics
common = [v for v in stats if stats[v]["n"] >= 60]
if len(common) >= 8:
    r1 = [rank1[v] for v in common]
    r2 = [rank2[v] for v in common]
    n = len(common)
    d2 = sum((a - b) ** 2 for a, b in zip(r1, r2))
    rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
    print(f"\n前半順位と後半順位のスピアマン相関(60R以上の{n}場): {rho:+.2f}")
    print("  (+1=完全持続 / 0=無関係 / 負=逆転。場の実力が本物なら正に出る)")

SCOPES = {
    "現5場": lambda v: v in TARGET_VENUE_CODES,
    "他19場": lambda v: v not in TARGET_VENUE_CODES,
    "全24場": lambda v: True,
}

for capped in (True, False):
    label = "日cap6(実運用形)" if capped else "capなし(帯の素の力)"
    print(f"\n===== {label} =====")
    print(f"{'月':<9}" + "".join(f"{s:<16}" for s in SCOPES))
    totals = {s: [0, 0, 0] for s in SCOPES}
    for m in MONTHS:
        row = f"{m}  "
        for sname, fn in SCOPES.items():
            rs = [r for r in records if r["month"] == m and fn(r["venue"])]
            if capped:
                by_day = defaultdict(list)
                for r in rs:
                    by_day[r["date"]].append(r)
                sel = []
                for d, day_rs in by_day.items():
                    day_rs.sort(key=lambda r: r["top"])
                    sel.extend(day_rs[:6])
                rs = sel
            st = sum(r["stake"] for r in rs)
            rt = sum(r["ret"] for r in rs)
            totals[sname][0] += st
            totals[sname][1] += rt
            totals[sname][2] += len(rs)
            row += f"{(rt/st if st else 0):>7.1%}({len(rs):>3}R)  "
        print(row)
    row = "合計     "
    for sname in SCOPES:
        st, rt, n = totals[sname]
        row += f"{(rt/st if st else 0):>7.1%}({n:>4}R) "
    print(row)
    for sname in SCOPES:
        st, rt, n = totals[sname]
        print(f"  {sname:<8} 投資{st:>10,}円 損益{rt-st:>+10,}円")
