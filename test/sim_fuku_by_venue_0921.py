# -*- coding: utf-8 -*-
"""「複」で買うとどの場で勝てるか(2026-09-21ケンさん指示)
  A 複勝: 予想1位の艇の複勝 1点500円
  B 3連複: 予想1-2-3位の3連複 1点500円(ドンピシャ1位の着順不問版)

    py -X utf8 test/sim_fuku_by_venue_0921.py

予想順位 = sim_donpisha_all_records_0921.py が保存した月次walk-forwardの1年分
(2025-10〜2026-09・全24場・全帯)。返還対象艇を含む券は返還。
抽出条件(事前固定): 期待値500円超・3期間中2期間以上黒字・最大1本除外後も500円超。
"""
import sqlite3
from collections import defaultdict

import pandas as pd

WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡",
         8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門",
         15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋",
         22: "福岡", 23: "唐津", 24: "大村"}
YEN = 500
PERIODS = [("25/10〜26/01", "2025-10", "2026-01"), ("26/02〜05", "2026-02", "2026-05"),
           ("26/06〜09", "2026-06", "2026-09")]
BANDS = [("全帯", 0, 1.01), ("〜22.5%", 0, .225), ("22.5〜30%", .225, .30),
         ("30〜40%", .30, .40), ("40〜55%", .40, .55), ("55%〜", .55, 1.01)]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2025-10-01' "
        "AND p.bet_type IN ('複勝','3連複')"):
    pay[rid][(bt, comb)] = max(pay[rid].get((bt, comb), 0), amt or 0)
conn.close()

L = pd.read_csv(WF_LEDGER, encoding="utf-8-sig", dtype={"refund": str})
rows = []
for rid, m, v, p1, pred, rf in zip(L["race_id"], L["month"], L["venue"], L["p1"],
                                   L["pred"], L["refund"].fillna("")):
    lanes = [int(x) for x in pred.split("-")]
    refund = {int(x) for x in rf.split("-") if x}
    p = pay.get(rid, {})
    fuku = YEN if lanes[0] in refund else p.get(("複勝", str(lanes[0])), 0) * YEN // 100
    t = sorted(lanes[:3])
    trio = (YEN if set(t) & refund
            else p.get(("3連複", f"{t[0]}={t[1]}={t[2]}"), 0) * YEN // 100)
    rows.append((m, v, p1, fuku, trio))
D = pd.DataFrame(rows, columns=["month", "venue", "p1", "A 複勝(予想1位)", "B 3連複(予想1-2-3位)"])
print(f"対象: {len(D):,}R(2025-10〜2026-09・全24場)")

for col in ("A 複勝(予想1位)", "B 3連複(予想1-2-3位)"):
    print(f"\n########## {col}・1点{YEN}円 ##########")
    print("― 帯別の全場計 ―")
    for name, lo, hi in BANDS:
        s = D[(D["p1"] >= lo) & (D["p1"] < hi)]
        hit = (s[col] > YEN).mean()
        print(f"  {name:<10}{len(s):>7,}R 的中率{hit:>6.1%} 期待値{s[col].mean():>5,.0f}円 "
              f"(回収率{s[col].mean() / YEN:>6.1%}) 損益{s[col].sum() - len(s) * YEN:>+12,.0f}円")
    for name, lo, hi in BANDS:
        s = D[(D["p1"] >= lo) & (D["p1"] < hi)]
        out, picked = [], []
        for v, g in s.groupby("venue"):
            n = len(g)
            per = []
            for _nm, a, b in PERIODS:
                t = g[(g["month"] >= a) & (g["month"] <= b)]
                per.append(t[col].sum() / (len(t) * YEN) if len(t) else float("nan"))
            out.append((g[col].mean(), v, n, (g[col] > YEN).mean(),
                        (g[col].sum() - g[col].max()) / n, per, g[col].sum() - n * YEN))
        out.sort(reverse=True)
        print(f"\n― {name}: 場別(上位8・下位3) ―")
        print(f"{'場':<6}{'R数':>6}{'的中率':>7}{'期待値':>8}{'損益':>11}{'最大1本除く':>10}"
              + "".join(f"{p[0]:>13}" for p in PERIODS) + f"{'黒字期間':>8}")
        for i, (evv, v, n, hr, exb, per, pl) in enumerate(out):
            ok = sum(1 for x in per if x == x and x > 1)
            if evv > YEN and ok >= 2 and exb > YEN:
                picked.append(VENUE[v])
            if i < 8 or i >= len(out) - 3:
                print(f"{VENUE[v]:<6}{n:>6,}{hr:>7.1%}{evv:>7,.0f}円{pl:>+10,.0f}円{exb:>9,.0f}円"
                      + "".join(f"{x:>13.0%}" for x in per) + f"{ok:>6}/3")
        print("  抽出: " + ("、".join(picked) if picked else "該当なし"))
