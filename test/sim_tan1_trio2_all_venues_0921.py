# -*- coding: utf-8 -*-
"""ドンピシャ1位の3連単(予想1-2-3位)+ドンピシャ2位の3連複(予想1=2=4位)を全24場で試算
(2026-09-21ケンさん指示)

    py -X utf8 test/sim_tan1_trio2_all_venues_0921.py

予想順位は1年分のwalk-forward台帳(2025-10〜2026-09)。1レース1,000円。返還反映。
比較用に 3連複(予想1=2=3位)との組み合わせも並べる。
抽出条件(事前固定・600/400配分): 回収率100%超・3期間中2期間以上黒字・最大1本除外後も100%超。
"""
import sqlite3
from collections import defaultdict

import pandas as pd

WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡",
         8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門",
         15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋",
         22: "福岡", 23: "唐津", 24: "大村"}
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
        "AND p.bet_type IN ('3連単','3連複')"):
    pay[rid][(bt, comb)] = amt or 0
conn.close()

L = pd.read_csv(WF_LEDGER, encoding="utf-8-sig", dtype={"refund": str})
rows = []
for rid, m, v, p1, pred, rf in zip(L["race_id"], L["month"], L["venue"], L["p1"],
                                   L["pred"], L["refund"].fillna("")):
    l = [int(x) for x in pred.split("-")]
    refund = {int(x) for x in rf.split("-") if x}
    p = pay.get(rid, {})

    def trio(ix):
        s = sorted(l[i] for i in ix)
        return 100 if set(s) & refund else p.get(("3連複", f"{s[0]}={s[1]}={s[2]}"), 0)
    tan = 100 if set(l[:3]) & refund else p.get(("3連単", f"{l[0]}-{l[1]}-{l[2]}"), 0)
    rows.append((m, v, p1, tan, trio((0, 1, 2)), trio((0, 1, 3))))
D = pd.DataFrame(rows, columns=["month", "venue", "p1", "tan", "t123", "t124"])
print(f"対象: {len(D):,}R(2025-10〜2026-09・全24場)")


def roi(g, a, b, col):
    return (g["tan"] * a + g[col] * b).sum() / (len(g) * (a + b) * 100) if len(g) else float("nan")


print("\n===== 帯別の全場計(回収率) =====")
print(f"{'帯':<10}{'R数':>7}{'3連単1位':>9}{'3複1=2=4':>9}{'3複1=2=3':>9}"
      f"{'単600+複124 400':>15}{'単300+複124 700':>15}{'単600+複123 400':>15}")
for name, lo, hi in BANDS:
    s = D[(D["p1"] >= lo) & (D["p1"] < hi)]
    print(f"{name:<10}{len(s):>7,}{roi(s, 1, 0, 't124'):>9.1%}{roi(s, 0, 1, 't124'):>9.1%}"
          f"{roi(s, 0, 1, 't123'):>9.1%}{roi(s, 6, 4, 't124'):>15.1%}"
          f"{roi(s, 3, 7, 't124'):>15.1%}{roi(s, 6, 4, 't123'):>15.1%}")

for name, lo, hi in (("22.5〜30%", .225, .30), ("全帯", 0, 1.01)):
    s = D[(D["p1"] >= lo) & (D["p1"] < hi)]
    print(f"\n===== 場別: 1位勝率 {name}(1レース1,000円) =====")
    print(f"{'場':<6}{'R数':>6}{'単1位':>7}{'複124':>7}{'複123':>7} |{'単600+複124 400':>14}{'年間損益':>11}"
          f"{'最大1本除く':>10}{'払戻あり':>8}" + "".join(f"{p[0]:>13}" for p in PERIODS)
          + f"{'黒字':>5} |{'単300+複124 700':>14}{'単600+複123 400':>14}")
    out, picked = [], []
    for v, g in s.groupby("venue"):
        gain = (g["tan"] * 6 + g["t124"] * 4)
        st = len(g) * 1000
        per = [roi(g[(g["month"] >= a) & (g["month"] <= b)], 6, 4, "t124") for _n, a, b in PERIODS]
        out.append((gain.sum() / st, v, g, gain, st, per))
    for r, v, g, gain, st, per in sorted(out, key=lambda x: -x[0]):
        ok = sum(1 for x in per if x == x and x > 1)
        exb = (gain.sum() - gain.max()) / st
        if r > 1 and ok >= 2 and exb > 1:
            picked.append(VENUE[v])
        print(f"{VENUE[v]:<6}{len(g):>6,}{roi(g, 1, 0, 't124'):>7.0%}{roi(g, 0, 1, 't124'):>7.0%}"
              f"{roi(g, 0, 1, 't123'):>7.0%} |{r:>14.1%}{gain.sum() - st:>+10,.0f}円{exb:>10.1%}"
              f"{(gain > 0).mean():>8.1%}" + "".join(f"{x:>13.0%}" for x in per) + f"{ok:>3}/3 |"
              f"{roi(g, 3, 7, 't124'):>14.1%}{roi(g, 6, 4, 't123'):>14.1%}")
    print("  抽出(単600+複124 400): " + ("、".join(picked) if picked else "該当なし"))
