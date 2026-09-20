# -*- coding: utf-8 -*-
"""尼崎・若松 × 1位勝率22.5〜30%: 3連単ドンピシャ+「複」の組み合わせ1,000円
(2026-09-21ケンさん案「ドンピシャ300円+ドンピシャ複勝700円」)

    py -X utf8 test/sim_amagasaki_wakamatsu_mix_0921.py

予想順位は1年分のwalk-forward台帳(data_raw/wf_ledger_all_202510_202609.csv)。
単=3連単 予想1-2-3位 / 複勝=予想1位の複勝 / 3複=3連複 予想1-2-3位。返還反映。
"""
import sqlite3
from collections import defaultdict

import pandas as pd

WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
VENUES = {13: "尼崎", 20: "若松"}
PERIODS = [("25/10〜26/01", "2025-10", "2026-01"), ("26/02〜05", "2026-02", "2026-05"),
           ("26/06〜09", "2026-06", "2026-09")]
MIXES = [("単300+複勝700 (ケン案)", 300, 700, 0), ("単300+3複700", 300, 0, 700),
         ("単500+複勝500", 500, 500, 0), ("単700+複勝300", 700, 300, 0),
         ("単300+複勝400+3複300", 300, 400, 300), ("単1000", 1000, 0, 0),
         ("複勝1000", 0, 1000, 0), ("3複1000", 0, 0, 1000)]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2025-10-01' "
        "AND r.venue_code IN (13, 20) AND p.bet_type IN ('3連単','複勝','3連複')"):
    pay[rid][(bt, comb)] = max(pay[rid].get((bt, comb), 0), amt or 0)
conn.close()

L = pd.read_csv(WF_LEDGER, encoding="utf-8-sig", dtype={"refund": str})
L = L[L["venue"].isin(VENUES) & (L["p1"] >= 0.225) & (L["p1"] < 0.30)]
rows = []   # (month, venue, 100円あたり払戻: 単, 複勝, 3複)  返還は100
for rid, m, v, pred, rf in zip(L["race_id"], L["month"], L["venue"], L["pred"],
                               L["refund"].fillna("")):
    l = [int(x) for x in pred.split("-")]
    refund = {int(x) for x in rf.split("-") if x}
    p = pay.get(rid, {})
    t = sorted(l[:3])
    bad3 = bool(set(l[:3]) & refund)
    rows.append((m, v,
                 100 if bad3 else p.get(("3連単", f"{l[0]}-{l[1]}-{l[2]}"), 0),
                 100 if l[0] in refund else p.get(("複勝", str(l[0])), 0),
                 100 if bad3 else p.get(("3連複", f"{t[0]}={t[1]}={t[2]}"), 0)))
D = pd.DataFrame(rows, columns=["month", "venue", "tan", "fuku", "trio"])
print(f"対象: {len(D)}R(尼崎{(D['venue'] == 13).sum()}R・若松{(D['venue'] == 20).sum()}R / "
      f"1年・月{len(D) / 12:.0f}Rペース)")
print(f"{'配分(1R 1,000円)':<24}{'回収率':>8}{'年間損益':>11}{'最大1本除く':>10}{'払戻あり':>8}"
      f"{'元本割れ的中':>10}{'最長連敗':>8}" + "".join(f"{p[0]:>13}" for p in PERIODS)
      + f"{'尼崎':>7}{'若松':>7}")
for name, a, b, c in MIXES:
    g = (D["tan"] * a + D["fuku"] * b + D["trio"] * c) // 100
    st = a + b + c
    streak = best = 0
    for x in g:
        streak = streak + 1 if x < st else 0
        best = max(best, streak)
    per = "".join(
        f"{g[(D['month'] >= lo) & (D['month'] <= hi)].sum() / (((D['month'] >= lo) & (D['month'] <= hi)).sum() * st):>13.0%}"
        for _n, lo, hi in PERIODS)
    ven = "".join(f"{g[D['venue'] == v].sum() / ((D['venue'] == v).sum() * st):>7.0%}"
                  for v in VENUES)
    print(f"{name:<24}{g.sum() / (len(D) * st):>8.1%}{g.sum() - len(D) * st:>+10,}円"
          f"{(g.sum() - g.max()) / (len(D) * st):>10.1%}{(g > 0).mean():>8.1%}"
          f"{((g > 0) & (g < st)).mean():>10.1%}{best:>7}R{per}{ven}")
