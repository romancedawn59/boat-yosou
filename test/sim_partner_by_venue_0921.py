# -*- coding: utf-8 -*-
"""若松・尼崎それぞれで、3連単ドンピシャ(r1-r2-r3)の相方を探す(2026-09-21ケンさん指示)
3連単1,000円(相方なし)も必ず比較に入れる。

    py -X utf8 test/sim_partner_by_venue_0921.py

1位勝率22.5〜30%帯。予想順位は1年分のwalk-forward台帳(2025-10〜2026-09)。返還反映。
相方候補: 単勝r1 / 複勝r1 / 2連単r1-r2 / 2連複r1=r2 / ワイドr1=r2 / 3連複r1=r2=r3 /
          3連単r1-r3-r2 / 3連単r1-r2-r4 / 3連単r1-r4-r3
配分: 3連単/相方 = 1000/0, 700/300, 500/500, 300/700(1レース1,000円)
"""
import sqlite3
from collections import defaultdict

import pandas as pd

WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
VENUES = {20: "若松", 13: "尼崎"}
PERIODS = [("2025-10", "2026-01"), ("2026-02", "2026-05"), ("2026-06", "2026-09")]
SPLITS = [(700, 300), (500, 500), (300, 700)]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2025-10-01' "
        "AND r.venue_code IN (13, 20)"):
    pay[rid][(bt, comb)] = max(pay[rid].get((bt, comb), 0), amt or 0)
conn.close()


def eq(*x):
    return "=".join(map(str, sorted(x)))


L = pd.read_csv(WF_LEDGER, encoding="utf-8-sig", dtype={"refund": str})
L = L[L["venue"].isin(VENUES) & (L["p1"] >= 0.225) & (L["p1"] < 0.30)]
rows = []
for rid, m, v, pred, rf in zip(L["race_id"], L["month"], L["venue"], L["pred"],
                               L["refund"].fillna("")):
    l = [int(x) for x in pred.split("-")]
    refund = {int(x) for x in rf.split("-") if x}
    p = pay.get(rid, {})
    r1, r2, r3, r4 = l[:4]

    def got(lanes, key):
        return 100 if set(lanes) & refund else p.get(key, 0)
    rows.append({
        "month": m, "venue": v,
        "3連単r1-r2-r3": got((r1, r2, r3), ("3連単", f"{r1}-{r2}-{r3}")),
        "単勝r1": got((r1,), ("単勝", str(r1))),
        "複勝r1": got((r1,), ("複勝", str(r1))),
        "2連単r1-r2": got((r1, r2), ("2連単", f"{r1}-{r2}")),
        "2連複r1=r2": got((r1, r2), ("2連複", eq(r1, r2))),
        "ワイドr1=r2": got((r1, r2), ("拡連複", eq(r1, r2))),
        "3連複r1=r2=r3": got((r1, r2, r3), ("3連複", eq(r1, r2, r3))),
        "3連単r1-r3-r2": got((r1, r2, r3), ("3連単", f"{r1}-{r3}-{r2}")),
        "3連単r1-r2-r4": got((r1, r2, r4), ("3連単", f"{r1}-{r2}-{r4}")),
        "3連単r1-r4-r3": got((r1, r3, r4), ("3連単", f"{r1}-{r4}-{r3}")),
    })
D = pd.DataFrame(rows)
MAIN = "3連単r1-r2-r3"
PARTNERS = [c for c in D.columns if c not in ("month", "venue", MAIN)]


def line(name, g, st, sub):
    k = best = 0
    for val in g:
        k = k + 1 if val < st else 0
        best = max(best, k)
    per = " ".join(f"{g[(sub['month'] >= a) & (sub['month'] <= b)].sum() / (((sub['month'] >= a) & (sub['month'] <= b)).sum() * st):>5.0%}"
                   for a, b in PERIODS)
    ok = sum(1 for a, b in PERIODS
             if g[(sub["month"] >= a) & (sub["month"] <= b)].sum()
             > ((sub["month"] >= a) & (sub["month"] <= b)).sum() * st)
    print(f"{name:<30}{g.sum() / (len(sub) * st):>8.1%}{g.sum() - len(sub) * st:>+10,}円"
          f"{(g.sum() - g.max()) / (len(sub) * st):>9.1%}{(g > 0).mean():>8.1%}"
          f"{((g > 0) & (g < st)).mean():>7.1%}{best:>6}R   {per}  {ok}/3")


for v, vn in VENUES.items():
    sub = D[D["venue"] == v].reset_index(drop=True)
    print(f"\n########## {vn} {len(sub)}R(1年・月{len(sub) / 12:.0f}Rペース) ##########")
    print("― 券ごとの単体の実力(100円あたり) ―")
    print(f"{'券':<18}{'的中':>5}{'的中率':>7}{'平均配当':>9}{'回収率':>8}{'最大1本除く':>10}   期間①  期間②  期間③")
    for c in [MAIN] + PARTNERS:
        h = sub[sub[c] > 100][c]
        per = " ".join(f"{sub[(sub['month'] >= a) & (sub['month'] <= b)][c].mean() / 100:>5.0%}"
                       for a, b in PERIODS)
        print(f"{c:<18}{len(h):>5}{len(h) / len(sub):>7.1%}{(h.mean() if len(h) else 0):>8,.0f}円"
              f"{sub[c].mean() / 100:>8.1%}{(sub[c].sum() - sub[c].max()) / (len(sub) * 100):>10.1%}   {per}")
    print("― 配分別(1レース1,000円) ―")
    print(f"{'配分':<30}{'回収率':>8}{'年間損益':>11}{'最大1本除く':>9}{'払戻あり':>8}{'ガミ':>7}{'最長連敗':>7}"
          f"   期間①  期間②  期間③ 黒字")
    line("3連単1,000(相方なし)", sub[MAIN] * 10, 1000, sub)
    for c in PARTNERS:
        for a, b in SPLITS:
            line(f"3連単{a}+{c}{b}", (sub[MAIN] * a + sub[c] * b) // 100, 1000, sub)
