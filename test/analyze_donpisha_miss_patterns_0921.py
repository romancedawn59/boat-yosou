# -*- coding: utf-8 -*-
"""ドンピシャ1位(r1-r2-r3)が外れたとき、結果は予想順位でどう並んだか(2026-09-21ケンさん指示)

    py -X utf8 test/analyze_donpisha_miss_patterns_0921.py

予想順位は1年分のwalk-forward台帳(2025-10〜2026-09・全24場)。
結果の3連単を予想順位(r1〜r6)に置き換えて集計する。回収率はその並びを毎回100円買った場合。
"""
import sqlite3
from collections import Counter

import pandas as pd

WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay3 = {rid: (comb, amt or 0) for rid, comb, amt in conn.execute(
    "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2025-10-01' "
    "AND p.bet_type='3連単'")}
conn.close()

L = pd.read_csv(WF_LEDGER, encoding="utf-8-sig", dtype={"refund": str})
rows = []
for rid, v, p1, pred in zip(L["race_id"], L["venue"], L["p1"], L["pred"]):
    if rid not in pay3:
        continue
    l = [int(x) for x in pred.split("-")]
    comb, amt = pay3[rid]
    try:
        pt = tuple(l.index(int(x)) + 1 for x in comb.split("-"))
    except ValueError:
        continue
    rows.append((v, p1, pt, amt))
D = pd.DataFrame(rows, columns=["venue", "p1", "pt", "amt"])


def report(sub, title):
    n = len(sub)
    miss = sub[sub["pt"] != (1, 2, 3)]
    print(f"\n===== {title}: {n:,}R / ドンピシャ的中{n - len(miss)}本({(n - len(miss)) / n:.1%})"
          f" / 外れ{len(miss):,}R =====")
    print("― 外れたときの並び(多い順・上位15) ―")
    print(f"{'並び':<12}{'回数':>6}{'外れの中の割合':>12}{'平均配当':>10}{'毎回100円の回収率':>15}")
    cnt = Counter(miss["pt"])
    for pt, c in cnt.most_common(15):
        a = miss[miss["pt"] == pt]["amt"]
        print(f"r{pt[0]}-r{pt[1]}-r{pt[2]:<6}{c:>6}{c / len(miss):>12.1%}{a.mean():>9,.0f}円"
              f"{a.sum() / (n * 100):>15.1%}")
    m = len(miss)
    print("― 外れ方のまとめ(外れレースに占める割合) ―")
    w = Counter(p[0] for p in miss["pt"])
    print("  1着に来た予想順位: " + " ".join(f"r{k}={w[k] / m:.0%}" for k in range(1, 7)))
    pos = Counter(("1着" if p[0] == 1 else "2着" if p[1] == 1 else "3着" if p[2] == 1 else "圏外")
                  for p in miss["pt"])
    print("  予想1位(r1)の結果: " + " ".join(f"{k}={pos[k] / m:.0%}" for k in ("1着", "2着", "3着", "圏外")))
    kinds = Counter()
    for p in miss["pt"]:
        s = set(p)
        if s == {1, 2, 3}:
            kinds["上位3艇の並び替え(3連複は的中)"] += 1
        elif max(p) == 4:
            kinds["r4が割り込み(r5・r6なし)"] += 1
        elif 6 in s:
            kinds["r6が絡む"] += 1
        else:
            kinds["r5が絡む(r6なし)"] += 1
    for k, c in kinds.most_common():
        a = miss[[(set(p) == {1, 2, 3}) if k.startswith("上位3") else
                  (max(p) == 4) if k.startswith("r4") else
                  (6 in p) if k.startswith("r6") else
                  (5 in p and 6 not in p) for p in miss["pt"]]]["amt"]
        print(f"  {k:<26}{c:>6}R {c / m:>6.1%}  平均配当{a.mean():>8,.0f}円")


band = D[(D["p1"] >= 0.225) & (D["p1"] < 0.30)]
report(band, "1位勝率22.5〜30%・全24場")
report(band[band["venue"].isin([13, 20])], "1位勝率22.5〜30%・尼崎+若松")
report(D, "全帯・全24場")
