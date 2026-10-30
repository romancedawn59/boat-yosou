# -*- coding: utf-8 -*-
"""何点まで絞ればプラスになるか(2026-10-31ケンさん「100点買うのか、30点買うのか2点まで絞れば良いのか」)

    py -X utf8 test/sim_points_k_1031.py

見るもの(基準は結果を見る前に固定):
  券種 = 3連単(120通り)・3連複(20)・2連単(30)・2連複(15)
  並べ方 = ①モデル確率の高い順(朝6:30の配信だけで決まる・全レース)
           ②期待値順 = モデル確率×15分前オッズ(昼のオッズがあるレースだけ)
  点数K = 1点から全点まで段階的に。金額は均等(100円/点)。3連単は確率比例の配分も試す
  型 = 1位勝率帯(<15 / 15〜22.5 / 22.5〜30 / 30〜40 / 40%〜)
  手順 = 前半(7/7〜9/14)で帯ごとに最良の(券種・並べ方・K)を決め、後半(9/15〜10/30)でそれを採点
  合格 = 後半で回収率>100%・的中30本以上・最大1本除いても100%超
  あわせて「n番目の買い目だけ買ったときの回収率」を出す。これが100%を割る順位から先は、何点足しても平均を下げる
"""
import glob
import json
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
from predictors import trifecta_probs  # noqa: E402

SPLIT = "20260915"
BANDS = [("<15%", 0, .15), ("15〜22.5%", .15, .225), ("22.5〜30%", .225, .30), ("30〜40%", .30, .40), ("40%〜", .40, 1.01)]
KS = {"3連単": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120],
      "3連複": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
      "2連単": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30],
      "2連複": [1, 2, 3, 4, 5, 6, 8, 10, 15]}
RANK_BUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 60), (61, 120)]

print("データ準備中...", flush=True)
conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
odds = defaultdict(dict)
for rid, bt, cb, o in conn.execute("SELECT race_id, bet_type, combination, odds FROM odds "
                                   "WHERE odds IS NOT NULL AND odds > 0 AND fetched_at != 'final-backfill'"):
    odds[(rid, bt)][cb] = o
pay = defaultdict(dict)
for rid, bt, cb, a in conn.execute("SELECT race_id, bet_type, combination, amount_yen FROM payouts "
                                   "WHERE race_id >= '20260707' AND bet_type IN ('3連単','3連複','2連単','2連複')"):
    pay[rid][(bt, cb)] = a or 0
conn.close()


def band_of(p1):
    for name, lo, hi in BANDS:
        if lo <= p1 < hi:
            return name


# レースごとに: 券種→並べ方→(当たり目の順位, 当たり目の確率, 払戻/100円, 上位K点の確率累積)
recs = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid = r["race_id"]
        if not r.get("ranked") or len(r["ranked"]) < 6 or rid not in pay:
            continue
        probs = {int(l): float(p) for l, p in r["ranked"]}
        tri = trifecta_probs(probs)
        z = sum(tri.values())
        P = {"3連単": {"-".join(map(str, k)): v / z for k, v in tri.items()}}
        trio, ex, qn = defaultdict(float), defaultdict(float), defaultdict(float)
        for (a, b, c), v in tri.items():
            trio["=".join(map(str, sorted((a, b, c))))] += v / z
            ex[f"{a}-{b}"] += v / z
            qn["=".join(map(str, sorted((a, b))))] += v / z
        P["3連複"], P["2連単"], P["2連複"] = dict(trio), dict(ex), dict(qn)
        rec = dict(rid=rid, period=("後半" if rid >= SPLIT else "前半"), band=band_of(float(r["ranked"][0][1])), has_odds=False)
        for bt, pr in P.items():
            win = [cb for (t, cb) in pay[rid] if t == bt]
            if len(win) != 1:  # 同着などは除く
                rec[bt] = None
                continue
            win = win[0]
            amt = pay[rid][(bt, win)]
            order = sorted(pr, key=pr.get, reverse=True)
            cum = np.cumsum([pr[cb] for cb in order])
            rec[bt] = {"model": (order.index(win) + 1, pr[win], amt, cum)}
            o = odds.get((rid, bt))
            if o and len(o) >= len(pr) * 0.9:
                ev = {cb: pr[cb] * o.get(cb, 0) for cb in pr}
                order2 = sorted(ev, key=ev.get, reverse=True)
                rec[bt]["ev"] = (order2.index(win) + 1, pr[win], amt, None)
                if bt == "3連単":
                    rec["has_odds"] = True
        recs.append(rec)
print(f"対象 {len(recs):,}R(前半{sum(r['period'] == '前半' for r in recs):,}・後半{sum(r['period'] == '後半' for r in recs):,})、"
      f"うち15分前オッズあり {sum(r['has_odds'] for r in recs):,}R")


def roi_k(rs, bt, order, k, weight="equal"):
    """上位k点を買ったときの (回収率, 的中数, 最大1本除く回収率, R数)。均等=100円/点、prop=k*100円を確率比例で配分"""
    rets, n = [], 0
    for r in rs:
        x = r.get(bt)
        if not x or order not in x:
            continue
        rank, pw, amt, cum = x[order]
        n += 1
        if rank <= k:
            if weight == "equal":
                rets.append(amt)
            else:
                rets.append(amt * (pw / cum[k - 1]) * k)
        else:
            rets.append(0)
    if n == 0:
        return np.nan, 0, np.nan, 0
    inv = n * k * 100
    tot = sum(rets)
    return tot / inv, sum(1 for v in rets if v > 0), (tot - max(rets)) / inv, n


def sel(rs, period=None, band=None):
    return [r for r in rs if (period is None or r["period"] == period) and (band is None or r["band"] == band)]


# ===== ① 全体: 券種×K(モデル順・均等) 前半/後半 =====
print("\n===== ① 型を見ずに全レース: 券種×点数K(モデル確率順・100円/点) =====")
for bt in KS:
    print(f"\n[{bt}]  K: 前半回収率(的中) / 後半回収率(的中)")
    line = []
    for k in KS[bt]:
        a = roi_k(sel(recs, "前半"), bt, "model", k)
        b = roi_k(sel(recs, "後半"), bt, "model", k)
        line.append(f"{k:>3}点: {a[0]:6.1%}({a[1]:>5}) / {b[0]:6.1%}({b[1]:>5})")
    print("\n".join(line))

# ===== ② 順位別の単独回収率 =====
print("\n===== ② n番目の買い目『だけ』を買ったときの回収率(3連単・100円) — 100%を割る順位から先は足すほど平均が下がる =====")
for order, label, rs_all in (("model", "モデル確率順(全レース)", recs), ("ev", "期待値順(オッズありレース)", [r for r in recs if r["has_odds"]])):
    print(f"\n[{label}]")
    hdr = "順位".rjust(8) + " | " + " | ".join(b[0].rjust(10) for b in BANDS) + " | " + "全体".rjust(10)
    print("    " + hdr)
    for period in ("前半", "後半"):
        for lo, hi in RANK_BUCKETS:
            cells = []
            for name in [b[0] for b in BANDS] + [None]:
                rs = sel(rs_all, period, name)
                tot, n = 0.0, 0
                for r in rs:
                    x = r.get("3連単")
                    if not x or order not in x:
                        continue
                    n += 1
                    rank, pw, amt, cum = x[order]
                    if lo <= rank <= hi:
                        tot += amt
                cells.append(f"{tot / (n * (hi - lo + 1) * 100):9.0%}" if n else "-".rjust(9))
            print(f"{period} {lo:>3}〜{hi:<3} | " + " | ".join(cells))

# ===== ③ 帯別: 券種×並べ方×Kを前半で選び、後半で採点 =====
print("\n===== ③ 帯ごとに前半で最良の(券種・並べ方・K・配分)を選び、後半で採点(合格=後半>100%・的中30本以上・最大1本除いても100%超) =====")
rows = []
for name, *_ in BANDS:
    cands = []
    for bt in KS:
        for order in ("model", "ev"):
            for k in KS[bt]:
                for w in (("equal", "prop") if bt == "3連単" and order == "model" else ("equal",)):
                    a = roi_k(sel(recs, "前半", name), bt, order, k, w)
                    if a[1] >= 20 and not np.isnan(a[0]):
                        cands.append((a[0], bt, order, k, w, a[1], a[3]))
    if not cands:
        continue
    cands.sort(reverse=True)
    for rank_c, (ra, bt, order, k, w, ha, na) in enumerate(cands[:3], 1):
        b = roi_k(sel(recs, "後半", name), bt, order, k, w)
        ok = (b[0] > 1) and (b[1] >= 30) and (b[2] > 1)
        rows.append(dict(帯=name, 前半順位=rank_c, 券種=bt, 並べ方=("モデル順" if order == "model" else "期待値順"), K=k, 配分=("均等" if w == "equal" else "確率比例"),
                         前半R=na, 前半回収率=ra, 前半的中=ha, 後半R=b[3], 後半回収率=b[0], 後半的中=b[1], 後半最大1本除く=b[2], 合格=("○" if ok else "×")))
tab = pd.DataFrame(rows)
print(tab.to_string(index=False, formatters={k: "{:.1%}".format for k in ("前半回収率", "後半回収率", "後半最大1本除く")}))

# ===== ④ 帯22.5〜30%の K曲線 =====
print("\n===== ④ 帯22.5〜30%: 点数Kの曲線(前半 / 後半) =====")
for bt, order in (("3連単", "model"), ("3連単", "ev"), ("3連複", "model"), ("3連複", "ev"), ("2連単", "model")):
    print(f"\n[{bt}・{'モデル順' if order == 'model' else '期待値順'}]")
    for k in KS[bt]:
        a = roi_k(sel(recs, "前半", "22.5〜30%"), bt, order, k)
        b = roi_k(sel(recs, "後半", "22.5〜30%"), bt, order, k)
        print(f"{k:>3}点: 前半{a[0]:6.1%}({a[1]:>3}/{a[3]}) 後半{b[0]:6.1%}({b[1]:>3}/{b[3]}) 後半最大1本除く{b[2]:6.1%}")

# ===== ⑤ 3連単 確率比例配分 vs 均等(全体・モデル順) =====
print("\n===== ⑤ 3連単 モデル順: 均等 vs 確率比例(全レース・後半) =====")
for k in (3, 5, 10, 20, 40, 120):
    e = roi_k(sel(recs, "後半"), "3連単", "model", k, "equal")
    p = roi_k(sel(recs, "後半"), "3連単", "model", k, "prop")
    print(f"{k:>3}点: 均等{e[0]:6.1%} 確率比例{p[0]:6.1%}")
