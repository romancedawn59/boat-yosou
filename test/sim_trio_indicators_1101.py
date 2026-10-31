# -*- coding: utf-8 -*-
"""3連複に特化した5指標の一括検証(2026-11-01ケンさん「①継続 で5指標の一括検証に着手」)

    py -X utf8 test/sim_trio_indicators_1101.py

指標(朝6:30の配信だけで計算できるもの①②③⑤、昼のオッズが要るもの④):
  ① 3連複確率   = モデルの3連複1位の組(r1r2r3)が3着以内を独占する確率
  ② 崖          = 3着以内確率の3位と4位の差(大きいほど「3艇で決まる」)
  ③ 絡み度      = r1の(3着以内確率 − 1着確率)(大きいほど「勝ちきれないが絡む」)
  ④ 市場乖離    = 組ごとの モデル3連複確率 ÷ 市場3連複確率(締切前の3連複オッズの1/オッズを正規化)
  ⑤ 外枠度      = 4〜6号艇の3着以内確率の合計
買い方(100円/点):
  ①②③⑤ … 3連複モデル順1点 / 3点 / 5点 / 4艇ボックス4点(3着以内確率上位4艇) / r1軸流し6点(r1+上位4艇から2艇)
  ④     … モデル順上位K点(K=1,3,5,10,20)のうち 乖離>=th(th=1.0,1.2,1.5,2.0)の組だけ買う(該当なしのレースは買わない)
手順(結果を見る前に固定):
  ①②③⑤は前半(7/7〜9/14)の5分位で区切りを決め、5分位×5買い方=25マス。④は5×4=20マス。全120マス
  指標ごとに前半の回収率上位3マス(的中20本以上)を選び、後半(9/15〜)で採点
  合格 = 後半で回収率>100%・的中30本以上・最大1本除いても100%超
  合格マスには、後半で同じ点数をランダムに買った場合との比較(シャッフル2,000回)を付ける
  あわせて指標ごとに5分位×買い方の表を前半・後半で並べ、向き(単調さ)が両期間でそろうかを見る
"""
import glob
import json
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
from predictors import trifecta_probs  # noqa: E402

SPLIT = "20260915"
MIN_HITS_A, MIN_HITS_B = 20, 30
rng = np.random.default_rng(1101)
pd.set_option("display.width", 220)

print("データ準備中...", flush=True)
conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
trio_pay = defaultdict(dict)
for rid, cb, a in conn.execute("SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連複' AND race_id>='20260707' AND amount_yen>0"):
    trio_pay[rid][cb] = a
trio_pay = {r: next(iter(v.items())) for r, v in trio_pay.items() if len(v) == 1}
mkt = defaultdict(dict)
for rid, cb, o in conn.execute("SELECT race_id, combination, odds FROM odds WHERE bet_type='3連複' AND odds>0 AND fetched_at!='final-backfill'"):
    mkt[rid][cb] = o
conn.close()


def key3(xs):
    return "=".join(map(str, sorted(xs)))


races = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid = r["race_id"]; rk = r.get("ranked")
        if not rk or len(rk) < 6 or rid not in trio_pay:
            continue
        probs = {int(l): float(p) for l, p in rk}
        tri = trifecta_probs(probs); z = sum(tri.values())
        top3 = defaultdict(float); trio = defaultdict(float)
        for (a, b, c), v in tri.items():
            for x in (a, b, c):
                top3[x] += v / z
            trio[key3((a, b, c))] += v / z
        order = sorted(trio, key=trio.get, reverse=True)
        lanes_by_t3 = sorted(top3, key=top3.get, reverse=True)
        t3 = [top3[l] for l in lanes_by_t3]
        r1 = int(rk[0][0]); p1 = float(rk[0][1])
        others4 = [l for l in lanes_by_t3 if l != r1][:4]
        methods = {
            "モデル順1点": order[:1],
            "モデル順3点": order[:3],
            "モデル順5点": order[:5],
            "4艇ボックス4点": [key3(c) for c in combinations(lanes_by_t3[:4], 3)],
            "r1軸流し6点": [key3((r1, a, b)) for a, b in combinations(others4, 2)],
        }
        q = None
        o = mkt.get(rid)
        if o and len(o) >= 18:
            inv = {c: 1 / v for c, v in o.items()}
            s = sum(inv.values())
            q = {c: inv.get(c, 0) / s for c in trio}
        comb, pay = trio_pay[rid]
        races.append(dict(rid=rid, period=("後半" if rid >= SPLIT else "前半"), comb=comb, pay=pay,
                          i1=trio[order[0]], i2=t3[2] - t3[3], i3=top3[r1] - p1, i5=top3[4] + top3[5] + top3[6],
                          methods=methods, order=order, trio=dict(trio), q=q))
A = [x for x in races if x["period"] == "前半"]; B = [x for x in races if x["period"] == "後半"]
print(f"対象 {len(races):,}R(前半{len(A):,}・後半{len(B):,})、締切前3連複オッズあり 前半{sum(x['q'] is not None for x in A):,}・後半{sum(x['q'] is not None for x in B):,}")


def score(xs, picker):
    """picker(x) → 買う組のリスト。(回収率, 的中, 最大1本除く, 買ったR, 点数合計)"""
    n_pts = 0; tot = 0; hits = 0; top = 0; n_r = 0
    for x in xs:
        picks = picker(x)
        if not picks:
            continue
        n_r += 1; n_pts += len(picks)
        if x["comb"] in picks:
            tot += x["pay"]; hits += 1; top = max(top, x["pay"])
    if n_pts == 0:
        return np.nan, 0, np.nan, 0, 0
    return tot / (n_pts * 100), hits, (tot - top) / (n_pts * 100), n_r, n_pts


def shuffle_p(xs, picker, real, n=2000):
    """同じレースで同じ点数を20通りから無作為に買ったときに、実際以上の回収率が出る確率"""
    rows = [(len(picker(x)), x["pay"]) for x in xs if picker(x)]
    k = np.array([a for a, _ in rows]); pay = np.array([b for _, b in rows]); pts = k.sum()
    sims = np.array([(pay * (rng.random(len(k)) < k / 20)).sum() / (pts * 100) for _ in range(n)])
    return (sims >= real).mean()


IND = {"i1": "①3連複確率", "i2": "②崖", "i3": "③絡み度", "i5": "⑤外枠度"}
QL = ["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"]
METHODS = list(races[0]["methods"])
results = []
for ind, label in IND.items():
    cuts = np.quantile([x[ind] for x in A], [.2, .4, .6, .8])

    def qi(x, cuts=cuts, ind=ind):
        return int(np.searchsorted(cuts, x[ind], side="right"))

    print(f"\n===== {label}(前半の5分位の区切り: {', '.join(f'{c:.3f}' for c in cuts)}) 回収率 前半 / 後半 =====")
    tab = []
    cells = []
    for m in METHODS:
        row = {"買い方": m}
        for q_ in range(5):
            pick = (lambda x, q_=q_, m=m, qi=qi: x["methods"][m] if qi(x) == q_ else None)
            a = score(A, pick); b = score(B, pick)
            row[QL[q_]] = f"{a[0]:.0%} / {b[0]:.0%}"
            cells.append((a, b, m, q_, pick))
        tab.append(row)
    print(pd.DataFrame(tab).to_string(index=False))
    cells = [c for c in cells if c[0][1] >= MIN_HITS_A]
    cells.sort(key=lambda c: c[0][0], reverse=True)
    for rank, (a, b, m, q_, pick) in enumerate(cells[:3], 1):
        ok = b[0] > 1 and b[1] >= MIN_HITS_B and b[2] > 1
        results.append(dict(指標=label, 前半順位=rank, 条件=QL[q_], 買い方=m, 前半R=a[3], 前半回収率=a[0], 前半的中=a[1],
                            後半R=b[3], 後半回収率=b[0], 後半的中=b[1], 後半最大1本除く=b[2], 合格=("○" if ok else "×"),
                            p=(shuffle_p(B, pick, b[0]) if ok else np.nan)))

# ===== ④ 市場乖離 =====
print("\n===== ④市場乖離: モデル順上位K点のうち 乖離(モデル÷市場)>=th の組だけ買う 回収率 前半 / 後半(買ったR) =====")
A4 = [x for x in A if x["q"]]; B4 = [x for x in B if x["q"]]
tab = []; cells = []
for K in (1, 3, 5, 10, 20):
    row = {"K": f"上位{K}点"}
    for th in (1.0, 1.2, 1.5, 2.0):
        pick = (lambda x, K=K, th=th: [c for c in x["order"][:K] if x["q"].get(c, 0) > 0 and x["trio"][c] / x["q"][c] >= th])
        a = score(A4, pick); b = score(B4, pick)
        row[f"乖離>={th}"] = f"{a[0]:.0%} / {b[0]:.0%} ({a[3]}/{b[3]}R)"
        cells.append((a, b, f"上位{K}点", th, pick))
    tab.append(row)
print(pd.DataFrame(tab).to_string(index=False))
cells = [c for c in cells if c[0][1] >= MIN_HITS_A]
cells.sort(key=lambda c: c[0][0], reverse=True)
for rank, (a, b, K, th, pick) in enumerate(cells[:3], 1):
    ok = b[0] > 1 and b[1] >= MIN_HITS_B and b[2] > 1
    results.append(dict(指標="④市場乖離", 前半順位=rank, 条件=f"乖離>={th}", 買い方=K, 前半R=a[3], 前半回収率=a[0], 前半的中=a[1],
                        後半R=b[3], 後半回収率=b[0], 後半的中=b[1], 後半最大1本除く=b[2], 合格=("○" if ok else "×"),
                        p=(shuffle_p(B4, pick, b[0]) if ok else np.nan)))

# 参考: 乖離の大きさ別に「その組だけ1点」(モデル順上位20点の全組を乖離で分類)
print("\n[参考] 組単位: 乖離の大きさ別に1点ずつ買った回収率(モデル順上位20点の全組を分類)")
bins = [0, .5, .8, 1.0, 1.2, 1.5, 2.0, 99]
for label_p, xs in (("前半", A4), ("後半", B4)):
    acc = defaultdict(lambda: [0, 0, 0])
    for x in xs:
        for c in x["order"]:
            qv = x["q"].get(c, 0)
            if qv <= 0:
                continue
            b_ = int(np.searchsorted(bins, x["trio"][c] / qv, side="right")) - 1
            acc[b_][0] += 1; acc[b_][1] += (c == x["comb"]); acc[b_][2] += x["pay"] if c == x["comb"] else 0
    print(f"  {label_p}: " + " | ".join(f"{bins[k]}〜{bins[k + 1]}: {v[0]:,}点 的中{v[1]} {v[2] / (v[0] * 100):.0%}" for k, v in sorted(acc.items())))

# ===== まとめ =====
print("\n===== 指標ごとに前半の上位3マスを選び、後半で採点(全120マス・合格=後半>100%・的中30本以上・最大1本除いても100%超) =====")
res = pd.DataFrame(results)
print(res.to_string(index=False, formatters={"前半回収率": "{:.1%}".format, "後半回収率": "{:.1%}".format, "後半最大1本除く": "{:.1%}".format,
                                              "p": lambda v: "-" if pd.isna(v) else f"{v:.3f}"}))
print(f"\n合格 {(res['合格'] == '○').sum()} / {len(res)}")
print("\n[比較の基準] 後半の全レース:")
for m in METHODS:
    b = score(B, lambda x, m=m: x["methods"][m])
    print(f"  {m}: {b[0]:.1%}(的中{b[1]:,}・{b[3]:,}R)")
