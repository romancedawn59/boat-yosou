# -*- coding: utf-8 -*-
"""中穴(当選組が20〜40番人気)で決まるレースを事前に見分けられるか
(2026-09-21ケンさん指示「結果の人気20番から40番くらいを拾える可能性を高める
 事前レース判定方法を考えて」)

    py -X utf8 test/verify_midlongshot_race_filter.py

■ 定義
ラベル = 当選3連単が締切前オッズ(odds/odds_final)で20〜40番人気。
材料 = 本番配信picks JSONの6艇の1着確率 + entriesの出走表情報 + 場/R/グレード。
すべてレース前に確定している情報のみ。

■ 方法(事前登録)
学習 2026-07-21〜08-31 / 採点 2026-09-01〜09-19(学習に未使用)。
LightGBM二値分類。採点期間でスコア上位20%・上位10%のレースの
「20〜40番人気決着率」を全体平均と比べる。
合格基準: 上位20%で全体平均の1.3倍以上。
補足として単独条件別の決着率と、判定レースで当たった並び(予想順位空間)を出す。
"""
import glob
import json
import math
import sqlite3
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

LO, HI = 20, 40
CLASS = {"B2": 0, "B1": 1, "A2": 2, "A1": 3}
GRADE = {"一般": 0, "G3": 1, "G2": 2, "G1": 3, "SG": 4}

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
arr = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id=res.race_id WHERE r.date>='2026-07-21' "
        "AND res.arrival_order IS NOT NULL"):
    arr[rid][lane] = ao
pay3 = {}
for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2026-07-21' "
        "AND p.bet_type='3連単'"):
    pay3[rid] = (comb, amt or 0)
odds = defaultdict(dict)
for tbl in ("odds", "odds_final"):
    for rid, comb, o in conn.execute(
            f"SELECT o.race_id, o.combination, o.odds FROM {tbl} o "
            f"JOIN races r ON r.race_id=o.race_id WHERE r.date>='2026-07-21' "
            f"AND o.bet_type='3連単' AND o.odds IS NOT NULL"):
        odds[rid][comb] = o
ent = defaultdict(dict)
for rid, lane, cls, nw, n2, lw, m2, st, f in conn.execute(
        "SELECT e.race_id, e.lane, e.racer_class, e.national_win_rate, "
        "e.national_2rate, e.local_win_rate, e.motor_2rate, e.avg_st, "
        "e.flying_count FROM entries e JOIN races r ON r.race_id=e.race_id "
        "WHERE r.date>='2026-07-21'"):
    ent[rid][lane] = (CLASS.get(cls), nw, n2, lw, m2, st, f or 0)
grade = {rid: GRADE.get(g, 0) for rid, g in conn.execute(
    "SELECT race_id, grade FROM races WHERE date>='2026-07-21'")}
conn.close()

rows = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-0[789]-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        rid = r["race_id"]
        if not r.get("ranked") or len(r["ranked"]) < 6 or rid not in pay3:
            continue
        o = odds.get(rid)
        comb, amt = pay3[rid]
        if not o or len(o) < 110 or comb not in o or not amt:
            continue
        e = ent.get(rid)
        if not e or len(e) < 6:
            continue
        rank = sorted(o.values()).index(o[comb]) + 1
        lanes = [x[0] for x in r["ranked"]]
        ps = [x[1] for x in r["ranked"]]
        tot = sum(ps)
        nps = [p / tot for p in ps]
        ent_h = -sum(p * math.log(p) for p in nps if p > 0)
        nw = [e[l][1] or 0 for l in range(1, 7)]
        cls = [e[l][0] if e[l][0] is not None else 1 for l in range(1, 7)]
        mot = [e[l][4] or 0 for l in range(1, 7)]
        st = [e[l][5] or 0.17 for l in range(1, 7)]
        top3 = sorted(arr[rid], key=arr[rid].get)[:3] if len(arr.get(rid, {})) >= 3 else None
        if not top3:
            continue
        rows.append({
            "date": pk["date"], "race_id": rid, "y": int(LO <= rank <= HI),
            "rank": rank, "amt": amt,
            "pattern": tuple(lanes.index(l) + 1 for l in top3),
            "p1": ps[0], "p2": ps[1], "p3": ps[2], "p4": ps[3], "p5": ps[4], "p6": ps[5],
            "gap12": ps[0] - ps[1], "gap23": ps[1] - ps[2], "gap34": ps[2] - ps[3],
            "entropy": ent_h, "p_lane1": dict(r["ranked"])[1],
            "rank_lane1": lanes.index(1) + 1, "fav_lane": lanes[0],
            "lane1_nw": nw[0], "lane1_cls": cls[0], "lane1_st": st[0],
            "lane1_mot": mot[0], "n_a1": sum(1 for c in cls if c == 3),
            "n_b": sum(1 for c in cls if c <= 1),
            "a1_outer": sum(1 for i in range(3, 6) if cls[i] == 3),
            "nw_max": max(nw), "nw_spread": max(nw) - min(nw),
            "nw_std": float(np.std(nw)), "best_nw_lane": int(np.argmax(nw)) + 1,
            "mot_max_lane": int(np.argmax(mot)) + 1, "mot_spread": max(mot) - min(mot),
            "st_min_outer": min(st[2:]), "st_edge": st[0] - min(st[2:]),
            "n_f": sum(1 for l in range(1, 7) if e[l][6] > 0),
            "venue_code": r["venue_code"], "race_no": r["race_no"],
            "grade": grade.get(rid, 0),
        })
D = pd.DataFrame(rows)
FEATS = [c for c in D.columns if c not in ("date", "race_id", "y", "rank", "amt", "pattern")]
base = D["y"].mean()
print(f"対象: {len(D):,}R(7/21〜9/19・オッズ記録あり) / {LO}〜{HI}番人気で決着: {base:.1%}")
z = D[D["y"] == 1]["amt"]
print(f"  そのときの3連単配当: 中央値{int(z.median()):,}円 "
      f"(25%点{int(z.quantile(.25)):,}円〜75%点{int(z.quantile(.75)):,}円)")

print("\n― 単独条件ごとの中穴決着率 ―")


def show(label, mask):
    sub = D[mask]
    if len(sub) >= 80:
        print(f"  {label:<26}{len(sub):>6}R  {sub['y'].mean():>6.1%}  "
              f"(平均比{sub['y'].mean() / base:.2f}倍)")


for lo, hi in ((0, .20), (.20, .30), (.30, .40), (.40, .50), (.50, .65), (.65, 1.01)):
    show(f"1位勝率 {lo:.0%}〜{hi:.0%}", (D["p1"] >= lo) & (D["p1"] < hi))
for k in (1, 2, 3):
    show(f"1号艇の予想順位 {k}位", D["rank_lane1"] == k)
show("1号艇の予想順位 4位以下", D["rank_lane1"] >= 4)
for k in (0, 1, 2):
    show(f"A1の人数 {k}人", D["n_a1"] == k)
show("A1の人数 3人以上", D["n_a1"] >= 3)
show("外枠(4-6)にA1がいる", D["a1_outer"] >= 1)
show("1号艇がB級", D["lane1_cls"] <= 1)
show("1号艇がA1", D["lane1_cls"] == 3)
show("最高勝率の艇が4〜6枠", D["best_nw_lane"] >= 4)
show("F持ちが2人以上", D["n_f"] >= 2)
for g, name in ((0, "一般戦"), (1, "G3"),):
    show(f"グレード {name}", D["grade"] == g)
show("グレード G2以上", D["grade"] >= 2)

tr = D[D["date"] <= "2026-08-31"]
te = D[D["date"] >= "2026-09-01"].copy()
print(f"\n― 判定モデル: 学習{len(tr):,}R(〜8/31) → 採点{len(te):,}R(9/1〜) ―")
clf = lgb.train({"objective": "binary", "verbosity": -1, "learning_rate": 0.03,
                 "num_leaves": 15, "min_data_in_leaf": 60,
                 "feature_fraction": 0.8, "seed": 7},
                lgb.Dataset(tr[FEATS], label=tr["y"],
                            categorical_feature=["venue_code", "race_no"]),
                num_boost_round=200)
te["score"] = clf.predict(te[FEATS])
te = te.sort_values("score", ascending=False)
tbase = te["y"].mean()
print(f"  採点期間の全体平均: {tbase:.1%}")
for pct in (10, 20, 30, 50):
    k = int(len(te) * pct / 100)
    top = te.head(k)
    print(f"  スコア上位{pct:>2}%({k:>4}R): 中穴決着率{top['y'].mean():>6.1%} "
          f"(平均比{top['y'].mean() / tbase:.2f}倍) / 3連単配当の中央値"
          f"{int(top['amt'].median()):,}円")
bot = te.tail(int(len(te) * 0.2))
print(f"  スコア下位20%: 中穴決着率{bot['y'].mean():.1%}")
imp = sorted(zip(FEATS, clf.feature_importance("gain")), key=lambda x: -x[1])
tot = sum(v for _, v in imp)
print("  効いた材料(上位8): " + " / ".join(f"{f}{v / tot:.0%}" for f, v in imp[:8]))
k20 = int(len(te) * 0.2)
ok = te.head(k20)["y"].mean() / tbase >= 1.3
print(f"  判定: {'合格' if ok else '不合格'}(基準=上位20%で平均の1.3倍以上)")

print("\n― 採点期間の上位20%レースで、中穴決着したときの並び(予想順位) ―")
hit = te.head(k20)
pc = Counter(hit[hit["y"] == 1]["pattern"])
n_hit = int(hit["y"].sum())
for pt, v in pc.most_common(8):
    print(f"  r{pt[0]}-r{pt[1]}-r{pt[2]}: {v}回")
w = Counter(p[0] for p in hit[hit["y"] == 1]["pattern"])
print(f"  勝者の予想順位(中穴決着{n_hit}R): "
      + " ".join(f"{k}位{w[k] / max(1, n_hit):.0%}" for k in range(1, 7)))
