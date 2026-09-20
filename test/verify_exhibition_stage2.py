# -*- coding: utf-8 -*-
"""展示情報(展示タイム・チルト・体重)で着順の予想精度は上がるか(2026-09-21ケンさん承認)

    py -X utf8 test/verify_exhibition_stage2.py

本番配信の確率(picks JSON)を土台に、展示情報を足した2段目モデルで順位を付け直す。
展示情報は締切直前にしか出ないので、使えるのは昼の買い直し枠だけ(朝の一括購入には使えない)。
学習 2026-07-21〜08-31 / 採点 2026-09-01〜(学習に未使用)。展示データのある場・レースだけ。
  土台   : 配信の確率をレース内で合計1に正規化しただけ
  2段目A : 土台の確率+順位+枠 だけで学習(展示なし=比較の基準)
  2段目B : A + 展示タイムのレース内順位・平均との差・チルト・体重差
■ 合格基準(事前登録): BがAに対して 採点期間のloglossを0.002以上改善 かつ
  予想1位の1着率+1.5pt以上 かつ ドンピシャ(1-2-3位の並び的中)率が下がらない。
"""
import glob
import json
import math
import os
import sqlite3
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
exh = defaultdict(dict)
for rid, lane, w, t, tilt in conn.execute(
        "SELECT race_id, lane, weight_kg, exhibition_time, tilt FROM exhibition"):
    exh[rid][lane] = (w, t, tilt)
arr = defaultdict(dict)
for rid, lane, ao in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id=res.race_id WHERE r.date>='2026-07-21'"):
    arr[rid][lane] = ao
conn.close()

rows = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    d = os.path.basename(path)[6:16]
    if d < "2026-07-21":
        continue
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid = r["race_id"]
        e, a = exh.get(rid), arr.get(rid)
        if not r.get("ranked") or len(r["ranked"]) < 6 or not e or len(e) < 6 or not a:
            continue
        if any(e[l][1] is None for l in range(1, 7)) or 1 not in a.values():
            continue
        tot = sum(p for _l, p in r["ranked"])
        times = [e[l][1] for l in range(1, 7)]
        ws = [e[l][0] or np.nan for l in range(1, 7)]
        order = sorted(range(1, 7), key=lambda l: e[l][1])
        for i, (lane, p) in enumerate(r["ranked"]):
            rows.append({
                "date": d, "race_id": rid, "venue": r["venue_code"], "lane": lane,
                "p_norm": p / tot, "p_raw": p, "rank": i + 1, "p_sum": tot,
                "exh_rank": order.index(lane) + 1,
                "exh_diff": e[lane][1] - float(np.mean(times)),
                "exh_gap_best": e[lane][1] - min(times),
                "tilt": e[lane][2] if e[lane][2] is not None else np.nan,
                "w_diff": (e[lane][0] - float(np.nanmean(ws))) if e[lane][0] else np.nan,
                "win": int(a.get(lane) == 1), "ao": a.get(lane) or 9,
            })
D = pd.DataFrame(rows)
tr, te = D[D["date"] < "2026-09-01"], D[D["date"] >= "2026-09-01"].copy()
print(f"展示データのあるレース: 学習{tr['race_id'].nunique():,}R / 採点{te['race_id'].nunique():,}R "
      f"(場: {sorted(D['venue'].unique())})")

BASE = ["p_norm", "p_raw", "rank", "p_sum", "lane"]
EXH = ["exh_rank", "exh_diff", "exh_gap_best", "tilt", "w_diff"]
PARAMS = {"objective": "binary", "verbosity": -1, "learning_rate": 0.03, "num_leaves": 7,
          "min_data_in_leaf": 100, "seed": 7}


def fit_predict(cols):
    m = lgb.train(PARAMS, lgb.Dataset(tr[cols], label=tr["win"]), num_boost_round=200)
    p = m.predict(te[cols])
    imp = sorted(zip(cols, m.feature_importance("gain")), key=lambda x: -x[1])
    return p, imp


te["s_base"] = te["p_norm"]
te["s_A"], _ = fit_predict(BASE)
te["s_B"], imp = fit_predict(BASE + EXH)
tot = sum(v for _, v in imp)
print("2段目Bで効いた材料: " + " / ".join(f"{c}{v / tot:.0%}" for c, v in imp[:7]))

print(f"\n{'':<10}{'logloss':>9}{'予想1位の1着率':>13}{'予想1位が3着内':>13}{'ドンピシャ率':>10}{'上位3艇で決着':>12}")
res = {}
for name, col in (("土台", "s_base"), ("2段目A", "s_A"), ("2段目B", "s_B")):
    s = te[col] / te.groupby("race_id")[col].transform("sum")
    ll = -np.mean(np.where(te["win"] == 1, np.log(s.clip(1e-6)), np.log((1 - s).clip(1e-6))))
    n = r1 = r1in3 = don = set3 = 0
    for _rid, g in te.assign(s=s).groupby("race_id"):
        gs = g.sort_values("s", ascending=False)
        ao = list(gs["ao"])
        n += 1
        r1 += ao[0] == 1
        r1in3 += ao[0] <= 3
        don += ao[:3] == [1, 2, 3]
        set3 += sorted(ao[:3]) == [1, 2, 3]
    res[name] = (ll, r1 / n, don / n)
    print(f"{name:<10}{ll:>9.4f}{r1 / n:>13.1%}{r1in3 / n:>13.1%}{don / n:>10.1%}{set3 / n:>12.1%}")
a, b = res["2段目A"], res["2段目B"]
ok = (a[0] - b[0] >= 0.002) and (b[1] - a[1] >= 0.015) and (b[2] >= a[2])
print(f"\n判定: {'合格' if ok else '不合格'} (logloss改善{a[0] - b[0]:+.4f} / "
      f"1位の1着率{(b[1] - a[1]) * 100:+.1f}pt / ドンピシャ率{(b[2] - a[2]) * 100:+.1f}pt)")
