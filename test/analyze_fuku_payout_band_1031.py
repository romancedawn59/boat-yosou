# -*- coding: utf-8 -*-
"""複勝r1の配当(200円未満が多い)と帯の関係(2026-10-31ケンさん「結果のオッズが200円切ってることと帯の関連はどう分析する？」)

    py -X utf8 test/analyze_fuku_payout_band_1031.py

分解: 回収率 = 的中率 × 的中時平均配当 ÷ 100。損益分岐配当 = 100 ÷ 的中率。
  ① 帯ごとに的中率・平均配当・損益分岐配当を並べる(帯が下がるほど配当は上がるが的中率が下がり、積はほぼ一定)
  ② 帯22.5〜30%の配当の内訳(どの配当帯が回収率を作っているか)
  ③ 買う前に分かる条件で帯の中を割る: 1号艇/外艇、市場の1着人気順位(締切前3連単オッズから)、モデル−市場の1着確率差
  ④ 事前宣言後(9/22〜)だけで同じ表
配当は市場(人気)が決めるので、帯(モデルの確率)だけでは配当を制御できない。配当を選ぶなら市場側の条件が要る。
"""
import glob
import json
import sqlite3

import numpy as np
import pandas as pd

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
fuku = {}
for rid, cb, a in conn.execute("SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='複勝' AND race_id>='20260707'"):
    fuku.setdefault(rid, {})[int(cb)] = a or 0
mkt = {}
for rid, cb, o in conn.execute("SELECT race_id, combination, odds FROM odds WHERE bet_type='3連単' AND odds>0 AND fetched_at!='final-backfill'"):
    mkt.setdefault(rid, {}).setdefault(int(cb[0]), 0.0)
    mkt[rid][int(cb[0])] += 1 / o
conn.close()

rows = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rk = r.get("ranked"); rid = r["race_id"]
        if not rk or len(rk) < 6 or rid not in fuku:
            continue
        r1, p1 = int(rk[0][0]), float(rk[0][1])
        pay = fuku[rid].get(r1, 0)
        m = mkt.get(rid)
        mrank = mq = None
        if m and len(m) == 6:
            z = sum(m.values()); mq = m[r1] / z
            mrank = 1 + sum(1 for v in m.values() if v > m[r1])
        rows.append(dict(rid=rid, p1=p1, r1=r1, in1=(r1 == 1), pay=pay, hit=pay > 0, mrank=mrank, mq=mq, after=(rid >= "20260922")))
df = pd.DataFrame(rows)
df["band"] = pd.cut(df["p1"], [0, .15, .225, .30, .40, 1], labels=["<15%", "15〜22.5", "22.5〜30", "30〜40", "40%〜"], right=False)


def summ(d):
    h = d["hit"].mean(); mp = d.loc[d["hit"], "pay"].mean() if d["hit"].any() else np.nan
    return pd.Series(dict(R=len(d), 的中率=h, 的中時平均配当=mp, 損益分岐配当=100 / h if h else np.nan, 回収率=d["pay"].sum() / (len(d) * 100),
                          配当200円未満の割合=(d.loc[d["hit"], "pay"] < 200).mean() if d["hit"].any() else np.nan))


F = {"的中率": "{:.1%}".format, "的中時平均配当": "{:.0f}円".format, "損益分岐配当": "{:.0f}円".format, "回収率": "{:.1%}".format, "配当200円未満の割合": "{:.0%}".format}
print("===== ① 帯ごとの複勝r1: 的中率×配当の積が回収率(全期間 7/7〜) =====")
print(df.groupby("band", observed=True).apply(summ).to_string(formatters=F))
B = df[df["band"] == "22.5〜30"]
print(f"\n===== ② 帯22.5〜30%の配当の内訳({len(B)}R) =====")
bins = pd.cut(B["pay"], [-1, 0, 119, 149, 199, 299, 499, 99999], labels=["外れ", "100〜110", "120〜140", "150〜190", "200〜290", "300〜490", "500〜"])
t = B.groupby(bins, observed=True).agg(R=("pay", "size"), 払戻計=("pay", "sum"))
t["割合"] = t["R"] / len(B); t["回収率への寄与"] = t["払戻計"] / (len(B) * 100)
print(t.to_string(formatters={"割合": "{:.1%}".format, "回収率への寄与": "{:.1%}".format}))
print("\n===== ③ 帯22.5〜30%を買う前に分かる条件で割る =====")
print("[1号艇か外艇か]"); print(B.groupby("in1").apply(summ).rename(index={True: "1号艇", False: "外艇"}).to_string(formatters=F))
Bo = B[B["mrank"].notna()].copy()
print(f"\n[市場の1着人気順位(締切前3連単オッズから・{len(Bo)}R)]")
print(Bo.groupby("mrank").apply(summ).to_string(formatters=F))
Bo["diff"] = pd.cut(Bo["p1"] - Bo["mq"], [-1, -.10, -.05, 0, .05, .10, 1], labels=["市場より-10pt超", "-10〜-5", "-5〜0", "0〜+5", "+5〜+10", "+10pt超"])
print("\n[モデル1着確率−市場1着確率]"); print(Bo.groupby("diff", observed=True).apply(summ).to_string(formatters=F))
print("\n===== ④ 9/22以降(事前宣言後)の帯22.5〜30% =====")
A = B[B["after"]]
print(summ(A).to_frame("全体").T.to_string(formatters=F))
print(A.groupby("in1").apply(summ).rename(index={True: "1号艇", False: "外艇"}).to_string(formatters=F))
