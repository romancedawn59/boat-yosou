# -*- coding: utf-8 -*-
"""r1〜r4の3連複(4艇から3艇=4通り)のうち、r4を必ず含む組の帯と点数の調査(2026-11-04ケンさん指示)

    py -X utf8 test/sim_trio_r1234_r4_1104.py

設計(結果を見る前に固定):
  発見用 = 1年台帳(月次walk-forward)の 2025-10〜2026-06(9か月)
  確認用 = 本番配信picks の 2026-07-07〜11-03(発見用と期間が重ならない)
  帯     = 1位勝率 <15 / 15〜20 / 20〜22.5 / 22.5〜25 / 25〜27.5 / 27.5〜30 / 30〜35 / 35〜40 / 40〜50 / 50%〜
  点数   = r4を含む3連複(100円/点): 1点 124・134・234 / 2点 124+134・124+234・134+234 / 3点 124+134+234
           / 4点 r1〜r4ボックス(123も含む) / 比較用 1点 123(r4なし)
  手順   = 発見用で全マスを採点 → 回収率上位5マス(的中30本以上)を確認用で採点
  合格   = 確認用で 回収率>100%・的中30本以上・最大1本除いても100%超 かつ 発見用でも100%超
  補足   = 組に欠場艇が入っていれば返還(100円戻り)
"""
import glob
import json
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
BANDS = [0, .15, .20, .225, .25, .275, .30, .35, .40, .50, 1.01]
BLAB = ["<15%", "15〜20", "20〜22.5", "22.5〜25", "25〜27.5", "27.5〜30", "30〜35", "35〜40", "40〜50", "50%〜"]
SETS = {"1点 124": [(1, 2, 4)], "1点 134": [(1, 3, 4)], "1点 234": [(2, 3, 4)],
        "2点 124+134": [(1, 2, 4), (1, 3, 4)], "2点 124+234": [(1, 2, 4), (2, 3, 4)], "2点 134+234": [(1, 3, 4), (2, 3, 4)],
        "3点 r4を含む全部": [(1, 2, 4), (1, 3, 4), (2, 3, 4)], "4点 r1〜r4ボックス": [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)],
        "比較 1点 123": [(1, 2, 3)]}

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
fuku = defaultdict(dict)
for rid, cb, a in conn.execute("SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連複'"):
    fuku[rid][cb] = a or 0
started = defaultdict(set)
for rid, lane in conn.execute("SELECT race_id, lane FROM results WHERE arrival_order IS NOT NULL OR st_time IS NOT NULL"):
    started[rid].add(lane)
conn.close()


def race_rows(source):
    out = []
    if source == "発見用":
        led = pd.read_csv(r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv")
        led = led[led["month"] <= "2026-06"]
        for rid, p1, pred in zip(led["race_id"], led["p1"], led["pred"]):
            out.append((rid, float(p1), [int(x) for x in pred.split("-")]))
    else:
        for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
            if path[-15:-5] < "2026-07-07" or path[-15:-5] > "2026-11-03":
                continue
            for r in json.load(open(path, encoding="utf-8"))["races"]:
                if r.get("ranked") and len(r["ranked"]) == 6:
                    out.append((r["race_id"], float(r["ranked"][0][1]), [int(l) for l, _ in r["ranked"]]))
    recs = []
    for rid, p1, order in out:
        if rid not in fuku or len(order) < 6:
            continue
        band = BLAB[int(np.searchsorted(BANDS, p1, side="right")) - 1]
        rec = dict(rid=rid, month=rid[:6], band=band, r4=order[3])
        for name, combos in SETS.items():
            ret = 0
            for cmb in combos:
                lanes = [order[k - 1] for k in cmb]
                if started.get(rid) and any(l not in started[rid] for l in lanes):
                    ret += 100            # 欠場を含む組は返還
                else:
                    ret += fuku[rid].get("=".join(map(str, sorted(lanes))), 0)
            rec[name] = ret
        rec["r4_hit"] = int(rec["3点 r4を含む全部"] > 0)
        rec["r4_pay"] = rec["3点 r4を含む全部"]
        recs.append(rec)
    return pd.DataFrame(recs)


A = race_rows("発見用"); B = race_rows("確認用")
print(f"発見用 {len(A):,}R(2025-10〜2026-06・1年台帳) / 確認用 {len(B):,}R(2026-07-07〜11-03・本番配信)")


def score(d, name):
    k = len(SETS[name]); v = d[name].values; inv = len(d) * k * 100
    if len(d) == 0:
        return dict(R=0, 的中=0, 回収率=np.nan, 最大1本除く=np.nan)
    return dict(R=len(d), 的中=int((v > 0).sum()), 回収率=v.sum() / inv, 最大1本除く=(v.sum() - v.max()) / inv)


print("\n===== 基本: r4を含む3連複3点のどれかが当たる率と配当(帯別)。回収率は投資300円あたり =====")
for lab, d in (("発見用", A), ("確認用", B)):
    t = d.groupby("band").apply(lambda g: pd.Series({"R": len(g), "r4絡みの的中率": g["r4_hit"].mean(),
                                                      "的中時の平均配当": g.loc[g["r4_hit"] == 1, "r4_pay"].mean(), "回収率": g["r4_pay"].sum() / (len(g) * 300)}))
    t = t.reindex([b for b in BLAB if b in t.index])
    print(f"[{lab}]"); print(t.to_string(formatters={"R": "{:,.0f}".format, "r4絡みの的中率": "{:.1%}".format, "的中時の平均配当": "{:,.0f}円".format, "回収率": "{:.1%}".format}))

print("\n===== 発見用: 帯×点数の回収率(100マス) =====")
tab = pd.DataFrame({name: {b: score(A[A["band"] == b], name)["回収率"] for b in BLAB} for name in SETS})
print(tab.to_string(float_format=lambda x: f"{x:.0%}"))
print("\n===== 確認用(参考・全マス): 帯×点数の回収率 =====")
tabB = pd.DataFrame({name: {b: score(B[B["band"] == b], name)["回収率"] for b in BLAB} for name in SETS})
print(tabB.to_string(float_format=lambda x: f"{x:.0%}"))

print("\n===== 発見用の上位5マス(的中30本以上)を確認用で採点 =====")
cells = []
for b in BLAB:
    for name in SETS:
        s = score(A[A["band"] == b], name)
        if s["的中"] >= 30:
            cells.append((s["回収率"], b, name, s))
cells.sort(reverse=True)
rows = []
for ra, b, name, sa in cells[:5]:
    sb = score(B[B["band"] == b], name)
    ok = sa["回収率"] > 1 and sb["回収率"] > 1 and sb["的中"] >= 30 and sb["最大1本除く"] > 1
    rows.append(dict(帯=b, 点数=name, 発見R=sa["R"], 発見回収率=sa["回収率"], 発見的中=sa["的中"], 発見最大1本除く=sa["最大1本除く"],
                     確認R=sb["R"], 確認回収率=sb["回収率"], 確認的中=sb["的中"], 確認最大1本除く=sb["最大1本除く"], 合格=("○" if ok else "×")))
print(pd.DataFrame(rows).to_string(index=False, formatters={c: "{:.1%}".format for c in ("発見回収率", "発見最大1本除く", "確認回収率", "確認最大1本除く")}))

print("\n===== 帯をまとめた場合(全帯)の点数別 =====")
for lab, d in (("発見用", A), ("確認用", B)):
    print(f"[{lab}] " + " / ".join(f"{name} {score(d, name)['回収率']:.1%}" for name in SETS))

print("\n===== 確認用の月別(発見用の1位マス) =====")
if rows:
    b, name = rows[0]["帯"], rows[0]["点数"]
    d = B[B["band"] == b]
    print(f"{b}×{name}: " + " / ".join(f"{m[4:]}月 {score(g, name)['回収率']:.0%}({score(g, name)['R']}R)" for m, g in d.groupby("month")))
