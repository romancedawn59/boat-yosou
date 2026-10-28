# -*- coding: utf-8 -*-
"""展望(レースの型)×券種の組み合わせ検証(2026-10-29ケンさん「レースの展望と適切な買い方は組み合わせられないの？」)

    py -X utf8 test/sim_outlook_x_ticket_1029.py

やること(基準は結果を見る前に固定):
  レースの型 = 朝6:30の配信にある情報だけで決める
    軸1 1位勝率帯: <15 / 15〜22.5 / 22.5〜30 / 30〜40 / 40%〜
    軸2 2着の絞りやすさ: 2位と3位の1着確率の差 gap23 >= 0.08 なら「2着絞れる」、未満なら「2着流動」
    (補助) 1位が1号艇かどうか
  券種メニュー = 1レース1,000円で15通り(金額は点数で等分・ケン式は配信どおりの配分)
  手順 = 前半(7/7〜9/14)で型ごとに回収率最大の券種を決める(的中20本以上の券種から)→後半(9/15〜10/28)でその対応表を採点
  合格 = 後半で「型→券種」の回収率 > 100% かつ 後半の一律複勝r1・後半の一律最良券種より高い。後半の型ごとの最良(後出し)は参考値
"""
import glob
import json
import sqlite3
from itertools import permutations

import numpy as np
import pandas as pd

SPLIT = "20260915"
conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
pay = {}
for r, b, c, a in conn.execute("SELECT race_id, bet_type, combination, amount_yen FROM payouts WHERE race_id >= '20260707'"):
    pay.setdefault(r, {})[(b, c)] = a or 0
conn.close()


def key2(a, b):
    return "=".join(map(str, sorted((a, b))))


def key3(a, b, c):
    return "=".join(map(str, sorted((a, b, c))))


def ret(p, bets):
    """bets=[(券種, 組番, 金額)] → 払戻合計"""
    return sum(p.get((t, c), 0) * y / 100 for t, c, y in bets)


def menu(r):
    rk = r["ranked"]; a, b, c = rk[0][0], rk[1][0], rk[2][0]
    others = [x for x in range(1, 7) if x not in (a, b, c)]
    m = {
        "単勝r1": [("単勝", str(a), 1000)],
        "複勝r1": [("複勝", str(a), 1000)],
        "2連単r1-r2": [("2連単", f"{a}-{b}", 1000)],
        "2連複r1r2": [("2連複", key2(a, b), 1000)],
        "拡連複r1r2": [("拡連複", key2(a, b), 1000)],
        "3連複r1r2r3": [("3連複", key3(a, b, c), 1000)],
        "3単r1-r2-r3": [("3連単", f"{a}-{b}-{c}", 1000)],
        "3単r1-r2r3(2点)": [("3連単", f"{a}-{b}-{c}", 500), ("3連単", f"{a}-{c}-{b}", 500)],
        "3単box6": [("3連単", "-".join(map(str, p)), 1000 / 6) for p in permutations((a, b, c))],
        "3単r1-r2r3-全(8点)": [("3連単", f"{a}-{b}-{x}", 125) for x in range(1, 7) if x not in (a, b)]
                            + [("3連単", f"{a}-{c}-{x}", 125) for x in range(1, 7) if x not in (a, c)],
        "3単r1流し20点": [("3連単", f"{a}-{x}-{y}", 50) for x in range(1, 7) for y in range(1, 7) if len({a, x, y}) == 3],
        "3複r1流し10点": [("3連複", key3(a, x, y), 100) for x in range(1, 7) for y in range(x + 1, 7) if a not in (x, y)],
        "3複r1r2-全(4点)": [("3連複", key3(a, b, x), 250) for x in range(1, 7) if x not in (a, b)],
    }
    bl = r.get("b") or []
    if len(bl) >= 10:
        m["モデル上位5点"] = [(t, cmb, 200) for t, cmb, _ in bl[:5]]
        m["モデル上位10点"] = [(t, cmb, 100) for t, cmb, _ in bl[:10]]
    ken = r.get("ken") or []
    tot = sum(k[2] for k in ken)
    if ken and tot > 0:
        m["ケン式(配信)"] = [(t, cmb, y * 1000 / tot) for t, cmb, y, _ in ken]
    return m


def band(p1):
    return "<15%" if p1 < .15 else "15〜22.5%" if p1 < .225 else "22.5〜30%" if p1 < .30 else "30〜40%" if p1 < .40 else "40%〜"


rows = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rk = r.get("ranked")
        if not rk or len(rk) < 6 or r["race_id"] not in pay:
            continue
        p = pay[r["race_id"]]
        p1, p2, p3 = rk[0][1], rk[1][1], rk[2][1]
        row = dict(race_id=r["race_id"], p1=p1, band=band(p1), nikaku="2着絞れる" if p2 - p3 >= .08 else "2着流動",
                   in1=("1号艇" if rk[0][0] == 1 else "外艇"), period=("後半" if r["race_id"] >= SPLIT else "前半"))
        for name, bets in menu(r).items():
            row[name] = ret(p, bets)
        rows.append(row)
df = pd.DataFrame(rows)
TICKETS = [c for c in df.columns if c not in ("race_id", "p1", "band", "nikaku", "in1", "period")]
df["type"] = df["band"] + "×" + df["nikaku"]
A, B = df[df["period"] == "前半"], df[df["period"] == "後半"]
print(f"対象 {len(df):,}R(前半{len(A):,}R 7/7〜9/14・後半{len(B):,}R 9/15〜10/28)・券種{len(TICKETS)}通り・1R1,000円")


def roi(d, t):
    s = d[t]
    return s.sum() / (s.notna().sum() * 1000) if s.notna().sum() else np.nan


def hits(d, t):
    return int((d[t] > 0).sum())


# ---- 一律(型を見ない)の基準 ----
print("\n===== 一律に同じ券種を買った場合(型を見ない) =====")
base = pd.DataFrame({"前半回収率": [roi(A, t) for t in TICKETS], "前半的中": [hits(A, t) for t in TICKETS],
                     "後半回収率": [roi(B, t) for t in TICKETS], "後半的中": [hits(B, t) for t in TICKETS]}, index=TICKETS)
print(base.to_string(formatters={"前半回収率": "{:.1%}".format, "後半回収率": "{:.1%}".format}))

# ---- 型×券種の表(前半で選ぶ) ----
print("\n===== 型(1位勝率帯×2着の絞り)ごとに前半で最良の券種を選び、後半で採点 =====")
out = []
for ty, dA in A.groupby("type"):
    dB = B[B["type"] == ty]
    cands = [(roi(dA, t), t) for t in TICKETS if hits(dA, t) >= 20]
    if not cands or len(dB) == 0:
        continue
    bestA_roi, bestA = max(cands)
    oracleB_roi, oracleB = max((roi(dB, t), t) for t in TICKETS if hits(dB, t) >= 5) if any(hits(dB, t) >= 5 for t in TICKETS) else (np.nan, "-")
    out.append(dict(型=ty, 前半R=len(dA), 前半最良=bestA, 前半回収率=bestA_roi, 後半R=len(dB), 後半回収率=roi(dB, bestA), 後半的中=hits(dB, bestA),
                    後半複勝r1=roi(dB, "複勝r1"), 後半最良_後出し=oracleB, 後半最良回収率_後出し=oracleB_roi))
tab = pd.DataFrame(out)
print(tab.to_string(index=False, formatters={k: "{:.1%}".format for k in ("前半回収率", "後半回収率", "後半複勝r1", "後半最良回収率_後出し")}))

# ---- 対応表どおりに後半を買った合計 ----
mapping = dict(zip(tab["型"], tab["前半最良"]))
Bm = B[B["type"].isin(mapping)].copy()
Bm["ret"] = [row[mapping[row["type"]]] for _, row in Bm.iterrows()]
top = Bm["ret"].sort_values(ascending=False).values
print(f"\n後半を『型→券種』の対応表どおりに買う: {len(Bm):,}R 投資{len(Bm) * 1000:,}円 払戻{Bm['ret'].sum():,.0f}円 回収率{Bm['ret'].sum() / (len(Bm) * 1000):.1%} "
      f"的中{(Bm['ret'] > 0).sum()}本 最大1/3本除く {(Bm['ret'].sum() - top[:1].sum()) / (len(Bm) * 1000):.1%}/{(Bm['ret'].sum() - top[:3].sum()) / (len(Bm) * 1000):.1%}")
print(f"同じ後半を一律複勝r1: {roi(Bm, '複勝r1'):.1%} / 一律の後半最良({base['後半回収率'].idxmax()}): {base['後半回収率'].max():.1%}")
Bm["month"] = Bm["race_id"].str[:6]
print(Bm.groupby("month").apply(lambda d: f"{len(d)}R {d['ret'].sum() / (len(d) * 1000):.1%}").to_string())

# ---- 帯22.5〜30%の中だけ、型を細かく(2着の絞り×1号艇) ----
print("\n===== 帯22.5〜30%の中で型を分けたとき(前半で選び後半で採点) =====")
for ax in ("nikaku", "in1"):
    for v, dA in A[A["band"] == "22.5〜30%"].groupby(ax):
        dB = B[(B["band"] == "22.5〜30%") & (B[ax] == v)]
        cands = [(roi(dA, t), t) for t in TICKETS if hits(dA, t) >= 10]
        if not cands or not len(dB):
            continue
        ra, t = max(cands)
        print(f"{ax}={v}: 前半{len(dA)}R 最良{t} {ra:.1%} → 後半{len(dB)}R {roi(dB, t):.1%}(的中{hits(dB, t)}) | 後半複勝r1 {roi(dB, '複勝r1'):.1%} 後半3複r1r2r3 {roi(dB, '3連複r1r2r3'):.1%}")

# ---- 参考: 券種間の相対(市場の値付けが型で歪んでいるか) 後半・型別の全券種 ----
print("\n===== 参考: 後半の型別×全券種の回収率(後出し・眺めるだけ) =====")
ref = B.groupby("type").apply(lambda d: pd.Series({t: roi(d, t) for t in TICKETS} | {"R": len(d)}))
print(ref.to_string(float_format=lambda x: f"{x:.0%}" if x < 10 else f"{x:.0f}"))
