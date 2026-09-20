# -*- coding: utf-8 -*-
"""若松・尼崎プランの監査(2026-09-21ケンさん指示「バイアスを捨てて問題・設計ミスをチェック」)

    py -X utf8 test/audit_wakamatsu_amagasaki_plan_0921.py

① 偶然でも同じ見た目になるか: 22.5〜30%帯の場ラベルをシャッフルして、
   「抽出条件を通る場が2場以上」「最良の場の回収率が265%以上」になる頻度を数える
② 後出しなしの場選び: 前の期間までの成績で抽出条件を通った場だけを次の期間に買う
③ 最大1本/2本/3本を除いた回収率(プラン全体)
④ 検証の予想と本番配信の予想は同じものか: 2026-07-21〜08-31の若松・尼崎で、
   帯の一致と予想1-2-3位の一致を数える
"""
import glob
import json
import random
import sqlite3
from collections import defaultdict

import pandas as pd

WF_LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
PERIODS = [("2025-10", "2026-01"), ("2026-02", "2026-05"), ("2026-06", "2026-09")]
VEN = {13: "尼崎", 20: "若松"}

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True,
                       timeout=120)
pay = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id=p.race_id WHERE r.date>='2025-10-01' "
        "AND p.bet_type IN ('3連単','単勝')"):
    pay[rid][(bt, comb)] = amt or 0
conn.close()

L = pd.read_csv(WF_LEDGER, encoding="utf-8-sig", dtype={"refund": str})
B = L[(L["p1"] >= 0.225) & (L["p1"] < 0.30)].copy()
tan, t132, win = [], [], []
for rid, pred, rf in zip(B["race_id"], B["pred"], B["refund"].fillna("")):
    l = [int(x) for x in pred.split("-")]
    refund = {int(x) for x in rf.split("-") if x}
    p = pay.get(rid, {})
    bad = bool(set(l[:3]) & refund)
    tan.append(100 if bad else p.get(("3連単", f"{l[0]}-{l[1]}-{l[2]}"), 0))
    t132.append(100 if bad else p.get(("3連単", f"{l[0]}-{l[2]}-{l[1]}"), 0))
    win.append(100 if l[0] in refund else p.get(("単勝", str(l[0])), 0))
B["tan"], B["t132"], B["win"] = tan, t132, win
B["period"] = [next(i for i, (a, b) in enumerate(PERIODS) if a <= m <= b) for m in B["month"]]


def passing(df):
    """抽出条件を通る場の集合と、最良の場の回収率"""
    ok, best = set(), 0
    for v, g in df.groupby("venue"):
        roi = g["tan"].mean() / 100
        best = max(best, roi)
        per = g.groupby("period")["tan"].mean() / 100
        exb = (g["tan"].sum() - g["tan"].max()) / (len(g) * 100)
        if roi > 1 and (per > 1).sum() >= 2 and exb > 1:
            ok.add(v)
    return ok, best


real_ok, real_best = passing(B)
print(f"① 実データ: 通過{len(real_ok)}場 / 最良の場の回収率{real_best:.0%}")
random.seed(7)
N = 2000
ge_n = ge_best = ge_33 = 0
venues = B["venue"].tolist()
real_33 = 0
for v, g in B.groupby("venue"):
    real_33 += int(((g.groupby("period")["tan"].mean() / 100) > 1).sum() == 3)
for _ in range(N):
    sh = B.copy()
    # 期間の構成を保つため、期間の中で場ラベルだけを入れ替える
    sh["venue"] = sh.groupby("period")["venue"].transform(
        lambda s: random.sample(s.tolist(), len(s)))
    ok, best = passing(sh)
    ge_n += len(ok) >= len(real_ok)
    ge_best += best >= real_best
    n33 = sum(int(((g.groupby("period")["tan"].mean() / 100) > 1).sum() == 3)
              for _v, g in sh.groupby("venue"))
    ge_33 += n33 >= real_33
print(f"   場に差が無いと仮定して場ラベルを{N}回シャッフル:")
print(f"   通過が{len(real_ok)}場以上になる確率 {ge_n / N:.1%} / 最良の場が{real_best:.0%}以上になる確率 {ge_best / N:.1%}"
      f" / 3期間とも黒字の場が{real_33}場以上になる確率 {ge_33 / N:.1%}")

print("\n② 後出しなしの場選び(前の期間までで抽出条件のうち『回収率100%超・最大1本除外後も100%超』を満たした場を次の期間に買う・3連単1点)")
for k in (1, 2):
    hist = B[B["period"] < k]
    pick = set()
    for v, g in hist.groupby("venue"):
        if g["tan"].mean() > 100 and (g["tan"].sum() - g["tan"].max()) / len(g) > 100:
            pick.add(v)
    te = B[(B["period"] == k) & B["venue"].isin(pick)]
    al = B[B["period"] == k]
    print(f"   期間{k + 1}: 選ばれた{len(pick)}場 → {len(te)}R 回収率{te['tan'].mean() / 100:.1%}"
          f"(同じ期間の全24場 {al['tan'].mean() / 100:.1%}) 若松入り={20 in pick} 尼崎入り={13 in pick}")

print("\n③ プラン全体(若松: 3連単500+単勝500 / 尼崎: 3連単700+r1-r3-r2 300)")
P = B[B["venue"].isin(VEN)].copy()
P["gain"] = [(t * 5 + w * 5) if v == 20 else (t * 7 + x * 3)
             for v, t, w, x in zip(P["venue"], P["tan"], P["win"], P["t132"])]
st = len(P) * 1000
g = sorted(P["gain"], reverse=True)
print(f"   {len(P)}R 回収率{sum(g) / st:.1%} 損益{sum(g) - st:+,}円 / 最大1本除く{sum(g[1:]) / st:.1%}"
      f" / 2本除く{sum(g[2:]) / st:.1%} / 3本除く{sum(g[3:]) / st:.1%} / 5本除く{sum(g[5:]) / st:.1%}")
print(f"   上位5本の払戻: {g[:5]} → 利益{sum(g) - st:,}円のうち上位5本が{sum(g[:5]):,}円")
for v, vn in VEN.items():
    s = P[P["venue"] == v]
    print(f"   {vn}: 月別損益 " + " ".join(
        f"{m[2:]}:{int(x):+,}" for m, x in (s.groupby("month")["gain"].sum()
                                            - s.groupby("month").size() * 1000).items()))

print("\n④ 検証の予想 vs 本番配信の予想(2026-07-21〜08-31・若松と尼崎)")
prod = {}
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-0[78]-*.json")):
    pk = json.load(open(path, encoding="utf-8"))
    for r in pk["races"]:
        if r.get("ranked") and len(r["ranked"]) >= 3 and r["venue_code"] in VEN:
            prod[r["race_id"]] = (r["ranked"][0][1], [x[0] for x in r["ranked"][:3]])
W = L[L["venue"].isin(VEN) & L["race_id"].isin(prod)]
both = wf_only = prod_only = same3 = same1 = 0
for rid, p1, pred in zip(W["race_id"], W["p1"], W["pred"]):
    pp1, ptop = prod[rid]
    a, b = 0.225 <= p1 < 0.30, 0.225 <= pp1 < 0.30
    both += a and b
    wf_only += a and not b
    prod_only += b and not a
    if a and b:
        top = [int(x) for x in pred.split("-")[:3]]
        same3 += top == ptop
        same1 += top[0] == ptop[0]
print(f"   比較できたレース {len(W)}R / 帯に入る: 両方{both}R・検証だけ{wf_only}R・本番だけ{prod_only}R")
if both:
    print(f"   両方が帯に入れたレースのうち 予想1位が同じ {same1}/{both}・予想1-2-3位が同じ {same3}/{both}")
