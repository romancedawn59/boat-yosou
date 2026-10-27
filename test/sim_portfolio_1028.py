# -*- coding: utf-8 -*-
"""買い方の組み合わせ5案と金額のシミュレーション(2026-10-28ケンさん依頼
「回収率がマックスになる買い方の組み合わせと金額を5組み合わせ選定してシミュレーション」)

    py -X utf8 test/sim_portfolio_1028.py

部品(これまでの検証で100%前後に出たものだけ。いずれも買う前に見える情報で決まる):
  複勝r1 / 単勝r1 / 3単r1-r2-r3 / 3単{r1,r2}-{r1,r2}-r3(2点) / 3複r1r2r3
  中穴5点 = 15分前オッズで人気20〜40位の目からモデル3連単確率の上位5点
  乖離2点 = 1着確率ボトム2艇を頭から外し、モデル確率÷市場想定確率(確率1%以上)の上位2点
  3複EV1点 = モデル3連複確率×15分前オッズの上位1点
帯 = 本番配信の1位勝率。
表A 本番配信4,748R(2026-07〜10・本番確率×15分前オッズ×払戻がそろうレース)で全5案を採点。
表B 全記録台帳52,693R(2025-10〜2026-09・月次WF)で、オッズを使わない案だけ1年分を採点。
注意: 部品も帯も「これまでの結果を見て」選んでいる(後出し)。表Aの数字は選んだ同じデータの上の数字。
"""
import glob
import json
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
from predictors import trifecta_probs

LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
BUDGET_RACE, BUDGET_DAY = 1500, 8000

print("データ準備中...", flush=True)
conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
odds3, oddsT = defaultdict(dict), defaultdict(dict)
for rid, bt, cb, o in conn.execute("SELECT race_id, bet_type, combination, odds FROM odds "
                                   "WHERE odds IS NOT NULL AND odds > 0 AND fetched_at != 'final-backfill'"):
    (odds3 if bt == "3連単" else oddsT)[rid][cb] = o
pay = defaultdict(dict)
for rid, bt, cb, a in conn.execute("SELECT race_id, bet_type, combination, amount_yen FROM payouts"):
    pay[rid][(bt, cb)] = a or 0
conn.close()

races = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid = r["race_id"]
        if not r.get("ranked") or len(r["ranked"]) < 6 or len(odds3.get(rid, {})) < 110 or rid not in pay:
            continue
        probs = {int(l): float(p) for l, p in r["ranked"]}
        tri = trifecta_probs(probs)
        z = sum(tri.values())
        tri = {"-".join(map(str, k)): v / z for k, v in tri.items()}
        o3 = odds3[rid]
        q = {cb: 1 / o for cb, o in o3.items()}
        zq = sum(q.values())
        q = {cb: v / zq for cb, v in q.items()}
        rank_mkt = {cb: i + 1 for i, cb in enumerate(sorted(o3, key=o3.get))}
        trio = defaultdict(float)
        for cb, v in tri.items():
            trio["=".join(sorted(cb.split("-")))] += v
        ranked = [int(l) for l, _ in r["ranked"]]
        races.append(dict(rid=rid, date=rid[:8], month=rid[:6], p1=float(r["ranked"][0][1]), r=ranked, tri=tri, o3=o3,
                          q=q, rank_mkt=rank_mkt, trio=dict(trio), oT=oddsT.get(rid, {}), bottom2=set(ranked[-2:]),
                          pay=pay[rid]))
print(f"表Aの対象 {len(races):,}R", flush=True)


# ---------- 部品: (券種, 買い目) のリストを返す ----------
def fuku_r1(x): return [("複勝", str(x["r"][0]))]
def tan_r1(x): return [("単勝", str(x["r"][0]))]
def tri123(x): r = x["r"]; return [("3連単", f"{r[0]}-{r[1]}-{r[2]}")]
def tri_box12_3(x): r = x["r"]; return [("3連単", f"{r[0]}-{r[1]}-{r[2]}"), ("3連単", f"{r[1]}-{r[0]}-{r[2]}")]
def trio123(x): r = x["r"]; return [("3連複", "=".join(map(str, sorted(r[:3]))))]
def mid5(x):
    cand = [c for c in x["tri"] if 20 <= x["rank_mkt"].get(c, 999) <= 40]
    return [("3連単", c) for c in sorted(cand, key=x["tri"].get, reverse=True)[:5]]
def kairi2(x):
    rt = {c: x["tri"][c] / x["q"][c] for c in x["tri"]
          if c in x["q"] and x["tri"][c] >= .01 and int(c.split("-")[0]) not in x["bottom2"]}
    return [("3連単", c) for c in sorted(rt, key=rt.get, reverse=True)[:2]]
def trio_ev1(x):
    ev = {c: x["trio"][c] * x["oT"][c] for c in x["trio"] if c in x["oT"]}
    return [("3連複", c) for c in sorted(ev, key=ev.get, reverse=True)[:1]] if ev else []


def band(x, lo, hi): return lo <= x["p1"] < hi


# ---------- 5案: 帯の条件 → [(部品, 1点あたりの金額)] ----------
PLANS = {
    "P0 現行紙上: 帯22.5〜30% 3単123 500円": [
        (lambda x: band(x, .225, .30), [(tri123, 500)])],
    "P1 本命複合: 帯22.5〜30% 複勝r1 700円+3単123 300円": [
        (lambda x: band(x, .225, .30), [(fuku_r1, 700), (tri123, 300)])],
    "P2 中穴複合: 帯22.5〜30% 中穴5点×100円+複勝r1 500円": [
        (lambda x: band(x, .225, .30), [(mid5, 100), (fuku_r1, 500)])],
    "P3 帯で使い分け: 22.5〜30%は中穴5点×100+複勝r1 500 / 30〜50%は乖離2点×200": [
        (lambda x: band(x, .225, .30), [(mid5, 100), (fuku_r1, 500)]),
        (lambda x: band(x, .30, .50), [(kairi2, 200)])],
    "P4 乖離だけ全帯: 乖離2点×500円": [
        (lambda x: True, [(kairi2, 500)])],
    "P5 寄せ集め(最も後出し): 22.5〜25%は複勝r1 1000+3単123 500 / 25〜30%は中穴5点×100 / 30〜50%は乖離2点×100": [
        (lambda x: band(x, .225, .25), [(fuku_r1, 1000), (tri123, 500)]),
        (lambda x: band(x, .25, .30), [(mid5, 100)]),
        (lambda x: band(x, .30, .50), [(kairi2, 100)])],
}


def run_plan(plan, rs):
    rows = []
    for x in rs:
        cost = ret = 0; hit = False
        for cond, parts in plan:
            if not cond(x):
                continue
            for fn, yen in parts:
                for bt, cb in fn(x):
                    cost += yen
                    a = x["pay"].get((bt, cb), 0)
                    if a:
                        ret += a * yen / 100; hit = True
        if cost:
            rows.append((x["date"], x["month"], cost, ret, hit))
    return pd.DataFrame(rows, columns=["date", "month", "cost", "ret", "hit"])


def summarize(d, n_all):
    inv = d["cost"].sum(); r = d["ret"].sum()
    daily = d.groupby("date")[["cost", "ret"]].sum()
    cum = (d["ret"] - d["cost"]).cumsum()
    dd = (cum.cummax() - cum).max()
    mon = d.groupby("month")[["cost", "ret"]].sum()
    half = d["date"] < sorted(d["date"])[len(d) // 2]
    return dict(買うR=len(d), 全体比=len(d) / n_all, 的中R=int(d["hit"].sum()), 投資=inv, 払戻=r, 回収率=r / inv,
                最大1本除く=(r - d["ret"].max()) / inv, 黒字月=f"{int((mon['ret'] > mon['cost']).sum())}/{len(mon)}",
                前半=d.loc[half, "ret"].sum() / d.loc[half, "cost"].sum(), 後半=d.loc[~half, "ret"].sum() / d.loc[~half, "cost"].sum(),
                日数=len(daily), 日平均投資=daily["cost"].mean(), 日最大投資=daily["cost"].max(),
                日1500超R=int((d["cost"] > BUDGET_RACE).sum()), 最大落ち込み=dd, 月別="  ".join(f"{m[4:]}月{v:.0%}" for m, v in (mon["ret"] / mon["cost"]).items()))


pd.set_option("display.width", 300, "display.max_colwidth", 200, "display.unicode.east_asian_width", True)
pct = "{:.1%}".format
yen = "{:,.0f}".format
fmt = {"全体比": pct, "回収率": pct, "最大1本除く": pct, "前半": pct, "後半": pct, "投資": yen, "払戻": yen, "日平均投資": yen, "日最大投資": yen, "最大落ち込み": yen}

print("\n===== 表A 本番配信 2026-07〜10(4,748R): 5案の採点 =====")
A = {}
for name, plan in PLANS.items():
    d = run_plan(plan, races)
    A[name] = summarize(d, len(races))
dfA = pd.DataFrame(A).T
print(dfA.drop(columns=["月別"]).to_string(formatters=fmt))
print("\n月別の回収率:")
for name, s in A.items():
    print(f"  {name[:12]:<14} {s['月別']}")

print("\n===== 表A 部品ごとの単独成績(同じ4,748R・1点100円) =====")
PARTS = {"複勝r1": fuku_r1, "単勝r1": tan_r1, "3単r1-r2-r3": tri123, "3単{r1,r2}box-r3": tri_box12_3, "3複r1r2r3": trio123,
         "中穴5点": mid5, "乖離2点": kairi2, "3複EV1点": trio_ev1}
BANDS = {"全帯": (0, 1.01), "22.5〜25%": (.225, .25), "25〜30%": (.25, .30), "22.5〜30%": (.225, .30), "30〜50%": (.30, .50), "50%〜": (.50, 1.01)}
rows = []
for pn, fn in PARTS.items():
    for bn, (lo, hi) in BANDS.items():
        d = run_plan([(lambda x, lo=lo, hi=hi: band(x, lo, hi), [(fn, 100)])], races)
        if len(d) == 0:
            continue
        rows.append(dict(部品=pn, 帯=bn, R=len(d), 的中=int(d["hit"].sum()), 回収率=d["ret"].sum() / d["cost"].sum(),
                         最大1本除く=(d["ret"].sum() - d["ret"].max()) / d["cost"].sum()))
print(pd.DataFrame(rows).pivot(index="部品", columns="帯", values="回収率").reindex(list(PARTS))[list(BANDS)].to_string(float_format=pct))
print("(的中本数)")
print(pd.DataFrame(rows).pivot(index="部品", columns="帯", values="的中").reindex(list(PARTS))[list(BANDS)].to_string())

# ---------- 表B: 全記録台帳(1年)でオッズを使わない案 ----------
led = pd.read_csv(LEDGER)
led["r"] = led["pred"].apply(lambda s: [int(x) for x in s.split("-")])
led_races = [dict(rid=r.race_id, date=r.race_id[:8], month=r.race_id[:6], p1=r.p1, r=r.r, pay=pay.get(r.race_id, {}))
             for r in led.itertuples()]
led_races = [x for x in led_races if x["pay"]]
print(f"\n===== 表B 全記録台帳 2025-10〜2026-09({len(led_races):,}R・月次WF): オッズを使わない案 =====")
PLANS_B = {k: v for k, v in PLANS.items() if k.startswith(("P0", "P1"))}
PLANS_B["P1' 複勝だけ: 帯22.5〜30% 複勝r1 1000円"] = [(lambda x: band(x, .225, .30), [(fuku_r1, 1000)])]
PLANS_B["P5' 寄せ集めの複勝部分: 22.5〜25% 複勝r1 1000+3単123 500"] = [(lambda x: band(x, .225, .25), [(fuku_r1, 1000), (tri123, 500)])]
B = {name: summarize(run_plan(plan, led_races), len(led_races)) for name, plan in PLANS_B.items()}
dfB = pd.DataFrame(B).T
print(dfB.drop(columns=["月別"]).to_string(formatters=fmt))
print("\n月別の回収率:")
for name, s in B.items():
    print(f"  {name[:12]:<14} {s['月別']}")
