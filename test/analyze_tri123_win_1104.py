# -*- coding: utf-8 -*-
"""10/29〜11/3の本命(帯22.5〜30%)で 3連単r1-r2-r3 500円が181.5%になった要因の精査(2026-11-04ケンさん指示)

    py -X utf8 test/analyze_tri123_win_1104.py

① 期待と実際: モデルがr1-r2-r3につけた確率の合計(期待的中数)と実際の的中数。期待どおりなら何本・何%か
② 長い目で見た同じ買い方: 本番配信7/7〜11/3の帯22.5〜30%で 3単r1-r2-r3 の回収率・月別・
   連続48レースの窓で181.5%以上になる頻度(偶然でどれくらい起きるか)
③ 的中4本の中身: オッズ(締切前があれば)・モデルの確率・市場の確率・期待値
④ 11/1再学習の前後で帯の中身(1号艇がr1の割合・r1-r2-r3の確率)が変わったか
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

V = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
     13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}
conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
tri_pay = defaultdict(dict)
for rid, cb, a in conn.execute("SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連単' AND race_id>='20260707'"):
    tri_pay[rid][cb] = a or 0
odds = defaultdict(dict)
for rid, cb, o in conn.execute("SELECT race_id, combination, odds FROM odds WHERE bet_type='3連単' AND odds>0 AND fetched_at!='final-backfill' AND race_id>='20260707'"):
    odds[rid][cb] = o
conn.close()

rows = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rk = r.get("ranked"); rid = r["race_id"]
        if not rk or len(rk) < 6 or rid not in tri_pay or not (.225 <= rk[0][1] < .30):
            continue
        probs = {int(l): float(p) for l, p in rk}
        tri = trifecta_probs(probs); z = sum(tri.values())
        a, b, c = int(rk[0][0]), int(rk[1][0]), int(rk[2][0]); key = f"{a}-{b}-{c}"
        o = odds.get(rid, {})
        q = None
        if len(o) >= 108:
            s = sum(1 / v for v in o.values()); q = (1 / o[key]) / s if key in o else None
        rows.append(dict(rid=rid, date=rid[:8], month=rid[:6], venue=V[r["venue_code"]], rno=r["race_no"], p1=rk[0][1], r1=a, key=key,
                         p_model=tri[(a, b, c)] / z, pay=tri_pay[rid].get(key, 0), shobusho=r.get("shobusho"),
                         odds_pre=o.get(key), q_mkt=q))
D = pd.DataFrame(rows).sort_values("rid").reset_index(drop=True)
D["hit"] = (D["pay"] > 0).astype(int)
W = D[(D["date"] >= "20261029") & (D["date"] <= "20261103")]

print(f"===== ① 期待と実際(10/29〜11/3・{len(W)}R) =====")
exp_hits = W["p_model"].sum()
print(f"モデルが見込んだ的中数 {exp_hits:.2f}本(r1-r2-r3の確率 平均{W['p_model'].mean():.1%}) / 実際 {W['hit'].sum()}本")
lam = exp_hits
from math import exp, factorial  # noqa: E402
p_ge4 = 1 - sum(exp(-lam) * lam ** k / factorial(k) for k in range(4))
print(f"見込みどおりの確率で4本以上当たる確率 {p_ge4:.1%}(ポアソン近似)")
avg_pay = D.loc[D["hit"] == 1, "pay"].mean()
print(f"長期の的中時平均配当 {avg_pay:,.0f}円(100円あたり) → 見込みの的中数×平均配当で期待回収率 {exp_hits * avg_pay / (len(W) * 100):.1%}")
print(f"今回の的中時平均配当 {W.loc[W['hit'] == 1, 'pay'].mean():,.0f}円")

print(f"\n===== ② 長い目で見た同じ買い方(本番配信7/7〜11/3・帯22.5〜30%・{len(D):,}R) =====")
tot = D["pay"].sum() / (len(D) * 100)
print(f"全期間 的中{D['hit'].sum()}本({D['hit'].mean():.1%}) 回収率{tot:.1%} / モデルの見込み的中率{D['p_model'].mean():.1%}")
print(D.groupby("month").apply(lambda g: pd.Series({"R": len(g), "的中": g["hit"].sum(), "見込み": g["p_model"].sum(), "回収率": g["pay"].sum() / (len(g) * 100)}))
      .to_string(formatters={"R": "{:.0f}".format, "的中": "{:.0f}".format, "見込み": "{:.1f}".format, "回収率": "{:.1%}".format}))
pays = D["pay"].values
win = np.array([pays[i:i + len(W)].sum() / (len(W) * 100) for i in range(len(D) - len(W) + 1)])
print(f"連続{len(W)}レースの窓 {len(win):,}個: 回収率の中央値{np.median(win):.1%} / 181.5%以上の窓 {(win >= 1.815).mean():.1%} / 100%以上の窓 {(win >= 1.0).mean():.1%}")
rng = np.random.default_rng(4)
boot = np.array([rng.choice(pays, len(W)).sum() / (len(W) * 100) for _ in range(20000)])
print(f"全期間から{len(W)}レースを無作為に選んだ場合: 181.5%以上になる確率 {(boot >= 1.815).mean():.1%}")
before = D[D["date"] < "20261029"]
print(f"10/28までの同じ買い方 {len(before):,}R 回収率{before['pay'].sum() / (len(before) * 100):.1%}(的中{before['hit'].sum()}本)")

print("\n===== ③ 的中4本の中身 =====")
for _, x in W[W["hit"] == 1].iterrows():
    od = x["pay"] / 100
    q = f"{x['q_mkt']:.1%}" if pd.notna(x["q_mkt"]) else "記録なし"
    print(f"  {x['date'][4:6]}/{x['date'][6:]} {x['venue']}{x['rno']}R {x['key']}: 払戻{x['pay']:,}円(500円で{x['pay'] * 5:,}円) 確定オッズ{od:.1f}倍"
          f" / モデルの確率{x['p_model']:.1%}(期待値 {x['p_model'] * od:.2f}) / 締切前オッズ{x['odds_pre'] if pd.notna(x['odds_pre']) else '記録なし'} 市場の確率{q}")
print("  [外れた44本で、締切前オッズがあるレースの r1-r2-r3 のオッズ]")
miss = W[(W["hit"] == 0) & W["odds_pre"].notna()]
print(f"   {len(miss)}R オッズ中央値{miss['odds_pre'].median():.1f}倍 / モデル確率×オッズ(期待値)の平均 {(miss['p_model'] * miss['odds_pre']).mean():.2f}")
ww = W[W["odds_pre"].notna()]
print(f"  締切前オッズがある{len(ww)}Rでの 期待値(モデル確率×オッズ)の平均 {(ww['p_model'] * ww['odds_pre']).mean():.2f} / 市場確率の平均 {ww['q_mkt'].mean():.1%} / モデル確率の平均 {ww['p_model'].mean():.1%}")
lo = D[D["odds_pre"].notna()]
print(f"  参考 全期間の帯で締切前オッズがある{len(lo):,}R: 期待値の平均 {(lo['p_model'] * lo['odds_pre']).mean():.2f} / 的中時のオッズ中央値 {lo.loc[lo['hit'] == 1, 'pay'].median() / 100:.1f}倍")

print("\n===== ④ 11/1再学習の前後 =====")
for lab, g in (("10/1〜10/31(10/1学習)", D[(D["date"] >= "20261001") & (D["date"] <= "20261031")]), ("11/1〜11/3(11/1学習)", D[D["date"] >= "20261101"])):
    print(f"  {lab}: 帯{len(g)}R・1日{len(g) / g['date'].nunique():.1f}件 / r1が1号艇 {(g['r1'] == 1).mean():.0%} / r1-r2-r3の確率 平均{g['p_model'].mean():.2%}"
          f" / 的中{g['hit'].sum()}本 回収率{g['pay'].sum() / (len(g) * 100):.1%}")
print("\n[的中4本の共通点] r1が1号艇: " + ", ".join(f"{x['venue']}{x['rno']}R={'1号艇' if x['r1'] == 1 else str(x['r1']) + '号艇'}" for _, x in W[W["hit"] == 1].iterrows()))
w1 = W[W["r1"] == 1]
print(f"  期間中のr1=1号艇 {len(w1)}R: 1号艇の1着 {sum(1 for _, x in w1.iterrows() if x['key'][0] == '1' and x['hit']) }本的中 / r1が他艇 {len(W) - len(w1)}R: 的中{W.loc[W['r1'] != 1, 'hit'].sum()}本")
