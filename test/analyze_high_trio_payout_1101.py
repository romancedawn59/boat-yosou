# -*- coding: utf-8 -*-
"""3連複の高配当レースの調査(2026-10-31ケンさん指示)
  ① レース前展望: 高配当になったレースは朝6:30の配信データでどう見えていたか
  ② 結果論: 高配当レースでは何が起きていたか(1年分のDB: 決まり手・勝者コース・1号艇・進入・級別)
  ③ ①②の情報で「高配当になるレース」を事前に当てられるか。当てられたとして、買って勝てるか

    py -X utf8 test/analyze_high_trio_payout_1101.py

定義(結果を見る前に固定): 高配当 = 3連複の払戻 3,000円以上(100円あたり・1年分の上位約9%)。参考に 10,000円以上も出す
③の手順: 前半(7/7〜9/14)で高配当か否かの分類モデル(LightGBM・朝に分かる情報だけ)を学習し、後半(9/15〜)でAUCと十分位ごとの高配当率を見る。
     さらに後半の「荒れ予測上位20%」のレースだけで買い方(3連複モデル順K点・全20点・1号艇抜き10点・r1流し10点)を採点。
合格 = 後半で回収率100%超・的中30本以上・最大1本除いても100%超。
"""
import glob
import json
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
from predictors import trifecta_probs  # noqa: E402

HIGH, VHIGH = 3000, 10000
SPLIT = "20260915"
TECH = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "抜き", 6: "恵まれ"}
pd.set_option("display.width", 200)

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
trio_pay = {}
for rid, cb, a in conn.execute("SELECT race_id, combination, amount_yen FROM payouts WHERE bet_type='3連複' AND race_id>='20251001' AND amount_yen>0"):
    trio_pay.setdefault(rid, {})[cb] = a
trio_pay = {r: v for r, v in trio_pay.items() if len(v) == 1}
races_meta = {r: (g or "一般", rn, t) for r, g, rn, t in conn.execute("SELECT race_id, grade, race_no, winning_technique_number FROM races WHERE date>='2025-10-01'")}
res = defaultdict(dict)
for rid, lane, course, ao, st in conn.execute("SELECT race_id, lane, course, arrival_order, st_time FROM results WHERE race_id>='20251001'"):
    res[rid][lane] = (course, ao, st)
ent = defaultdict(dict)
for rid, lane, cls, avg_st, nwr, m2 in conn.execute("SELECT race_id, lane, racer_class, avg_st, national_win_rate, motor_2rate FROM entries WHERE race_id>='20260707'"):
    ent[rid][lane] = (cls, avg_st, nwr, m2)
mkt = defaultdict(dict)
for rid, cb, o in conn.execute("SELECT race_id, combination, odds FROM odds WHERE bet_type='3連複' AND odds>0 AND fetched_at!='final-backfill'"):
    mkt[rid][cb] = o
conn.close()

# ===== 定義と分布 =====
allpay = np.array([list(v.values())[0] for v in trio_pay.values()])
print(f"===== 3連複の払戻(1年・{len(allpay):,}R) 中央値{np.median(allpay):,.0f}円 / 高配当(>= {HIGH:,}円)は{(allpay >= HIGH).mean():.1%} / >= {VHIGH:,}円は{(allpay >= VHIGH).mean():.1%} =====")

# ===== ② 結果論(1年) =====
print(f"\n===== ② 結果論: 高配当レースで何が起きていたか(1年分・高配当{(allpay >= HIGH).sum():,}R vs それ以外) =====")
rows = []
for rid, v in trio_pay.items():
    pay = list(v.values())[0]; comb = list(v.keys())[0]
    r = res.get(rid); meta = races_meta.get(rid)
    if not r or len(r) < 6 or not meta:
        continue
    lanes3 = set(map(int, comb.split("=")))
    winner = next((l for l, (c, ao, st) in r.items() if ao == 1), None)
    if winner is None:
        continue
    wcourse = r[winner][0]
    c1 = r[1]
    rows.append(dict(rid=rid, pay=pay, high=pay >= HIGH, vhigh=pay >= VHIGH, grade=meta[0], race_no=meta[1], tech=TECH.get(meta[2], "不明"),
                     lane1_in3=(1 in lanes3), lane1_win=(winner == 1), lane1_course=c1[0], lane1_st=c1[2], winner_lane=winner, winner_course=wcourse,
                     outer_in3=sum(1 for l in lanes3 if l >= 4), wakunari=all(r[l][0] == l for l in r if r[l][0] is not None),
                     lane1_last=(c1[1] is not None and c1[1] >= 5)))
R = pd.DataFrame(rows)


def share(d, col):
    return d.groupby(col).size() / len(d)


for label, mask in (("高配当", R["high"]), ("それ以外", ~R["high"]), ("1万円以上", R["vhigh"])):
    d = R[mask]
    print(f"\n[{label} {len(d):,}R] 1号艇3着以内 {d['lane1_in3'].mean():.0%} / 1号艇1着 {d['lane1_win'].mean():.0%} / 1号艇5着以下 {d['lane1_last'].mean():.0%} / 枠なり進入 {d['wakunari'].mean():.0%} / 3着以内に4〜6号艇が2艇以上 {(d['outer_in3'] >= 2).mean():.0%}")
    print("  決まり手: " + " ".join(f"{k}{v:.0%}" for k, v in share(d, "tech").sort_values(ascending=False).items()))
    print("  勝者の枠: " + " ".join(f"{k}号{v:.0%}" for k, v in share(d, "winner_lane").items()) + " | 勝者のコース: " + " ".join(f"{k}C{v:.0%}" for k, v in share(d, "winner_course").items() if k))
    print("  グレード: " + " ".join(f"{k}{v:.0%}" for k, v in share(d, "grade").items()) + f" | 1号艇の平均ST {d['lane1_st'].mean():.3f}")
print("\n[高配当の割合をレース番号別に]")
print((R.groupby("race_no")["high"].mean()).map("{:.1%}".format).to_string())

# ===== ① レース前展望(picks) =====
print("\n===== ① レース前展望: 朝6:30の配信データで高配当レースはどう見えていたか(7/7〜) =====")
feat = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        rid = r["race_id"]; rk = r.get("ranked")
        if not rk or len(rk) < 6 or rid not in trio_pay or rid not in ent or len(ent[rid]) < 6:
            continue
        probs = {int(l): float(p) for l, p in rk}
        tri = trifecta_probs(probs); z = sum(tri.values())
        top3 = defaultdict(float); trio = defaultdict(float)
        for (a, b, c), v in tri.items():
            for x in (a, b, c):
                top3[x] += v / z
            trio["=".join(map(str, sorted((a, b, c))))] += v / z
        order = sorted(trio, key=trio.get, reverse=True)
        comb, pay = next(iter(trio_pay[rid].items()))
        p = np.array(sorted(probs.values(), reverse=True))
        t3 = sorted(top3.values(), reverse=True)
        e = ent[rid]
        a1 = sum(1 for l in e if e[l][0] == "A1")
        nwr = [e[l][2] or 0 for l in sorted(e)]
        m2 = [e[l][3] or 0 for l in sorted(e)]
        meta = races_meta.get(rid, ("一般", r["race_no"], None))
        o = mkt.get(rid)
        feat.append(dict(rid=rid, period=("後半" if rid >= SPLIT else "前半"), pay=pay, high=pay >= HIGH, venue=r["venue_code"], race_no=r["race_no"],
                         grade={"SG": 4, "G1": 3, "G2": 2, "G3": 1}.get(meta[0], 0),
                         p1=p[0], p2=p[1], p3=p[2], p_lane1=probs[1], r1_is_1=int(int(rk[0][0]) == 1), p_trio=trio[order[0]],
                         cliff=t3[2] - t3[3], ent1=-(p * np.log(p + 1e-9)).sum(), outer_top3=sum(top3[l] for l in (4, 5, 6)),
                         top3_1=top3[1], a1=a1, cls1=int(e[1][0] == "A1"), st1=e[1][1] or 0.2, nwr1=nwr[0], nwr_max_other=max(nwr[1:]),
                         nwr_spread=max(nwr) - min(nwr), m2_1=m2[0], m2_max=max(m2),
                         model_rank_win=order.index(comb) + 1, model_p_win=trio[comb],
                         mkt_rank_win=(1 + sum(1 for c_, v in o.items() if v < o.get(comb, 1e9))) if o and comb in o else np.nan,
                         mkt_odds_win=(o.get(comb) if o else np.nan), n_odds=len(o) if o else 0,
                         order=order, trio=dict(trio), r1=int(rk[0][0])))
Fd = pd.DataFrame(feat)
print(f"対象 {len(Fd):,}R(前半{(Fd['period'] == '前半').sum():,}・後半{(Fd['period'] == '後半').sum():,}) 高配当 {Fd['high'].mean():.1%}")
cols = {"p1": "1位勝率", "p_trio": "3連複r1r2r3確率", "cliff": "3着以内の崖", "ent1": "1着分布の散らばり", "outer_top3": "外枠3着以内確率計", "top3_1": "1号艇3着以内確率",
        "p_lane1": "1号艇1着確率", "r1_is_1": "1号艇がr1", "a1": "A1の人数", "nwr_spread": "勝率の幅", "grade": "グレード", "race_no": "レース番号"}
t = Fd.groupby("high")[list(cols)].mean().T.rename(columns={True: "高配当", False: "それ以外"}).rename(index=cols)
print(t.to_string(float_format=lambda x: f"{x:.3f}"))
print("\n[当たった3連複の組はモデルの20通りで何番目だったか]")
for label, d in (("高配当", Fd[Fd["high"]]), ("それ以外", Fd[~Fd["high"]])):
    c = Counter(pd.cut(d["model_rank_win"], [0, 1, 3, 5, 10, 20], labels=["1位", "2〜3位", "4〜5位", "6〜10位", "11〜20位"]))
    print(f"  {label}: " + " ".join(f"{k}{v / len(d):.0%}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))) + f" | モデルがその組につけていた確率の中央値 {d['model_p_win'].median():.1%}")
dd = Fd[Fd["n_odds"] >= 18]
print(f"\n[締切前オッズがある{len(dd):,}R: 当たった組の市場人気順位] 高配当: 中央値{dd[dd['high']]['mkt_rank_win'].median():.0f}位 / それ以外: {dd[~dd['high']]['mkt_rank_win'].median():.0f}位")
print("[1位勝率帯ごとの高配当率]")
print(Fd.groupby(pd.cut(Fd["p1"], [0, .15, .225, .30, .40, 1], right=False))["high"].agg(["size", "mean"]).rename(columns={"size": "R", "mean": "高配当率"}).to_string(formatters={"高配当率": "{:.1%}".format}))

# ===== ③ 事前予測 =====
print("\n===== ③ 朝の情報で高配当レースを当てられるか(LightGBM・前半で学習→後半で評価) =====")
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

X_COLS = ["venue", "race_no", "grade", "p1", "p2", "p3", "p_lane1", "r1_is_1", "p_trio", "cliff", "ent1", "outer_top3", "top3_1", "a1", "cls1", "st1",
          "nwr1", "nwr_max_other", "nwr_spread", "m2_1", "m2_max"]
A, B = Fd[Fd["period"] == "前半"], Fd[Fd["period"] == "後半"]
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, verbose=-1)
clf.fit(A[X_COLS], A["high"], categorical_feature=["venue"])
B = B.copy(); B["score"] = clf.predict_proba(B[X_COLS])[:, 1]
A = A.copy(); A["score"] = clf.predict_proba(A[X_COLS])[:, 1]
print(f"AUC 後半 {roc_auc_score(B['high'], B['score']):.3f}(前半=学習データ {roc_auc_score(A['high'], A['score']):.3f}) ※0.5=当てずっぽう、1.0=完全")
imp = pd.Series(clf.feature_importances_, index=X_COLS).sort_values(ascending=False)
print("重要度上位: " + ", ".join(f"{k}{v}" for k, v in imp.head(8).items()))
B["decile"] = pd.qcut(B["score"], 5, labels=["下位20%", "20〜40", "40〜60", "60〜80", "荒れ予測上位20%"])
t = B.groupby("decile", observed=True).agg(R=("high", "size"), 高配当率=("high", "mean"), 平均払戻=("pay", "mean"), 中央値払戻=("pay", "median"))
print(t.to_string(formatters={"高配当率": "{:.1%}".format, "平均払戻": "{:,.0f}円".format, "中央値払戻": "{:,.0f}円".format}))


def strat_returns(d, name):
    out = []
    for _, x in d.iterrows():
        order, trio, pay_c = x["order"], x["trio"], next(iter(trio_pay[x["rid"]].items()))
        comb, pay = pay_c
        if name.startswith("モデル順"):
            k = int(name[len("モデル順"):-1]); picks = order[:k]
        elif name == "全20点":
            picks = order
        elif name == "1号艇抜き10点":
            picks = [c for c in order if "1" not in c.split("=")]
        elif name == "r1流し10点":
            picks = [c for c in order if str(x["r1"]) in c.split("=")]
        elif name == "モデル6〜20位(人気薄15点)":
            picks = order[5:]
        out.append((len(picks), pay if comb in picks else 0))
    n = sum(k for k, _ in out); tot = sum(p for _, p in out); hits = sum(1 for _, p in out if p > 0)
    top = max((p for _, p in out), default=0)
    return tot / (n * 100) if n else np.nan, hits, (tot - top) / (n * 100) if n else np.nan, len(out)


print("\n[後半: 荒れ予測上位20%のレースだけで買う(100円/点) / 比較に後半全レース]")
rows = []
for name in ("モデル順1点", "モデル順3点", "モデル順5点", "モデル順10点", "全20点", "1号艇抜き10点", "r1流し10点", "モデル6〜20位(人気薄15点)"):
    top = strat_returns(B[B["decile"] == "荒れ予測上位20%"], name)
    allb = strat_returns(B, name)
    ok = top[0] > 1 and top[1] >= 30 and top[2] > 1
    rows.append(dict(買い方=name, 荒れ上位20pct_回収率=top[0], 的中=top[1], 最大1本除く=top[2], R=top[3], 後半全体_回収率=allb[0], 合格=("○" if ok else "×")))
print(pd.DataFrame(rows).to_string(index=False, formatters={k: "{:.1%}".format for k in ("荒れ上位20pct_回収率", "最大1本除く", "後半全体_回収率")}))
print("\n[参考(後出し): 前半=学習データで同じ表。ここが良くて後半が悪ければ過学習]")
rows = []
A["decile"] = pd.qcut(A["score"], 5, labels=["下位20%", "20〜40", "40〜60", "60〜80", "荒れ予測上位20%"])
for name in ("モデル順3点", "全20点", "1号艇抜き10点", "モデル6〜20位(人気薄15点)"):
    top = strat_returns(A[A["decile"] == "荒れ予測上位20%"], name)
    rows.append(dict(買い方=name, 荒れ上位20pct_回収率=top[0], 的中=top[1], R=top[3]))
print(pd.DataFrame(rows).to_string(index=False, formatters={"荒れ上位20pct_回収率": "{:.1%}".format}))
