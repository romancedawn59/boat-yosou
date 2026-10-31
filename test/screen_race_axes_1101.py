# -*- coding: utf-8 -*-
"""帯(1位勝率)以外のレース傾向の軸をふるいにかける(2026-11-01ケンさん「帯以外のレース傾向分析方法を探して」)

    py -X utf8 test/screen_race_axes_1101.py

材料: data_raw/div_2b_market_1101.pkl(締切前3連単オッズがある7/1〜10/31の5,471R。案2bの着順確率と市場の確率と着順)
      + DBの番組表・出走表・結果履歴
軸ごとにレースを区分けし、区分ごとに次の2種類を見る
  (1) レースの性格: 1号艇の1着率・3連単払戻の中央値・万舟率(1万円以上)・市場の3連単1番人気の的中率
  (2) 独自システムが市場に勝てる場所か: log-loss(市場) − log-loss(案2b)。正なら案2bが市場より正確。95%区間つき
ふるいの見方: (1)の差が大きい軸は「読み」に使える。(2)で区間が0より上の区分は「市場に勝てる候補地」。
記述的なふるいで事前宣言ではない。区分の数が多いので、偶然で1〜2区分は0より上に出る前提で読む
"""
import pickle
import sqlite3
from itertools import permutations

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
pd.set_option("display.width", 220)
P3 = np.array(list(permutations(range(6), 3)))
IDX3 = {tuple(p): i for i, p in enumerate(P3)}
rows = pickle.load(open(r"Y:\マイドライブ\boat\data_raw\div_2b_market_1101.pkl", "rb"))
rids = [x["rid"] for x in rows]

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
races = pd.read_sql_query("SELECT race_id, date, venue_code, race_no, grade, day_label, subtitle, title, deadline_time, wind_speed_m, wave_height_cm FROM races", conn).set_index("race_id")
ent = pd.read_sql_query("SELECT race_id, lane, reg_no, racer_class, national_win_rate, avg_st, motor_2rate, age FROM entries WHERE race_id>='20260701'", conn)
hist = pd.read_sql_query("""SELECT r.date, e.reg_no, e.lane, res.course, res.arrival_order, r.winning_technique_number AS tech
                            FROM entries e JOIN races r ON r.race_id=e.race_id JOIN results res ON res.race_id=e.race_id AND res.lane=e.lane""", conn)
lane1 = pd.read_sql_query("SELECT r.venue_code, r.date, res.arrival_order FROM results res JOIN races r ON r.race_id=res.race_id WHERE res.lane=1 AND res.arrival_order IS NOT NULL", conn)
tri_pay = {r: a for r, a in conn.execute("SELECT race_id, amount_yen FROM payouts WHERE bet_type='3連単' AND race_id>='20260701'")}
conn.close()

# 場のイン強度(7/1より前の1号艇1着率)
vin = lane1[lane1["date"] < "2026-07-01"].groupby("venue_code")["arrival_order"].apply(lambda s: (s == 1).mean())
# 選手の展開履歴(前日まで): 1コース1着率・まくり勝ち率・前付け率
h = hist.dropna(subset=["reg_no"]).copy(); h["reg_no"] = h["reg_no"].astype(int)
h["start"] = h["arrival_order"].notna().astype(int)
h["c1"] = (h["course"] == 1).astype(int); h["c1w"] = ((h["course"] == 1) & (h["arrival_order"] == 1)).astype(int)
h["mk"] = ((h["arrival_order"] == 1) & h["tech"].isin([3, 4])).astype(int)
h["mae"] = (h["course"].notna() & (h["course"] < h["lane"])).astype(int)
cum = h.groupby(["reg_no", "date"])[["start", "c1", "c1w", "mk", "mae"]].sum().groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0)
sty = pd.DataFrame({"c1w_rate": (cum["c1w"] + 20 * .55) / (cum["c1"] + 20), "mk_rate": (cum["mk"] + 20 * .05) / (cum["start"] + 20),
                    "mae_rate": (cum["mae"] + 20 * .03) / (cum["start"] + 20)}).reset_index()
ent = ent.merge(races[["date"]], left_on="race_id", right_index=True).merge(sty, on=["reg_no", "date"], how="left")
E = {rid: g.set_index("lane") for rid, g in ent[ent["race_id"].isin(rids)].groupby("race_id")}

recs = []
for x in rows:
    rid, t, q, act = x["rid"], x["t"], x["q"], x["act"]
    if rid not in E or len(E[rid]) < 6 or rid not in races.index:
        continue
    e, r = E[rid], races.loc[rid]
    a, b, c = act[:3]
    def ll(v):
        pa = v[P3[:, 0] == a].sum(); pab = v[(P3[:, 0] == a) & (P3[:, 1] == b)].sum()
        return -np.log(max(pa, 1e-9)) - np.log(max(pab / pa, 1e-9)) - np.log(max(v[IDX3[(a, b, c)]] / pab, 1e-9))
    q1 = np.array([q[P3[:, 0] == k].sum() for k in range(6)])
    cls = e["racer_class"].fillna("B2")
    nwr = e["national_win_rate"].fillna(0); st = e["avg_st"].fillna(.18); mot = e["motor_2rate"].fillna(30)
    hour = int(r["deadline_time"][11:13]) if r["deadline_time"] else 12
    recs.append(dict(
        rid=rid, d_ll=ll(q) - ll(t), lane1_win=int(a == 0), pay=tri_pay.get(rid, 0), mkt_top_hit=int(np.argmax(q) == IDX3[(a, b, c)]),
        場のイン強度=pd.cut([vin.get(r["venue_code"], .55)], [0, .52, .57, 1], labels=["イン弱(52%未満)", "中", "イン強(57%以上)"])[0],
        レース種別=("優勝戦・準優" if r["subtitle"] in ("優勝戦", "準優勝戦") else "特選・選抜・ドリーム" if r["subtitle"] and any(k in r["subtitle"] for k in ("特", "選抜", "ドリーム", "記者")) else "予選・一般"),
        節の日目=("初日" if r["day_label"] == "初日" else "最終日" if r["day_label"] == "最終日" else "中日"),
        グレード=("一般" if r["grade"] in (None, "一般") else "G3以上"),
        時間帯=("朝(〜11時)" if hour < 11 else "昼(11〜16時)" if hour < 16 else "夕(16〜18時)" if hour < 18 else "ナイター(18時〜)"),
        女子戦=("女子戦" if r["title"] and any(k in r["title"] for k in ("女子", "レディース", "ヴィーナス", "クイーン", "オールレディ")) else "混合"),
        風速=("〜2m" if (r["wind_speed_m"] or 0) <= 2 else "3〜4m" if (r["wind_speed_m"] or 0) <= 4 else "5m以上"),
        波高=("〜2cm" if (r["wave_height_cm"] or 0) <= 2 else "3〜5cm" if (r["wave_height_cm"] or 0) <= 5 else "6cm以上"),
        A1の人数=min(int((cls == "A1").sum()), 3),
        一号艇の級=("A1" if cls[1] == "A1" else "A2" if cls[1] == "A2" else "B級"),
        一号艇と他艇の勝率差=nwr[1] - nwr.drop(1).max(),
        勝率の幅=nwr.max() - nwr.min(),
        一号艇のイン1着率=e.loc[1, "c1w_rate"],
        一号艇とカドのST差=st[1] - st[[2, 3, 4]].min(),
        カドまくり屋=e.loc[[3, 4], "mk_rate"].max(),
        前付け選手=("あり" if e.loc[[4, 5, 6], "mae_rate"].max() > .10 else "なし"),
        一号艇のモーター順位=int(mot.rank(ascending=False)[1]),
        外枠の最強モーター=mot[[4, 5, 6]].max(),
        市場の1号艇1着確率=q1[0],
        市場の3単1番人気の確率=q.max(),
        市場の散らばり=-(q * np.log(q)).sum(),
    ))
D = pd.DataFrame(recs)
print(f"対象 {len(D):,}R(7/1〜10/31・締切前オッズあり)。全体: 1号艇1着{D['lane1_win'].mean():.1%}・3単払戻中央値{D['pay'].median():,.0f}円・万舟{(D['pay'] >= 10000).mean():.1%}・"
      f"市場3単1番人気的中{D['mkt_top_hit'].mean():.1%}・log-loss(市場−案2b){D['d_ll'].mean():+.4f}")

QUANT = {"一号艇と他艇の勝率差": 4, "勝率の幅": 4, "一号艇のイン1着率": 4, "一号艇とカドのST差": 4, "カドまくり屋": 4, "外枠の最強モーター": 4,
         "市場の1号艇1着確率": 5, "市場の3単1番人気の確率": 5, "市場の散らばり": 5}
CAT = ["場のイン強度", "レース種別", "節の日目", "グレード", "時間帯", "女子戦", "風速", "波高", "A1の人数", "一号艇の級", "前付け選手", "一号艇のモーター順位"]


def ci(v):
    v = np.asarray(v); idx = rng.integers(0, len(v), size=(1000, len(v))); mm = v[idx].mean(1)
    return v.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


summary = []
for ax in CAT + list(QUANT):
    col = D[ax]
    grp = pd.qcut(col, QUANT[ax], duplicates="drop") if ax in QUANT else col
    print(f"\n===== {ax} =====")
    out = []
    for k, g in D.groupby(grp, observed=True):
        if len(g) < 40:
            continue
        mu, lo, hi = ci(g["d_ll"])
        mark = "◎案2bが正確" if lo > 0 else ("▲市場が正確" if hi < 0 else "")
        out.append(dict(区分=str(k), R=len(g), 一号艇1着=g["lane1_win"].mean(), 払戻中央値=g["pay"].median(), 万舟率=(g["pay"] >= 10000).mean(),
                        市場1番人気的中=g["mkt_top_hit"].mean(), 市場から案2bを引いた差=f"{mu:+.3f}[{lo:+.3f},{hi:+.3f}]", 判定=mark))
    t = pd.DataFrame(out)
    print(t.to_string(index=False, formatters={"一号艇1着": "{:.1%}".format, "払戻中央値": "{:,.0f}円".format, "万舟率": "{:.1%}".format, "市場1番人気的中": "{:.1%}".format}))
    summary.append(dict(軸=ax, 一号艇1着の幅=t["一号艇1着"].max() - t["一号艇1着"].min(), 万舟率の幅=t["万舟率"].max() - t["万舟率"].min(),
                        案2bが正確な区分=int((t["判定"] == "◎案2bが正確").sum()), 市場が正確な区分=int((t["判定"] == "▲市場が正確").sum()), 区分数=len(t)))

print("\n===== まとめ: 軸ごとの『性格の分け方』と『市場に勝てる区分』 =====")
S = pd.DataFrame(summary).sort_values("万舟率の幅", ascending=False)
print(S.to_string(index=False, formatters={"一号艇1着の幅": "{:.1%}".format, "万舟率の幅": "{:.1%}".format}))
