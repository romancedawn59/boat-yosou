# -*- coding: utf-8 -*-
"""事前宣言(2026-11-04・結果を見る前に固定し、実行前にコミットする)
直前情報(展示タイム・展示ST・展示進入・風と波)を入れると、4つの的中率がどれだけ上がるか

    py -X utf8 test/prereg_chokuzen_hitrate_1104.py

■ 目的(ケンさん指定): 市場人気・オッズは使わない。単勝1点・2連複1点・3連複1点・3連単5点の的中率を上げる
■ 作り方(2段構え)
  1段目 = 本番配信の1着確率(朝6:30・picks JSON)。日ごとに出した予想なので、どの期間でも学習に使っていない値
  2段目 = 1段目の確率に材料を足して1着確率を補正するLightGBM(1着か否かの2値)。レース内で合計1に正規化し、
          2着以下は本番と同じ展開式(Harville・λ2=0.70/λ3=0.50)で3連単120通りの確率にする
  対照   = 2段目の材料が「1段目の確率・その順位・枠」だけ(補正そのものの効果を差し引くため)
  直前あり = 対照の材料 + 直前情報
     展示タイム系: 展示タイム・レース内順位・レース平均との差・最速との差、チルト、展示時の体重、安定板の有無
     風と波: レース時の風速・風向・波高(結果データの値を直前情報の代わりに使う)
     展示ST・展示進入(副検証のみ): 展示ST・レース内順位・最速との差、展示進入コース・枠との差(前付け)・1コースか
■ データ
  主検証 = 展示タイム+風と波。学習 2026-07-07〜09-30 / 採点 2026-10-01〜11-03(展示タイムが6艇そろうレース)
  副検証 = 主検証の材料+展示ST・展示進入。学習 2026-07-07〜10-17 / 採点 2026-10-18〜11-03(展示STが6艇そろうレース)
           展示ST・展示進入は9月末からしか記録がないため標本が小さい。参考扱い
■ 物差し(採点期間・同じレースで対照と直前ありを比べる)
  単勝1点 = 1着確率1位の艇 / 2連複1点 = 2連複確率1位の組 / 3連複1点 = 3連複確率1位の組 / 3連単5点 = 3連単確率の上位5組
  差はレースごとの対の差、95%区間はブートストラップ2,000回。副指標として1着のlog-loss
■ 合格の条件(主検証で判定)
  4つの的中率のうち少なくとも1つで、直前ありが対照より高く差の95%区間が0より上。かつ4つとも有意に下がらない(区間の上端が0より上)
  副検証は同じ物差しで記録だけする(判定には使わない)
"""
import glob
import json
import sqlite3
import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
from predictors import trifecta_probs  # noqa: E402

rng = np.random.default_rng(20261104)
pd.set_option("display.width", 220)

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
ex = pd.read_sql_query("SELECT race_id, lane, weight_kg AS ex_weight, exhibition_time, tilt, ex_course, ex_st, stabilizer FROM exhibition", conn)
rc = pd.read_sql_query("SELECT race_id, wind_speed_m, wind_direction_number, wave_height_cm FROM races WHERE date>='2026-07-01'", conn)
res = pd.read_sql_query("SELECT race_id, lane, arrival_order FROM results WHERE race_id>='20260701'", conn)
conn.close()

prod = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    day = path[-15:-5]
    if not ("2026-07-07" <= day <= "2026-11-03"):
        continue
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        if r.get("ranked") and len(r["ranked"]) == 6:
            for k, (l, p) in enumerate(r["ranked"]):
                prod.append((r["race_id"], day, int(l), float(p), k + 1))
D = pd.DataFrame(prod, columns=["race_id", "date", "lane", "p_prod", "prod_rank"])
D = D.merge(ex, on=["race_id", "lane"], how="inner").merge(rc, on="race_id", how="left").merge(res, on=["race_id", "lane"], how="left")
g = D.groupby("race_id")
ok = (g["lane"].transform("count") == 6) & g["exhibition_time"].transform(lambda s: s.notna().all())
fin = D.groupby("race_id")["arrival_order"].agg(lambda s: sorted(s.dropna().astype(int).tolist())[:3] == [1, 2, 3])
D = D[ok & D["race_id"].map(fin)].copy().sort_values(["race_id", "lane"]).reset_index(drop=True)
D["win"] = (D["arrival_order"] == 1).astype(int)
D["logit_p"] = np.log(D["p_prod"].clip(1e-4, 1 - 1e-4) / (1 - D["p_prod"].clip(1e-4, 1 - 1e-4)))
g = D.groupby("race_id")
D["ext_rank"] = g["exhibition_time"].rank(method="min")
D["ext_diff_mean"] = D["exhibition_time"] - g["exhibition_time"].transform("mean")
D["ext_diff_best"] = D["exhibition_time"] - g["exhibition_time"].transform("min")
D["stab"] = g["stabilizer"].transform("max")
D["exst_rank"] = g["ex_st"].rank(method="min")
D["exst_diff_best"] = D["ex_st"] - g["ex_st"].transform("min")
D["course_shift"] = D["lane"] - D["ex_course"]
D["ex_c1"] = (D["ex_course"] == 1).astype(float).where(D["ex_course"].notna())
D["has_st"] = g["ex_st"].transform(lambda s: s.notna().all())

CTRL = ["logit_p", "prod_rank", "lane"]
EXT = ["exhibition_time", "ext_rank", "ext_diff_mean", "ext_diff_best", "tilt", "ex_weight", "stab", "wind_speed_m", "wind_direction_number", "wave_height_cm"]
STC = ["ex_st", "exst_rank", "exst_diff_best", "ex_course", "course_shift", "ex_c1"]
PARAMS = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100,
          "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "seed": 7}


def fit_predict(tr, te, cols):
    dates = np.sort(tr["date"].unique()); cut = dates[int(len(dates) * .85)]
    a, b = tr[tr["date"] < cut], tr[tr["date"] >= cut]
    ds = lgb.Dataset(a[cols], label=a["win"]); vs = lgb.Dataset(b[cols], label=b["win"], reference=ds)
    m = lgb.train(PARAMS, ds, 2000, valid_sets=[vs], callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(te[cols]).reshape(-1, 6)
    return p / p.sum(axis=1, keepdims=True)


def hits(P, te):
    """P: [レース, 枠1〜6] の1着確率 → 4つの的中(0/1)と1着log-loss"""
    rids = te["race_id"].values[::6]
    pos = te["arrival_order"].values.reshape(-1, 6)
    out = []
    for i in range(len(rids)):
        probs = {l + 1: float(P[i, l]) for l in range(6)}
        tri = trifecta_probs(probs)
        q2 = defaultdict(float); q3 = defaultdict(float)
        for (a, b, c), v in tri.items():
            q2[frozenset((a, b))] += v; q3[frozenset((a, b, c))] += v
        act = [int(np.where(pos[i] == k)[0][0]) + 1 for k in (1, 2, 3)]
        top5 = sorted(tri, key=tri.get, reverse=True)[:5]
        out.append(dict(tan=int(max(probs, key=probs.get) == act[0]), q2=int(max(q2, key=q2.get) == frozenset(act[:2])),
                        q3=int(max(q3, key=q3.get) == frozenset(act)), t5=int(tuple(act) in top5), ll=-np.log(max(probs[act[0]], 1e-9))))
    return pd.DataFrame(out)


def ci(v):
    v = np.asarray(v, dtype=float); idx = rng.integers(0, len(v), size=(2000, len(v))); mm = v[idx].mean(1)
    return v.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


LAB = [("tan", "単勝1点"), ("q2", "2連複1点"), ("q3", "3連複1点"), ("t5", "3連単5点")]


def report(title, tr, te, arms):
    print(f"\n===== {title} =====")
    print(f"学習 {tr['race_id'].nunique():,}R({tr['date'].min()}〜{tr['date'].max()}) / 採点 {te['race_id'].nunique():,}R({te['date'].min()}〜{te['date'].max()})")
    H = {"本番そのまま": hits(te["p_prod"].values.reshape(-1, 6) / te["p_prod"].values.reshape(-1, 6).sum(1, keepdims=True), te)}
    for name, cols in arms.items():
        H[name] = hits(fit_predict(tr, te, cols), te)
    t = pd.DataFrame({name: {**{lab: h[k].mean() for k, lab in LAB}, "1着log-loss": h["ll"].mean()} for name, h in H.items()}).T
    print(t.to_string(formatters={**{lab: "{:.1%}".format for _, lab in LAB}, "1着log-loss": "{:.4f}".format}))
    names = list(arms)
    base, test = H[names[0]], H[names[-1]]
    res = {}
    print(f"  差({names[-1]} − {names[0]})・95%区間:")
    for k, lab in LAB:
        mu, lo, hi = ci(test[k] - base[k]); res[k] = (mu, lo, hi)
        print(f"    {lab}: {mu * 100:+.2f}pt [{lo * 100:+.2f}, {hi * 100:+.2f}]" + ("  ◎上がった" if lo > 0 else ("  ▲下がった" if hi < 0 else "")))
    mu, lo, hi = ci(base["ll"] - test["ll"])
    print(f"    1着log-loss(正なら改善): {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return res


main_tr = D[(D["date"] >= "2026-07-07") & (D["date"] <= "2026-09-30")]
main_te = D[(D["date"] >= "2026-10-01") & (D["date"] <= "2026-11-03")]
r = report("主検証: 展示タイム系+風と波", main_tr, main_te, {"対照(補正のみ)": CTRL, "直前あり(展示タイム+風波)": CTRL + EXT})
up = any(r[k][1] > 0 for k, _ in LAB); down = any(r[k][2] < 0 for k, _ in LAB)
print(f"\n判定: {'合格' if up and not down else '不合格'}(上がった指標 {'あり' if up else 'なし'} / 有意に下がった指標 {'あり' if down else 'なし'})")

sub_tr = D[(D["date"] >= "2026-07-07") & (D["date"] <= "2026-10-17")]
sub_te = D[(D["date"] >= "2026-10-18") & (D["date"] <= "2026-11-03") & D["has_st"]]
report("副検証(参考): 展示タイム系+風と波+展示ST・展示進入", sub_tr, sub_te,
       {"対照(補正のみ)": CTRL, "直前あり(展示タイム+風波)": CTRL + EXT, "直前あり(+展示ST・進入)": CTRL + EXT + STC})
