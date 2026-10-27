# -*- coding: utf-8 -*-
"""買い方の総当たり検証(2026-10-28ケンさん依頼「自由な組み合わせで検証」・基準は結果を見る前に固定)

    py -X utf8 test/sim_free_combinations_1028.py

台帳: data_raw/wf_ledger_all_202510_202609.csv(月次WF・52,693R・予想順位r1〜r6と1位勝率p1)
払戻: boat.db payouts(全7券種)。返還艇を含む買い目は投資・払戻とも0とする。
第1段階 券種×点数の構造(約50通り)×1位勝率の帯(2.5%刻み+広い帯)を1点100円均等で採点する。
第2段階 15分前オッズ(3連単・3連複、2026-07〜09)で「オッズがX倍以上のときだけ買う」を足す。

合格(事前固定): 的中30本以上・回収率100%超・最大1本除いても100%超・黒字月が7/12以上。
合格したマスは帯ラベルのシャッフル検定(2,000回)で偶然との区別を見る。
マスの総数が多いので、5%水準なら総数の5%は偶然で合格する。それを超えて残るかで判断する。
出力: data_raw/free_combos_1028.csv(全マス)
"""
import sys
from itertools import combinations, permutations

import numpy as np
import pandas as pd

sys.path.insert(0, r"Y:\マイドライブ\boat\src")
import db
from config import DB_PATH

LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
OUT = r"Y:\マイドライブ\boat\data_raw\free_combos_1028.csv"
STAKE = 100
MIN_HITS, MIN_PROFIT_MONTHS = 30, 7

print("データ準備中...", flush=True)
led = pd.read_csv(LEDGER)
led["r"] = led["pred"].apply(lambda s: [int(x) for x in s.split("-")])
led["ref"] = led["refund"].apply(lambda s: set() if pd.isna(s) else {int(x) for x in str(s).split("-")})
conn = db.connect(DB_PATH)
ids = set(led["race_id"])
pay = {}
for rid, bt, cb, amt in conn.execute("SELECT race_id, bet_type, combination, amount_yen FROM payouts"):
    if rid in ids:
        pay[(rid, bt, cb)] = amt or 0
odds = {}
for rid, bt, cb, o in conn.execute("SELECT race_id, bet_type, combination, odds FROM odds WHERE bet_type IN ('3連単','3連複')"):
    if rid in ids and o:
        odds[(rid, bt, cb)] = float(o)
conn.close()
print(f"台帳 {len(led):,}R / 払戻 {len(pay):,}行 / 15分前オッズあり {len({k[0] for k in odds}):,}R", flush=True)


# ---------- 買い目の構造 ----------
def tan(r, i): return [("単勝", f"{r[i]}")]
def fuku(r, i): return [("複勝", f"{r[i]}")]
def ex(r, a, bs): return [("2連単", f"{r[a]}-{r[b]}") for b in bs]
def qn(r, pairs): return [("2連複", "=".join(map(str, sorted((r[a], r[b]))))) for a, b in pairs]
def wide(r, pairs): return [("拡連複", "=".join(map(str, sorted((r[a], r[b]))))) for a, b in pairs]
def trio(r, sets): return [("3連複", "=".join(map(str, sorted((r[a], r[b], r[c]))))) for a, b, c in sets]
def tri(r, seq): return [("3連単", f"{r[a]}-{r[b]}-{r[c]}") for a, b, c in seq]
def tri_form(r, firsts, seconds, thirds):
    return tri(r, [(a, b, c) for a in firsts for b in seconds for c in thirds if len({a, b, c}) == 3])
def box_trio(r, idx): return trio(r, list(combinations(idx, 3)))
def box_tri(r, idx): return tri(r, list(permutations(idx, 3)))

ALL = [0, 1, 2, 3, 4, 5]
STRUCTS = {
    # 単勝・複勝
    "単勝r1": lambda r: tan(r, 0), "単勝r2": lambda r: tan(r, 1),
    "複勝r1": lambda r: fuku(r, 0), "複勝r2": lambda r: fuku(r, 1), "複勝r3": lambda r: fuku(r, 2),
    # 2連単
    "2単r1-r2": lambda r: ex(r, 0, [1]), "2単r1-r3": lambda r: ex(r, 0, [2]), "2単r2-r1": lambda r: ex(r, 1, [0]),
    "2単r1-{r2,r3}": lambda r: ex(r, 0, [1, 2]), "2単r1-{r2〜r4}": lambda r: ex(r, 0, [1, 2, 3]),
    "2単r1-{r2〜r5}": lambda r: ex(r, 0, [1, 2, 3, 4]), "2単{r1,r2}box": lambda r: ex(r, 0, [1]) + ex(r, 1, [0]),
    # 2連複・拡連複
    "2複r1=r2": lambda r: qn(r, [(0, 1)]), "2複r1=r3": lambda r: qn(r, [(0, 2)]), "2複r2=r3": lambda r: qn(r, [(1, 2)]),
    "2複r1={r2,r3}": lambda r: qn(r, [(0, 1), (0, 2)]), "2複r1={r2〜r4}": lambda r: qn(r, [(0, 1), (0, 2), (0, 3)]),
    "拡r1=r2": lambda r: wide(r, [(0, 1)]), "拡r1=r3": lambda r: wide(r, [(0, 2)]), "拡r2=r3": lambda r: wide(r, [(1, 2)]),
    # 3連複
    "3複r1r2r3": lambda r: trio(r, [(0, 1, 2)]), "3複r1r2r4": lambda r: trio(r, [(0, 1, 3)]), "3複r1r3r4": lambda r: trio(r, [(0, 2, 3)]),
    "3複r1r2-{r3,r4}": lambda r: trio(r, [(0, 1, 2), (0, 1, 3)]), "3複r1r2-{r3〜r5}": lambda r: trio(r, [(0, 1, 2), (0, 1, 3), (0, 1, 4)]),
    "3複r1-{r2〜r4}": lambda r: trio(r, [(0, 1, 2), (0, 1, 3), (0, 2, 3)]), "3複box r1〜r4": lambda r: box_trio(r, [0, 1, 2, 3]),
    "3複r1-{r2〜r5}": lambda r: trio(r, [(0,) + c for c in combinations([1, 2, 3, 4], 2)]),
    "3複r2r3r4": lambda r: trio(r, [(1, 2, 3)]),
    # 3連単
    "3単r1-r2-r3": lambda r: tri(r, [(0, 1, 2)]), "3単r1-r2-r4": lambda r: tri(r, [(0, 1, 3)]), "3単r1-r3-r2": lambda r: tri(r, [(0, 2, 1)]),
    "3単r2-r1-r3": lambda r: tri(r, [(1, 0, 2)]), "3単r1-r3-r4": lambda r: tri(r, [(0, 2, 3)]),
    "3単r1-r2-{r3,r4}": lambda r: tri_form(r, [0], [1], [2, 3]), "3単r1-{r2,r3}-{r2,r3}": lambda r: tri_form(r, [0], [1, 2], [1, 2]),
    "3単r1-{r2,r3}-{r2〜r4}": lambda r: tri_form(r, [0], [1, 2], [1, 2, 3]), "3単r1-r2-{r3〜r5}": lambda r: tri_form(r, [0], [1], [2, 3, 4]),
    "3単r1-{r2〜r4}-{r2〜r4}": lambda r: tri_form(r, [0], [1, 2, 3], [1, 2, 3]), "3単r1-r2-全": lambda r: tri_form(r, [0], [1], ALL),
    "3単r1-全-r2": lambda r: tri_form(r, [0], ALL, [1]), "3単{r1,r2}-{r1,r2}-r3": lambda r: tri_form(r, [0, 1], [0, 1], [2]),
    "3単box r1r2r3": lambda r: box_tri(r, [0, 1, 2]), "3単r1-{r2〜r5}-{r2〜r5}": lambda r: tri_form(r, [0], [1, 2, 3, 4], [1, 2, 3, 4]),
    "3単r1-{r2,r3}-全": lambda r: tri_form(r, [0], [1, 2], ALL), "3単{r2,r3}-r1-{r2〜r4}": lambda r: tri_form(r, [1, 2], [0], [1, 2, 3]),
    "3単r1-{r2,r3}-{r4,r5}": lambda r: tri_form(r, [0], [1, 2], [3, 4]), "3単r1-{r4,r5}-{r2,r3}": lambda r: tri_form(r, [0], [3, 4], [1, 2]),
    # 併用
    "併用 3単123+3複123": lambda r: tri(r, [(0, 1, 2)]) + trio(r, [(0, 1, 2)]),
    "併用 3単123+2単12": lambda r: tri(r, [(0, 1, 2)]) + ex(r, 0, [1]),
    "併用 3単123+単勝r1": lambda r: tri(r, [(0, 1, 2)]) + tan(r, 0),
    "併用 3複123+複勝r1": lambda r: trio(r, [(0, 1, 2)]) + fuku(r, 0),
    "併用 3単12-{3,4}+3複123": lambda r: tri_form(r, [0], [1], [2, 3]) + trio(r, [(0, 1, 2)]),
    "併用 3単1-{2,3}-{2〜4}+2単1-{2,3}": lambda r: tri_form(r, [0], [1, 2], [1, 2, 3]) + ex(r, 0, [1, 2]),
    "併用 3単123+3単124+3複123": lambda r: tri(r, [(0, 1, 2), (0, 1, 3)]) + trio(r, [(0, 1, 2)]),
}

# ---------- 帯 ----------
edges = np.arange(0.15, 0.625, 0.025)
BANDS = {"全帯": (0, 1.01), "〜22.5%": (0, .225), "22.5〜30%": (.225, .30), "30〜40%": (.30, .40),
         "40〜50%": (.40, .50), "50〜60%": (.50, .60), "60%〜": (.60, 1.01), "〜15%": (0, .15)}
for lo in edges:
    BANDS[f"{lo:.1%}〜{lo + .025:.1%}"] = (round(lo, 3), round(lo + .025, 3))
BANDS["62.5%〜"] = (.625, 1.01)

# ---------- 第1段階: 構造ごとに1レースずつ投資・払戻を出す ----------
rids, months, p1s, refs, preds = led["race_id"].values, led["month"].values, led["p1"].values, led["ref"].values, led["r"].values
cost_tab, ret_tab, hit_tab = {}, {}, {}
for name, fn in STRUCTS.items():
    cost = np.zeros(len(led)); ret = np.zeros(len(led)); hit = np.zeros(len(led), dtype=bool)
    for i in range(len(led)):
        rid, ref = rids[i], refs[i]
        for bt, cb in fn(preds[i]):
            lanes = {int(x) for x in cb.replace("=", "-").split("-")}
            if lanes & ref:
                continue
            cost[i] += STAKE
            a = pay.get((rid, bt, cb), 0)
            if a:
                ret[i] += a * STAKE / 100
                hit[i] = True
    cost_tab[name], ret_tab[name], hit_tab[name] = cost, ret, hit
print(f"構造 {len(STRUCTS)}通り × 帯 {len(BANDS)}通り = {len(STRUCTS) * len(BANDS):,}マス", flush=True)


def score(cost, ret, hit, mask, mons):
    c, r, h = cost[mask], ret[mask], hit[mask]
    n, inv = int(mask.sum()), c.sum()
    if n == 0 or inv == 0:
        return None
    mdf = pd.DataFrame({"m": mons[mask], "c": c, "r": r}).groupby("m").sum()
    mdf = mdf[mdf["c"] > 0]
    return dict(R=n, 点数=round(c.mean() / STAKE, 1), 的中R=int(h.sum()), 的中率=h.mean(), 回収率=r.sum() / inv,
                最大1本除く=(r.sum() - r.max()) / inv, 黒字月=int((mdf["r"] > mdf["c"]).sum()), 月数=len(mdf),
                最大払戻=r.max())


rows = []
for name in STRUCTS:
    for band, (lo, hi) in BANDS.items():
        mask = (p1s >= lo) & (p1s < hi)
        s = score(cost_tab[name], ret_tab[name], hit_tab[name], mask, months)
        if s:
            rows.append(dict(段階=1, 構造=name, 帯=band, 条件="", **s))

# ---------- 第2段階: 15分前オッズで絞る(3単123・3複123・3単124・3単132) ----------
has_odds = np.array([(rid, "3連単", "-".join(map(str, preds[i][:3]))) in odds for i, rid in enumerate(rids)])
print(f"第2段階: 15分前オッズがある台帳レース {has_odds.sum():,}R", flush=True)
ODDS_STRUCTS = {"3単r1-r2-r3": ("3連単", lambda r: f"{r[0]}-{r[1]}-{r[2]}"), "3単r1-r2-r4": ("3連単", lambda r: f"{r[0]}-{r[1]}-{r[3]}"),
                "3単r1-r3-r2": ("3連単", lambda r: f"{r[0]}-{r[2]}-{r[1]}"), "3複r1r2r3": ("3連複", lambda r: "=".join(map(str, sorted(r[:3]))))}
BROAD2 = {"全帯": (0, 1.01), "〜22.5%": (0, .225), "22.5〜30%": (.225, .30), "30〜40%": (.30, .40), "40〜50%": (.40, .50), "50%〜": (.50, 1.01)}
for name, (bt, cbf) in ODDS_STRUCTS.items():
    o = np.array([odds.get((rid, bt, cbf(preds[i])), np.nan) for i, rid in enumerate(rids)])
    conds = [("オッズ記録あり", ~np.isnan(o))] + [(f"オッズ{t}倍以上", o >= t) for t in (5, 10, 15, 20, 30, 50)] + \
            [(f"オッズ{t}倍未満", o < t) for t in (10, 20, 30)]
    for cond, cmask in conds:
        for band, (lo, hi) in BROAD2.items():
            mask = cmask & (p1s >= lo) & (p1s < hi)
            s = score(cost_tab[name], ret_tab[name], hit_tab[name], mask, months)
            if s:
                rows.append(dict(段階=2, 構造=name, 帯=band, 条件=cond, **s))
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding="utf-8-sig")

# ---------- 合格判定とシャッフル検定 ----------
df["合格"] = (df["的中R"] >= MIN_HITS) & (df["回収率"] > 1) & (df["最大1本除く"] > 1) & (df["黒字月"] >= MIN_PROFIT_MONTHS)
rng = np.random.default_rng(7)


def shuffle_p(name, lo, hi, base_mask, n_iter=2000):
    """帯ラベルをシャッフルしても同じ回収率以上が出る確率(構造は固定、帯の割り当てだけを崩す)"""
    c, r, p = cost_tab[name][base_mask], ret_tab[name][base_mask], p1s[base_mask]
    inb = (p >= lo) & (p < hi)
    real = r[inb].sum() / c[inb].sum()
    k, ge = int(inb.sum()), 0
    for _ in range(n_iter):
        idx = rng.choice(len(p), k, replace=False)
        ge += r[idx].sum() / max(c[idx].sum(), 1) >= real
    return ge / n_iter


pd.set_option("display.width", 250, "display.max_rows", 500, "display.unicode.east_asian_width", True)
fmt = {"的中率": "{:.1%}".format, "回収率": "{:.1%}".format, "最大1本除く": "{:.1%}".format, "最大払戻": "{:,.0f}".format}
wide_b = ["〜22.5%", "22.5〜30%", "30〜40%", "40〜50%", "50〜60%", "60%〜"]

print("\n===== 第1段階 全帯(帯で絞らない)の構造別成績 =====")
print(df[(df["段階"] == 1) & (df["帯"] == "全帯")].drop(columns=["段階", "帯", "条件", "合格"]).to_string(index=False, formatters=fmt))

print("\n===== 第1段階 広い帯 × 構造(回収率の高い順・的中30本以上・上位30) =====")
w = df[(df["段階"] == 1) & df["帯"].isin(wide_b) & (df["的中R"] >= MIN_HITS)].sort_values("回収率", ascending=False)
print(w.head(30).drop(columns=["段階", "条件", "合格"]).to_string(index=False, formatters=fmt))

print("\n===== 第1段階 帯22.5〜30% の構造別成績(回収率の高い順) =====")
print(df[(df["段階"] == 1) & (df["帯"] == "22.5〜30%")].sort_values("回収率", ascending=False)
      .drop(columns=["段階", "帯", "条件", "合格"]).to_string(index=False, formatters=fmt))

print("\n===== 第1段階 2.5%刻みの帯 × 構造(回収率の高い順・的中30本以上・上位30) =====")
n = df[(df["段階"] == 1) & ~df["帯"].isin(wide_b + ["全帯", "〜15%"]) & (df["的中R"] >= MIN_HITS)].sort_values("回収率", ascending=False)
print(n.head(30).drop(columns=["段階", "条件", "合格"]).to_string(index=False, formatters=fmt))

print("\n===== 第2段階 15分前オッズで絞る(回収率の高い順・的中30本以上・上位30) =====")
s2 = df[(df["段階"] == 2) & (df["的中R"] >= MIN_HITS)].sort_values("回収率", ascending=False)
print(s2.head(30).drop(columns=["段階", "合格"]).to_string(index=False, formatters=fmt))

print(f"\n===== 合格したマス(的中{MIN_HITS}本以上・回収率>100%・最大1本除いても>100%・黒字月{MIN_PROFIT_MONTHS}/12以上) =====")
total = len(df)
print(f"マス総数 {total:,}(第1段階 {(df['段階'] == 1).sum():,} / 第2段階 {(df['段階'] == 2).sum():,})。5%水準で偶然に合格する期待数 {total * .05:.0f}")
ok = df[df["合格"]].copy()
if ok.empty:
    print("合格なし")
else:
    pv = []
    for _, x in ok.iterrows():
        lo, hi = (BANDS if x["段階"] == 1 else BROAD2)[x["帯"]]
        if x["帯"] == "全帯" or (x["段階"] == 2 and x["条件"] != "オッズ記録あり"):
            pv.append(np.nan)
            continue
        pv.append(shuffle_p(x["構造"], lo, hi, np.ones(len(led), dtype=bool) if x["段階"] == 1 else has_odds))
    ok["シャッフルp"] = pv
    print(ok.drop(columns=["段階", "合格"]).to_string(index=False, formatters={**fmt, "シャッフルp": "{:.3f}".format}))
    print(f"合格 {len(ok)}マス、うちシャッフルp<0.05は {(ok['シャッフルp'] < .05).sum()}マス(帯なし・オッズ条件付きはp未計算)")
