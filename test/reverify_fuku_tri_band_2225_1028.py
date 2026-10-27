# -*- coding: utf-8 -*-
"""再検証: 1位勝率22.5〜25%だけ 複勝r1 1000円 + 3連単r1-r2-r3 500円(2026-10-28ケンさん指示)

    py -X utf8 test/reverify_fuku_tri_band_2225_1028.py

見るもの(基準は結果を見る前に固定):
  ① 1年台帳(月次WF・2025-10〜2026-09)の月別・部品別(複勝/3単)・高配当の寄与(上位1/3/5本を除く)
  ② 隣の帯(20〜22.5 / 25〜27.5 / 27.5〜30)と境界を±0.5%ずらしたときの回収率(帯の端に依存していないか)
  ③ 隣接帯(15〜40%)を土台にしたシャッフル検定(2,000回)
  ④ 本番配信の予想(picks JSON・2026-07-07〜)で同じ買い方。全期間と、帯を見つけた9/21より後(9/22〜)だけ
  ⑤ 場別(上位・下位)…場の優位は偶然と区別できない前提で、偏りの確認だけ
合格の目安: ①最大3本除いても100%超 ②隣の帯も境界ずらしも大きく崩れない ③p<0.05 ④本番で100%超(9/22〜は標本が小さいので参考)
"""
import glob
import json
import sqlite3

import numpy as np
import pandas as pd

LEDGER = r"Y:\マイドライブ\boat\data_raw\wf_ledger_all_202510_202609.csv"
FUKU_YEN, TRI_YEN = 1000, 500
LO, HI = .225, .25
VENUE = {1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡", 8: "常滑", 9: "津", 10: "三国",
         11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山", 19: "下関",
         20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"}

conn = sqlite3.connect(r"file:Y:\マイドライブ\boat\boat.db?mode=ro", uri=True, timeout=120)
pay = {(r, b, c): a or 0 for r, b, c, a in conn.execute(
    "SELECT race_id, bet_type, combination, amount_yen FROM payouts WHERE bet_type IN ('複勝','3連単')")}
has_pay = {r for r, b, c in pay}
conn.close()


def attach(df):
    """race_id, p1, r1, r2, r3 を持つ表に 複勝・3単の払戻(100円あたり)と買い方の投資・払戻を付ける"""
    df = df[df["race_id"].isin(has_pay)].copy()
    df["fuku"] = [pay.get((r, "複勝", str(a)), 0) for r, a in zip(df["race_id"], df["r1"])]
    df["tri"] = [pay.get((r, "3連単", f"{a}-{b}-{c}"), 0) for r, a, b, c in zip(df["race_id"], df["r1"], df["r2"], df["r3"])]
    df["ret_f"] = df["fuku"] * FUKU_YEN / 100
    df["ret_t"] = df["tri"] * TRI_YEN / 100
    df["ret"] = df["ret_f"] + df["ret_t"]
    df["cost"] = FUKU_YEN + TRI_YEN
    df["month"] = df["race_id"].str[:6]
    df["venue"] = df["race_id"].str[9:11].astype(int)
    return df


def roi(d, col="ret"):
    return d[col].sum() / d["cost"].sum() if len(d) else float("nan")


def block(d, label):
    if len(d) == 0:
        print(f"{label}: 0R"); return
    top = d["ret"].sort_values(ascending=False).values
    inv = d["cost"].sum()
    print(f"{label}: {len(d):,}R 投資{inv:,.0f}円 払戻{d['ret'].sum():,.0f}円 回収率{roi(d):.1%} "
          f"| 複勝分{d['ret_f'].sum() / (len(d) * FUKU_YEN):.1%}({(d['fuku'] > 0).sum()}本) 3単分{d['ret_t'].sum() / (len(d) * TRI_YEN):.1%}({(d['tri'] > 0).sum()}本) "
          f"| 上位1/3/5本除く {(d['ret'].sum() - top[:1].sum()) / inv:.1%}/{(d['ret'].sum() - top[:3].sum()) / inv:.1%}/{(d['ret'].sum() - top[:5].sum()) / inv:.1%}")


# ================= ① 1年台帳 =================
led = pd.read_csv(LEDGER)
led[["r1", "r2", "r3"]] = led["pred"].str.split("-", expand=True).iloc[:, :3].astype(int)
L = attach(led)
B = L[(L["p1"] >= LO) & (L["p1"] < HI)]
print("===== ① 1年台帳(月次WF 2025-10〜2026-09) 帯22.5〜25% 複勝r1 1000円+3単r1-r2-r3 500円 =====")
block(B, "全期間")
mon = B.groupby("month").apply(lambda d: pd.Series(dict(R=len(d), 複勝的中=(d["fuku"] > 0).sum(), 三単的中=(d["tri"] > 0).sum(),
                                                          回収率=roi(d), 複勝分=roi(d, "ret_f") * 1.5, 三単分=roi(d, "ret_t") * 3,
                                                          最大払戻=d["ret"].max())))
# 複勝分・3単分は各部品の投資に対する回収率(1.5=1500/1000, 3=1500/500)
print(mon.to_string(formatters={"回収率": "{:.1%}".format, "複勝分": "{:.1%}".format, "三単分": "{:.1%}".format, "最大払戻": "{:,.0f}".format}))
print(f"黒字月 {(mon['回収率'] > 1).sum()}/{len(mon)}、月の回収率の中央値 {mon['回収率'].median():.1%}")
print("\n高配当の上位8本(複勝の払戻は100円あたり):")
print(B.sort_values("ret", ascending=False).head(8)[["race_id", "p1", "pred", "fuku", "tri", "ret"]].to_string(index=False))

# ================= ② 隣の帯・境界ずらし =================
print("\n===== ② 隣の帯と境界ずらし(1年台帳) =====")
for lo, hi in ((.20, .225), (.225, .25), (.25, .275), (.275, .30), (.30, .325), (.22, .245), (.23, .255), (.22, .25), (.225, .255), (.225, .30)):
    block(L[(L["p1"] >= lo) & (L["p1"] < hi)], f"帯{lo:.1%}〜{hi:.1%}")

# ================= ③ シャッフル検定 =================
print("\n===== ③ 帯ラベルのシャッフル検定(土台=1位勝率15〜40%・2,000回) =====")
base = L[(L["p1"] >= .15) & (L["p1"] < .40)]
rng = np.random.default_rng(7)
for col, label in (("ret", "複勝+3単"), ("ret_f", "複勝だけ"), ("ret_t", "3単だけ")):
    real = B[col].sum() / B["cost"].sum()
    pool = base[col].values; k = len(B)
    sims = np.array([pool[rng.choice(len(pool), k, replace=False)].sum() / (k * 1500) for _ in range(2000)])
    print(f"{label}: 帯の回収率{real:.1%} / 土台{pool.sum() / (len(pool) * 1500):.1%} / 偶然で帯以上が出る確率 p={(sims >= real).mean():.3f}")
print("(複勝だけ・3単だけは1,500円のうちの各部品の寄与で比べている)")

# ================= ④ 本番配信 =================
rows = []
for path in sorted(glob.glob(r"Y:\マイドライブ\boat\docs\data\picks_2026-*.json")):
    for r in json.load(open(path, encoding="utf-8"))["races"]:
        if r.get("ranked") and len(r["ranked"]) >= 6:
            rk = r["ranked"]
            rows.append((r["race_id"], float(rk[0][1]), int(rk[0][0]), int(rk[1][0]), int(rk[2][0])))
P = attach(pd.DataFrame(rows, columns=["race_id", "p1", "r1", "r2", "r3"]))
P["pred"] = P["r1"].astype(str) + "-" + P["r2"].astype(str) + "-" + P["r3"].astype(str)
PB = P[(P["p1"] >= LO) & (P["p1"] < HI)]
print(f"\n===== ④ 本番配信の予想(picks JSON {P['race_id'].min()[:8]}〜{P['race_id'].max()[:8]}・{len(P):,}R) =====")
block(PB, "帯22.5〜25% 全期間")
block(PB[PB["race_id"] >= "20260922"], "帯22.5〜25% 9/22〜(帯を見つけた後)")
block(PB[PB["race_id"] < "20260922"], "帯22.5〜25% 〜9/21")
mon = PB.groupby("month").apply(lambda d: pd.Series(dict(R=len(d), 複勝的中=(d["fuku"] > 0).sum(), 三単的中=(d["tri"] > 0).sum(), 回収率=roi(d))))
print(mon.to_string(formatters={"回収率": "{:.1%}".format}))
print("参考: 本番の隣の帯")
for lo, hi in ((.20, .225), (.25, .275), (.275, .30), (.225, .30)):
    block(P[(P["p1"] >= lo) & (P["p1"] < hi)], f"帯{lo:.1%}〜{hi:.1%}")

# ================= ⑤ 場別 =================
print("\n===== ⑤ 場別(1年台帳・帯22.5〜25%)・R数が20以上の場 =====")
v = B.groupby("venue").apply(lambda d: pd.Series(dict(R=len(d), 回収率=roi(d), 複勝分=roi(d, "ret_f") * 1.5)))
v = v[v["R"] >= 20].sort_values("回収率", ascending=False)
v.index = [VENUE[i] for i in v.index]
print(v.to_string(formatters={"回収率": "{:.1%}".format, "複勝分": "{:.1%}".format}))
print(f"場の回収率が100%超: {(v['回収率'] > 1).sum()}/{len(v)}場")
