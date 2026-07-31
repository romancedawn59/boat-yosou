# -*- coding: utf-8 -*-
"""超混戦: E/F単の「家族拡張」(削らず+4点400円)の月次検証(2026-08-01ケンさん発案)

    py -X utf8 test/verify_konsen_family_extend.py

■ 提案(ケンさんの真意=G単H単の正体)
E単(r3-r1-r2)とF単(r4-r1-r2)の各トリオについて「次に来そうな並び」を足す。
定義(事前固定): 各トリオの3連単6通りをBenter確率順に並べ、既購入(E/F)を除く
上位2つずつ=計4点×100円を、案1×2(2,000円)に【削らず】追加(計2,400円)。

■ 既知の背景
単体5fold: Atop1 204.6% / Atop2 90.2% / Btop1 276.8% / Btop2 313.2%。
削って足す形(変形ハ)は5foldは通ったが月次切りで-39,000円(頑健性なし・紙上判定中)。
今回は「削らない」ため別物として月次切り(2025-12〜2026-05+参考6-7月)で測る。

■ 判定(事前固定)
追加4点の限界ROI(追加分だけの回収率)が194%(厚くの限界効率)を頑健に超え、
かつ全体ROIが案1×2を月次過半で上回る場合のみ採用候補。
150-194%は紙上観測(W2と同席)。150%未満は棄却。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH
from features import FEATURE_COLUMNS, build_training_set

MONTHS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07"]

conn = db.connect(DB_PATH)
df = build_training_set(conn)
payout_map = defaultdict(dict)
for rid, bt, comb, amt in conn.execute(
    "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
    "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-12-01'"):
    payout_map[rid][(bt, comb)] = amt or 0
conn.close()

agg = defaultdict(lambda: [0, 0])          # {(arm, month): [st, rt]}
ext_only = defaultdict(lambda: [0, 0, 0])  # {month: [st, rt, hits]} 追加4点だけ
n = 0
for m in MONTHS:
    train_df = df[df["date"] < f"{m}-01"]
    month_df = df[df["date"].str.startswith(m)].copy()
    if month_df.empty:
        continue
    print(f"{m}: 学習{len(train_df):,}行", flush=True)
    booster = train_fold(train_df)
    month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])
    for rid, g in month_df.groupby("race_id"):
        if not payout_map[rid]:
            continue
        g_sorted = g.sort_values("pred", ascending=False)
        ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                  for _, r in g_sorted.iterrows()]
        if len(ranked) < 5 or ranked[0]["prob"] >= 0.20:
            continue
        n += 1
        lanes = [r["lane"] for r in ranked]
        r1, r2, r3, r4, r5 = lanes[:5]
        probs = P.normalize_probs(ranked)
        tri = P.trifecta_probs(probs)

        def trio(a, b, c):
            s = sorted([a, b, c])
            return f"{s[0]}={s[1]}={s[2]}"

        def next2(members, exclude):
            cands = sorted(((o, p) for o, p in tri.items()
                            if set(o) == set(members)), key=lambda x: -x[1])
            out = []
            for (a, b, c), _p in cands:
                comb = f"{a}-{b}-{c}"
                if comb != exclude:
                    out.append(comb)
                if len(out) == 2:
                    break
            return out

        base = [("3連複", trio(r1, r2, r3), 600), ("3連複", trio(r1, r2, r4), 400),
                ("3連単", f"{r3}-{r1}-{r2}", 400), ("3連単", f"{r4}-{r1}-{r2}", 400),
                ("3連複", trio(r3, r4, r5), 200)]
        ext = [("3連単", cmb, 100)
               for cmb in next2((r1, r2, r3), f"{r3}-{r1}-{r2}")
               + next2((r1, r2, r4), f"{r4}-{r1}-{r2}")]
        pay = payout_map[rid]
        for arm, bets in (("案1×2(現行)", base), ("家族拡張(+4点2,400円)", base + ext)):
            st = sum(y for _, _, y in bets)
            rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y in bets)
            a = agg[(arm, m)]
            a[0] += st
            a[1] += rt
        st = sum(y for _, _, y in ext)
        rt = sum(pay.get((bt, comb), 0) * y // 100 for bt, comb, y in ext)
        e = ext_only[m]
        e[0] += st
        e[1] += rt
        e[2] += sum(1 for bt, comb, y in ext if pay.get((bt, comb), 0))

print(f"\n超混戦帯: {n:,}R")
print(f"\n{'月':<9}{'案1×2':>9}{'家族拡張':>9}{'追加4点のみ':>11}")
tb = te = 0
for m in MONTHS:
    b = agg[("案1×2(現行)", m)]
    x = agg[("家族拡張(+4点2,400円)", m)]
    e = ext_only[m]
    if not b[0]:
        continue
    print(f"{m:<9}{b[1]/b[0]:>9.1%}{x[1]/x[0]:>9.1%}{(e[1]/e[0] if e[0] else 0):>11.1%}")
st_b = sum(agg[('案1×2(現行)', m)][0] for m in MONTHS)
rt_b = sum(agg[('案1×2(現行)', m)][1] for m in MONTHS)
st_x = sum(agg[('家族拡張(+4点2,400円)', m)][0] for m in MONTHS)
rt_x = sum(agg[('家族拡張(+4点2,400円)', m)][1] for m in MONTHS)
st_e = sum(e[0] for e in ext_only.values())
rt_e = sum(e[1] for e in ext_only.values())
hits = sum(e[2] for e in ext_only.values())
print(f"{'合計':<9}{rt_b/st_b:>9.1%}{rt_x/st_x:>9.1%}{rt_e/st_e:>11.1%}")
print(f"\n案1×2: 投資{st_b:,}円 損益{rt_b-st_b:+,}円")
print(f"家族拡張: 投資{st_x:,}円 損益{rt_x-st_x:+,}円 (差{(rt_x-st_x)-(rt_b-st_b):+,}円)")
print(f"追加4点のみ: 投資{st_e:,}円 回収{rt_e:,}円({rt_e/st_e:.1%}) 的中{hits}本")
mon_win = sum(1 for m in MONTHS if agg[('家族拡張(+4点2,400円)', m)][0]
              and agg[('家族拡張(+4点2,400円)', m)][1]/agg[('家族拡張(+4点2,400円)', m)][0]
              > agg[('案1×2(現行)', m)][1]/agg[('案1×2(現行)', m)][0])
print(f"月次で案1×2超え: {mon_win}/8か月")
print(f"\n判定: 追加分194%超+月次過半で採用候補 / 150-194%=紙上 / 150%未満=棄却")
