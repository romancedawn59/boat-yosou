# -*- coding: utf-8 -*-
"""v2.1 vs v2 の同一条件シミュレーション(2025-11〜2026-04・月次再学習方式)

    py -X utf8 test/verify_v21_vs_v2_sim.py

■ 比較(選別・モデルは共通、配分だけ差し替え)
  v2  : 本命=V2構成6点(6点目=C勝万舟100円) / 超混戦=Q案7点
  v2.1: 本命=V2構成6点(6点目=保険複r2r3r4) / 超混戦=案1「拾える複厚」5点
選別は現行v2ルールで固定: 本命=5場×1位生値20〜30%×日cap6(低い順)、
超混戦=全場×20%未満(本命に選ばれた分を除く)。
学習は本番の月次再学習と同じ「各月をその前月末までの全データで学習」。
注意: 2025-11の学習データは約3.5か月(DBが2025-07-15開始)でモデルが若い。

出力: 月別×(本命/超混戦/合計)の レース数・何か当たる率・回収率・損益。
"""
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"]
KONSEN_MAX, HONMEI_MAX, CAP = 0.20, 0.30, 6


def old_konsen_plan(lanes):
    r1, r2, r3, r4, r5 = lanes[:5]

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"
    return [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 100),
            ("3連複", trio(r1, r3, r4), 100), ("3連複", trio(r2, r3, r4), 100),
            ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200),
            ("3連複", trio(r3, r4, r5), 100)]


def old_honmei_plan(lanes, probs):
    r1, r2, r3, r4 = lanes[:4]

    def trio(a, b, x):
        s = sorted([a, b, x])
        return f"{s[0]}={s[1]}={s[2]}"
    plan = [("3連複", trio(r1, r2, r3), 200), ("3連複", trio(r1, r2, r4), 200),
            ("3連複", trio(r1, r3, r4), 100),
            ("3連単", f"{r3}-{r1}-{r2}", 200), ("3連単", f"{r4}-{r1}-{r2}", 200)]
    existing = {(bt, comb) for bt, comb, _y in plan}
    for bt, comb, _p in P.picks_katsu(probs):
        if (bt, comb) not in existing:
            plan.append((bt, comb, 100))
            break
    return plan


def new_plan(ranked, konsen):
    plan = P.ken_portfolio("荒れ注意", ranked, [],
                           P.picks_katsu(P.normalize_probs(ranked)), konsen=konsen)
    return [(bt, comb, y) for bt, comb, y, _s in plan]


def main():
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)
    payout_map = defaultdict(dict)
    for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id WHERE r.date >= '2025-11-01'"):
        payout_map[rid][(bt, comb)] = amt or 0
    conn.close()

    # {(arm, seg, month): [stake, ret, races, hit_races]}
    agg = defaultdict(lambda: [0, 0, 0, 0])

    for m in MONTHS:
        train_df = df[df["date"] < f"{m}-01"]
        month_df = df[df["date"].str.startswith(m)].copy()
        if month_df.empty:
            continue
        print(f"{m}: 学習{len(train_df):,}行 → 予測{len(month_df):,}行", flush=True)
        booster = train_fold(train_df)
        month_df["pred"] = booster.predict(month_df[FEATURE_COLUMNS])

        # レース文脈を組み立て
        ctx_by_day = defaultdict(list)
        for rid, g in month_df.groupby("race_id"):
            if not payout_map[rid]:
                continue
            g_sorted = g.sort_values("pred", ascending=False)
            ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                      for _, r in g_sorted.iterrows()]
            if len(ranked) < 5:
                continue
            ctx_by_day[g["date"].iloc[0]].append({
                "rid": rid, "ranked": ranked, "top": ranked[0]["prob"],
                "in5": int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES,
            })

        # 選別(現行v2ルール): 超混戦→本命cap6(重複は本命扱い)
        for d, cs in ctx_by_day.items():
            konsen = {c["rid"] for c in cs if c["top"] < KONSEN_MAX}
            pool = sorted((c for c in cs if c["in5"] and c["top"] < HONMEI_MAX),
                          key=lambda c: c["top"])
            honmei = {c["rid"] for c in pool[:CAP]}
            for c in cs:
                if c["rid"] in honmei:
                    seg = "本命"
                elif c["rid"] in konsen:
                    seg = "超混戦"
                else:
                    continue
                lanes = [r["lane"] for r in c["ranked"]]
                probs = P.normalize_probs(c["ranked"])
                plans = {
                    "v2": (old_honmei_plan(lanes, probs) if seg == "本命"
                           else old_konsen_plan(lanes)),
                    "v2.1": new_plan(c["ranked"], konsen=(seg == "超混戦")),
                }
                pay = payout_map[c["rid"]]
                for arm, plan in plans.items():
                    st = sum(y for _, _, y in plan)
                    rt = sum(pay.get((bt, comb), 0) * y // 100
                             for bt, comb, y in plan)
                    for key in ((arm, seg, m), (arm, seg, "合計"),
                                (arm, "全体", m), (arm, "全体", "合計")):
                        a = agg[key]
                        a[0] += st
                        a[1] += rt
                        a[2] += 1
                        a[3] += 1 if rt else 0

    print("\n===== 月別比較(全体=本命+超混戦) =====")
    print(f"{'月':<9}{'R数':>5} | {'v2 的中率':>9}{'回収率':>8}{'損益':>10} | "
          f"{'v2.1 的中率':>10}{'回収率':>8}{'損益':>10} | {'損益差':>9}")
    for m in MONTHS + ["合計"]:
        a2 = agg[("v2", "全体", m)]
        a21 = agg[("v2.1", "全体", m)]
        if not a2[2]:
            continue
        d2, d21 = a2[1] - a2[0], a21[1] - a21[0]
        print(f"{m:<9}{a2[2]:>5,} | {a2[3]/a2[2]:>9.1%}{a2[1]/a2[0]:>8.1%}"
              f"{d2:>+10,} | {a21[3]/a21[2]:>10.1%}{a21[1]/a21[0]:>8.1%}"
              f"{d21:>+10,} | {d21-d2:>+9,}")

    print("\n===== セグメント別(2025-11〜2026-04合計) =====")
    for seg in ("本命", "超混戦"):
        print(f"--- {seg} ---")
        for arm in ("v2", "v2.1"):
            st, rt, n, h = agg[(arm, seg, "合計")]
            if st:
                print(f"  {arm:<5} {n:,}R 投資{st:,}円 何か当たる率{h/n:.1%} "
                      f"回収率{rt/st:.1%} 損益{rt-st:+,}円")


if __name__ == "__main__":
    main()
