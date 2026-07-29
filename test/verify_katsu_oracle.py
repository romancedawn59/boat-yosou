# -*- coding: utf-8 -*-
"""C勝万舟×展開予想のオラクル検証(2026-07-29ケンさん発案)

    py -X utf8 test/verify_katsu_oracle.py

■ 何を確かめるか(天井の測定)
万舟決着の正体はまくり系(万舟の58.9%・超万舟の66.3%、逃げは11.5%)で、
1号艇は48.1%が4着以下に沈む。一方、現行C(万舟圏の確率上位5点)は
Harville独立近似のため「頭だけ穴・2-3着に上位残り」の形に寄る。
そこで【決まり手を後知恵(オラクル)で与えたら】並びの選び直しでどこまで
的中率が上がるかを測る。後知恵ですら絞れないならZ1-2(まくり予測)を
作ってもCは改善しない=この方向は死んでいると分かる。

■ 方式
- テンプレ: P(枠番の並び a-b-c | 決まり手) を TEST_START より前の期間
  (2025-07-15〜2025-11-30)だけから推定(ラプラース平滑化 α=1、120通り)。
  評価期間のデータはテンプレに一切使わない
- walk-forward: backtest.py と同一fold。各レースでモデル勝率→万舟圏
  (発生確率0.5%以下)の候補集合を作る(ここまで現行Cと完全に同じ)
- 比較する3つの選び方(同じ候補集合から5点・各100円):
    C現行   = 発生確率の高い順に5点(picks_katsuと同一)
    オラクル = そのレースの【正解の】決まり手kでテンプレP(並び|k)の高い順に5点
               (同率はモデル確率で決着)
    ランダム = 候補から無作為5点(seed=42固定・下限の対照群)

■【事前登録】判定基準(実行前に固定。マスを見てから動かさない)
主要評価: 評価期間の万舟決着レース(3連単払戻1万円以上・全場・決まり手あり)
  合格1: オラクルの的中率がC現行の1.5倍以上
  合格2: オラクルの的中時平均払戻がC現行の70%以上
         (絞り込みが人気側に寄っただけではないことの担保)
両方合格→Z1-2(まくり予測)へ進む価値がある(天井=オラクルとの差が予測精度の上限)
片方でも不合格→「決まり手が分かっても並びは絞れない」= C改善のこの方向は打ち止め
副次(参考のみ・採否に使わない): 全評価レースのROI、5場スコープ、
  C×オラクルの重複点数、重複点/非重複点の的中率(ケンさんの④一致仮説の予備観察)

■ 注意
- オラクルは実運用不可能(決まり手はレース後にしか分からない)。これは天井の測定
- 本番コードには一切触れない(v1凍結: 新規ファイル+test/出力のみ)
出力: test/verify_katsu_oracle_results.json
"""
import json
import random
import sys
from collections import defaultdict

sys.path.insert(0, r"Y:\マイドライブ\boat\src")

import db
import predictors as P
from backtest import N_FOLDS, TEST_START, train_fold
from config import DB_PATH, TARGET_VENUE_CODES
from features import FEATURE_COLUMNS, build_training_set

MANSHU_PAYOUT_MIN = 10000   # 万舟決着の定義(3連単払戻1万円以上)
ALPHA = 1.0                 # ラプラース平滑化
N_PICKS = 5
SEED = 42
KIMARITE = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "抜き", 6: "恵まれ"}
RESULT_JSON = r"Y:\マイドライブ\boat\test\verify_katsu_oracle_results.json"


def build_template(conn):
    """P(並び a-b-c | 決まり手) を TEST_START より前だけから推定する"""
    counts = defaultdict(lambda: defaultdict(int))
    top3 = defaultdict(dict)
    tech = {}
    for rid, k, lane, ao in conn.execute(
        "SELECT r.race_id, r.winning_technique_number, res.lane, res.arrival_order "
        "FROM races r JOIN results res ON res.race_id = r.race_id "
        "WHERE r.date < ? AND r.winning_technique_number IS NOT NULL "
        "AND res.arrival_order IN (1,2,3)", (TEST_START,),
    ):
        top3[rid][ao] = lane
        tech[rid] = k
    n_races = defaultdict(int)
    for rid, t3 in top3.items():
        if len(t3) == 3:
            order = (t3[1], t3[2], t3[3])
            counts[tech[rid]][order] += 1
            n_races[tech[rid]] += 1

    def prob(k, order):
        return (counts[k].get(order, 0) + ALPHA) / (n_races[k] + ALPHA * 120)

    return prob, {KIMARITE[k]: n for k, n in sorted(n_races.items())}


def main():
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)

    template_prob, template_n = build_template(conn)
    print("テンプレ学習期間(〜%s)の決まり手別レース数: %s" % (TEST_START, template_n),
          flush=True)

    actual = defaultdict(dict)
    for rid, lane, o in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id = res.race_id "
        "WHERE r.date >= ? AND res.arrival_order IN (1,2,3)", (TEST_START,),
    ):
        actual[rid][o] = lane
    payout_tri = defaultdict(dict)
    for rid, comb, amt in conn.execute(
        "SELECT p.race_id, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id "
        "WHERE r.date >= ? AND p.bet_type = '3連単'", (TEST_START,),
    ):
        payout_tri[rid][comb] = amt or 0
    race_tech = dict(conn.execute(
        "SELECT r.race_id, r.winning_technique_number FROM races r "
        "WHERE r.date >= ? AND r.winning_technique_number IS NOT NULL",
        (TEST_START,)))
    conn.close()

    test_df = df[df["date"] >= TEST_START]
    dates = sorted(test_df["date"].unique())
    fold_size = len(dates) // N_FOLDS
    boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

    rng = random.Random(SEED)
    METHODS = ("C現行", "オラクル", "ランダム")
    # {(スコープ, 集合, 手法): [投資, 回収, レース数, 的中数]}
    agg = defaultdict(lambda: [0, 0, 0, 0])
    pay_hits = defaultdict(list)          # {(スコープ, 集合, 手法): [的中時払戻]}
    overlap_hist = defaultdict(int)       # C×オラクルの重複点数の分布
    joint = defaultdict(lambda: [0, 0])   # {区分: [点数, 的中]} ④一致仮説の予備観察
    tech_hit = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # 決まり手別

    for i in range(N_FOLDS):
        f_start, f_end = boundaries[i], boundaries[i + 1]
        train_df = df[df["date"] < f_start]
        fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()
        print(f"fold{i+1} 学習中...", flush=True)
        booster = train_fold(train_df)
        fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])

        for rid, g in fold_df.groupby("race_id"):
            t3 = actual.get(rid, {})
            k = race_tech.get(rid)
            if len(t3) != 3 or k is None:
                continue
            win_comb = f"{t3[1]}-{t3[2]}-{t3[3]}"
            win_pay = payout_tri.get(rid, {}).get(win_comb)
            if win_pay is None:
                continue

            g_sorted = g.sort_values("pred", ascending=False)
            ranked = [{"lane": int(r["lane"]), "prob": float(r["pred"])}
                      for _, r in g_sorted.iterrows()]
            probs = P.normalize_probs(ranked)
            if len(probs) < 4:
                continue
            tri = P.trifecta_probs(probs)
            cands = [(o, p) for o, p in tri.items() if p <= P.MANSHU_PROB_MAX]
            if not cands:
                continue

            c5 = [o for o, _ in sorted(cands, key=lambda x: -x[1])[:N_PICKS]]
            o5 = [o for o, _ in sorted(
                cands, key=lambda x: (-template_prob(k, x[0]), -x[1]))[:N_PICKS]]
            r5 = [o for o, _ in
                  rng.sample(cands, min(N_PICKS, len(cands)))]
            picks = {"C現行": c5, "オラクル": o5, "ランダム": r5}

            win_order = (t3[1], t3[2], t3[3])
            manshu = win_pay >= MANSHU_PAYOUT_MIN
            scopes = ["全場"]
            if int(g["venue_code"].iloc[0]) in TARGET_VENUE_CODES:
                scopes.append("5場")
            subsets = ["全評価R"] + (["万舟決着"] if manshu else [])

            for m in METHODS:
                hit = win_order in picks[m]
                ret = win_pay if hit else 0
                for sc in scopes:
                    for ss in subsets:
                        a = agg[(sc, ss, m)]
                        a[0] += 100 * len(picks[m])
                        a[1] += ret
                        a[2] += 1
                        a[3] += 1 if hit else 0
                        if hit:
                            pay_hits[(sc, ss, m)].append(win_pay)
            if manshu:
                kn = KIMARITE[k]
                for m in ("C現行", "オラクル"):
                    th = tech_hit[kn][m]
                    th[0] += 1
                    th[1] += 1 if win_order in picks[m] else 0

            # ④一致仮説の予備観察(万舟決着レースのみ・点単位)
            if manshu:
                both = set(c5) & set(o5)
                overlap_hist[len(both)] += 1
                for o in set(c5) | set(o5):
                    key = ("両方が選んだ点" if o in both
                           else "C現行のみ" if o in c5 else "オラクルのみ")
                    joint[key][0] += 1
                    joint[key][1] += 1 if o == win_order else 0

    # ---------------------------------------------------------------- 出力
    def line(sc, ss, m):
        st, rt, n, h = agg[(sc, ss, m)]
        if not n:
            return None
        avg = (sum(pay_hits[(sc, ss, m)]) / h) if h else 0
        return (f"  {m:<6} 的中 {h:>4}/{n:,}R ({h/n:6.2%})  回収率 {rt/st:7.1%}  "
                f"的中時平均払戻 {avg:>9,.0f}円")

    for sc in ("全場", "5場"):
        for ss in ("万舟決着", "全評価R"):
            n = agg[(sc, ss, "C現行")][2]
            tag = "★主要評価" if (sc, ss) == ("全場", "万舟決着") else "参考"
            print(f"\n=== [{tag}] {sc}・{ss}({n:,}R) ===")
            for m in METHODS:
                s = line(sc, ss, m)
                if s:
                    print(s)

    st_c = agg[("全場", "万舟決着", "C現行")]
    st_o = agg[("全場", "万舟決着", "オラクル")]
    hr_c = st_c[3] / st_c[2] if st_c[2] else 0
    hr_o = st_o[3] / st_o[2] if st_o[2] else 0
    ap_c = (sum(pay_hits[("全場", "万舟決着", "C現行")]) / st_c[3]) if st_c[3] else 0
    ap_o = (sum(pay_hits[("全場", "万舟決着", "オラクル")]) / st_o[3]) if st_o[3] else 0
    ok1 = hr_o >= 1.5 * hr_c
    ok2 = ap_o >= 0.7 * ap_c if ap_c else False
    print("\n===== 事前登録基準の判定(全場・万舟決着) =====")
    print(f"  合格1(的中率1.5倍以上): オラクル {hr_o:.2%} vs C {hr_c:.2%} "
          f"(比 {hr_o/hr_c if hr_c else float('nan'):.2f}倍) → {'○' if ok1 else '×'}")
    print(f"  合格2(平均払戻70%以上): オラクル {ap_o:,.0f}円 vs C {ap_c:,.0f}円 "
          f"(比 {ap_o/ap_c if ap_c else float('nan'):.1%}) → {'○' if ok2 else '×'}")
    print(f"  総合: {'両方合格 → Z1-2(まくり予測)に進む価値あり' if ok1 and ok2 else '不合格 → この方向のC改善は打ち止め'}")

    print("\n----- 参考: 決まり手別の的中率(全場・万舟決着) -----")
    for kn, d in sorted(tech_hit.items(), key=lambda x: -x[1]["C現行"][0]):
        n = d["C現行"][0]
        print(f"  {kn:<6} {n:>5,}R  C {d['C現行'][1]/n:6.2%} → "
              f"オラクル {d['オラクル'][1]/n:6.2%}")

    print("\n----- 参考: ④一致仮説の予備観察(万舟決着・点単位) -----")
    print(f"  C×オラクル重複点数の分布: "
          + " ".join(f"{k}点:{v}R" for k, v in sorted(overlap_hist.items())))
    for key, (n, h) in sorted(joint.items()):
        if n:
            print(f"  {key:<10} {n:>6,}点  的中 {h/n:.3%}")

    out = {
        "pre_registered": {
            "primary": "全場・万舟決着(3連単1万円以上)レース",
            "pass1": "オラクル的中率 >= C現行の1.5倍",
            "pass2": "オラクル的中時平均払戻 >= C現行の70%",
        },
        "template_n": template_n,
        "results": {f"{sc}|{ss}|{m}": agg[(sc, ss, m)]
                    for sc in ("全場", "5場") for ss in ("万舟決着", "全評価R")
                    for m in METHODS},
        "hit_rate": {"C": hr_c, "oracle": hr_o},
        "avg_payout_hits": {"C": ap_c, "oracle": ap_o},
        "pass1": ok1, "pass2": ok2,
        "tech_hit": {kn: {m: v for m, v in d.items()}
                     for kn, d in tech_hit.items()},
        "overlap_hist": dict(overlap_hist),
        "joint": {k: v for k, v in joint.items()},
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {RESULT_JSON}")


if __name__ == "__main__":
    main()
