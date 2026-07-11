# -*- coding: utf-8 -*-
"""検証⑨: 進入コース特徴量(イン屋/アウト屋傾向)の有効性検証

現行モデルは枠番(lane)しか見ておらず、選手のコース取り傾向は特徴量にない。
results.course(進入コース、約1年分・98%充足)から選手ごとの傾向を
リークなし(各行はその行より前のレースのみ参照・shift(1)済み)で算出し、
現行特徴量に追加した場合のモデルを検証⑦⑧と同じ枠組みで比較する。

    py -X utf8 test/verify_course_features.py

枠組み: 2026-05-01より前のデータで学習 → 2026-05〜06の対象5場を予測。
比較軸: AUC / 1位的中率 / ken現行構成のバケット別回収率 / 本命勝負所の回収率
(最大1発を除いた頑健性チェックつき)。

追加する特徴量(いずれも直近の実績からの派生。当日の進入コースは使わない):
- form_course_diff:        直近20走の平均(進入コース - 枠番)。負=前づけ(イン屋)
- form_course_attack_rate: 直近20走の前づけ率(コース < 枠番 の割合)
- form_course_edge:        直近20走の平均(取ったコースの全国1着率 - 枠番の全国1着率)。
                           コース取りで得ている勝率上の有利さ。基準1着率は学習期間から算出
- form_lane_avg_course:    この枠番からの過去平均進入コース(選手×枠番、全期間)
"""
import json
import sys
from collections import defaultdict
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lightgbm as lgb
import pandas as pd

import db
import predictors as P
from backtest import PARAMS
from config import DB_PATH, PROJECT_DIR, TARGET_VENUE_CODES, VENUE_NAMES
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set

EVAL_START, EVAL_END = "2026-05-01", "2026-06-30"
TEST_DIR = PROJECT_DIR / "test"

COURSE_COLUMNS = [
    "form_course_diff",
    "form_course_attack_rate",
    "form_course_edge",
    "form_lane_avg_course",
]


def compute_course_features(conn, baseline_wr: dict) -> pd.DataFrame:
    """選手ごとの進入コース傾向特徴量。(race_id, lane)キー。

    compute_form_featuresと同じ流儀: 選手ごとに時系列で並べ、shift(1)してから
    集計するため、各行は「その行より前のレース」の実績しか参照しない。
    """
    h = pd.read_sql_query(
        """
        SELECT r.race_id, r.date, r.race_no, e.lane, e.reg_no,
               res.course
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
        """,
        conn,
    )
    h = h.sort_values(["reg_no", "date", "race_no"]).reset_index(drop=True)

    has_course = h["course"].notna()
    h["_diff"] = (h["course"] - h["lane"]).where(has_course)
    h["_attack"] = (h["course"] < h["lane"]).astype(float).where(has_course)
    h["_edge"] = (h["course"].map(baseline_wr) - h["lane"].map(baseline_wr)).where(has_course)

    g = h.groupby("reg_no", sort=False)

    def prev_rolling_mean(col: str, window: int = 20, min_periods: int = 3) -> pd.Series:
        return g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
        )

    h["form_course_diff"] = prev_rolling_mean("_diff")
    h["form_course_attack_rate"] = prev_rolling_mean("_attack")
    h["form_course_edge"] = prev_rolling_mean("_edge")
    h["form_lane_avg_course"] = h.groupby(["reg_no", "lane"], sort=False)["course"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )
    return h[["race_id", "lane", *COURSE_COLUMNS]]


def train(train_df: pd.DataFrame, feature_cols: list[str]) -> lgb.Booster:
    """backtest.train_foldと同じ学習手順(特徴量リストだけ差し替え可能に)"""
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr, va = train_df[train_df["date"] < cutoff], train_df[train_df["date"] >= cutoff]
    train_set = lgb.Dataset(tr[feature_cols], label=tr["is_winner"],
                            categorical_feature=CATEGORICAL_FEATURES)
    valid_set = lgb.Dataset(va[feature_cols], label=va["is_winner"], reference=train_set)
    return lgb.train(PARAMS, train_set, valid_sets=[valid_set], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def auc_score(y: pd.Series, p: pd.Series) -> float:
    """順位ベースのAUC(依存ライブラリ追加を避けるため自前計算)"""
    r = p.rank()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    print("学習データ構築中...")
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)

    # コース別の基準1着率は学習期間(2026-05-01より前)のみから算出(リーク防止)
    baseline_wr = dict(conn.execute(
        "SELECT res.course, AVG(res.arrival_order = 1) FROM results res "
        "JOIN races r ON r.race_id = res.race_id "
        "WHERE r.date < ? AND res.course IS NOT NULL AND res.arrival_order IS NOT NULL "
        "GROUP BY res.course", (EVAL_START,),
    ).fetchall())
    print("コース別基準1着率:", {k: round(v, 3) for k, v in sorted(baseline_wr.items())})

    print("コース特徴量算出中...")
    course_feat = compute_course_features(conn, baseline_wr)
    df = df.merge(course_feat, on=["race_id", "lane"], how="left")

    actual = defaultdict(dict)
    for rid, lane, order in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id = res.race_id "
        "WHERE r.date BETWEEN ? AND ? AND res.arrival_order IS NOT NULL",
        (EVAL_START, EVAL_END),
    ):
        actual[rid][order] = lane

    payout_map = defaultdict(dict)
    for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id WHERE r.date BETWEEN ? AND ?",
        (EVAL_START, EVAL_END),
    ):
        payout_map[rid][(bt, comb)] = amt or 0
    conn.close()

    train_df = df[df["date"] < EVAL_START]
    eval_df = df[(df["date"] >= EVAL_START) & (df["date"] <= EVAL_END)
                 & (df["venue_code"].isin(TARGET_VENUE_CODES))].copy()
    print(f"学習 {len(train_df):,}行 / 評価 {len(eval_df):,}行")

    variants = {
        "base": FEATURE_COLUMNS,
        "course": FEATURE_COLUMNS + COURSE_COLUMNS,
    }
    boosters = {}
    for name, cols in variants.items():
        print(f"モデル学習中({name}: {len(cols)}特徴量)...")
        boosters[name] = train(train_df, cols)
        eval_df[f"pred_{name}"] = boosters[name].predict(eval_df[cols])

    results = {"baseline_wr": {str(k): v for k, v in baseline_wr.items()},
               "train_rows": len(train_df)}

    # 新特徴量の重要度(gain)と全特徴量中の順位
    bst = boosters["course"]
    imp = pd.Series(bst.feature_importance("gain"), index=bst.feature_name())
    imp_rank = imp.rank(ascending=False).astype(int)
    results["importance"] = {
        c: {"gain_pct": float(imp[c] / imp.sum()), "rank": int(imp_rank[c]),
            "total": len(imp)}
        for c in COURSE_COLUMNS
    }

    def evaluate(name: str) -> dict:
        pred_col = f"pred_{name}"
        races = []
        for rid, g in eval_df.groupby("race_id"):
            if 1 not in actual[rid]:
                continue
            g_sorted = g.sort_values(pred_col, ascending=False)
            ranked = [{"lane": int(r["lane"]), "prob": float(r[pred_col])}
                      for _, r in g_sorted.iterrows()]
            probs = P.normalize_probs(ranked)
            conf = P.bucket_of(ranked[0]["prob"])
            b_picks = P.picks_yamada(probs) if len(probs) >= 4 else []
            c_picks = P.picks_katsu(probs) if len(probs) >= 4 else []
            plan = P.ken_portfolio(conf, ranked, b_picks, c_picks)
            stake = sum(y for _, _, y, _ in plan)
            ret = sum(payout_map[rid].get((bt, comb), 0) * yen // 100
                      for bt, comb, yen, _ in plan)
            races.append({
                "race_id": rid, "date": str(g["date"].iloc[0]),
                "venue_code": int(g["venue_code"].iloc[0]),
                "race_no": int(g["race_no"].iloc[0]),
                "ranked": ranked, "bets": {"confidence": conf, "plan": plan},
                "stake": stake, "return": ret,
                "top1_hit": ranked[0]["lane"] == actual[rid][1],
            })
        races.sort(key=lambda r: (r["date"], r["venue_code"], r["race_no"]))
        for d, grp in groupby(races, key=lambda r: r["date"]):
            P.select_shobusho(list(grp), max_races=10)

        def agg(rs):
            n = len(rs)
            hits = sum(1 for r in rs if r["return"])
            stake = sum(r["stake"] for r in rs)
            ret = sum(r["return"] for r in rs)
            return {"n": n, "hits": hits, "stake": stake, "ret": ret,
                    "roi": ret / stake if stake else 0.0}

        out = {
            "auc": auc_score(eval_df["is_winner"], eval_df[pred_col]),
            "top1": sum(r["top1_hit"] for r in races) / len(races),
            "n_races": len(races),
            "bucket": {b: agg([r for r in races if r["bets"]["confidence"] == b
                               and r["bets"]["plan"]])
                       for b in ("堅め", "標準", "荒れ注意")},
            "shobusho": {m: agg([r for r in races if r["shobusho"] == m])
                         for m in ("本命", "準")},
            "monthly_honmei": {
                mo: agg([r for r in races if r["shobusho"] == "本命"
                         and r["date"].startswith(mo)])
                for mo in ("2026-05", "2026-06")
            },
        }
        # 頑健性: 本命勝負所から最大払戻の1レースを除いた回収率
        hon = [r for r in races if r["shobusho"] == "本命"]
        if hon:
            top = max(hon, key=lambda r: r["return"])
            rest = [r for r in hon if r is not top]
            out["honmei_excl_max"] = {**agg(rest), "excluded": {
                "race": f"{top['date']} {VENUE_NAMES[top['venue_code']]}{top['race_no']}R",
                "return": top["return"]}}
        return out, races

    all_races = {}
    for name in variants:
        results[name], all_races[name] = evaluate(name)
        r = results[name]
        print(f"\n===== {name} =====")
        print(f"AUC {r['auc']:.4f} / 1位的中率 {r['top1']:.1%} / 評価 {r['n_races']}レース")
        for b, s in r["bucket"].items():
            print(f"  {b}: {s['n']}R 回収率{s['roi']:.1%} 損益{s['ret']-s['stake']:+,}円")
        for m, s in r["shobusho"].items():
            print(f"  勝負所{m}: {s['n']}R 的中{s['hits']} 回収率{s['roi']:.1%} 損益{s['ret']-s['stake']:+,}円")
        for mo, s in r["monthly_honmei"].items():
            print(f"    {mo} 本命: {s['n']}R 回収率{s['roi']:.1%}")
        ex = r["honmei_excl_max"]
        print(f"  本命(最大1発 {ex['excluded']['race']} {ex['excluded']['return']:,}円 を除く): "
              f"回収率{ex['roi']:.1%}")

    # CSV: コース特徴量モデルの全レース予想(ベースラインの勝負所判定も併記)
    base_sho = {r["race_id"]: r["shobusho"] for r in all_races["base"]}
    csv_path = TEST_DIR / "predictions_202605-06_course_features.csv"
    rows = []
    for r in all_races["course"]:
        res = actual[r["race_id"]]
        plan_str = " / ".join(f"{bt}{comb}:{yen}円({src})"
                              for bt, comb, yen, src in r["bets"]["plan"])
        rows.append({
            "race_id": r["race_id"], "date": r["date"],
            "venue": VENUE_NAMES[r["venue_code"]], "race_no": r["race_no"],
            "confidence": r["bets"]["confidence"],
            "shobusho_course": r["shobusho"] or "",
            "shobusho_base": base_sho.get(r["race_id"]) or "",
            "top_prob": round(r["ranked"][0]["prob"], 3),
            "result": "-".join(str(res.get(k, "?")) for k in (1, 2, 3)),
            "plan": plan_str, "stake": r["stake"], "return": r["return"],
            "profit": r["return"] - r["stake"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV出力: {csv_path}")

    json_path = TEST_DIR / "verify_course_features_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"集計JSON出力: {json_path}")


if __name__ == "__main__":
    main()
