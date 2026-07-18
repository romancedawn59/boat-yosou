# -*- coding: utf-8 -*-
"""検証⑫: 選手×場の特徴量強化(v2判断会2026-07-30〜31の主材料・紙上専用)

    py -X utf8 test/verify_racer_venue.py

目的(notes/HANDOVER.md §8の中心課題「問題は荒れるはず→順当だった。これに尽きる」):
荒れ判定「1位勝率35%未満」には
  A型=本当の混戦(市場も予測不能。エッジの源泉。〜20%帯・ガミ率37%)
  B型=モデルの無知(人間は勝者を知っている→順当決着→ガミ。20〜35%帯)
が混在し、実戦はB型多数派。「人間が知っている情報(選手×場・機材の直近・番組の質)」を
モデルに教えてB型を消し、荒れ注意をA型に純化させるのがこの検証の狙い。

枠組み: backtest.pyと同一のウォークフォワード(2025-12-01以降を5分割、各期間は
それより前のデータだけで学習したfoldモデルで予測)。選別=チャンピオン
(1位勝率生値35%未満・1日上限10)、買い目=検証済みV2構成(ken_portfolio荒れ注意)に
base/new両モデルとも統一し、変数を「特徴量」だけに絞る。本番は不変(v1凍結)。

事前登録の合格基準(※実行前にコードとレポートに固定。AUCではない):
1. 荒れ注意選別レースの順当決着率(的中のうち払戻<掛金)がbase比-5pt以上下がる
   (基準52〜58%→37%方向。-5ptはfold間σ≈5.1%相当の明確な低下)
2. 荒れ注意の回収率が最大1発除きでもbaseを上回る
3. AUCは参考値(上がらなくてもよい)
補助確認: 1位勝率帯の層別(境界20/25/30/35%は2026-07-18層別分析の丸め値で固定)で、
選別レースが〜20%帯(本物の混戦)に寄っているか=純化の確認。
判定: 1と2の両方を満たせば合格(v2候補の本命)、片方のみは部分合格として判断会に
委ね、どちらも満たさなければ不採用として記録する(気象・進入・潮汐と同じ扱い)。

追加特徴量(9個。番組の質以外はすべて自前DB resultsからshift(1)でリーク防止):
[選手×場] rv_win_rate / rv_top3_rate: この場での直近30走の1着率・3連対率
[場×枠]   vl_win_rate: この場×この枠の1着率(累積) /
          vl_dev: 全場同枠の1着率との乖離(イン優位度の場差をモデルに明示)
[機材]    motor_recent_top2 / boat_recent_top2:
          この場のこのモーター/ボートの直近10走2連対率(期別成績より鮮度が高い)
[番組の質] rq_wr_edge: 自分の全国勝率-他艇の最大全国勝率(正=断然格上) /
          rq_class_edge: 級別ordの同差 / rq_n_a1: レース内A1級人数
          (番組表由来でレース前に確定している=リークなし・shift不要)

出力: test/backtest_report_racer_venue.html /
      test/verify_racer_venue_results.json / test/predictions_racer_venue.csv
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lightgbm as lgb
import pandas as pd

import challengers as CH
import db
import predictors as P
from backtest import N_FOLDS, PARAMS, TEST_START
from config import DB_PATH, JST, PROJECT_DIR, TARGET_VENUE_CODES, VENUE_NAMES
from features import (CATEGORICAL_FEATURES, CLASS_ORDER, FEATURE_COLUMNS,
                      build_training_set)

TEST_DIR = PROJECT_DIR / "test"

NEW_COLUMNS = [
    "rv_win_rate", "rv_top3_rate",            # 選手×場
    "vl_win_rate", "vl_dev",                  # 場×枠
    "motor_recent_top2", "boat_recent_top2",  # 機材の直近
    "rq_wr_edge", "rq_class_edge", "rq_n_a1", # 番組の質
]

# 1位勝率(生値)の層別境界。2026-07-18層別分析の丸め値で固定(成績での最適化はしない)
BANDS = [(0.00, 0.20, "〜20%"), (0.20, 0.25, "20〜25%"),
         (0.25, 0.30, "25〜30%"), (0.30, 0.35, "30〜35%")]

# 事前登録: 順当決着率の「改善」と認める最小低下幅(fold間σ≈5.1%相当)
JUNTO_IMPROVE_PT = 0.05


def band_of(top_prob: float) -> str:
    for lo, hi, label in BANDS:
        if lo <= top_prob < hi:
            return label
    return "35%〜"


def compute_new_features(conn) -> pd.DataFrame:
    """9個の新特徴量。(race_id, lane)キー。

    features.compute_form_featuresと同じ流儀: 時系列に並べてからshift(1)して
    集計するため、各行は「その行より前のレース」の実績しか参照しない。
    番組の質(rq_*)のみ番組表由来のレース内比較で、レース前に確定しておりshift不要。
    """
    h = pd.read_sql_query(
        """
        SELECT r.race_id, r.date, r.race_no, r.venue_code, e.lane, e.reg_no,
               e.motor_no, e.boat_no, e.racer_class, e.national_win_rate,
               res.arrival_order
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
        """,
        conn,
    )
    # 全体を時系列に並べる(groupbyは元の並びを保持するため各グループも時系列になる)
    h = h.sort_values(["date", "venue_code", "race_no"]).reset_index(drop=True)

    has_res = h["arrival_order"].notna()
    h["_win"] = (h["arrival_order"] == 1).astype(float).where(has_res)
    h["_top2"] = (h["arrival_order"] <= 2).astype(float).where(has_res)
    h["_top3"] = (h["arrival_order"] <= 3).astype(float).where(has_res)

    def roll(keys, col, window, minp):
        return h.groupby(keys, sort=False)[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=minp).mean())

    def expand(keys, col, minp):
        return h.groupby(keys, sort=False)[col].transform(
            lambda s: s.shift(1).expanding(min_periods=minp).mean())

    h["rv_win_rate"] = roll(["reg_no", "venue_code"], "_win", 30, 5)
    h["rv_top3_rate"] = roll(["reg_no", "venue_code"], "_top3", 30, 5)
    h["vl_win_rate"] = expand(["venue_code", "lane"], "_win", 30)
    h["vl_dev"] = h["vl_win_rate"] - expand(["lane"], "_win", 30)
    h["motor_recent_top2"] = roll(["venue_code", "motor_no"], "_top2", 10, 3)
    h["boat_recent_top2"] = roll(["venue_code", "boat_no"], "_top2", 10, 3)

    h["_cls"] = h["racer_class"].map(CLASS_ORDER)

    def edge_in_race(col: str) -> pd.Series:
        """自分の値 - 同レース他艇の最大値。同値トップが2艇なら双方0になる"""
        rank = h.groupby("race_id", sort=False)[col].rank(
            method="first", ascending=False)
        m1 = h[col].where(rank == 1).groupby(h["race_id"]).transform("max")
        m2 = h[col].where(rank == 2).groupby(h["race_id"]).transform("max")
        others_best = m1.where(rank != 1, m2)
        return h[col] - others_best

    h["rq_wr_edge"] = edge_in_race("national_win_rate")
    h["rq_class_edge"] = edge_in_race("_cls")
    h["rq_n_a1"] = ((h["racer_class"] == "A1").astype(float)
                    .groupby(h["race_id"]).transform("sum"))
    return h[["race_id", "lane", *NEW_COLUMNS]]


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


def grade(ctx: dict, actual: dict, payout_map: dict) -> dict:
    """1レースの紙上採点(検証⑪と同じ計算経路: V2構成・決着3分類)"""
    plan = P.ken_portfolio("荒れ注意", ctx["ranked"], [], P.picks_katsu(ctx["probs"]))
    pay = payout_map[ctx["race_id"]]
    stake = sum(y for _, _, y, _ in plan)
    ret = sum(pay.get((bt, comb), 0) * yen // 100 for bt, comb, yen, _ in plan)
    res = actual[ctx["race_id"]]
    santan = pay.get(("3連単", f"{res.get(1)}-{res.get(2)}-{res.get(3)}"), 0)
    return {**ctx, "stake": stake, "ret": ret,
            "outcome": CH.classify_outcome(stake, ret, santan)}


def select_races(ctxs: list[dict]) -> set[str]:
    """チャンピオン選別(1位勝率生値35%未満)+1日上限10。両モデル共通の規則"""
    selected: set[str] = set()
    ctxs = sorted(ctxs, key=lambda c: (c["date"], c["race_id"]))
    for _d, grp in groupby(ctxs, key=lambda c: c["date"]):
        cands = []
        for c in grp:
            score = CH.champion_score(c["ranked"])
            if score is not None:
                cands.append((score, c["race_id"]))
        selected.update(CH.daily_cap(cands))
    return selected


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    hits = [r for r in rows if r["ret"]]
    stake = sum(r["stake"] for r in rows)
    ret = sum(r["ret"] for r in rows)
    top = max(rows, key=lambda r: r["ret"], default=None)
    dist = {k: sum(1 for r in hits if r["outcome"] == k) for k in ("順当", "中波乱", "万舟")}
    out = {
        "n": n, "hits": len(hits), "hit_rate": len(hits) / n if n else 0.0,
        "stake": stake, "ret": ret, "profit": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "roi_excl_max": (ret - (top["ret"] if top else 0)) / stake if stake else 0.0,
        "dist": dist,
        "junto_rate": dist["順当"] / len(hits) if hits else 0.0,
    }
    if top and top["ret"]:
        out["max_hit"] = {"race": f"{top['date']} {VENUE_NAMES[top['venue_code']]}"
                                  f"{top['race_no']}R", "ret": top["ret"]}
    return out


def main():
    print("学習データ構築中...")
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)
    print("新特徴量算出中...")
    df = df.merge(compute_new_features(conn), on=["race_id", "lane"], how="left")

    actual = defaultdict(dict)
    for rid, lane, order in conn.execute(
        "SELECT res.race_id, res.lane, res.arrival_order FROM results res "
        "JOIN races r ON r.race_id = res.race_id "
        "WHERE r.date >= ? AND res.arrival_order IS NOT NULL", (TEST_START,),
    ):
        actual[rid][order] = lane
    payout_map = defaultdict(dict)
    for rid, bt, comb, amt in conn.execute(
        "SELECT p.race_id, p.bet_type, p.combination, p.amount_yen FROM payouts p "
        "JOIN races r ON r.race_id = p.race_id WHERE r.date >= ?", (TEST_START,),
    ):
        payout_map[rid][(bt, comb)] = amt or 0
    conn.close()

    test_df = df[df["date"] >= TEST_START]
    dates = sorted(test_df["date"].unique())
    fold_size = len(dates) // N_FOLDS
    boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

    variants = {"base": FEATURE_COLUMNS, "new": FEATURE_COLUMNS + NEW_COLUMNS}
    contexts: dict[str, list[dict]] = {v: [] for v in variants}
    eval_parts: dict[str, list[pd.DataFrame]] = {v: [] for v in variants}
    importances: list[pd.Series] = []

    for i in range(N_FOLDS):
        f_start, f_end = boundaries[i], boundaries[i + 1]
        train_df = df[df["date"] < f_start]
        fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)
                     & (df["venue_code"].isin(TARGET_VENUE_CODES))].copy()
        for name, cols in variants.items():
            print(f"fold{i+1} 学習中({name}: {len(cols)}特徴量, 学習{len(train_df):,}行)...")
            booster = train(train_df, cols)
            fold_df[f"pred_{name}"] = booster.predict(fold_df[cols])
            eval_parts[name].append(fold_df[["is_winner", f"pred_{name}"]]
                                    .rename(columns={f"pred_{name}": "pred"}))
            if name == "new":
                imp = pd.Series(booster.feature_importance("gain"),
                                index=booster.feature_name())
                importances.append(imp / imp.sum())

        for rid, g in fold_df.groupby("race_id"):
            if 1 not in actual[rid]:
                continue
            for name in variants:
                g_sorted = g.sort_values(f"pred_{name}", ascending=False)
                ranked = [{"lane": int(r["lane"]), "prob": float(r[f"pred_{name}"])}
                          for _, r in g_sorted.iterrows()]
                probs = P.normalize_probs(ranked)
                if len(probs) < 4:
                    continue
                contexts[name].append({
                    "race_id": rid, "date": str(g["date"].iloc[0]), "fold": i + 1,
                    "venue_code": int(g["venue_code"].iloc[0]),
                    "race_no": int(g["race_no"].iloc[0]),
                    "ranked": ranked, "probs": probs,
                    "top_prob": ranked[0]["prob"],
                })

    n_days = len({c["date"] for c in contexts["base"]})
    results = {"train_note": f"walk-forward {TEST_START}〜 {N_FOLDS}fold・5場・{n_days}日",
               "criteria": {
                   "1_junto": f"順当決着率がbase比-{JUNTO_IMPROVE_PT:.0%}pt以上低下",
                   "2_roi": "最大1発除き回収率がbaseを上回る",
                   "3_auc": "AUCは参考値",
               },
               "new_columns": NEW_COLUMNS}

    # 新特徴量の重要度(gain比率のfold平均と順位)
    imp_mean = pd.concat(importances, axis=1).mean(axis=1)
    imp_rank = imp_mean.rank(ascending=False).astype(int)
    results["importance"] = {
        c: {"gain_pct": float(imp_mean[c]), "rank": int(imp_rank[c]),
            "total": len(imp_mean)}
        for c in NEW_COLUMNS
    }

    graded: dict[str, dict[str, dict]] = {}
    selected: dict[str, set[str]] = {}
    for name in variants:
        sel = select_races(contexts[name])
        selected[name] = sel
        graded[name] = {c["race_id"]: grade(c, actual, payout_map)
                       for c in contexts[name] if c["race_id"] in sel}
        rows = list(graded[name].values())
        ev = pd.concat(eval_parts[name])
        total = agg(rows)
        total["per_day"] = total["n"] / n_days if n_days else 0.0
        total["auc"] = auc_score(ev["is_winner"], ev["pred"])
        total["bands"] = {}
        for _lo, _hi, label in BANDS:
            seg = [r for r in rows if band_of(r["top_prob"]) == label]
            b = agg(seg)
            b["share"] = b["n"] / total["n"] if total["n"] else 0.0
            total["bands"][label] = b
        total["folds"] = [agg([r for r in rows if r["fold"] == k])
                          for k in range(1, N_FOLDS + 1)]
        results[name] = total
        print(f"\n===== {name} =====")
        print(f"AUC {total['auc']:.4f} / 選別 {total['n']}R({total['per_day']:.1f}/日) "
              f"的中{total['hit_rate']:.1%} 回収率{total['roi']:.1%} "
              f"(最大1発除き{total['roi_excl_max']:.1%}) 順当率{total['junto_rate']:.1%} "
              f"損益{total['profit']:+,}円")
        for label, b in total["bands"].items():
            print(f"  {label}: {b['n']}R({b['share']:.0%}) 回収率{b['roi']:.1%} "
                  f"(除き{b['roi_excl_max']:.1%}) 順当率{b['junto_rate']:.1%}")

    # 差分レース: newが外したレース(B型が消えたか)とnewが新たに拾ったレース
    only_base = selected["base"] - selected["new"]
    only_new = selected["new"] - selected["base"]
    inter = len(selected["base"] & selected["new"])
    union = len(selected["base"] | selected["new"])
    results["diff"] = {
        "jaccard": inter / union if union else 0.0,
        "dropped": agg([graded["base"][r] for r in only_base]),   # base側の買い目で採点
        "added": agg([graded["new"][r] for r in only_new]),       # new側の買い目で採点
    }

    # 事前登録基準の判定
    b, n = results["base"], results["new"]
    results["verdict"] = {
        "crit1_junto": {"base": b["junto_rate"], "new": n["junto_rate"],
                        "pass": n["junto_rate"] <= b["junto_rate"] - JUNTO_IMPROVE_PT},
        "crit2_roi_excl_max": {"base": b["roi_excl_max"], "new": n["roi_excl_max"],
                               "pass": n["roi_excl_max"] > b["roi_excl_max"]},
        "ref_auc": {"base": b["auc"], "new": n["auc"]},
        "ref_purify_20pct": {"base": b["bands"]["〜20%"]["share"],
                             "new": n["bands"]["〜20%"]["share"]},
    }
    v = results["verdict"]
    n_pass = int(v["crit1_junto"]["pass"]) + int(v["crit2_roi_excl_max"]["pass"])
    results["verdict"]["overall"] = ("合格" if n_pass == 2 else
                                     "部分合格" if n_pass == 1 else "不合格")
    print(f"\n===== 事前登録基準の判定: {results['verdict']['overall']} =====")
    print(f"基準1 順当決着率: base {v['crit1_junto']['base']:.1%} → "
          f"new {v['crit1_junto']['new']:.1%} "
          f"({'合格' if v['crit1_junto']['pass'] else '不合格'})")
    print(f"基準2 最大1発除き回収率: base {v['crit2_roi_excl_max']['base']:.1%} → "
          f"new {v['crit2_roi_excl_max']['new']:.1%} "
          f"({'合格' if v['crit2_roi_excl_max']['pass'] else '不合格'})")
    print(f"参考 AUC: {v['ref_auc']['base']:.4f} → {v['ref_auc']['new']:.4f} / "
          f"〜20%帯シェア: {v['ref_purify_20pct']['base']:.1%} → "
          f"{v['ref_purify_20pct']['new']:.1%}")

    # CSV: 全レースの両モデル比較(8月末の層別深掘りにも使える生データ)
    by_rid = {c["race_id"]: c for c in contexts["base"]}
    rows = []
    for rid, cb in sorted(by_rid.items(), key=lambda x: (x[1]["date"], x[0])):
        cn = next((c for c in contexts["new"] if c["race_id"] == rid), None)
        if cn is None:
            continue
        res = actual[rid]
        gb, gn = graded["base"].get(rid), graded["new"].get(rid)
        rows.append({
            "race_id": rid, "date": cb["date"],
            "venue": VENUE_NAMES[cb["venue_code"]], "race_no": cb["race_no"],
            "top_prob_base": round(cb["top_prob"], 4),
            "top_prob_new": round(cn["top_prob"], 4),
            "band_base": band_of(cb["top_prob"]), "band_new": band_of(cn["top_prob"]),
            "sel_base": int(rid in selected["base"]),
            "sel_new": int(rid in selected["new"]),
            "result": "-".join(str(res.get(k, "?")) for k in (1, 2, 3)),
            "outcome_base": (gb or {}).get("outcome") or "",
            "outcome_new": (gn or {}).get("outcome") or "",
            "profit_base": gb["ret"] - gb["stake"] if gb else "",
            "profit_new": gn["ret"] - gn["stake"] if gn else "",
        })
    csv_path = TEST_DIR / "predictions_racer_venue.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV出力: {csv_path}")

    results["updated"] = datetime.now(JST).isoformat(timespec="seconds")
    json_path = TEST_DIR / "verify_racer_venue_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"集計JSON出力: {json_path}")

    (TEST_DIR / "backtest_report_racer_venue.html").write_text(
        render_report(results), encoding="utf-8")
    print(f"レポート出力: {TEST_DIR / 'backtest_report_racer_venue.html'}")


# ===== レポート =====

FEATURE_LABELS = {
    "rv_win_rate": "選手×場: 直近30走1着率",
    "rv_top3_rate": "選手×場: 直近30走3連対率",
    "vl_win_rate": "場×枠: 1着率(累積)",
    "vl_dev": "場×枠: 全場同枠との乖離",
    "motor_recent_top2": "モーター: 直近10走2連対率",
    "boat_recent_top2": "ボート: 直近10走2連対率",
    "rq_wr_edge": "番組の質: 全国勝率の他艇最大との差",
    "rq_class_edge": "番組の質: 級別の他艇最大との差",
    "rq_n_a1": "番組の質: レース内A1人数",
}


def _row(label: str, s: dict, extra: str = "") -> str:
    roi_cls = "pos" if s["roi"] >= 1 else "neg"
    ex_cls = "pos" if s["roi_excl_max"] >= 1 else "neg"
    d = s["dist"]
    return (f"<tr><td>{label}</td><td class='num'>{s['n']}</td>"
            f"<td class='num'>{s['hit_rate']:.1%}</td>"
            f"<td class='num {roi_cls}'>{s['roi']:.1%}</td>"
            f"<td class='num {ex_cls}'>{s['roi_excl_max']:.1%}</td>"
            f"<td class='num'>{s['junto_rate']:.1%}</td>"
            f"<td class='num'>{d['順当']}/{d['中波乱']}/{d['万舟']}</td>"
            f"<td class='num'>{s['profit']:+,}円</td>{extra}</tr>")


def render_report(r: dict) -> str:
    v = r["verdict"]
    verdict_color = {"合格": "#1a7f37", "部分合格": "#9a6700", "不合格": "#cf222e"}
    main_rows = "".join(
        _row(name, r[name]) for name in ("base", "new"))

    band_rows = []
    for name in ("base", "new"):
        for label, b in r[name]["bands"].items():
            if not b["n"]:
                continue
            band_rows.append(
                f"<tr><td>{name}</td><td>{label}</td>"
                f"<td class='num'>{b['n']} ({b['share']:.0%})</td>"
                f"<td class='num'>{b['hit_rate']:.1%}</td>"
                f"<td class='num {'pos' if b['roi'] >= 1 else 'neg'}'>{b['roi']:.1%}</td>"
                f"<td class='num'>{b['roi_excl_max']:.1%}</td>"
                f"<td class='num'>{b['junto_rate']:.1%}</td></tr>")

    fold_rows = []
    for name in ("base", "new"):
        for i, s in enumerate(r[name]["folds"], 1):
            if not s["n"]:
                continue
            fold_rows.append(
                f"<tr><td>fold{i}</td><td>{name}</td><td class='num'>{s['n']}</td>"
                f"<td class='num'>{s['hit_rate']:.1%}</td>"
                f"<td class='num {'pos' if s['roi'] >= 1 else 'neg'}'>{s['roi']:.1%}</td>"
                f"<td class='num'>{s['junto_rate']:.1%}</td></tr>")

    imp_rows = "".join(
        f"<tr><td>{c}</td><td>{FEATURE_LABELS[c]}</td>"
        f"<td class='num'>{d['gain_pct']:.2%}</td>"
        f"<td class='num'>{d['rank']}/{d['total']}</td></tr>"
        for c, d in sorted(r["importance"].items(),
                           key=lambda x: x[1]["rank"]))

    diff = r["diff"]
    drop, add = diff["dropped"], diff["added"]

    def _cell(x):
        return (f"{x['n']}R / 回収率{x['roi']:.1%} / 順当率{x['junto_rate']:.1%}"
                if x["n"] else "0R")

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>検証⑫ 選手×場の特徴量強化(紙上)</title>
<style>
  body {{ font-family: sans-serif; margin: 0 auto; padding: 12px; background: #f6f8fa; max-width: 900px; }}
  h1 {{ font-size: 1.2rem; margin: 10px 4px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 14px; margin-bottom: 14px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #f6f8fa; text-align: left; padding: 6px; border-bottom: 2px solid #d0d7de; }}
  td {{ padding: 6px; border-bottom: 1px solid #eee; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pos {{ color: #1a7f37; font-weight: bold; }}
  .neg {{ color: #cf222e; }}
  .note {{ font-size: .78rem; color: #57606a; margin: 6px 4px; }}
  .verdict {{ font-size: 1.05rem; font-weight: bold; padding: 10px 14px; border-radius: 8px;
             background: #fff; border-left: 6px solid {verdict_color[v['overall']]};
             margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
</style>
</head>
<body>
<h1>検証⑫ 選手×場の特徴量強化(すべて紙上・本番はv1のまま)</h1>
<p class="note">生成: {r['updated']} / 枠組み: {r['train_note']}
(backtest.pyと同一fold・選別=チャンピオン・買い目=検証済みV2構成で統一。
変数は特徴量のみ)。導入判断は2026-07-30〜31のv2判断会。</p>

<div class="verdict">事前登録基準の判定:
<span style="color:{verdict_color[v['overall']]}">{v['overall']}</span></div>

<div class="card">
  <h2 style="margin-top:0">事前登録の合格基準(実行前に固定・AUCではない)</h2>
  <table>
    <tr><th>基準</th><th>base</th><th>new</th><th>判定</th></tr>
    <tr><td>1. 順当決着率(的中のうち払戻&lt;掛金)がbase比-5pt以上低下<br>
      <span class="note">中心課題「荒れるはず→順当だった」の直接指標。-5ptはfold間σ≈5.1%相当</span></td>
      <td class="num">{v['crit1_junto']['base']:.1%}</td>
      <td class="num">{v['crit1_junto']['new']:.1%}</td>
      <td class="{'pos' if v['crit1_junto']['pass'] else 'neg'}">
        {'合格' if v['crit1_junto']['pass'] else '不合格'}</td></tr>
    <tr><td>2. 最大1発除き回収率がbaseを上回る</td>
      <td class="num">{v['crit2_roi_excl_max']['base']:.1%}</td>
      <td class="num">{v['crit2_roi_excl_max']['new']:.1%}</td>
      <td class="{'pos' if v['crit2_roi_excl_max']['pass'] else 'neg'}">
        {'合格' if v['crit2_roi_excl_max']['pass'] else '不合格'}</td></tr>
    <tr><td>3. AUC(参考値・合否に使わない)</td>
      <td class="num">{v['ref_auc']['base']:.4f}</td>
      <td class="num">{v['ref_auc']['new']:.4f}</td><td>—</td></tr>
    <tr><td>補助: 〜20%帯(本物の混戦)の選別シェア=純化の確認</td>
      <td class="num">{v['ref_purify_20pct']['base']:.1%}</td>
      <td class="num">{v['ref_purify_20pct']['new']:.1%}</td><td>—</td></tr>
  </table>
</div>

<div class="card">
  <h2 style="margin-top:0">荒れ注意選別の全fold合計</h2>
  <table>
    <tr><th>モデル</th><th class="num">選別数</th><th class="num">的中率</th>
        <th class="num">回収率</th><th class="num">最大1発除き</th>
        <th class="num">順当決着率</th><th class="num">順当/中波乱/万舟</th>
        <th class="num">損益</th></tr>
    {main_rows}
  </table>
  <p class="note">base最大1発: {r['base'].get('max_hit', {}).get('race', '—')}
  {r['base'].get('max_hit', {}).get('ret', 0):,}円 /
  new最大1発: {r['new'].get('max_hit', {}).get('race', '—')}
  {r['new'].get('max_hit', {}).get('ret', 0):,}円</p>
</div>

<div class="card">
  <h2 style="margin-top:0">1位勝率帯の層別(境界20/25/30/35%固定)</h2>
  <table>
    <tr><th>モデル</th><th>帯</th><th class="num">選別数(シェア)</th>
        <th class="num">的中率</th><th class="num">回収率</th>
        <th class="num">最大1発除き</th><th class="num">順当決着率</th></tr>
    {''.join(band_rows)}
  </table>
  <p class="note">狙い: newの選別が〜20%帯(エッジの本体)に寄り、
  20〜35%帯(B型=モデルの無知)が減っていれば純化に成功。</p>
</div>

<div class="card">
  <h2 style="margin-top:0">差分レース(newは何を外し、何を拾ったか)</h2>
  <table>
    <tr><th>区分</th><th>成績</th></tr>
    <tr><td>baseのみが選んだ(newが外した)レース<br>
      <span class="note">ここの順当率が高ければ「B型を正しく外せた」証拠(base側買い目で採点)</span></td>
      <td class="num">{_cell(drop)}</td></tr>
    <tr><td>newのみが選んだレース(new側買い目で採点)</td>
      <td class="num">{_cell(add)}</td></tr>
  </table>
  <p class="note">選別の重なり(Jaccard): {diff['jaccard']:.2f}</p>
</div>

<div class="card">
  <h2 style="margin-top:0">新特徴量の重要度(gain比率・5fold平均)</h2>
  <table>
    <tr><th>特徴量</th><th>内容</th><th class="num">gain比率</th><th class="num">順位</th></tr>
    {imp_rows}
  </table>
  <p class="note">過去の教訓(気象・進入・潮汐): 「重要度は付くが回収率悪化」がありうるため、
  重要度は採否の根拠にしない。</p>
</div>

<div class="card">
  <h2 style="margin-top:0">fold別</h2>
  <table>
    <tr><th>fold</th><th>モデル</th><th class="num">選別数</th>
        <th class="num">的中率</th><th class="num">回収率</th><th class="num">順当決着率</th></tr>
    {''.join(fold_rows)}
  </table>
</div>

<p class="note">検算: baseは検証⑪-Aチャンピオン(180.3%/最大1発除き142.5%/順当52.3%)と
同一計算経路。DBがその後の日次収集で伸びているため完全一致はしないが同水準になることを
確認して整合とみなす。<br>
再実行: py -X utf8 test/verify_racer_venue.py(冪等・全期間を毎回再計算)</p>
</body>
</html>
"""


if __name__ == "__main__":
    main()
