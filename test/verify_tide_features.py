# -*- coding: utf-8 -*-
"""検証⑩: 潮汐特徴量の有効性検証(検証のみ・本番導入禁止)

    py -X utf8 test/fetch_tide.py          # 先に潮位データを取得しておく
    py -X utf8 test/verify_tide_features.py

場マスタ調査に基づく事前登録予測(レポートにも明記):
- 江戸川(3)=汽水・河川 / 平和島(4)=海水・運河 / 若松(20)=海水・洞海湾直結
  → 干満の影響を受ける「潮汐場」
- 常滑(8)=水門で干満皆無 / 尼崎(13)=淡水プール → 影響を受けない「対照場」
- 潮汐特徴量が本物なら効果は潮汐3場でのみ現れ、対照2場では現れないはず。
  5場一様に"効いた/悪化した"場合はノイズと判定する。

枠組みは検証⑦⑧⑨と同一: 2026-05-01より前で学習 → 2026-05〜06の対象5場を評価。
結果が良くても本番導入はしない(2026-09-01のv2判断材料。features.py本体は不変)。

追加する特徴量(締切時刻の潮位。潮汐は天文計算の推算値なので事前に確定しており
リークなし。対象5場以外の場はNaN=LightGBMの欠損分岐に任せる):
- tide_level:     締切時刻の潮位(cm)
- tide_delta:     締切時刻から1時間の潮位変化(cm/h)。正=上げ潮、負=下げ潮
- tide_range_day: その日の潮位差(最高-最低、cm)。大潮/小潮の代理変数
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
from fetch_tide import VENUE_STATION

EVAL_START, EVAL_END = "2026-05-01", "2026-06-30"
TEST_DIR = PROJECT_DIR / "test"

TIDE_COLUMNS = ["tide_level", "tide_delta", "tide_range_day"]
TIDAL_VENUES = [3, 4, 20]    # 事前登録: 効果が出るならここだけ
CONTROL_VENUES = [8, 13]     # 対照群: ここで"効いた"らノイズの証拠


def tide_features_for(deadline: str | None, venue_code: int,
                      tide_map: dict[tuple[str, str], float]) -> dict:
    """1レース分の潮汐特徴量。地点対応がない場・欠損は空dict(=NaN)"""
    station = VENUE_STATION.get(venue_code)
    if not station or not deadline or len(deadline) < 13:
        return {}
    day = deadline[:10]
    hour = int(deadline[11:13])

    def level(h: int) -> float | None:
        return tide_map.get((station, f"{day} {h:02d}:00:00"))

    out = {}
    lv = level(hour)
    if lv is not None:
        out["tide_level"] = lv
        nxt = level(hour + 1) if hour < 23 else None
        if nxt is not None:
            out["tide_delta"] = nxt - lv
    day_levels = [x for x in (level(h) for h in range(24)) if x is not None]
    if day_levels:
        out["tide_range_day"] = max(day_levels) - min(day_levels)
    return out


def build_tide_df(conn) -> pd.DataFrame:
    """全レースの潮汐特徴量(レース単位)。地点対応のない場は行を作らない"""
    tide_map = {(st, dt): lv for st, dt, lv in
                conn.execute("SELECT station, datetime, level_cm FROM tide")}
    rows = []
    for rid, vc, dl in conn.execute(
        "SELECT race_id, venue_code, deadline_time FROM races"
    ):
        feats = tide_features_for(dl, vc, tide_map)
        if feats:
            rows.append({"race_id": rid, **feats})
    return pd.DataFrame(rows)


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
    r = p.rank()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    print("学習データ構築中...")
    conn = db.connect(DB_PATH)
    n_tide = conn.execute("SELECT COUNT(*) FROM tide").fetchone()[0]
    if not n_tide:
        print("tideテーブルが空です。先に test/fetch_tide.py を実行してください。")
        conn.close()
        return
    df = build_training_set(conn)
    tide_df = build_tide_df(conn)
    df = df.merge(tide_df, on="race_id", how="left")

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
    cover = eval_df["tide_level"].notna().mean()
    print(f"学習 {len(train_df):,}行 / 評価 {len(eval_df):,}行 / 評価行の潮位充足率 {cover:.1%}")

    variants = {"base": FEATURE_COLUMNS, "tide": FEATURE_COLUMNS + TIDE_COLUMNS}
    boosters = {}
    for name, cols in variants.items():
        print(f"モデル学習中({name}: {len(cols)}特徴量)...")
        boosters[name] = train(train_df, cols)
        eval_df[f"pred_{name}"] = boosters[name].predict(eval_df[cols])

    results = {"train_rows": len(train_df), "tide_coverage": cover,
               "hypothesis": {"tidal": TIDAL_VENUES, "control": CONTROL_VENUES}}

    bst = boosters["tide"]
    imp = pd.Series(bst.feature_importance("gain"), index=bst.feature_name())
    imp_rank = imp.rank(ascending=False).astype(int)
    results["importance"] = {
        c: {"gain_pct": float(imp[c] / imp.sum()), "rank": int(imp_rank[c]),
            "total": len(imp)}
        for c in TIDE_COLUMNS
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
            races.append({
                "race_id": rid, "date": str(g["date"].iloc[0]),
                "venue_code": int(g["venue_code"].iloc[0]),
                "ranked": ranked, "bets": {"confidence": conf, "plan": plan},
                "stake": sum(y for _, _, y, _ in plan),
                "return": sum(payout_map[rid].get((bt, comb), 0) * yen // 100
                              for bt, comb, yen, _ in plan),
            })
        races.sort(key=lambda r: (r["date"], r["venue_code"]))
        for d, grp in groupby(races, key=lambda r: r["date"]):
            P.select_shobusho(list(grp), max_races=10)

        def agg(rs):
            n = len(rs)
            stake = sum(r["stake"] for r in rs)
            ret = sum(r["return"] for r in rs)
            return {"n": n, "hits": sum(1 for r in rs if r["return"]),
                    "stake": stake, "ret": ret,
                    "roi": ret / stake if stake else 0.0}

        # 事前登録予測の検証: 場別AUC(行レベル)と場別の荒れ注意ROI
        by_venue = {}
        for vc in TARGET_VENUE_CODES:
            seg = eval_df[eval_df["venue_code"] == vc]
            v_races = [r for r in races if r["venue_code"] == vc]
            by_venue[vc] = {
                "auc": auc_score(seg["is_winner"], seg[pred_col]),
                "are": agg([r for r in v_races
                            if r["bets"]["confidence"] == "荒れ注意" and r["bets"]["plan"]]),
            }
        return {
            "auc": auc_score(eval_df["is_winner"], eval_df[pred_col]),
            "honmei": agg([r for r in races if r["shobusho"] == "本命"]),
            "are_all": agg([r for r in races
                            if r["bets"]["confidence"] == "荒れ注意" and r["bets"]["plan"]]),
            "by_venue": by_venue,
        }

    for name in variants:
        results[name] = evaluate(name)
        r = results[name]
        print(f"\n===== {name} =====")
        print(f"AUC {r['auc']:.4f} / 本命勝負所 {r['honmei']['n']}R "
              f"回収率{r['honmei']['roi']:.1%} 損益{r['honmei']['ret']-r['honmei']['stake']:+,}円")
        for vc, s in r["by_venue"].items():
            print(f"  {VENUE_NAMES[vc]}: AUC {s['auc']:.4f} / "
                  f"荒れ注意 {s['are']['n']}R 回収率{s['are']['roi']:.1%}")

    # 事前登録予測の判定サマリー(場別AUC差)
    print("\n===== 場別AUC差(tide - base) =====")
    for vc in TARGET_VENUE_CODES:
        d_auc = results["tide"]["by_venue"][vc]["auc"] - results["base"]["by_venue"][vc]["auc"]
        group = "潮汐場" if vc in TIDAL_VENUES else "対照場"
        print(f"  {VENUE_NAMES[vc]}({group}): {d_auc:+.4f}")

    json_path = TEST_DIR / "verify_tide_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"\n集計JSON出力: {json_path}")


if __name__ == "__main__":
    main()
