"""ウォークフォワード検証CLI: 買い方別の的中率・回収率を複数期間で検証する

    python backtest.py

2025-12-01以降をN_FOLDS個の期間に分割し、各期間について
「その期間より前のデータだけで学習したモデル」で予測して買い目のROIを計測する。
予算は1レース1000円、100円単位で均等配分(余りは先頭の目に上乗せ)、再投資なし。
"""
from collections import defaultdict

import lightgbm as lgb
import pandas as pd

import db
from config import DB_PATH
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set

TEST_START = "2025-12-01"
N_FOLDS = 5
BUDGET = 1000

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
}

BUCKETS = ("堅め", "標準", "荒れ注意")


def bucket_of(top_prob: float) -> str:
    return "堅め" if top_prob >= 0.5 else ("標準" if top_prob >= 0.35 else "荒れ注意")


def allocate(n: int) -> list[int]:
    """予算を n 点に100円単位で配分(先頭優先で上乗せ)"""
    base = BUDGET // n // 100 * 100
    rest = BUDGET - base * n
    alloc = []
    for _ in range(n):
        extra = 100 if rest > 0 else 0
        rest -= extra
        alloc.append(base + extra)
    return alloc


def strategies(r: list[int]) -> dict[str, list[tuple[str, str]]]:
    """予測勝率降順の枠番リスト -> {戦略名: [(券種, 組み合わせ), ...]}"""
    def ex(a, b): return ("2連単", f"{a}-{b}")
    def qn(a, b): return ("2連複", f"{min(a, b)}={max(a, b)}")
    def tf(a, b, c): return ("3連単", f"{a}-{b}-{c}")
    def tr(a, b, c):
        s = sorted([a, b, c])
        return ("3連複", f"{s[0]}={s[1]}={s[2]}")

    if len(r) < 4:
        return {}
    r1, r2, r3, r4 = r[:4]
    return {
        "2連複1点(1=2位)": [qn(r1, r2)],
        "3連単1点(1-2-3位)": [tf(r1, r2, r3)],
        "3連複軸1流し3点": [tr(r1, r2, r3), tr(r1, r2, r4), tr(r1, r3, r4)],
        "3連単F6点": [tf(r1, a, b) for a in (r2, r3, r4) for b in (r2, r3, r4) if a != b],
        "3連単穴2点(3,4位頭)": [tf(r3, r1, r2), tf(r4, r1, r2)],
        "推奨:3連複流し600+穴400": [
            (*tr(r1, r2, r3),), (*tr(r1, r2, r4),), (*tr(r1, r3, r4),),
            tf(r3, r1, r2), tf(r4, r1, r2),
        ],
    }


def train_fold(train_df: pd.DataFrame) -> lgb.Booster:
    """フォールド内の末尾10%を早期停止用に使って学習"""
    train_df = train_df.sort_values("date")
    cutoff = train_df["date"].iloc[int(len(train_df) * 0.9)]
    tr, va = train_df[train_df["date"] < cutoff], train_df[train_df["date"] >= cutoff]
    train_set = lgb.Dataset(tr[FEATURE_COLUMNS], label=tr["is_winner"],
                            categorical_feature=CATEGORICAL_FEATURES)
    valid_set = lgb.Dataset(va[FEATURE_COLUMNS], label=va["is_winner"], reference=train_set)
    return lgb.train(PARAMS, train_set, valid_sets=[valid_set], num_boost_round=500,
                     callbacks=[lgb.early_stopping(30, verbose=False)])


def main():
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)

    test_df = df[df["date"] >= TEST_START]

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

    dates = sorted(test_df["date"].unique())
    fold_size = len(dates) // N_FOLDS
    boundaries = [dates[i * fold_size] for i in range(N_FOLDS)] + [dates[-1] + "z"]

    # {(バケット, 戦略): [投資, 回収, レース数, 的中数]}
    total = defaultdict(lambda: [0, 0, 0, 0])

    for i in range(N_FOLDS):
        f_start, f_end = boundaries[i], boundaries[i + 1]
        train_df = df[df["date"] < f_start]
        fold_df = df[(df["date"] >= f_start) & (df["date"] < f_end)].copy()

        booster = train_fold(train_df)
        fold_df["pred"] = booster.predict(fold_df[FEATURE_COLUMNS])

        fold_stat = defaultdict(lambda: [0, 0, 0, 0])
        for rid, g in fold_df.groupby("race_id"):
            if 1 not in actual[rid]:
                continue
            ranked = g.sort_values("pred", ascending=False)["lane"].astype(int).tolist()
            b = bucket_of(g["pred"].max())
            for name, bets in strategies(ranked).items():
                alloc = allocate(len(bets))
                ret = sum(
                    payout_map[rid].get((bt, comb), 0) * stake // 100
                    for (bt, comb), stake in zip(bets, alloc)
                )
                for stat in (fold_stat, total):
                    s = stat[(b, name)]
                    s[0] += sum(alloc)
                    s[1] += ret
                    s[2] += 1
                    s[3] += 1 if ret else 0

        period = f"{f_start}〜{fold_df['date'].max()}"
        n_are = fold_stat[('荒れ注意', '3連複軸1流し3点')][2]
        print(f"\n--- fold{i+1} {period} (学習 {len(train_df):,}行 / 荒れ注意 {n_are}レース) ---")
        for name in ("3連複軸1流し3点", "3連単F6点", "3連単穴2点(3,4位頭)", "推奨:3連複流し600+穴400"):
            s = fold_stat[("荒れ注意", name)]
            if s[0]:
                print(f"  荒れ注意 {name:<22} 的中 {s[3]/s[2]:6.1%}  回収率 {s[1]/s[0]:7.1%}")

    print("\n===== 全fold合計 (バケット × 戦略) =====")
    for b in BUCKETS:
        n = next((total[(b, k)][2] for k in strategies([1, 2, 3, 4]) if total[(b, k)][2]), 0)
        print(f"\n[{b}] {n}レース")
        for name in strategies([1, 2, 3, 4]):
            stake, ret, races, hits = total[(b, name)]
            if stake:
                print(f"  {name:<24} 的中 {hits/races:6.1%}  回収率 {ret/stake:7.1%}  損益 {ret-stake:+,}円")


if __name__ == "__main__":
    main()
