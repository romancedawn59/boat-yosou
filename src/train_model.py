"""1着確率を予測するLightGBMモデルを学習するCLI

    python train_model.py

日付順に並べ、直近15%を検証用に分割する(時系列分割でリークを防ぐ)。
"""
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import db
from config import DB_PATH, MODEL_PATH
from features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_set

MIN_TRAINING_ROWS = 1000
VALID_FRACTION = 0.15


def main():
    conn = db.connect(DB_PATH)
    df = build_training_set(conn)
    conn.close()

    if len(df) < MIN_TRAINING_ROWS:
        print(f"学習データが少なすぎます({len(df)}件)。collect.pyでデータを増やしてください。")
        return

    df = df.sort_values("date").reset_index(drop=True)
    cutoff = df["date"].iloc[int(len(df) * (1 - VALID_FRACTION))]
    train_df = df[df["date"] < cutoff]
    valid_df = df[df["date"] >= cutoff]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["is_winner"]
    X_valid, y_valid = valid_df[FEATURE_COLUMNS], valid_df["is_winner"]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 31,
    }
    booster = lgb.train(
        params, train_set,
        valid_sets=[valid_set],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    pred = booster.predict(X_valid)
    auc = roc_auc_score(y_valid, pred)

    valid_df = valid_df.copy()
    valid_df["pred"] = pred
    top1 = valid_df.loc[valid_df.groupby("race_id")["pred"].idxmax()]
    hit_rate = top1["is_winner"].mean()

    print(f"学習件数: {len(train_df)}  検証件数: {len(valid_df)}  "
          f"(検証期間: {cutoff} 〜 {df['date'].max()})")
    print(f"検証AUC: {auc:.4f}")
    print(f"検証データでの単勝的中率(モデルの最高確率艇が実際に1着だった割合): {hit_rate:.1%}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # LightGBMネイティブのsave_modelは日本語を含むパスに書けないため、Python側で書き込む
    MODEL_PATH.write_text(booster.model_to_string(), encoding="utf-8")
    print(f"モデルを保存しました: {MODEL_PATH}")


if __name__ == "__main__":
    main()
