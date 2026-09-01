"""1着確率モデルと3着内モデル(超混戦専用順位)を学習するCLI

    python train_model.py

日付順に並べ、直近15%を検証用に分割する(時系列分割でリークを防ぐ)。
3着内モデルはv2.2の並走機能(2026-09-01ケンさん承認): 超混戦タブの
レースだけ「3着以内に絡む艇」の順で並べ替えた専用順位を表示用に出す。
根拠: test/sim_konsen_top3_model.py(軸生存34.3%→41.3%・事前登録基準クリア)。
"""
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import db
from config import DB_PATH, MODEL_PATH, MODEL_TOP3_PATH
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

    # --- 3着内モデル(超混戦専用順位・表示用) ---
    df["is_top3"] = (df["arrival_order"] <= 3).astype(int)
    t3_train = lgb.Dataset(train_df[FEATURE_COLUMNS],
                           label=df.loc[train_df.index, "is_top3"],
                           categorical_feature=CATEGORICAL_FEATURES)
    t3_valid = lgb.Dataset(valid_df[FEATURE_COLUMNS],
                           label=df.loc[valid_df.index, "is_top3"],
                           reference=t3_train)
    booster_t3 = lgb.train(
        params, t3_train,
        valid_sets=[t3_valid],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    auc_t3 = roc_auc_score(df.loc[valid_df.index, "is_top3"],
                           booster_t3.predict(X_valid))
    print(f"3着内モデル 検証AUC: {auc_t3:.4f}")
    MODEL_TOP3_PATH.write_text(booster_t3.model_to_string(), encoding="utf-8")
    print(f"3着内モデルを保存しました: {MODEL_TOP3_PATH}")


if __name__ == "__main__":
    main()
