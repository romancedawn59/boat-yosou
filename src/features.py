"""DBの行 -> LightGBM学習・予測用の特徴量への変換

学習に使う列は「予測時点(レース前)に知り得る情報」だけに限定する。
- 番組表由来: 選手・モーター・ボートの期別成績等
- 自前DBのresultsから算出するローリング成績: 当該レースより前のレースのみを
  参照する(shiftしてから集計)ことでリークを防ぐ
気象(風速・波高等)はresults確定後の実測値でバックテストしたところ、重要度0.1〜0.4%
(31特徴量中23〜28位)と極めて弱く、推奨買い目構成の回収率がむしろ悪化した(122.0%→119.2%、
5fold中1つが100%割れ)ため特徴量には含めない。ただしOpen-Meteo予報(weather.py)は
レポート表示用の参考情報としてpredict.py側で別途利用する。
進入コース等、results由来の当日情報は予測時に代替が効かないため含めない。
"""
import pandas as pd

GRADE_ORDER = {"一般": 0, "G3": 1, "G2": 2, "G1": 3, "SG": 4}
CLASS_ORDER = {"B2": 0, "B1": 1, "A2": 2, "A1": 3}

FORM_COLUMNS = [
    "form_last10_win_rate",   # 直近10走の1着率
    "form_last10_top2_rate",  # 直近10走の2連対率
    "form_last10_avg_finish", # 直近10走の平均着順
    "form_last10_avg_st",     # 直近10走の平均ST
    "form_last3_win_rate",    # 直近3走の1着率(短期の勢い)
    "form_lane_win_rate",     # この枠番からの通算1着率(DB保持期間内)
    "form_days_since_last",   # 前走からの日数
]

FEATURE_COLUMNS = [
    "lane", "venue_code", "race_no", "racer_class_ord", "grade_ord", "distance_m",
    "age", "weight_kg", "flying_count", "late_count", "avg_st",
    "national_win_rate", "national_2rate", "national_3rate",
    "local_win_rate", "local_2rate", "local_3rate",
    "motor_2rate", "motor_3rate", "boat_2rate", "boat_3rate",
    *FORM_COLUMNS,
]

# 戸田の企画レース(5R/7R)のようにレース番号で1号艇の信頼度が激変するため、
# race_noは順序ではなくカテゴリとして扱う(場×レース番号の組で効かせる)
CATEGORICAL_FEATURES = ["venue_code", "race_no"]

_ENTRY_COLS = """
    e.lane, e.reg_no, e.racer_name, e.racer_class, e.age, e.weight_kg,
    e.flying_count, e.late_count, e.avg_st,
    e.national_win_rate, e.national_2rate, e.national_3rate,
    e.local_win_rate, e.local_2rate, e.local_3rate,
    e.motor_2rate, e.motor_3rate, e.boat_2rate, e.boat_3rate
"""


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["racer_class_ord"] = df["racer_class"].map(CLASS_ORDER)
    df["grade_ord"] = df["grade"].map(GRADE_ORDER)
    return df


def compute_form_features(conn) -> pd.DataFrame:
    """全出走履歴から選手ごとのローリング成績を算出する。

    戻り値は(race_id, lane)をキーにFORM_COLUMNSを持つDataFrame。
    各行の値は「その行のレースより前」の出走だけから計算される(shift(1)済み)ため、
    結果未確定の未来のレース行に対しても安全に使える。
    """
    h = pd.read_sql_query(
        """
        SELECT r.race_id, r.date, r.race_no, e.lane, e.reg_no,
               res.arrival_order, res.st_time
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        LEFT JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
        """,
        conn,
    )
    h["date_dt"] = pd.to_datetime(h["date"])
    h = h.sort_values(["reg_no", "date", "race_no"]).reset_index(drop=True)

    has_result = h["arrival_order"].notna()
    h["_win"] = (h["arrival_order"] == 1).astype(float).where(has_result)
    h["_top2"] = (h["arrival_order"] <= 2).astype(float).where(has_result)

    g = h.groupby("reg_no", sort=False)

    def prev_rolling_mean(col: str, window: int, min_periods: int) -> pd.Series:
        return g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
        )

    h["form_last10_win_rate"] = prev_rolling_mean("_win", 10, 3)
    h["form_last10_top2_rate"] = prev_rolling_mean("_top2", 10, 3)
    h["form_last10_avg_finish"] = prev_rolling_mean("arrival_order", 10, 3)
    h["form_last10_avg_st"] = prev_rolling_mean("st_time", 10, 3)
    h["form_last3_win_rate"] = prev_rolling_mean("_win", 3, 1)
    h["form_days_since_last"] = g["date_dt"].transform(lambda s: s.diff().dt.days)
    h["form_lane_win_rate"] = h.groupby(["reg_no", "lane"], sort=False)["_win"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean()
    )

    return h[["race_id", "lane", *FORM_COLUMNS]]


def build_training_set(conn) -> pd.DataFrame:
    """着順が確定している(results がある)レースのみ、ラベル(is_winner)付きで返す"""
    query = f"""
        SELECT
            r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
            {_ENTRY_COLS},
            res.arrival_order
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        JOIN results res ON res.race_id = e.race_id AND res.lane = e.lane
        WHERE res.arrival_order IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    df = _encode(df)
    df = df.merge(compute_form_features(conn), on=["race_id", "lane"], how="left")
    df["is_winner"] = (df["arrival_order"] == 1).astype(int)
    return df


def build_program_features(conn, race_ids: list[str]) -> pd.DataFrame:
    """指定レースIDの番組表由来特徴量(ラベルなし、予測時に使用)"""
    if not race_ids:
        return pd.DataFrame(columns=["race_id", "race_no", *FEATURE_COLUMNS])

    placeholders = ",".join("?" for _ in race_ids)
    query = f"""
        SELECT
            r.race_id, r.date, r.venue_code, r.race_no, r.grade, r.distance_m,
            r.wind_speed_m, r.wave_height_cm, r.temperature,
            {_ENTRY_COLS}
        FROM entries e
        JOIN races r ON r.race_id = e.race_id
        WHERE r.race_id IN ({placeholders})
        ORDER BY r.race_no, e.lane
    """
    df = pd.read_sql_query(query, conn, params=race_ids)
    df = _encode(df)
    return df.merge(compute_form_features(conn), on=["race_id", "lane"], how="left")
