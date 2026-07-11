"""SQLiteスキーマ定義とDB接続・UPSERTヘルパー"""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id           TEXT PRIMARY KEY,   -- YYYYMMDD_{venue_code:02d}_{race_no:02d}
    date              TEXT NOT NULL,      -- YYYY-MM-DD
    venue_code        INTEGER NOT NULL,
    race_no           INTEGER NOT NULL,
    title             TEXT,
    subtitle          TEXT,               -- 予選/準優勝戦/優勝戦 等
    grade             TEXT,               -- SG/G1/G2/G3/一般
    day_label         TEXT,               -- 初日/2日目/最終日 等
    distance_m        INTEGER,
    deadline_time     TEXT,               -- 締切予定時刻 YYYY-MM-DD HH:MM:SS
    -- 以下はresults側から埋まる(番組表のみの日はNULL)
    weather_number    INTEGER,            -- 天候コード
    wind_speed_m      REAL,
    wind_direction_number INTEGER,        -- 風向コード
    wave_height_cm    REAL,
    temperature       REAL,
    water_temperature REAL,
    winning_technique_number INTEGER      -- 決まり手コード
);

CREATE TABLE IF NOT EXISTS entries (
    race_id           TEXT NOT NULL REFERENCES races(race_id),
    lane              INTEGER NOT NULL,  -- 枠番 1-6
    reg_no            INTEGER NOT NULL,  -- 選手登録番号
    racer_name        TEXT,
    racer_class       TEXT,              -- A1/A2/B1/B2
    branch_number     INTEGER,           -- 支部(JIS都道府県コード)
    birthplace_number INTEGER,           -- 出身地(JIS都道府県コード)
    age               INTEGER,
    weight_kg         REAL,
    flying_count      INTEGER,           -- 当期F数
    late_count        INTEGER,           -- 当期L数
    avg_st            REAL,
    national_win_rate REAL,
    national_2rate    REAL,
    national_3rate    REAL,
    local_win_rate    REAL,
    local_2rate       REAL,
    local_3rate       REAL,
    motor_no          INTEGER,
    motor_2rate       REAL,
    motor_3rate       REAL,
    boat_no           INTEGER,
    boat_2rate        REAL,
    boat_3rate        REAL,
    PRIMARY KEY (race_id, lane)
);

CREATE TABLE IF NOT EXISTS results (
    race_id       TEXT NOT NULL REFERENCES races(race_id),
    lane          INTEGER NOT NULL,   -- 枠番
    course        INTEGER,            -- 実際の進入コース
    arrival_order INTEGER,            -- 着順(1-6)。失格・欠場等はNULL
    st_time       REAL,               -- スタートタイミング(負値=フライング)
    PRIMARY KEY (race_id, lane)
);

CREATE TABLE IF NOT EXISTS payouts (
    race_id      TEXT NOT NULL REFERENCES races(race_id),
    bet_type     TEXT NOT NULL,   -- 単勝/複勝/2連単/2連複/拡連複/3連単/3連複
    combination  TEXT NOT NULL,   -- 例 "1-2-3" "1=3=5"
    amount_yen   INTEGER,
    PRIMARY KEY (race_id, bet_type, combination)
);

-- 直前オッズ(boatrace.jpオッズページ)。10分おきの収集で締切直前の値に上書きされる。
-- fetched_atで取得時点を記録。過去日の'final-backfill'は確定最終オッズ。
CREATE TABLE IF NOT EXISTS odds (
    race_id      TEXT NOT NULL REFERENCES races(race_id),
    bet_type     TEXT NOT NULL,   -- 3連単/3連複
    combination  TEXT NOT NULL,   -- 例 "1-2-3" "1=3=5"
    odds         REAL,
    fetched_at   TEXT,
    PRIMARY KEY (race_id, bet_type, combination)
);

-- 確定最終オッズ(市場分析専用)。oddsテーブルの15分前スナップショット行を
-- 上書きしないよう、最終オッズは必ずこちらに分離して保存する。
-- test/collect_final_odds.py が過去日のオッズページから遡及取得して埋める。
CREATE TABLE IF NOT EXISTS odds_final (
    race_id      TEXT NOT NULL REFERENCES races(race_id),
    bet_type     TEXT NOT NULL,   -- 3連単/3連複
    combination  TEXT NOT NULL,   -- 例 "1-2-3" "1=3=5"
    odds         REAL,
    fetched_at   TEXT,
    PRIMARY KEY (race_id, bet_type, combination)
);

-- 潮位(気象庁の潮位表=天文潮汐の推算値。検証⑩用、本番予測には使わない)。
-- 1時間刻み。stationは気象庁の地点コード(TK=東京/NG=名古屋/OS=大阪/MO=門司)。
CREATE TABLE IF NOT EXISTS tide (
    station   TEXT NOT NULL,
    datetime  TEXT NOT NULL,   -- YYYY-MM-DD HH:00:00 (JST)
    level_cm  REAL,
    PRIMARY KEY (station, datetime)
);

-- 直前情報(boatrace.jp直前情報ページ)。締切約20分前に確定する。
-- 予測モデルには使わない(計測・保存のみ、将来の分析用)。
CREATE TABLE IF NOT EXISTS exhibition (
    race_id          TEXT NOT NULL REFERENCES races(race_id),
    lane             INTEGER NOT NULL,
    reg_no           INTEGER,
    weight_kg        REAL,    -- 当日計量後の体重(番組表時点の体重と差が出ることがある)
    exhibition_time  REAL,    -- 周回展示タイム(秒)
    tilt             REAL,    -- チルト角度
    PRIMARY KEY (race_id, lane)
);

CREATE INDEX IF NOT EXISTS idx_races_venue_date ON races(venue_code, date);
CREATE INDEX IF NOT EXISTS idx_entries_reg_no ON entries(reg_no);
"""

_PK_COLS = {
    "races": ("race_id",),
    "entries": ("race_id", "lane"),
    "results": ("race_id", "lane"),
    "payouts": ("race_id", "bet_type", "combination"),
    "exhibition": ("race_id", "lane"),
    "odds": ("race_id", "bet_type", "combination"),
    "odds_final": ("race_id", "bet_type", "combination"),
    "tide": ("station", "datetime"),
}


def make_race_id(date_str: str, venue_code: int, race_no: int) -> str:
    """racesテーブルの主キー形式 YYYYMMDD_{venue:02d}_{race:02d} を組み立てる"""
    return f"{date_str.replace('-', '')}_{venue_code:02d}_{race_no:02d}"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def _upsert(conn: sqlite3.Connection, table: str, row: dict):
    """渡された列だけを更新するUPSERT。既存行の他列は保持される。

    番組表とresultsが同じraces行を別々の列で埋めるため、
    INSERT OR REPLACEではなくON CONFLICT DO UPDATEを使う。
    """
    pk = _PK_COLS[table]
    cols = list(row.keys())
    placeholders = ", ".join(f":{c}" for c in cols)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    updates = [c for c in cols if c not in pk]
    if updates:
        sql += f" ON CONFLICT({', '.join(pk)}) DO UPDATE SET "
        sql += ", ".join(f"{c}=excluded.{c}" for c in updates)
    else:
        sql += f" ON CONFLICT({', '.join(pk)}) DO NOTHING"
    conn.execute(sql, row)


def upsert_race(conn, race: dict):
    _upsert(conn, "races", race)


def upsert_entry(conn, entry: dict):
    _upsert(conn, "entries", entry)


def upsert_result(conn, result: dict):
    _upsert(conn, "results", result)


def upsert_payout(conn, payout: dict):
    _upsert(conn, "payouts", payout)


def upsert_exhibition(conn, exhibition: dict):
    _upsert(conn, "exhibition", exhibition)


def upsert_odds(conn, odds_row: dict):
    _upsert(conn, "odds", odds_row)


def upsert_odds_final(conn, odds_row: dict):
    _upsert(conn, "odds_final", odds_row)


def upsert_tide(conn, tide_row: dict):
    _upsert(conn, "tide", tide_row)
