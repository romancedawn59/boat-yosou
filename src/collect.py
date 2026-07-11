"""BoatraceOpenAPIの日次JSONを取得しSQLiteへ格納するCLI

    python collect.py                        # 前回の続きから今日まで(初回は約1年遡る)
    python collect.py 2025-07-15 2025-07-31  # 日付範囲を指定
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import db
from config import DB_PATH, DEFAULT_LOOKBACK_DAYS, jst_today
from downloader import download_day
from parser_b import parse_program
from parser_k import parse_result


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_day(conn, d: date, force: bool = False) -> bool:
    """1日分を取得してDBへ格納。データが存在したらTrue"""
    paths = download_day(d, force=force)
    if paths["program"] is None and paths["result"] is None:
        return False

    if paths["program"]:
        program_data = parse_program(_load_json(paths["program"]))
        for race in program_data["races"]:
            db.upsert_race(conn, race)
        for entry in program_data["entries"]:
            db.upsert_entry(conn, entry)

    if paths["result"]:
        result_data = parse_result(_load_json(paths["result"]))
        for race in result_data["races"]:
            db.upsert_race(conn, race)
        for result in result_data["results"]:
            db.upsert_result(conn, result)
        for payout in result_data["payouts"]:
            db.upsert_payout(conn, payout)

    conn.commit()
    return True


def collect_range(start: date, end: date, force_first: bool = False):
    conn = db.connect(DB_PATH)
    ok, skipped, failed = 0, 0, 0

    for d in _daterange(start, end):
        try:
            found = collect_day(conn, d, force=(force_first and d == start))
        except Exception as e:
            conn.rollback()  # 失敗日の中途半端な行が後続日のcommitに混ざらないように
            failed += 1
            print(f"{d}: 失敗 ({e})")
            continue
        if found:
            ok += 1
            print(f"{d}: OK")
        else:
            skipped += 1

    conn.close()
    print(f"完了: 成功={ok} 開催なし/保持期間外={skipped} 失敗={failed}")


def auto_range() -> tuple[date, date, bool]:
    """引数なし実行時の収集範囲。

    収集済みなら最終日から(当日途中の不完全データを取り直すため同日を強制再取得)、
    未収集ならresultsの保持期間ぶんだけ遡る。
    """
    # クラウドランナーはUTC。date.today()は0:00〜8:59 JSTに前日を返し、
    # 当日分が収集済みだと start > end で空振りするため必ずJSTで判定する
    today = jst_today()
    conn = db.connect(DB_PATH)
    row = conn.execute("SELECT MAX(date) FROM races").fetchone()
    conn.close()

    if row and row[0]:
        return date.fromisoformat(row[0]), today, True
    return today - timedelta(days=DEFAULT_LOOKBACK_DAYS), today, False


if __name__ == "__main__":
    if len(sys.argv) == 1:
        start, end, force_first = auto_range()
        print(f"収集範囲: {start} 〜 {end}")
        collect_range(start, end, force_first=force_first)
    elif len(sys.argv) == 3:
        collect_range(date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]))
    else:
        print("usage: python collect.py [YYYY-MM-DD YYYY-MM-DD]")
        sys.exit(1)
