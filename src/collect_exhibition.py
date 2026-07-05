"""当日の対象場レースの直前情報(展示タイム)を締切前に取得しDBへ保存するCLI

締切のLOOKAHEAD_MIN分前になったレースから取得を試み、既に保存済みなら
スキップする(冪等)。1回の実行は一瞬で終わるので、タスクスケジューラ等で
レース開催中は数分おきに実行することを想定している。

    python collect_exhibition.py
"""
from datetime import date, datetime, timedelta

import db
import exhibition
from config import DB_PATH, TARGET_VENUE_CODES

LOOKAHEAD_MIN = 30  # 締切のこの分数前から取得を試みる(展示は締切約20分前に確定)


def collect_today(today: date, now: datetime) -> None:
    conn = db.connect(DB_PATH)
    ph = ",".join("?" * len(TARGET_VENUE_CODES))
    rows = conn.execute(
        f"SELECT race_id, venue_code, race_no, deadline_time FROM races "
        f"WHERE date = ? AND venue_code IN ({ph}) ORDER BY venue_code, race_no",
        (today.isoformat(), *TARGET_VENUE_CODES),
    ).fetchall()

    for race_id, venue_code, race_no, deadline_str in rows:
        if not deadline_str:
            continue
        deadline = datetime.strptime(deadline_str, "%Y-%m-%d %H:%M:%S")
        if now < deadline - timedelta(minutes=LOOKAHEAD_MIN):
            continue  # まだ早い

        already = conn.execute(
            "SELECT COUNT(*) FROM exhibition WHERE race_id = ?", (race_id,)
        ).fetchone()[0]
        if already:
            continue

        try:
            rows_ex = exhibition.fetch_exhibition(venue_code, race_no, today)
        except Exception as e:
            print(f"{race_id}: 取得失敗 ({e})")
            continue

        if not rows_ex:
            print(f"{race_id}: まだ展示データなし")
            continue

        for r in rows_ex:
            db.upsert_exhibition(conn, {"race_id": race_id, **r})
        conn.commit()
        print(f"{race_id}: 展示データ保存 ({len(rows_ex)}艇)")

    conn.close()


if __name__ == "__main__":
    collect_today(date.today(), datetime.now())
