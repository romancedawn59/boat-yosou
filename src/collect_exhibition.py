"""当日の対象場レースの直前情報(展示タイム)と直前オッズを締切前に取得しDBへ保存するCLI

締切のLOOKAHEAD_MIN分前になったレースを対象に、
- 展示タイム: 1回取得できたら以後スキップ(確定値のため)
- オッズ: 毎回上書き(変動するため。最後に取得した=締切に最も近い値が残る)
タスクスケジューラでレース開催中は10分おきに実行される想定。

    python collect_exhibition.py
"""
from datetime import date, datetime, timedelta

import db
import exhibition
import odds as odds_mod
from config import DB_PATH, TARGET_VENUE_CODES

LOOKAHEAD_MIN = 30  # 締切のこの分数前から取得を試みる(展示は締切約20分前に確定)


def _collect_race_odds(conn, race_id: str, venue_code: int, race_no: int, today: date, now: datetime) -> None:
    try:
        o = odds_mod.fetch_odds(venue_code, race_no, today)
    except Exception as e:
        print(f"{race_id}: オッズ取得失敗 ({e})")
        return
    n = 0
    for bt_name, sep in (("3連単", "-"), ("3連複", "=")):
        for key, val in o[bt_name].items():
            db.upsert_odds(conn, {
                "race_id": race_id, "bet_type": bt_name,
                "combination": sep.join(map(str, key)),
                "odds": val, "fetched_at": now.isoformat(timespec="seconds"),
            })
            n += 1
    conn.commit()
    if n:
        print(f"{race_id}: オッズ更新 ({n}件)")


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
        if now > deadline + timedelta(minutes=5):
            continue  # 締切済み(最後の上書きが直前オッズとして残る)

        # 展示タイム: 確定値なので取得済みならスキップ
        already = conn.execute(
            "SELECT COUNT(*) FROM exhibition WHERE race_id = ?", (race_id,)
        ).fetchone()[0]
        if not already:
            try:
                rows_ex = exhibition.fetch_exhibition(venue_code, race_no, today)
            except Exception as e:
                rows_ex = []
                print(f"{race_id}: 展示取得失敗 ({e})")
            if rows_ex:
                for r in rows_ex:
                    db.upsert_exhibition(conn, {"race_id": race_id, **r})
                conn.commit()
                print(f"{race_id}: 展示データ保存 ({len(rows_ex)}艇)")
            else:
                print(f"{race_id}: まだ展示データなし")

        # オッズ: 毎回上書き(締切に最も近い値を残す)
        _collect_race_odds(conn, race_id, venue_code, race_no, today, now)

    conn.close()


if __name__ == "__main__":
    collect_today(date.today(), datetime.now())
