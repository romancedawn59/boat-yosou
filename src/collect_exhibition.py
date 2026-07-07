"""当日の対象場レースの直前情報(展示タイム・直前オッズ)を締切15分前に1回だけ取得するCLI

方針(2026-07-09確定): 直前情報は保存のみ。予測・勝負所判定には一切反映しない。
数ヶ月分蓄積した後、朝予想との比較検証を行ってから利用可否を判断する。

- 取得タイミング: 締切の15分前〜締切の間に入った最初の実行で1回だけ
  (タスクスケジューラの10分おき実行なら、この窓に必ず1回だけ入る)
- 展示タイム・オッズとも取得済みのレースはスキップ(再取得しない)

    python collect_exhibition.py
"""
from datetime import date, datetime, timedelta

import db
import exhibition
import odds as odds_mod
from config import DB_PATH, TARGET_VENUE_CODES

LOOKAHEAD_MIN = 15  # 締切のこの分数前から取得対象(展示は締切約20分前に確定済み)


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
        print(f"{race_id}: 直前オッズ保存 ({n}件)")


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
        # 締切15分前〜締切の窓に入ったレースだけが対象
        if now < deadline - timedelta(minutes=LOOKAHEAD_MIN) or now > deadline:
            continue

        # 展示タイム: 1回だけ取得
        already_ex = conn.execute(
            "SELECT COUNT(*) FROM exhibition WHERE race_id = ?", (race_id,)
        ).fetchone()[0]
        if not already_ex:
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

        # オッズ: 1回だけ取得(締切15分前スナップショット)
        already_odds = conn.execute(
            "SELECT COUNT(*) FROM odds WHERE race_id = ?", (race_id,)
        ).fetchone()[0]
        if not already_odds:
            _collect_race_odds(conn, race_id, venue_code, race_no, today, now)

    conn.close()


if __name__ == "__main__":
    collect_today(date.today(), datetime.now())
