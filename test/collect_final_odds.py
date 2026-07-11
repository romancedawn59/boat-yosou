# -*- coding: utf-8 -*-
"""確定最終オッズの遡及取得CLI(市場分析基盤・約束①用)

    py -X utf8 test/collect_final_odds.py

対象: 「15分前スナップショット(oddsテーブル、fetched_at≠'final-backfill')があり、
かつ odds_final にまだ無い」過去日のレースのみ。
公式サイトの過去日付オッズページは最終オッズを表示し続ける(HANDOVERの裏技)ため、
既存 odds.fetch_odds() をそのまま過去日付で呼べば確定最終オッズが取れる。

方針:
- oddsテーブル(15分前スナップショットの純度が資産)には一切書かない。
  最終オッズは odds_final テーブルへ分離保存する
- 当日(JST)のレースは未確定のため対象外
- 冪等: odds_finalに1件でもあるレースはスキップ(再実行で差分のみ取得)
- REQUEST_INTERVAL_SEC を厳守してサーバー負荷に配慮する
"""
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
import odds as odds_mod
from config import DB_PATH, JST, REQUEST_INTERVAL_SEC, jst_today


def pick_targets(conn, today: date) -> list[tuple[str, int, int, str]]:
    """取得対象の(race_id, venue_code, race_no, date)。

    final-backfill行はスナップショットではないため対象にしない
    (比較対象の「15分前」が存在しないレースの最終オッズは分析に使えない)。
    """
    return conn.execute(
        """
        SELECT DISTINCT o.race_id, r.venue_code, r.race_no, r.date
        FROM odds o
        JOIN races r ON r.race_id = o.race_id
        WHERE o.fetched_at != 'final-backfill'
          AND r.date < ?
          AND NOT EXISTS (SELECT 1 FROM odds_final f WHERE f.race_id = o.race_id)
        ORDER BY r.date, r.venue_code, r.race_no
        """,
        (today.isoformat(),),
    ).fetchall()


def collect(conn, targets: list[tuple[str, int, int, str]]) -> int:
    ok = 0
    fetched_at = datetime.now(JST).isoformat(timespec="seconds")
    for race_id, venue_code, race_no, d_str in targets:
        try:
            o = odds_mod.fetch_odds(venue_code, race_no, date.fromisoformat(d_str))
        except Exception as e:
            print(f"{race_id}: 取得失敗 ({e})")
            time.sleep(REQUEST_INTERVAL_SEC)
            continue
        n = 0
        for bt_name, sep in (("3連単", "-"), ("3連複", "=")):
            for key, val in o[bt_name].items():
                db.upsert_odds_final(conn, {
                    "race_id": race_id, "bet_type": bt_name,
                    "combination": sep.join(map(str, key)),
                    "odds": val, "fetched_at": fetched_at,
                })
                n += 1
        conn.commit()
        if n:
            ok += 1
            print(f"{race_id}: 最終オッズ保存 ({n}件)")
        else:
            print(f"{race_id}: オッズページが空(中止等の可能性)")
        time.sleep(REQUEST_INTERVAL_SEC)
    return ok


if __name__ == "__main__":
    conn = db.connect(DB_PATH)
    targets = pick_targets(conn, jst_today())
    print(f"取得対象: {len(targets)}レース")
    done = collect(conn, targets)
    conn.close()
    print(f"完了: {done}/{len(targets)}レース")
