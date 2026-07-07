import sqlite3
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
from collect_exhibition import collect_today

TODAY = date(2026, 7, 4)
RACE_ID = "20260704_04_01"  # 平和島(対象5場のひとつ)


class _NoCloseConnection:
    """collect_today()内のconn.close()を無効化し、テストから中身を検証できるようにするラッパー"""

    def __init__(self, conn):
        self._conn = conn

    def close(self):
        pass

    def __getattr__(self, name):
        return getattr(self._conn, name)


class TestCollectExhibition(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(db.SCHEMA)
        db.upsert_race(self.conn, {
            "race_id": RACE_ID, "date": TODAY.isoformat(), "venue_code": 4, "race_no": 1,
            "deadline_time": "2026-07-04 10:47:00",
        })
        self.conn.commit()

        patcher = patch("collect_exhibition.db.connect", return_value=_NoCloseConnection(self.conn))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_skips_when_too_early(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch, \
             patch("collect_exhibition.odds_mod.fetch_odds") as mock_odds:
            collect_today(TODAY, datetime(2026, 7, 4, 9, 0, 0))  # 締切1時間47分前
        mock_fetch.assert_not_called()
        mock_odds.assert_not_called()

    def test_fetches_when_within_lookahead_window(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch, \
             patch("collect_exhibition.odds_mod.fetch_odds") as mock_odds:
            mock_fetch.return_value = [
                {"lane": 1, "reg_no": 1234, "weight_kg": 52.0, "exhibition_time": 6.8, "tilt": 0.0},
            ]
            mock_odds.return_value = {"3連単": {(1, 2, 3): 5.6}, "3連複": {(1, 2, 3): 2.3}}
            collect_today(TODAY, datetime(2026, 7, 4, 10, 30, 0))  # 締切17分前

        mock_fetch.assert_called_once_with(4, 1, TODAY)
        row = self.conn.execute(
            "SELECT lane, exhibition_time FROM exhibition WHERE race_id = ?", (RACE_ID,)
        ).fetchone()
        self.assertEqual(row, (1, 6.8))
        odds_rows = self.conn.execute(
            "SELECT bet_type, combination, odds FROM odds WHERE race_id = ? ORDER BY bet_type",
            (RACE_ID,),
        ).fetchall()
        self.assertEqual(odds_rows, [("3連単", "1-2-3", 5.6), ("3連複", "1=2=3", 2.3)])

    def test_exhibition_skips_but_odds_updates_when_already_saved(self):
        db.upsert_exhibition(self.conn, {
            "race_id": RACE_ID, "lane": 1, "reg_no": 1234,
            "weight_kg": 52.0, "exhibition_time": 6.8, "tilt": 0.0,
        })
        self.conn.commit()

        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch, \
             patch("collect_exhibition.odds_mod.fetch_odds") as mock_odds:
            mock_odds.return_value = {"3連単": {(1, 2, 3): 9.9}, "3連複": {}}
            collect_today(TODAY, datetime(2026, 7, 4, 10, 30, 0))

        mock_fetch.assert_not_called()   # 展示は確定値なのでスキップ
        mock_odds.assert_called_once()   # オッズは毎回上書き
        val = self.conn.execute(
            "SELECT odds FROM odds WHERE race_id=? AND combination='1-2-3'", (RACE_ID,)
        ).fetchone()[0]
        self.assertEqual(val, 9.9)

    def test_after_deadline_skips_everything(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch, \
             patch("collect_exhibition.odds_mod.fetch_odds") as mock_odds:
            collect_today(TODAY, datetime(2026, 7, 4, 11, 0, 0))  # 締切13分後
        mock_fetch.assert_not_called()
        mock_odds.assert_not_called()

    def test_no_data_yet_does_not_crash(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition", return_value=[]), \
             patch("collect_exhibition.odds_mod.fetch_odds",
                   return_value={"3連単": {}, "3連複": {}}):
            collect_today(TODAY, datetime(2026, 7, 4, 10, 30, 0))
        count = self.conn.execute("SELECT COUNT(*) FROM exhibition").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
