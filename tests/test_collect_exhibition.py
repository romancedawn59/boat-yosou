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
        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch:
            collect_today(TODAY, datetime(2026, 7, 4, 9, 0, 0))  # 締切1時間47分前
        mock_fetch.assert_not_called()

    def test_fetches_when_within_lookahead_window(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch:
            mock_fetch.return_value = [
                {"lane": 1, "reg_no": 1234, "weight_kg": 52.0, "exhibition_time": 6.8, "tilt": 0.0},
            ]
            collect_today(TODAY, datetime(2026, 7, 4, 10, 30, 0))  # 締切17分前

        mock_fetch.assert_called_once_with(4, 1, TODAY)
        row = self.conn.execute(
            "SELECT lane, exhibition_time FROM exhibition WHERE race_id = ?", (RACE_ID,)
        ).fetchone()
        self.assertEqual(row, (1, 6.8))

    def test_skips_when_already_saved(self):
        db.upsert_exhibition(self.conn, {
            "race_id": RACE_ID, "lane": 1, "reg_no": 1234,
            "weight_kg": 52.0, "exhibition_time": 6.8, "tilt": 0.0,
        })
        self.conn.commit()

        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch:
            collect_today(TODAY, datetime(2026, 7, 4, 10, 30, 0))
        mock_fetch.assert_not_called()

    def test_no_data_yet_does_not_crash(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition", return_value=[]):
            collect_today(TODAY, datetime(2026, 7, 4, 10, 30, 0))
        count = self.conn.execute("SELECT COUNT(*) FROM exhibition").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
