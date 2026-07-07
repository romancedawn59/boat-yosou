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
RACE_ID = "20260704_04_01"  # 平和島(対象5場のひとつ)、締切10:47


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

    def test_skips_before_15min_window(self):
        """締切16分以上前は取得しない(15分前のみ取得の方針)"""
        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch, \
             patch("collect_exhibition.odds_mod.fetch_odds") as mock_odds:
            collect_today(TODAY, datetime(2026, 7, 4, 10, 30, 0))  # 締切17分前
        mock_fetch.assert_not_called()
        mock_odds.assert_not_called()

    def test_fetches_once_within_15min_window(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch, \
             patch("collect_exhibition.odds_mod.fetch_odds") as mock_odds:
            mock_fetch.return_value = [
                {"lane": 1, "reg_no": 1234, "weight_kg": 52.0, "exhibition_time": 6.8, "tilt": 0.0},
            ]
            mock_odds.return_value = {"3連単": {(1, 2, 3): 5.6}, "3連複": {(1, 2, 3): 2.3}}
            collect_today(TODAY, datetime(2026, 7, 4, 10, 35, 0))  # 締切12分前

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

    def test_second_pass_does_not_refetch(self):
        """展示・オッズとも保存済みなら再取得しない(1回だけのスナップショット)"""
        db.upsert_exhibition(self.conn, {
            "race_id": RACE_ID, "lane": 1, "reg_no": 1234,
            "weight_kg": 52.0, "exhibition_time": 6.8, "tilt": 0.0,
        })
        db.upsert_odds(self.conn, {
            "race_id": RACE_ID, "bet_type": "3連単", "combination": "1-2-3",
            "odds": 5.6, "fetched_at": "2026-07-04T10:35:00",
        })
        self.conn.commit()

        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch, \
             patch("collect_exhibition.odds_mod.fetch_odds") as mock_odds:
            collect_today(TODAY, datetime(2026, 7, 4, 10, 45, 0))  # 窓内2回目の実行
        mock_fetch.assert_not_called()
        mock_odds.assert_not_called()
        # 最初のスナップショットが残っている
        val = self.conn.execute(
            "SELECT odds FROM odds WHERE race_id=? AND combination='1-2-3'", (RACE_ID,)
        ).fetchone()[0]
        self.assertEqual(val, 5.6)

    def test_after_deadline_skips_everything(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition") as mock_fetch, \
             patch("collect_exhibition.odds_mod.fetch_odds") as mock_odds:
            collect_today(TODAY, datetime(2026, 7, 4, 10, 48, 0))  # 締切1分後
        mock_fetch.assert_not_called()
        mock_odds.assert_not_called()

    def test_no_data_yet_does_not_crash(self):
        with patch("collect_exhibition.exhibition.fetch_exhibition", return_value=[]), \
             patch("collect_exhibition.odds_mod.fetch_odds",
                   return_value={"3連単": {}, "3連複": {}}):
            collect_today(TODAY, datetime(2026, 7, 4, 10, 40, 0))
        count = self.conn.execute("SELECT COUNT(*) FROM exhibition").fetchone()[0]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
