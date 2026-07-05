import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db


class TestDb(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmpdir.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def test_make_race_id(self):
        self.assertEqual(db.make_race_id("2025-07-15", 2, 1), "20250715_02_01")
        self.assertEqual(db.make_race_id("2025-07-15", 24, 12), "20250715_24_12")

    def test_upsert_race_merges_partial_columns(self):
        """番組表由来の列とresults由来の列が同じ行にマージされること"""
        db.upsert_race(self.conn, {
            "race_id": "20250715_02_01", "date": "2025-07-15",
            "venue_code": 2, "race_no": 1,
            "title": "テスト戦", "distance_m": 1800,
        })
        db.upsert_race(self.conn, {
            "race_id": "20250715_02_01", "date": "2025-07-15",
            "venue_code": 2, "race_no": 1,
            "weather_number": 2, "wind_speed_m": 4.0,
        })

        row = self.conn.execute(
            "SELECT title, distance_m, weather_number, wind_speed_m FROM races WHERE race_id = ?",
            ("20250715_02_01",),
        ).fetchone()
        self.assertEqual(row, ("テスト戦", 1800, 2, 4.0))

    def test_upsert_race_updates_existing_value(self):
        db.upsert_race(self.conn, {
            "race_id": "20250715_02_01", "date": "2025-07-15",
            "venue_code": 2, "race_no": 1, "title": "旧タイトル",
        })
        db.upsert_race(self.conn, {
            "race_id": "20250715_02_01", "date": "2025-07-15",
            "venue_code": 2, "race_no": 1, "title": "新タイトル",
        })

        row = self.conn.execute(
            "SELECT title FROM races WHERE race_id = ?", ("20250715_02_01",)
        ).fetchone()
        self.assertEqual(row[0], "新タイトル")
        count = self.conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        self.assertEqual(count, 1)

    def test_upsert_entry_and_result_composite_pk(self):
        for lane in (1, 2):
            db.upsert_entry(self.conn, {
                "race_id": "20250715_02_01", "lane": lane, "reg_no": 4000 + lane,
            })
            db.upsert_result(self.conn, {
                "race_id": "20250715_02_01", "lane": lane,
                "course": lane, "arrival_order": lane, "st_time": 0.15,
            })
        # 同じ(race_id, lane)への再投入は上書きであり行は増えない
        db.upsert_result(self.conn, {
            "race_id": "20250715_02_01", "lane": 1,
            "course": 1, "arrival_order": 2, "st_time": 0.20,
        })

        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 2)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM results").fetchone()[0], 2)
        order = self.conn.execute(
            "SELECT arrival_order FROM results WHERE race_id = ? AND lane = 1",
            ("20250715_02_01",),
        ).fetchone()[0]
        self.assertEqual(order, 2)

    def test_upsert_payout(self):
        db.upsert_payout(self.conn, {
            "race_id": "20250715_02_01", "bet_type": "3連単",
            "combination": "1-2-3", "amount_yen": 1000,
        })
        db.upsert_payout(self.conn, {
            "race_id": "20250715_02_01", "bet_type": "3連単",
            "combination": "1-2-3", "amount_yen": 1200,
        })

        rows = self.conn.execute("SELECT amount_yen FROM payouts").fetchall()
        self.assertEqual(rows, [(1200,)])


if __name__ == "__main__":
    unittest.main()
