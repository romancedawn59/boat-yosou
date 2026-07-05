import sqlite3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
from weather import estimate_wave_height_cm, lookup


class TestLookup(unittest.TestCase):
    def setUp(self):
        self.hourly = {
            "2026-07-05T10:00": (3.5, 28.0),
            "2026-07-05T11:00": (4.0, 28.5),
        }

    def test_rounds_down_to_the_hour(self):
        self.assertEqual(lookup(self.hourly, "2026-07-05 10:47:00"), (3.5, 28.0))

    def test_exact_hour(self):
        self.assertEqual(lookup(self.hourly, "2026-07-05 11:00:00"), (4.0, 28.5))

    def test_missing_hour_returns_none(self):
        self.assertIsNone(lookup(self.hourly, "2026-07-05 23:00:00"))


class TestEstimateWaveHeight(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(db.SCHEMA)
        # wave_cm = 2 * wind_speed_m の完全な線形関係を持つダミーデータ
        for i, wind in enumerate([0.0, 1.0, 2.0, 3.0, 4.0] * 10):
            rid = f"2025071{i:02d}_02_01"
            db.upsert_race(self.conn, {
                "race_id": rid, "date": "2025-07-15", "venue_code": 2, "race_no": 1,
                "wind_speed_m": wind, "wave_height_cm": wind * 2.0,
            })
        self.conn.commit()

    def test_fits_linear_relationship(self):
        est = estimate_wave_height_cm(self.conn, venue_code=2, wind_speed_m=2.5)
        self.assertAlmostEqual(est, 5.0, places=1)

    def test_different_venue_has_no_data_returns_zero(self):
        est = estimate_wave_height_cm(self.conn, venue_code=99, wind_speed_m=2.5)
        self.assertEqual(est, 0.0)

    def test_never_negative(self):
        est = estimate_wave_height_cm(self.conn, venue_code=2, wind_speed_m=-5.0)
        self.assertGreaterEqual(est, 0.0)


if __name__ == "__main__":
    unittest.main()
