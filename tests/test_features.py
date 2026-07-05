import sqlite3
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
from features import FEATURE_COLUMNS, build_program_features, build_training_set, compute_form_features


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(db.SCHEMA)
        db.upsert_race(self.conn, {
            "race_id": "20250715_02_01", "date": "2025-07-15",
            "venue_code": 2, "race_no": 1, "grade": "一般", "distance_m": 1800,
        })
        for lane, cls, order in [(1, "A1", 1), (2, "B1", 2)]:
            db.upsert_entry(self.conn, {
                "race_id": "20250715_02_01", "lane": lane, "reg_no": 1000 + lane,
                "racer_name": f"選手{lane}", "racer_class": cls, "age": 30, "weight_kg": 52.0,
                "flying_count": 0, "late_count": 0, "avg_st": 0.15,
                "national_win_rate": 6.0, "national_2rate": 40.0, "national_3rate": 60.0,
                "local_win_rate": 6.0, "local_2rate": 40.0, "local_3rate": 60.0,
                "motor_2rate": 30.0, "motor_3rate": 45.0, "boat_2rate": 30.0, "boat_3rate": 45.0,
            })
            db.upsert_result(self.conn, {
                "race_id": "20250715_02_01", "lane": lane, "course": lane,
                "arrival_order": order, "st_time": 0.15,
            })
        self.conn.commit()

    def test_build_training_set_labels_winner(self):
        df = build_training_set(self.conn)
        self.assertEqual(len(df), 2)

        winner = df[df["lane"] == 1].iloc[0]
        loser = df[df["lane"] == 2].iloc[0]
        self.assertEqual(winner["is_winner"], 1)
        self.assertEqual(loser["is_winner"], 0)
        self.assertEqual(winner["racer_class_ord"], 3)  # A1
        self.assertEqual(loser["racer_class_ord"], 1)   # B1
        self.assertEqual(winner["grade_ord"], 0)         # 一般

        for col in FEATURE_COLUMNS:
            self.assertIn(col, df.columns)

    def test_build_program_features_has_no_label(self):
        df = build_program_features(self.conn, ["20250715_02_01"])
        self.assertEqual(len(df), 2)
        self.assertNotIn("is_winner", df.columns)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df.columns)

    def test_build_program_features_empty_race_ids(self):
        df = build_program_features(self.conn, [])
        self.assertEqual(len(df), 0)


class TestFormFeatures(unittest.TestCase):
    """ローリング成績が「当該レースより前」だけから計算されること(リーク防止)"""

    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(db.SCHEMA)
        # 選手9999が4日連続で出走: 着順 1,3,1着 + 4走目(結果未確定)
        arrivals = [1, 3, 1, None]
        for i, arrival in enumerate(arrivals):
            rid = f"2025071{5+i}_02_01"
            db.upsert_race(self.conn, {
                "race_id": rid, "date": f"2025-07-1{5+i}",
                "venue_code": 2, "race_no": 1, "grade": "一般",
            })
            db.upsert_entry(self.conn, {"race_id": rid, "lane": 1, "reg_no": 9999})
            if arrival is not None:
                db.upsert_result(self.conn, {
                    "race_id": rid, "lane": 1, "course": 1,
                    "arrival_order": arrival, "st_time": 0.10 + i * 0.02,
                })
        self.conn.commit()
        self.form = compute_form_features(self.conn)

    def _row(self, race_id):
        return self.form[self.form["race_id"] == race_id].iloc[0]

    def test_form_excludes_current_race(self):
        # 4走目: 過去3走(1,3,1着)から計算。当該レースの結果は存在しない
        last = self._row("20250718_02_01")
        self.assertAlmostEqual(last["form_last3_win_rate"], 2 / 3)
        self.assertAlmostEqual(last["form_last10_avg_finish"], (1 + 3 + 1) / 3)
        self.assertAlmostEqual(last["form_last10_avg_st"], (0.10 + 0.12 + 0.14) / 3)
        self.assertAlmostEqual(last["form_lane_win_rate"], 2 / 3)
        self.assertEqual(last["form_days_since_last"], 1)

    def test_first_race_has_no_form(self):
        first = self._row("20250715_02_01")
        self.assertTrue(first[["form_last3_win_rate", "form_last10_win_rate",
                               "form_days_since_last"]].isna().all())

    def test_min_periods_guard(self):
        # 2走目: 過去1走のみ -> last3(min_periods=1)は出るが、last10(min_periods=3)はNaN
        second = self._row("20250716_02_01")
        self.assertAlmostEqual(second["form_last3_win_rate"], 1.0)
        self.assertTrue(pd.isna(second["form_last10_win_rate"]))


if __name__ == "__main__":
    unittest.main()
