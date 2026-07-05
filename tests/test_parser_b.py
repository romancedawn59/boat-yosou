import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser_b import parse_program

PROGRAM_FIXTURE = {
    "programs": [
        {
            "date": "2024-06-01",
            "stadium_number": 2,
            "number": 1,
            "closed_at": "2024-06-01 10:47:00",
            "day_label": "３日目",
            "grade_label": "G1",
            "grade_number": 2,
            "title": "戸田プリムローズ開設68周年記念",
            "subtitle": "予選",
            "distance": 1800,
            "boats": [
                {
                    "racer_boat_number": 1,
                    "racer_name": "石渡 鉄兵",
                    "racer_number": 3716,
                    "racer_class_number": 1,
                    "racer_branch_number": 13,
                    "racer_birthplace_number": 12,
                    "racer_age": 49,
                    "racer_weight": 52.7,
                    "racer_flying_count": 0,
                    "racer_late_count": 0,
                    "racer_average_start_timing": 0.13,
                    "racer_national_top_1_percent": 6.76,
                    "racer_national_top_2_percent": 45.14,
                    "racer_national_top_3_percent": 63.19,
                    "racer_local_top_1_percent": 6.67,
                    "racer_local_top_2_percent": 49.09,
                    "racer_local_top_3_percent": 65.45,
                    "racer_assigned_motor_number": 23,
                    "racer_assigned_motor_top_2_percent": 27.98,
                    "racer_assigned_motor_top_3_percent": 44.05,
                    "racer_assigned_boat_number": 69,
                    "racer_assigned_boat_top_2_percent": 33.68,
                    "racer_assigned_boat_top_3_percent": 46.32,
                },
                {
                    "racer_boat_number": 2,
                    "racer_name": "秋元 哲",
                    "racer_number": 4532,
                    "racer_class_number": 2,
                    "racer_branch_number": 11,
                    "racer_birthplace_number": 9,
                    "racer_age": 35,
                    "racer_weight": 51,
                    "racer_flying_count": 1,
                    "racer_late_count": 0,
                    "racer_average_start_timing": 0.14,
                    "racer_national_top_1_percent": 6.76,
                    "racer_national_top_2_percent": 52.59,
                    "racer_national_top_3_percent": 70.37,
                    "racer_local_top_1_percent": 5.98,
                    "racer_local_top_2_percent": 43.75,
                    "racer_local_top_3_percent": 61.72,
                    "racer_assigned_motor_number": 20,
                    "racer_assigned_motor_top_2_percent": 43.41,
                    "racer_assigned_motor_top_3_percent": 60.49,
                    "racer_assigned_boat_number": 72,
                    "racer_assigned_boat_top_2_percent": 36.31,
                    "racer_assigned_boat_top_3_percent": 52.51,
                },
            ],
        }
    ]
}


class TestParseProgram(unittest.TestCase):
    def setUp(self):
        self.parsed = parse_program(PROGRAM_FIXTURE)

    def test_race_row(self):
        races = self.parsed["races"]
        self.assertEqual(len(races), 1)
        race = races[0]
        self.assertEqual(race["race_id"], "20240601_02_01")
        self.assertEqual(race["date"], "2024-06-01")
        self.assertEqual(race["venue_code"], 2)
        self.assertEqual(race["race_no"], 1)
        self.assertEqual(race["title"], "戸田プリムローズ開設68周年記念")
        self.assertEqual(race["subtitle"], "予選")
        self.assertEqual(race["grade"], "G1")
        self.assertEqual(race["day_label"], "３日目")
        self.assertEqual(race["distance_m"], 1800)
        self.assertEqual(race["deadline_time"], "2024-06-01 10:47:00")

    def test_entry_rows(self):
        entries = self.parsed["entries"]
        self.assertEqual(len(entries), 2)

        e1 = entries[0]
        self.assertEqual(e1["race_id"], "20240601_02_01")
        self.assertEqual(e1["lane"], 1)
        self.assertEqual(e1["reg_no"], 3716)
        self.assertEqual(e1["racer_name"], "石渡 鉄兵")
        self.assertEqual(e1["racer_class"], "A1")
        self.assertEqual(e1["branch_number"], 13)
        self.assertEqual(e1["birthplace_number"], 12)
        self.assertEqual(e1["age"], 49)
        self.assertEqual(e1["weight_kg"], 52.7)
        self.assertEqual(e1["flying_count"], 0)
        self.assertEqual(e1["avg_st"], 0.13)
        self.assertEqual(e1["national_win_rate"], 6.76)
        self.assertEqual(e1["national_2rate"], 45.14)
        self.assertEqual(e1["national_3rate"], 63.19)
        self.assertEqual(e1["local_win_rate"], 6.67)
        self.assertEqual(e1["motor_no"], 23)
        self.assertEqual(e1["motor_2rate"], 27.98)
        self.assertEqual(e1["boat_no"], 69)
        self.assertEqual(e1["boat_2rate"], 33.68)

        e2 = entries[1]
        self.assertEqual(e2["racer_class"], "A2")
        self.assertEqual(e2["flying_count"], 1)

    def test_empty_input(self):
        parsed = parse_program({"programs": []})
        self.assertEqual(parsed, {"races": [], "entries": []})

    def test_null_racer_boat_is_skipped(self):
        """選手未確定(全項目null)の枠はentriesに含めない。レース行は残す"""
        fixture = {
            "programs": [{
                "date": "2026-05-17",
                "stadium_number": 13,
                "number": 1,
                "boats": [
                    {"racer_boat_number": None, "racer_number": None, "racer_name": None},
                    PROGRAM_FIXTURE["programs"][0]["boats"][0],
                ],
            }]
        }
        parsed = parse_program(fixture)
        self.assertEqual(len(parsed["races"]), 1)
        self.assertEqual(len(parsed["entries"]), 1)
        self.assertEqual(parsed["entries"][0]["reg_no"], 3716)


if __name__ == "__main__":
    unittest.main()
