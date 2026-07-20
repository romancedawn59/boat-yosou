import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from parser_k import parse_result

RESULT_FIXTURE = {
    "results": [
        {
            "date": "2025-07-15",
            "stadium_number": 1,
            "number": 1,
            "wind_speed": 4,
            "wind_direction_number": 10,
            "wave_height": 3,
            "weather_number": 2,
            "air_temperature": 26,
            "water_temperature": 27,
            "technique_number": 1,
            "boats": [
                {
                    "racer_boat_number": 1,
                    "racer_course_number": 1,
                    "racer_start_timing": 0.22,
                    "racer_place_number": 1,
                    "racer_number": 3860,
                    "racer_name": "松本 浩貴",
                },
                {
                    "racer_boat_number": 2,
                    "racer_course_number": 2,
                    "racer_start_timing": None,
                    "racer_place_number": None,  # 欠場・失格等
                    "racer_number": 3538,
                    "racer_name": "菊池 峰晴",
                },
            ],
            "payouts": {
                "trifecta": [{"combination": "1-5-3", "amount": 12690}],
                "trio": [{"combination": "1=3=5", "amount": 2110}],
                "exacta": [{"combination": "1-5", "amount": 1860}],
                "quinella": [{"combination": "1=5", "amount": 1190}],
                "quinella_place": [
                    {"combination": "1=5", "amount": 690},
                    {"combination": "1=3", "amount": 740},
                ],
                "win": [{"combination": "1", "amount": 290}],
                "place": [
                    {"combination": "1", "amount": 320},
                    {"combination": None, "amount": None},  # 不成立
                ],
            },
        }
    ]
}


class TestParseResult(unittest.TestCase):
    def setUp(self):
        self.parsed = parse_result(RESULT_FIXTURE)

    def test_race_weather_row(self):
        races = self.parsed["races"]
        self.assertEqual(len(races), 1)
        race = races[0]
        self.assertEqual(race["race_id"], "20250715_01_01")
        self.assertEqual(race["date"], "2025-07-15")
        self.assertEqual(race["venue_code"], 1)
        self.assertEqual(race["race_no"], 1)
        self.assertEqual(race["weather_number"], 2)
        self.assertEqual(race["wind_speed_m"], 4)
        self.assertEqual(race["wind_direction_number"], 10)
        self.assertEqual(race["wave_height_cm"], 3)
        self.assertEqual(race["temperature"], 26)
        self.assertEqual(race["water_temperature"], 27)
        self.assertEqual(race["winning_technique_number"], 1)

    def test_result_rows(self):
        results = self.parsed["results"]
        self.assertEqual(len(results), 2)

        r1 = results[0]
        self.assertEqual(r1["race_id"], "20250715_01_01")
        self.assertEqual(r1["lane"], 1)
        self.assertEqual(r1["course"], 1)
        self.assertEqual(r1["arrival_order"], 1)
        self.assertEqual(r1["st_time"], 0.22)

        # 着順が取れない艇はNULL扱い
        r2 = results[1]
        self.assertIsNone(r2["arrival_order"])
        self.assertIsNone(r2["st_time"])

    def test_payout_rows(self):
        payouts = self.parsed["payouts"]
        by_type = {}
        for p in payouts:
            by_type.setdefault(p["bet_type"], []).append(p)

        self.assertEqual(by_type["3連単"][0]["combination"], "1-5-3")
        self.assertEqual(by_type["3連単"][0]["amount_yen"], 12690)
        self.assertEqual(by_type["3連複"][0]["combination"], "1=3=5")
        self.assertEqual(by_type["2連単"][0]["amount_yen"], 1860)
        self.assertEqual(by_type["2連複"][0]["amount_yen"], 1190)
        self.assertEqual(len(by_type["拡連複"]), 2)
        self.assertEqual(by_type["単勝"][0]["amount_yen"], 290)
        # 不成立(combination=None)は除外され、複勝は1件のみ
        self.assertEqual(len(by_type["複勝"]), 1)

    def test_empty_input(self):
        parsed = parse_result({"results": []})
        self.assertEqual(parsed, {"races": [], "results": [], "payouts": []})


if __name__ == "__main__":
    unittest.main()
