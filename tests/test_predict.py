import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict import recommend_bets, render_html


def _ranked(probs: list[float]) -> list[dict]:
    """勝率降順のダミー艇リスト。lane=順位とずらして組み目を検証しやすくする"""
    lanes = [1, 3, 2, 4, 5, 6]
    return [
        {"lane": lanes[i], "name": f"選手{i}", "racer_class": "A1", "prob": p}
        for i, p in enumerate(probs[: len(lanes)])
    ]


class TestRecommendBets(unittest.TestCase):
    def test_areru_race_is_shobu_with_1000yen_plan(self):
        """荒れ注意 -> 勝負。3連複流し600円+3連単穴400円=1000円"""
        bets = recommend_bets(_ranked([0.25, 0.2, 0.2, 0.15, 0.1, 0.1]))
        self.assertEqual(bets["confidence"], "荒れ注意")
        self.assertEqual(bets["stance"], "勝負")
        self.assertEqual(bets["plan"], [
            ("3連複", "1=2=3", 200, "負けにくい"),
            ("3連複", "1=3=4", 200, "負けにくい"),
            ("3連複", "1=2=4", 200, "負けにくい"),
            ("3連単", "2-1-3", 200, "大穴"),
            ("3連単", "4-1-3", 200, "大穴"),
        ])
        self.assertEqual(sum(y for _, _, y, _ in bets["plan"]), 1000)

    def test_standard_race_is_skip_with_reference_plan(self):
        bets = recommend_bets(_ranked([0.40, 0.3, 0.1, 0.1, 0.05, 0.05]))
        self.assertEqual(bets["confidence"], "標準")
        self.assertEqual(bets["stance"], "見送り推奨")
        self.assertEqual([tag for _, _, _, tag in bets["plan"]], ["参考"] * 3)
        self.assertEqual(sum(y for _, _, y, _ in bets["plan"]), 300)

    def test_solid_race_is_skip_with_small_reference(self):
        bets = recommend_bets(_ranked([0.60, 0.2, 0.1, 0.05, 0.03, 0.02]))
        self.assertEqual(bets["confidence"], "堅め")
        self.assertEqual(bets["stance"], "見送り推奨")
        self.assertEqual(bets["plan"], [
            ("2連複", "1=3", 100, "参考"),
            ("3連単", "1-3-2", 100, "参考"),
        ])

    def test_short_field_areru_falls_back_to_skip(self):
        """荒れ注意でも4艇未満なら勝負プランは組めず見送り扱い"""
        bets = recommend_bets(_ranked([0.3, 0.3, 0.2]))
        self.assertEqual(bets["confidence"], "荒れ注意")
        self.assertEqual(bets["stance"], "見送り推奨")


class TestRenderHtml(unittest.TestCase):
    def _race(self, probs, venue_code=4, race_no=1, wx=None):
        from config import VENUE_NAMES
        ranked = _ranked(probs)
        return {
            "venue_code": venue_code,
            "venue_name": VENUE_NAMES[venue_code],
            "race_no": race_no,
            "deadline": "2026-07-05 10:47:00",
            "weather": wx,
            "ranked": ranked,
            "bets": recommend_bets(ranked),
        }

    def test_render_groups_by_venue_and_lists_shobu(self):
        races = [
            self._race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], venue_code=4, race_no=5),
            self._race([0.60, 0.2, 0.1, 0.05, 0.03, 0.02], venue_code=20, race_no=1),
        ]
        html = render_html(date(2026, 7, 5), races)
        self.assertIn("本日の勝負レース: <b>平和島5R</b>", html)
        self.assertIn("<h2>平和島</h2>", html)
        self.assertIn("<h2>若松</h2>", html)
        self.assertIn("買い目プラン(計1,000円)", html)
        self.assertIn("viewport", html)  # スマホ対応

    def test_render_no_shobu_day(self):
        races = [self._race([0.60, 0.2, 0.1, 0.05, 0.03, 0.02])]
        html = render_html(date(2026, 7, 5), races)
        self.assertIn("本日は勝負レースなし", html)
        self.assertIn("見送り推奨", html)
        self.assertIn("参考買い目", html)

    def test_render_with_weather_forecast(self):
        wx = {"wind_speed_m": 3.5, "wind_dir": "南東", "wave_height_cm": 2.4, "temperature": 28.0}
        races = [self._race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], wx=wx)]
        html = render_html(date(2026, 7, 5), races)
        self.assertIn("風速3.5m/s(南東の風)", html)
        self.assertIn("予測には未使用", html)

    def test_render_without_weather_forecast(self):
        """気象予報の取得に失敗した場合でも落ちない"""
        races = [self._race([0.60, 0.2, 0.1, 0.05, 0.03, 0.02], wx=None)]
        html = render_html(date(2026, 7, 5), races)
        self.assertNotIn("予報:", html)


if __name__ == "__main__":
    unittest.main()
