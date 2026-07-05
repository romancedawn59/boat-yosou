import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exhibition import _ROW_PATTERN, parse_exhibition_html

SAMPLE_HTML = """
<td><a href="/data/racersearch/profile?toban=3973">崎　　　利仁</a></td>
<td rowspan="2">53.8kg</td>
<td rowspan="4">6.88</td>
<td rowspan="4">-0.5</td>
</tr><tr><td rowspan="2">0.0</td></tr>
<td><a href="/data/racersearch/profile?toban=5077">滝沢　　　崚</a></td>
<td rowspan="2">51.5kg</td>
<td rowspan="4">6.87</td>
<td rowspan="4">0.0</td>
</tr><tr><td rowspan="2">0.0</td></tr>
"""


class TestExhibitionPattern(unittest.TestCase):
    def test_extracts_reg_no_weight_time_tilt_in_order(self):
        rows = _ROW_PATTERN.findall(SAMPLE_HTML)
        self.assertEqual(rows, [
            ("3973", "53.8", "6.88", "-0.5"),
            ("5077", "51.5", "6.87", "0.0"),
        ])

    def test_no_match_on_unrelated_html(self):
        self.assertEqual(_ROW_PATTERN.findall("<html><body>非開催</body></html>"), [])


class TestParseExhibitionHtml(unittest.TestCase):
    def test_returns_dicts_with_lane_assigned_in_order(self):
        rows = parse_exhibition_html(SAMPLE_HTML)
        self.assertEqual(rows, [
            {"lane": 1, "reg_no": 3973, "weight_kg": 53.8, "exhibition_time": 6.88, "tilt": -0.5},
            {"lane": 2, "reg_no": 5077, "weight_kg": 51.5, "exhibition_time": 6.87, "tilt": 0.0},
        ])

    def test_empty_html_returns_empty_list(self):
        self.assertEqual(parse_exhibition_html("<html></html>"), [])


if __name__ == "__main__":
    unittest.main()
