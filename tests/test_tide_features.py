import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

import db
from fetch_tide import VENUE_STATION, parse_tide_text
from verify_tide_features import tide_features_for


def _jma_line(values: list[int], yy: int, mm: int, dd: int, station: str) -> str:
    """気象庁潮位表テキストの1行(毎時24値×3桁+年月日+地点コード)を組み立てる"""
    assert len(values) == 24
    body = "".join(f"{v:3d}" for v in values)
    return f"{body}{yy:2d}{mm:2d}{dd:2d}{station:<2s}"


class TestParseTideText(unittest.TestCase):
    def test_parses_hourly_levels(self):
        values = list(range(100, 124))  # 100..123
        text = _jma_line(values, 26, 7, 1, "TK")
        rows = parse_tide_text(text)
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0], ("TK", "2026-07-01 00:00:00", 100.0))
        self.assertEqual(rows[23], ("TK", "2026-07-01 23:00:00", 123.0))

    def test_negative_levels_and_multiple_lines(self):
        # 干潮時の負値(例: -18cm)も潮位として読めること
        values = [-18] + [50] * 23
        text = "\n".join([
            _jma_line(values, 25, 12, 31, "MO"),
            "短すぎる行は無視される",
        ])
        rows = parse_tide_text(text)
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0], ("MO", "2025-12-31 00:00:00", -18.0))


class TestTideFeatures(unittest.TestCase):
    def setUp(self):
        # 東京(TK)の1日分: 潮位100→146(2cm/h上昇)の単調な上げ潮
        self.tide_map = {
            ("TK", f"2026-05-01 {h:02d}:00:00"): 100.0 + h * 2 for h in range(24)
        }

    def test_features_at_deadline_hour(self):
        feats = tide_features_for("2026-05-01 14:30:00", 4, self.tide_map)  # 平和島→TK
        self.assertEqual(feats["tide_level"], 128.0)   # 14時の潮位
        self.assertEqual(feats["tide_delta"], 2.0)     # 上げ潮=正
        self.assertEqual(feats["tide_range_day"], 46.0)

    def test_venue_without_station_returns_empty(self):
        # 対象5場以外(例: 桐生=1)は地点対応がないためNaN扱い(空dict)
        self.assertEqual(tide_features_for("2026-05-01 14:30:00", 1, self.tide_map), {})

    def test_missing_deadline_returns_empty(self):
        self.assertEqual(tide_features_for(None, 4, self.tide_map), {})

    def test_missing_tide_data_returns_empty(self):
        self.assertEqual(tide_features_for("2026-06-01 14:30:00", 4, self.tide_map), {})

    def test_station_mapping_matches_hypothesis(self):
        # 事前登録: 潮汐場(江戸川・平和島→東京、若松→門司)と対照場(常滑→名古屋、尼崎→大阪)
        self.assertEqual(VENUE_STATION[3], "TK")
        self.assertEqual(VENUE_STATION[4], "TK")
        self.assertEqual(VENUE_STATION[20], "MO")
        self.assertEqual(VENUE_STATION[8], "NG")
        self.assertEqual(VENUE_STATION[13], "OS")


class TestTideTable(unittest.TestCase):
    def test_upsert_tide_composite_pk(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        conn = db.connect(Path(tmpdir.name) / "test.db")
        self.addCleanup(conn.close)

        db.upsert_tide(conn, {"station": "TK", "datetime": "2026-05-01 00:00:00",
                              "level_cm": 100.0})
        # 同一(station, datetime)への再投入は上書きで行は増えない(冪等)
        db.upsert_tide(conn, {"station": "TK", "datetime": "2026-05-01 00:00:00",
                              "level_cm": 101.0})
        rows = conn.execute("SELECT level_cm FROM tide").fetchall()
        self.assertEqual(rows, [(101.0,)])


if __name__ == "__main__":
    unittest.main()
