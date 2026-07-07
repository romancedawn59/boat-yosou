import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import PAGES_URL
from predict import build_notify_text, recommend_bets, shobu_races


def _ranked(probs: list[float]) -> list[dict]:
    lanes = [1, 3, 2, 4, 5, 6]
    return [
        {"lane": lanes[i], "name": f"選手{i}", "racer_class": "A1", "prob": p}
        for i, p in enumerate(probs[: len(lanes)])
    ]


def _race(probs, venue_code, venue_name, race_no):
    ranked = _ranked(probs)
    return {
        "venue_code": venue_code,
        "venue_name": venue_name,
        "race_no": race_no,
        "deadline": "2026-07-05 10:47:00",
        "weather": None,
        "ranked": ranked,
        "bets": recommend_bets(ranked),
    }


class TestShobuRaces(unittest.TestCase):
    def test_collects_labels_and_budget(self):
        races = [
            _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], 4, "平和島", 5),
            _race([0.60, 0.2, 0.1, 0.05, 0.03, 0.02], 20, "若松", 1),
            _race([0.20, 0.2, 0.2, 0.2, 0.1, 0.1], 13, "尼崎", 2),
        ]
        labels, budget = shobu_races(races)
        self.assertEqual(labels, ["平和島5R", "尼崎2R"])
        self.assertEqual(budget, 2200)  # 勝負プランは万舟枠込みで1レース1,100円

    def test_no_shobu_returns_empty(self):
        races = [_race([0.60, 0.2, 0.1, 0.05, 0.03, 0.02], 20, "若松", 1)]
        labels, budget = shobu_races(races)
        self.assertEqual(labels, [])
        self.assertEqual(budget, 0)


class TestBuildNotifyText(unittest.TestCase):
    def test_includes_races_budget_and_url(self):
        races = [_race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], 4, "平和島", 5)]
        text = build_notify_text(date(2026, 7, 5), races)
        self.assertIn("2026-07-05", text)
        self.assertIn("平和島5R", text)
        self.assertIn("予算: 1,100円", text)
        self.assertIn(f"{PAGES_URL}/archive/2026-07-05_picks.html", text)

    def test_no_shobu_day_text(self):
        races = [_race([0.60, 0.2, 0.1, 0.05, 0.03, 0.02], 20, "若松", 1)]
        text = build_notify_text(date(2026, 7, 5), races)
        self.assertIn("本日は勝負レースなし", text)


class TestNotifyLineSend(unittest.TestCase):
    def test_skips_without_token(self):
        import notify_line
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(notify_line.send("test"))

    def test_posts_with_bearer_token(self):
        import notify_line
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        with patch.dict("os.environ", {"LINE_CHANNEL_ACCESS_TOKEN": "dummy-token"}), \
             patch("notify_line.urllib.request.urlopen", return_value=mock_resp) as mock_open:
            ok = notify_line.send("こんにちは")

        self.assertTrue(ok)
        sent_req = mock_open.call_args[0][0]
        self.assertEqual(sent_req.get_header("Authorization"), "Bearer dummy-token")
        self.assertIn(b"\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf", sent_req.data)


if __name__ == "__main__":
    unittest.main()
