import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


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
        self.assertIn("こんにちは".encode("utf-8"), sent_req.data)


if __name__ == "__main__":
    unittest.main()
