import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

from study_log import classify, score_of


class TestScore(unittest.TestCase):
    def test_100_percent_or_more_caps_at_100(self):
        # 払戻率100%以上は100点(2026-07-18ケンさん定義)
        self.assertEqual(score_of(1000, 1000), 100)
        self.assertEqual(score_of(1000, 74510), 100)

    def test_below_100_is_payout_rate(self):
        self.assertEqual(score_of(1000, 700), 70)   # ガミ許容ラインの70%相当
        self.assertEqual(score_of(1000, 180), 18)
        self.assertEqual(score_of(900, 450), 50)    # C空の900円プランにも対応

    def test_zero_guard(self):
        self.assertEqual(score_of(0, 0), 0)


class TestClassify(unittest.TestCase):
    def test_c_solo_hit_is_santan_only(self):
        # 3連単のみの的中=C勝万舟の単独的中(穴頭3連単は3連複と同時に当たるため)
        tag, memo = classify([("3連単", "5-4-6", 100, 8040)], 900, 8040)
        self.assertEqual(tag, "3連単のみ(C単独)")
        self.assertIn("合格点", memo)

    def test_c_solo_low_payout_flags_divergence(self):
        tag, memo = classify([("3連単", "5-1-2", 100, 800)], 1000, 800)
        self.assertEqual(tag, "3連単のみ(C単独)")
        self.assertIn("乖離", memo)  # 万舟圏想定なのに低配当→要研究

    def test_trio_only_gami_is_selection_issue(self):
        tag, memo = classify([("3連複", "1=2=3", 200, 480)], 1000, 480)
        self.assertEqual(tag, "3連複のみ")
        self.assertIn("選別の課題", memo)

    def test_snapshot_note_appended_only_below_100(self):
        note = "購入時点で市場は順当視(的中3連複は2.4倍)"
        _, memo = classify([("3連複", "1=2=3", 200, 480)], 1000, 480, note)
        self.assertIn("順当視", memo)
        _, memo_ok = classify([("3連単", "3-1-2", 200, 5000)], 1000, 5400, note)
        self.assertNotIn("順当視", memo_ok)  # 100点の行には注記しない


if __name__ == "__main__":
    unittest.main()
