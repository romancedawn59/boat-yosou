"""検証の作法の道具(test/validation_kit.py)のテスト"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test"))

import validation_kit as V


class ValidationKitTest(unittest.TestCase):
    def test_enough_hits_threshold(self):
        self.assertFalse(V.enough_hits(9))
        self.assertTrue(V.enough_hits(10))

    def test_wilson_contains_observed_rate(self):
        lo, hi = V.wilson(8, 95)
        self.assertLess(lo, 8 / 95)
        self.assertGreater(hi, 8 / 95)
        self.assertEqual(V.wilson(0, 0), (0.0, 0.0))

    def test_shuffle_pvalue_is_high_when_labels_carry_no_signal(self):
        # 24ラベルに同じ分布の払戻を割り振る → 最良ラベルの見た目は偶然でも出る
        import random
        rng = random.Random(1)
        labels = [i % 24 for i in range(2400)]
        payoffs = [5000.0 if rng.random() < 0.05 else 0.0 for _ in range(2400)]
        p = V.label_shuffle_pvalue(labels, payoffs, stake=500, n_iter=200)
        self.assertGreater(p, 0.05)

    def test_shuffle_pvalue_is_low_when_one_label_is_truly_better(self):
        labels = [0] * 300 + [1] * 300
        payoffs = [3000.0 if i % 3 == 0 else 0.0 for i in range(300)] + [0.0] * 300
        p = V.label_shuffle_pvalue(labels, payoffs, stake=500, n_iter=200)
        self.assertLess(p, 0.05)


if __name__ == "__main__":
    unittest.main()
