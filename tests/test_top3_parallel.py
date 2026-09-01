# -*- coding: utf-8 -*-
"""超混戦専用順位(3着内モデル)の並走表示のテスト(2026-09-01追加)

モデル未配置や確率欠損で配信が止まらないこと(top3_orderがNoneを返して
表示だけが消えること)が最重要の回帰ポイント。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict import top3_order


def boat(lane, p_win, p_top3):
    return {"lane": lane, "prob": p_win, "prob_top3": p_top3}


class TestTop3Order(unittest.TestCase):
    def test_orders_by_top3_prob(self):
        ranked = [boat(1, 0.19, 0.40), boat(3, 0.15, 0.70),
                  boat(5, 0.12, 0.55)]
        self.assertEqual(top3_order(ranked), [3, 5, 1])

    def test_none_when_model_missing(self):
        """prob_top3がNone(モデル未配置)なら表示なし=Noneを返す"""
        ranked = [boat(1, 0.19, None), boat(3, 0.15, 0.70)]
        self.assertIsNone(top3_order(ranked))

    def test_none_when_empty(self):
        self.assertIsNone(top3_order([]))

    def test_key_absent_is_safe(self):
        """古いJSON等でprob_top3キー自体が無くても落ちない"""
        ranked = [{"lane": 1, "prob": 0.19}]
        self.assertIsNone(top3_order(ranked))


if __name__ == "__main__":
    unittest.main()
