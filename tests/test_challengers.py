import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import challengers as C
import predictors as P


def _ranked(probs):
    return [{"lane": i + 1, "name": f"選手{i}", "racer_class": "A1", "prob": p}
            for i, p in enumerate(probs)]


FLAT = _ranked([0.20, 0.18, 0.17, 0.16, 0.15, 0.14])   # 混戦(荒れ注意)
HARD = _ranked([0.60, 0.15, 0.10, 0.07, 0.05, 0.03])   # 堅め


class TestSelectionMetrics(unittest.TestCase):
    def test_top_gap(self):
        self.assertAlmostEqual(C.top_gap(P.normalize_probs(FLAT)), 0.02)
        self.assertAlmostEqual(C.top_gap(P.normalize_probs(HARD)), 0.45)

    def test_entropy_bounds(self):
        # 6艇均等が最大 ln(6)。偏るほど小さい
        uniform = {i: 1 / 6 for i in range(1, 7)}
        self.assertAlmostEqual(C.entropy(uniform), math.log(6), places=9)
        self.assertLess(C.entropy(P.normalize_probs(HARD)), C.entropy(P.normalize_probs(FLAT)))

    def test_market_divergence(self):
        self.assertEqual(C.market_divergence([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]), 0)
        # 完全逆順のフットルール距離は18(6艇)
        self.assertEqual(C.market_divergence([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]), 18)
        # 上位2艇の入れ替わりのみ → 1+1=2
        self.assertEqual(C.market_divergence([1, 2, 3, 4, 5, 6], [2, 1, 3, 4, 5, 6]), 2)


class TestSelectors(unittest.TestCase):
    def test_champion_boundary(self):
        # 1位勝率35%が境界(35%ちょうどは標準=非選別)
        self.assertIsNotNone(C.champion_score(_ranked([0.349, 0.2, 0.15, 0.13, 0.1, 0.07])))
        self.assertIsNone(C.champion_score(_ranked([0.350, 0.2, 0.15, 0.13, 0.1, 0.07])))

    def test_gap_selector_boundary(self):
        probs = P.normalize_probs(FLAT)  # gap=0.02
        self.assertIsNotNone(C.gap_score(probs, threshold=0.021))
        self.assertIsNone(C.gap_score(probs, threshold=0.02))  # 未満条件なので同値は非選別

    def test_entropy_selector_boundary(self):
        probs = P.normalize_probs(FLAT)
        e = C.entropy(probs)
        self.assertIsNotNone(C.entropy_score(probs, threshold=e - 1e-9))
        self.assertIsNone(C.entropy_score(probs, threshold=e))  # 超条件なので同値は非選別

    def test_divergence_selector_none_when_no_market(self):
        # スナップショット無し=判定不能(選別しない)
        self.assertIsNone(C.divergence_score([1, 2, 3, 4, 5, 6], None))
        self.assertIsNotNone(C.divergence_score([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], threshold=7))
        self.assertIsNone(C.divergence_score([1, 2, 3, 4, 5, 6], [2, 1, 3, 4, 5, 6], threshold=7))

    def test_challenger3_is_stub(self):
        with self.assertRaises(NotImplementedError):
            C.challenger_c_condition(FLAT, [], {})

    def test_daily_cap(self):
        cands = [(float(i), f"r{i}") for i in range(15)]
        capped = C.daily_cap(cands, limit=10)
        self.assertEqual(len(capped), 10)
        self.assertEqual(capped[0], "r14")  # score降順


class TestCalibration(unittest.TestCase):
    def test_below_mode_matches_target_count(self):
        values = [i / 100 for i in range(1, 101)]  # 0.01..1.00
        th = C.calibrate_threshold(values, target=30, mode="below")
        n = sum(1 for v in values if v < th)
        self.assertAlmostEqual(n, 30, delta=3)  # ±10%以内

    def test_above_mode_matches_target_count(self):
        values = [i / 100 for i in range(1, 101)]
        th = C.calibrate_threshold(values, target=30, mode="above")
        n = sum(1 for v in values if v > th)
        self.assertAlmostEqual(n, 30, delta=3)

    def test_empty_values_select_nothing(self):
        # 較正データが無ければ「何も選別しない」安全側の閾値
        self.assertEqual(C.calibrate_threshold([], 10, "below"), float("-inf"))
        self.assertEqual(C.calibrate_threshold([], 10, "above"), float("inf"))


class TestOutcomeClassification(unittest.TestCase):
    def test_three_classes_and_miss(self):
        self.assertIsNone(C.classify_outcome(1000, 0, 50000))          # 不的中
        self.assertEqual(C.classify_outcome(1000, 600, 3000), "順当")   # ガミ
        self.assertEqual(C.classify_outcome(1000, 600, 50000), "順当")  # 万舟レースのガミも順当扱い
        self.assertEqual(C.classify_outcome(1000, 2400, 3000), "中波乱")
        self.assertEqual(C.classify_outcome(1000, 8000, 12000), "万舟")

    def test_boundary_values(self):
        # 払戻=掛金ちょうどは順当ではない(損していない)
        self.assertEqual(C.classify_outcome(1000, 1000, 5000), "中波乱")
        # 3連単10,000円ちょうどは万舟
        self.assertEqual(C.classify_outcome(1000, 5000, 10000), "万舟")


class TestShowProbability(unittest.TestCase):
    def test_matches_trifecta_marginalization(self):
        probs = P.normalize_probs(FLAT)
        tri = P.trifecta_probs(probs)
        expect = sum(p for k, p in tri.items() if 1 in k)
        self.assertAlmostEqual(C.show_probability(probs, 1), expect, places=9)

    def test_all_lanes_sum_to_three(self):
        # 各レースで3艇が複勝圏に入るので、6艇の複勝確率合計は3
        probs = P.normalize_probs(HARD)
        total = sum(C.show_probability(probs, lane) for lane in range(1, 7))
        self.assertAlmostEqual(total, 3.0, places=6)


class TestRiskMetrics(unittest.TestCase):
    def test_max_drawdown(self):
        # +5, -3, -4(累積-2, ピーク5からのDD=-7), +10
        self.assertAlmostEqual(C.max_drawdown([5, -3, -4, 10]), -7)
        self.assertEqual(C.max_drawdown([1, 2, 3]), 0)

    def test_longest_losing_streak(self):
        self.assertEqual(C.longest_losing_streak([-1, -1, 2, -1, -1, -1, 3]), 3)
        self.assertEqual(C.longest_losing_streak([1, 2]), 0)


class TestCompositions(unittest.TestCase):
    C_PICKS = [("3連単", "5-6-4", 0.004), ("3連単", "6-5-4", 0.003)]

    def _plan(self, name, ranked=None):
        return C.build_composition(name, ranked or FLAT, self.C_PICKS)

    def test_all_compositions_total_1000(self):
        for name in C.COMPOSITION_NAMES:
            plan = self._plan(name)
            self.assertEqual(sum(y for _, _, y, _ in plan), 1000, name)
            for _, _, yen, _ in plan:
                self.assertEqual(yen % 100, 0, name)  # 100円単位

    def test_current_equals_ken_portfolio(self):
        expect = P.ken_portfolio("荒れ注意", FLAT, [], self.C_PICKS)
        self.assertEqual(self._plan("現行"), expect)

    def test_attack_tickets(self):
        # 原案の250円×2は舟券の100円単位制約で購入不可のため300/200に事前調整済み
        plan = self._plan("攻撃型")
        combos = [(bt, comb, yen) for bt, comb, yen, _ in plan]
        self.assertIn(("3連複", "1=2=3", 300), combos)
        self.assertIn(("3連複", "1=2=4", 200), combos)
        for tf in ("3-1-2", "4-1-2", "3-1-4", "4-1-3"):
            self.assertIn(("3連単", tf, 100), combos)
        self.assertEqual(len(plan), 7)  # 6点+C

    def test_defense_has_no_trifecta(self):
        plan = self._plan("守備型")
        # 3連単はC勝万舟の1点のみ(構成本体は3連複4点)
        own = [x for x in plan if x[3] == "守備"]
        self.assertTrue(all(bt == "3連複" for bt, _, _, _ in own))
        self.assertEqual([y for _, _, y, _ in own], [300, 300, 200, 100])
        self.assertIn(("3連複", "2=3=4", 100), [(bt, comb, yen) for bt, comb, yen, _ in own])

    def test_axis_spread_replaces_one_trio(self):
        plan = self._plan("軸分散型")
        combos = [comb for _, comb, _, _ in plan]
        self.assertIn("2=3=4", combos)      # 予測1位不在の3連複に置換
        self.assertNotIn("1=3=4", combos)   # 現行の3点目は買わない
        self.assertIn("3-1-2", combos)      # 3連単2点は現行と同一
        self.assertIn("4-1-2", combos)

    def test_c_empty_gives_900(self):
        for name in C.COMPOSITION_NAMES:
            plan = C.build_composition(name, FLAT, [])
            self.assertEqual(sum(y for _, _, y, _ in plan), 900, name)

    def test_short_field_returns_empty(self):
        for name in C.COMPOSITION_NAMES:
            self.assertEqual(C.build_composition(name, FLAT[:3], self.C_PICKS), [], name)