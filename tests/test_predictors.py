import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predictors as P


def _ranked(probs):
    lanes = [1, 2, 3, 4, 5, 6]
    return [{"lane": lanes[i], "name": f"選手{i}", "racer_class": "A1", "prob": p}
            for i, p in enumerate(probs)]


RANKED = _ranked([0.50, 0.20, 0.12, 0.09, 0.06, 0.03])
PROBS = P.normalize_probs(RANKED)


class TestProbability(unittest.TestCase):
    def test_normalize_sums_to_one(self):
        self.assertAlmostEqual(sum(PROBS.values()), 1.0)

    def test_trifecta_probs_sum_to_one(self):
        tri = P.trifecta_probs(PROBS)
        self.assertEqual(len(tri), 120)
        self.assertAlmostEqual(sum(tri.values()), 1.0, places=6)

    def test_quinella_prob_matches_exacta_sum(self):
        tri = P.trifecta_probs(PROBS)
        # 2連複{1,2} = 3連単で1-2-*と2-1-*の合計
        expect = sum(p for (a, b, _c), p in tri.items() if {a, b} == {1, 2})
        self.assertAlmostEqual(P.quinella_prob(PROBS, 1, 2), expect, places=6)


class TestPicks(unittest.TestCase):
    def test_ishibashi_returns_top5_sorted_and_hard(self):
        picks = P.picks_ishibashi(PROBS)
        self.assertEqual(len(picks), 5)
        probs = [p for _, _, p in picks]
        self.assertEqual(probs, sorted(probs, reverse=True))
        # 最有力は本命2艇の2連複のはず
        self.assertEqual(picks[0][:2], ("2連複", "1=2"))
        for bt, _, _ in picks:
            self.assertIn(bt, ("2連複", "3連複"))

    def test_yamada_returns_top5_trifectas(self):
        picks = P.picks_yamada(PROBS)
        self.assertEqual(len(picks), 5)
        self.assertEqual(picks[0][:2], ("3連単", "1-2-3"))
        self.assertTrue(all(bt == "3連単" for bt, _, _ in picks))

    def test_katsu_all_below_threshold(self):
        picks = P.picks_katsu(PROBS)
        self.assertEqual(len(picks), 5)
        for _, _, p in picks:
            self.assertLessEqual(p, P.MANSHU_PROB_MAX)
        probs = [p for _, _, p in picks]
        self.assertEqual(probs, sorted(probs, reverse=True))


class TestKenPortfolio(unittest.TestCase):
    def _plans(self, confidence):
        a = P.picks_ishibashi(PROBS)
        b = P.picks_yamada(PROBS)
        c = P.picks_katsu(PROBS)
        return P.ken_portfolio(confidence, RANKED, a, b, c)

    def test_total_is_1000_and_includes_katsu_100(self):
        for conf in ("堅め", "標準", "荒れ注意"):
            plan = self._plans(conf)
            self.assertEqual(sum(y for _, _, y, _ in plan), 1000, conf)
            katsu = [x for x in plan if x[3] == "勝万舟"]
            self.assertEqual(len(katsu), 1, conf)
            self.assertEqual(katsu[0][2], 100, conf)

    def test_amounts_within_100_400(self):
        for conf in ("堅め", "標準", "荒れ注意"):
            for _, _, yen, _ in self._plans(conf):
                self.assertTrue(100 <= yen <= 400)
                self.assertEqual(yen % 100, 0)

    def test_areru_keeps_validated_core(self):
        plan = self._plans("荒れ注意")
        combos = [(bt, comb) for bt, comb, _, _ in plan]
        self.assertIn(("3連複", "1=2=3"), combos)
        self.assertIn(("3連複", "1=2=4"), combos)
        self.assertIn(("3連複", "1=3=4"), combos)
        self.assertIn(("3連単", "3-1-2"), combos)

    def test_short_field_returns_empty(self):
        short = _ranked([0.5, 0.3, 0.2])[:3]
        self.assertEqual(
            P.ken_portfolio("標準", short, [], [], [("3連単", "1-2-3", 0.001)]), [])


class TestShobusho(unittest.TestCase):
    def _race(self, conf, top_prob, has_plan=True):
        return {
            "bets": {"confidence": conf, "plan": [("x", "y", 100, "z")] if has_plan else []},
            "ranked": [{"lane": 1, "prob": top_prob}],
        }

    def test_areru_marked_honmei_standards_fill(self):
        races = [self._race("荒れ注意", 0.30), self._race("標準", 0.40),
                 self._race("堅め", 0.60), self._race("標準", 0.36)]
        P.select_shobusho(races, max_races=3)
        self.assertEqual(races[0]["shobusho"], "本命")
        self.assertEqual(races[3]["shobusho"], "準")   # 標準のうち1位勝率が低い方
        self.assertEqual(races[1]["shobusho"], "準")
        self.assertIsNone(races[2]["shobusho"])        # 堅めは選ばれない

    def test_cap_at_max(self):
        races = [self._race("荒れ注意", 0.2 + i * 0.01) for i in range(12)]
        P.select_shobusho(races, max_races=10)
        marked = [r for r in races if r["shobusho"] == "本命"]
        self.assertEqual(len(marked), 10)
        # 1位勝率が低い(=より荒れそうな)順に選ばれる
        self.assertIsNone(races[10]["shobusho"])
        self.assertIsNone(races[11]["shobusho"])


if __name__ == "__main__":
    unittest.main()
