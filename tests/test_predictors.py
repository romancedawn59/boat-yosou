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

    def test_yamada_returns_top10_trifectas(self):
        picks = P.picks_yamada(PROBS)
        self.assertEqual(len(picks), 10)
        self.assertEqual(picks[0][:2], ("3連単", "1-2-3"))
        self.assertTrue(all(bt == "3連単" for bt, _, _ in picks))
        probs = [p for _, _, p in picks]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_trio_top_returns_best_trios(self):
        trios = P.trio_top(PROBS, 2)
        self.assertEqual(len(trios), 2)
        self.assertEqual(trios[0][0], "1=2=3")  # 本命3艇の3連複が最有力
        self.assertGreaterEqual(trios[0][1], trios[1][1])

    def test_katsu_all_below_threshold(self):
        picks = P.picks_katsu(PROBS)
        self.assertEqual(len(picks), 5)
        for _, _, p in picks:
            self.assertLessEqual(p, P.MANSHU_PROB_MAX)
        probs = [p for _, _, p in picks]
        self.assertEqual(probs, sorted(probs, reverse=True))


class TestKenPortfolio(unittest.TestCase):
    def _plans(self, confidence):
        b = P.picks_yamada(PROBS)
        c = P.picks_katsu(PROBS)
        return P.ken_portfolio(confidence, RANKED, b, c)

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

    def test_no_quinella_in_any_plan(self):
        # 2連複は判断材料であり購入しない(検証⑦で採用)
        for conf in ("堅め", "標準", "荒れ注意"):
            for bt, _, _, _ in self._plans(conf):
                self.assertNotEqual(bt, "2連複", conf)

    def test_katame_and_hyojun_use_top_trios(self):
        for conf in ("堅め", "標準"):
            plan = self._plans(conf)
            trios = [(bt, comb) for bt, comb, _, src in plan if src == "本線"]
            self.assertEqual(trios[0], ("3連複", "1=2=3"), conf)
            self.assertEqual(len(trios), 2, conf)

    def test_short_field_returns_empty(self):
        short = _ranked([0.5, 0.3, 0.2])[:3]
        self.assertEqual(
            P.ken_portfolio("標準", short, [], [("3連単", "1-2-3", 0.001)]), [])


class TestFlatProbsRegression(unittest.TestCase):
    """確率が平坦なレースはC候補が0点になりうるが、kenプランは消えないこと(回帰テスト)。

    以前は ken_portfolio がC空でプラン全体を[]にし、最も荒れたレースが
    勝負所から静かに脱落するバグがあった。
    """

    FLAT = _ranked([0.20, 0.18, 0.17, 0.16, 0.15, 0.14])
    NORMAL = _ranked([0.30, 0.22, 0.16, 0.13, 0.10, 0.09])

    def test_flat_probs_make_katsu_empty(self):
        # 全120通りが万舟圏の閾値(0.005)を超えるためC候補は0点になる
        probs = P.normalize_probs(self.FLAT)
        self.assertEqual(P.picks_katsu(probs), [])

    def test_ken_returns_900yen_plan_without_katsu(self):
        probs = P.normalize_probs(self.FLAT)
        plan = P.ken_portfolio("荒れ注意", self.FLAT, P.picks_yamada(probs), [])
        self.assertEqual(len(plan), 5)  # 検証済み5点構成は維持される
        self.assertEqual(sum(y for _, _, y, _ in plan), 900)
        self.assertTrue(all(src == "検証済み" for _, _, _, src in plan))

    def test_normal_probs_keep_6point_1000yen_plan(self):
        probs = P.normalize_probs(self.NORMAL)
        c = P.picks_katsu(probs)
        self.assertTrue(c)  # 通常の荒れレースではC候補あり
        plan = P.ken_portfolio("荒れ注意", self.NORMAL, P.picks_yamada(probs), c)
        self.assertEqual(len(plan), 6)
        self.assertEqual(sum(y for _, _, y, _ in plan), 1000)
        self.assertEqual(len([x for x in plan if x[3] == "勝万舟"]), 1)


class TestShobusho(unittest.TestCase):
    """v2選別(ケンさん案): 本命=対象場×荒れ注意×上位cap / 超混戦=全場×top<20% /
    要注目=観測専用(本命の溢れ+標準の補充)"""

    VENUES = [3, 4, 8, 13, 20]

    def _race(self, conf, top_prob, venue=4, has_plan=True):
        return {
            "venue_code": venue,
            "bets": {"confidence": conf, "plan": [("x", "y", 100, "z")] if has_plan else []},
            "ranked": [{"lane": 1, "prob": top_prob}],
        }

    def _select(self, races, **kw):
        P.select_shobusho(races, honmei_venues=self.VENUES, **kw)

    def test_konsen_selected_from_any_venue(self):
        races = [self._race("荒れ注意", 0.18, venue=1),   # 対象外の場でも超混戦
                 self._race("荒れ注意", 0.30, venue=1)]   # 対象外の場×20%以上は選外
        self._select(races)
        self.assertEqual(races[0]["shobusho"], "超混戦")
        self.assertIsNone(races[1]["shobusho"])

    def test_honmei_cap_and_priority(self):
        # 対象場の荒れ注意7レース: 1位勝率が低い順にcap=6が本命、溢れは要注目
        races = [self._race("荒れ注意", 0.34 - i * 0.01) for i in range(7)]
        self._select(races)
        marks = [r["shobusho"] for r in races]
        self.assertEqual(marks.count("本命"), 6)
        self.assertEqual(races[0]["shobusho"], "要注目")  # 最も高い0.34が溢れる

    def test_target_venue_konsen_shows_as_honmei(self):
        # 対象場×20%未満は本命枠に入る(購入は1回・表示は本命を優先)
        races = [self._race("荒れ注意", 0.15)]
        self._select(races)
        self.assertEqual(races[0]["shobusho"], "本命")

    def test_attention_fills_from_standards(self):
        races = [self._race("荒れ注意", 0.30), self._race("標準", 0.40),
                 self._race("標準", 0.36), self._race("堅め", 0.60)]
        self._select(races, attention_cap=2)
        self.assertEqual(races[0]["shobusho"], "本命")
        self.assertEqual(races[2]["shobusho"], "要注目")  # 標準のうち1位勝率が低い方から
        self.assertEqual(races[1]["shobusho"], "要注目")
        self.assertIsNone(races[3]["shobusho"])           # 堅めは選ばれない

    def test_unbought_konsen_goes_to_attention(self):
        # プランが組めない超混戦は「購入0点」の要注目として観測に載せる(ユーザー指示)
        races = [self._race("荒れ注意", 0.15, has_plan=False)]
        self._select(races)
        self.assertEqual(races[0]["shobusho"], "要注目")


if __name__ == "__main__":
    unittest.main()
