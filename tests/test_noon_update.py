import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predictors as P
from noon_update import build_notify_text, build_odds_view
from predict import _render_odds_pane, render_venue_page


def _ranked(probs):
    lanes = [1, 2, 3, 4, 5, 6]
    return [{"lane": lanes[i], "name": f"選手{i}", "racer_class": "A1", "prob": p}
            for i, p in enumerate(probs)]


def _race(probs, venue_code=4, race_no=1):
    from config import VENUE_NAMES
    ranked = _ranked(probs)
    norm = P.normalize_probs(ranked)
    a, b, c = P.picks_ishibashi(norm), P.picks_yamada(norm), P.picks_katsu(norm)
    conf = P.bucket_of(ranked[0]["prob"])
    return {
        "race_id": f"20260705_{venue_code:02d}_{race_no:02d}",
        "venue_code": venue_code,
        "venue_name": VENUE_NAMES[venue_code],
        "race_no": race_no,
        "deadline": "2026-07-05 14:00:00",
        "weather": None,
        "ranked": ranked,
        "picks_a": a, "picks_b": b, "picks_c": c,
        "bets": {"confidence": conf, "plan": P.ken_portfolio(conf, ranked, a, b, c)},
        "shobusho": None,
    }


def _flat_odds(value=10.0):
    """全組み合わせ一律オッズのダミーデータ"""
    from itertools import combinations, permutations
    return {
        "3連単": {k: value for k in permutations(range(1, 7), 3)},
        "3連複": {k: value for k in combinations(range(1, 7), 3)},
    }


class TestBuildOddsView(unittest.TestCase):
    def setUp(self):
        self.race = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
        self.view = build_odds_view(self.race, _flat_odds(10.0), "12:00")

    def test_ken_rows_cover_all_plan_points(self):
        self.assertEqual(len(self.view["ken_rows"]), len(self.race["bets"]["plan"]))
        for bt, comb, o, est, ev in self.view["ken_rows"]:
            self.assertEqual(o, 10.0)
            self.assertGreater(est, 0)   # 想定払戻=オッズ×金額
            self.assertGreater(ev, 0)

    def test_estimated_return_is_odds_times_stake(self):
        plan = self.race["bets"]["plan"]
        for (bt, comb, o, est, ev), (_bt, _comb, yen, _src) in zip(self.view["ken_rows"], plan):
            self.assertEqual(est, int(10.0 * yen))

    def test_value_returns_top3(self):
        self.assertEqual(len(self.view["value"]), 3)

    def test_missing_odds_handled(self):
        view = build_odds_view(self.race, {"3連単": {}, "3連複": {}}, "12:00")
        for bt, comb, o, est, ev in view["ken_rows"]:
            self.assertIsNone(o)
            self.assertEqual(est, 0)
        self.assertEqual(view["value"], [])


class TestNotifyText(unittest.TestCase):
    def test_contains_time_venues_count_and_url(self):
        r1 = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], venue_code=4, race_no=1)
        r2 = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], venue_code=8, race_no=2)
        r3 = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], venue_code=8, race_no=3)
        panes = {r["race_id"]: "<div/>" for r in (r1, r2, r3)}
        text = build_notify_text("10:05", [r1, r2, r3], panes)

        self.assertIn("10:05", text)
        self.assertIn("平和島・常滑", text)     # 場コード順・重複なし
        self.assertIn("3レース", text)
        self.assertIn("boat-yosou", text)       # サイトURL

    def test_venues_limited_to_races_with_panes(self):
        r1 = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], venue_code=4, race_no=1)
        r2 = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], venue_code=8, race_no=2)
        panes = {r1["race_id"]: "<div/>"}  # 常滑は締切済みでオッズ無し
        text = build_notify_text("10:05", [r1, r2], panes)

        self.assertIn("平和島", text)
        self.assertNotIn("常滑", text)
        self.assertIn("1レース", text)


class TestTabsRendering(unittest.TestCase):
    def test_page_without_odds_has_no_tabs(self):
        races = [_race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])]
        html = render_venue_page(date(2026, 7, 5), 4, races)
        self.assertNotIn('class="tabbtn', html)  # タブボタン要素がない(CSS定義は常在)
        self.assertIn("予想屋ken のポートフォリオ", html)

    def test_page_with_odds_has_tabs_and_both_panes(self):
        race = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
        view = build_odds_view(race, _flat_odds(25.0), "12:00")
        panes = {race["race_id"]: _render_odds_pane(view)}
        html = render_venue_page(date(2026, 7, 5), 4, [race], panes)

        self.assertIn("朝の予想", html)
        self.assertIn("オッズ反映⏱", html)
        self.assertIn("成績対象外", html)
        self.assertIn("オッズ取得: 12:00 時点", html)
        self.assertIn("swTab", html)  # タブ切替JS
        self.assertIn("25.0倍", html)

    def test_only_races_with_odds_get_tabs(self):
        r1 = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], race_no=1)
        r2 = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1], race_no=2)
        view = build_odds_view(r1, _flat_odds(), "12:00")
        panes = {r1["race_id"]: _render_odds_pane(view)}
        html = render_venue_page(date(2026, 7, 5), 4, [r1, r2], panes)
        self.assertEqual(html.count("オッズ反映⏱"), 1)


if __name__ == "__main__":
    unittest.main()
