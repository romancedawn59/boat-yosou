import json
import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import noon_update
import predict
import predictors as P
from noon_update import _apply_morning_picks, build_notify_text, build_odds_view
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
        "bets": {"confidence": conf, "plan": P.ken_portfolio(conf, ranked, b, c)},
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


class TestApplyMorningPicks(unittest.TestCase):
    """朝のpicks JSONによる表示固定(schedule遅延でnoonが朝より先に走った場合の対策)"""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        root = Path(self.tmpdir.name)
        self.site_dir = root / "site"
        self.project_dir = root / "project"
        for p in (patch.object(predict, "SITE_DIR", self.site_dir),
                  patch.object(noon_update, "PROJECT_DIR", self.project_dir)):
            p.start()
            self.addCleanup(p.stop)
        self.d = date(2026, 7, 5)
        self.race = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])

    def _morning_entry(self, race_id):
        return {
            "race_id": race_id,
            "venue_code": 4,
            "race_no": 1,
            "confidence": "荒れ注意",
            "shobusho": "本命",
            "a": [["2連複", "1=2", 0.3]],
            "b": [["3連単", "1-2-3", 0.1]],
            "c": [["3連単", "6-5-4", 0.004]],
            "ken": [["3連複", "1=2=3", 200, "検証済み"]],
        }

    def _write_picks(self, base_dir, entries):
        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / f"picks_{self.d.isoformat()}.json").write_text(
            json.dumps({"date": self.d.isoformat(), "races": entries}, ensure_ascii=False),
            encoding="utf-8")

    def test_returns_false_when_json_missing(self):
        self.assertFalse(_apply_morning_picks([self.race], self.d))

    def test_overwrites_from_site_data(self):
        self._write_picks(self.site_dir, [self._morning_entry(self.race["race_id"])])
        self.assertTrue(_apply_morning_picks([self.race], self.d))
        self.assertEqual(self.race["bets"]["confidence"], "荒れ注意")
        self.assertEqual(self.race["shobusho"], "本命")
        self.assertEqual(self.race["bets"]["plan"], [("3連複", "1=2=3", 200, "検証済み")])
        self.assertEqual(self.race["picks_a"], [("2連複", "1=2", 0.3)])
        self.assertEqual(self.race["picks_b"], [("3連単", "1-2-3", 0.1)])
        self.assertEqual(self.race["picks_c"], [("3連単", "6-5-4", 0.004)])

    def test_falls_back_to_docs_data(self):
        # reports/site/data に無ければ docs/data(公開済みの朝版)を使う
        self._write_picks(self.project_dir / "docs", [self._morning_entry(self.race["race_id"])])
        self.assertTrue(_apply_morning_picks([self.race], self.d))
        self.assertEqual(self.race["shobusho"], "本命")

    def test_unknown_race_id_keeps_recomputed_values(self):
        self._write_picks(self.site_dir, [self._morning_entry("20260705_08_01")])
        before_conf = self.race["bets"]["confidence"]
        before_plan = list(self.race["bets"]["plan"])
        before_c = list(self.race["picks_c"])
        self.assertTrue(_apply_morning_picks([self.race], self.d))
        self.assertEqual(self.race["bets"]["confidence"], before_conf)
        self.assertEqual(self.race["bets"]["plan"], before_plan)
        self.assertEqual(self.race["picks_c"], before_c)
        self.assertIsNone(self.race["shobusho"])


class TestBuildOddsCheck(unittest.TestCase):
    """要オッズ確認の一桁オッズ判定(帯・場の制限なしの共通関数)"""

    @staticmethod
    def _rows(fuku_odds, tan_odds=()):
        rows = [("3連複", "1=2=3", o, 0, 0.0) for o in fuku_odds]
        rows += [("3連単", "3-1-2", o, 0, 0.0) for o in tan_odds]
        return rows

    def test_exactly_two_singles_is_chance(self):
        oc = noon_update.build_odds_check(self._rows([5.2, 9.9, 12.4, 55.0]))
        self.assertEqual(oc, {"singles": 2, "verdict": "chance", "n_fuku": 4})

    def test_zero_or_one_single_is_chaos(self):
        self.assertEqual(
            noon_update.build_odds_check(self._rows([12.0, 30.0]))["verdict"],
            "chaos")
        self.assertEqual(
            noon_update.build_odds_check(self._rows([5.0, 12.0]))["verdict"],
            "chaos")

    def test_three_singles_is_cheap(self):
        oc = noon_update.build_odds_check(self._rows([2.0, 5.0, 9.0, 20.0]))
        self.assertEqual(oc["verdict"], "cheap")

    def test_tan_only_or_missing_odds_returns_none(self):
        # 3連複が無い、またはオッズ欠落(None)のみなら判定不能
        self.assertIsNone(noon_update.build_odds_check(self._rows([], [7.0])))
        self.assertIsNone(noon_update.build_odds_check(self._rows([None, None])))


class TestAppendOddsCheckLog(unittest.TestCase):
    """全レース紙上記録の追記保存(2026-08-04〜)"""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        root = Path(self.tmpdir.name)
        self.site_dir = root / "site"
        self.project_dir = root / "project"
        for p in (patch.object(predict, "SITE_DIR", self.site_dir),
                  patch.object(noon_update, "PROJECT_DIR", self.project_dir)):
            p.start()
            self.addCleanup(p.stop)
        self.d = date(2026, 8, 4)

    def _read(self):
        path = self.site_dir / "data" / f"odds_check_{self.d.isoformat()}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_appends_and_replaces_snapshots(self):
        rec1 = [{"race_id": "20260804_04_03", "check": {"verdict": "chance"}}]
        noon_update._append_odds_check_log(self.d, "09:01", rec1)
        self.assertEqual([s["fetched"] for s in self._read()["snapshots"]],
                         ["09:01"])

        # 別時刻は追記される
        noon_update._append_odds_check_log(self.d, "10:32", rec1)
        self.assertEqual([s["fetched"] for s in self._read()["snapshots"]],
                         ["09:01", "10:32"])

        # 同時刻ラベルの再実行は上書き(重複しない)
        rec2 = [{"race_id": "20260804_04_09", "check": None}]
        noon_update._append_odds_check_log(self.d, "10:32", rec2)
        log = self._read()
        self.assertEqual([s["fetched"] for s in log["snapshots"]],
                         ["09:01", "10:32"])
        self.assertEqual(log["snapshots"][-1]["races"], rec2)

    def test_reads_existing_from_docs_fallback(self):
        # Actionsのまっさらなrunner: site/dataに無くdocs/dataに前回分がある場合
        docs_data = self.project_dir / "docs" / "data"
        docs_data.mkdir(parents=True)
        prior = {"date": self.d.isoformat(),
                 "snapshots": [{"fetched": "09:01", "races": []}]}
        (docs_data / f"odds_check_{self.d.isoformat()}.json").write_text(
            json.dumps(prior, ensure_ascii=False), encoding="utf-8")

        noon_update._append_odds_check_log(self.d, "10:32", [])
        self.assertEqual([s["fetched"] for s in self._read()["snapshots"]],
                         ["09:01", "10:32"])


class TestMergedOddsCheck(unittest.TestCase):
    def test_latest_snapshot_wins_and_sorted_by_deadline(self):
        log = {"snapshots": [
            {"fetched": "09:01", "races": [
                {"race_id": "a", "deadline": "2026-08-04 12:54:00",
                 "check": {"verdict": "chaos"}},
                {"race_id": "b", "deadline": "2026-08-04 11:55:00",
                 "check": {"verdict": "cheap"}},
            ]},
            {"fetched": "10:32", "races": [
                {"race_id": "a", "deadline": "2026-08-04 12:54:00",
                 "check": {"verdict": "chance"}},
            ]},
        ]}
        merged = noon_update.merged_odds_check(log)
        # 締切順(bが先)・aは10:32の判定で上書き・bは09:01の値が残る
        self.assertEqual([r["race_id"] for r in merged], ["b", "a"])
        self.assertEqual(merged[1]["check"]["verdict"], "chance")
        self.assertEqual(merged[1]["fetched"], "10:32")
        self.assertEqual(merged[0]["fetched"], "09:01")


class TestOddsCheckDisplay(unittest.TestCase):
    """検証済み帯=色付き表示・検証外帯=記録用の控えめ表示・indexのテーブル"""

    def test_validated_band_gets_colored_verdict(self):
        # 平和島(5場)×1位30〜35%帯 → validated
        race = _race([0.32, 0.2, 0.18, 0.15, 0.1, 0.05])
        odds = _flat_odds(25.0)
        # 3連複の先頭2点だけ一桁にして「ちょうど2点」を作る
        fuku_keys = [tuple(sorted(int(x) for x in comb.split("=")))
                     for bt, comb, _y, _s in race["bets"]["plan"] if bt == "3連複"]
        odds["3連複"][fuku_keys[0]] = 5.0
        odds["3連複"][fuku_keys[1]] = 8.0
        view = build_odds_view(race, odds, "10:30")
        self.assertTrue(view["odds_check"]["validated"])
        self.assertEqual(view["odds_check"]["verdict"], "chance")
        pane = _render_odds_pane(view)
        self.assertIn("○購入チャンス(裁量)", pane)

    def test_out_of_band_gets_muted_recording_note(self):
        # 1位50%超=検証外の帯 → 記録用表示のみ(色付き判定は出さない)
        race = _race([0.55, 0.15, 0.1, 0.1, 0.05, 0.05])
        view = build_odds_view(race, _flat_odds(25.0), "10:30")
        self.assertFalse(view["odds_check"]["validated"])
        pane = _render_odds_pane(view)
        self.assertIn("記録用・検証外の帯", pane)
        self.assertNotIn("購入チャンス(裁量)", pane)

    def test_non_target_venue_is_not_validated(self):
        # 桐生(venue_code=1)は30〜35帯でも検証外
        race = _race([0.32, 0.2, 0.18, 0.15, 0.1, 0.05], venue_code=1)
        view = build_odds_view(race, _flat_odds(25.0), "10:30")
        self.assertFalse(view["odds_check"]["validated"])

    def test_shopping_page_renders_odds_check_table(self):
        race = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
        records = [
            {"race_id": race["race_id"], "venue_code": 4, "race_no": 3,
             "shobusho": "要注目", "p1": 0.32,
             "deadline": "2026-08-04 12:54:00", "fetched": "10:32",
             "check": {"singles": 2, "verdict": "chance", "n_fuku": 4,
                       "validated": True}},
            {"race_id": "20260804_01_05", "venue_code": 1, "race_no": 5,
             "shobusho": None, "p1": 0.55,
             "deadline": "2026-08-04 13:30:00", "fetched": "10:32",
             "check": {"singles": 4, "verdict": "cheap", "n_fuku": 4,
                       "validated": False}},
        ]
        html = predict.render_shopping_page(
            date(2026, 8, 4), [race], None, records)
        self.assertIn("一桁オッズ判定・全レース記録", html)
        self.assertIn("🟢検証済み帯", html)          # validated行の強調
        self.assertIn("桐生", html)                  # 5場以外もテーブルに載る
        self.assertIn("○1 / △0 / ×1(全2レース)", html)

    def test_shopping_page_without_records_has_no_table(self):
        race = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
        html = predict.render_shopping_page(date(2026, 8, 4), [race])
        self.assertNotIn("一桁オッズ判定・全レース記録", html)

    def test_shopping_page_has_selrace_panel(self):
        # 選択レースパネル(2026-08-09): 朝版(オッズ記録なし)でも常設される
        race = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
        html = predict.render_shopping_page(date(2026, 8, 9), [race])
        self.assertIn("選択レース(オッズ追っかけ用)", html)
        self.assertIn('SEL_DATE = "2026-08-09"', html)
        self.assertIn("平和島", html)          # 本日の場がセレクトに入る

    def test_oc_target_section_lists_30_35_band_races(self):
        # 追っかけ対象一覧(2026-08-09): 5場×30〜35%帯を朝から表示・判定待ち表記
        target = _race([0.32, 0.2, 0.18, 0.15, 0.1, 0.05], race_no=3)   # 対象
        katame = _race([0.55, 0.15, 0.1, 0.1, 0.05, 0.05], race_no=5)   # 帯外
        html = predict.render_shopping_page(date(2026, 8, 9), [target, katame])
        self.assertIn("オッズ追っかけ対象", html)
        self.assertIn("判定待ち", html)
        sec = html.split("オッズ追っかけ対象")[1].split("🎯")[0]
        self.assertIn("平和島3R", sec)
        self.assertNotIn("平和島5R", sec)

    def test_oc_target_section_shows_latest_verdict(self):
        target = _race([0.32, 0.2, 0.18, 0.15, 0.1, 0.05], race_no=3)
        records = [{"race_id": target["race_id"], "venue_code": 4, "race_no": 3,
                    "shobusho": "要注目", "p1": 0.32,
                    "deadline": "2026-08-09 12:54:00", "fetched": "12:11",
                    "check": {"singles": 2, "verdict": "chance", "n_fuku": 4,
                              "validated": True}}]
        html = predict.render_shopping_page(
            date(2026, 8, 9), [target], None, records)
        self.assertIn("○購入チャンス(一桁ちょうど2点) [12:11時点]", html)

    def test_odds_check_rows_are_clickable(self):
        race = _race([0.25, 0.2, 0.2, 0.15, 0.1, 0.1])
        records = [
            {"race_id": race["race_id"], "venue_code": 4, "race_no": 3,
             "shobusho": None, "p1": 0.32,
             "deadline": "2026-08-09 12:54:00", "fetched": "10:32",
             "check": {"singles": 2, "verdict": "chance", "n_fuku": 4,
                       "validated": True}},
        ]
        html = predict.render_shopping_page(
            date(2026, 8, 9), [race], None, records)
        self.assertIn("class='ocrow' data-v='4' data-r='3'", html)

    def test_kento_warehouse_shell_and_buttons(self):
        # 購入検討(倉庫・2026-08-09): シェル常設+各表に買い目取得ボタン
        target = _race([0.32, 0.2, 0.18, 0.15, 0.1, 0.05], race_no=3)
        records = [{"race_id": target["race_id"], "venue_code": 4, "race_no": 3,
                    "shobusho": None, "p1": 0.32,
                    "deadline": "2026-08-09 12:54:00", "fetched": "10:32",
                    "check": {"singles": 2, "verdict": "chance", "n_fuku": 4,
                              "validated": True}}]
        html = predict.render_shopping_page(
            date(2026, 8, 9), [target], None, records)
        self.assertIn("🛒 購入検討(倉庫", html)
        self.assertIn("kentoRender", html)
        # 追っかけ一覧・全レース表の両方に買い目取得ボタン
        self.assertGreaterEqual(
            html.count("class='kentobtn' data-v='4' data-r='3'"), 2)
        self.assertIn("買い目予想", html)   # 旧「取得(時刻)」列の置き換え


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
