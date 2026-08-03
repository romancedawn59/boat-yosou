"""番組表の欠落検知(2026-07-24の配信停止バグの回帰テスト)

朝の収集時点で番組表が未公開(404)だと、結果JSONからレース枠だけが作られ
entriesが空になる。races件数で開催判定していたため予測が素通りし、
特徴量0件のままLightGBMに渡ってクラッシュした。
"""
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
import predict

D = date(2026, 7, 24)


class TestEnsureProgram(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmpdir.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def _add_race(self, race_no=1):
        """結果JSON由来のレース枠(出走表なし)を1件作る"""
        race_id = db.make_race_id(D.isoformat(), 22, race_no)
        db.upsert_race(self.conn, {
            "race_id": race_id, "date": D.isoformat(),
            "venue_code": 22, "race_no": race_no,
        })
        self.conn.commit()
        return race_id

    def _add_entries(self, race_id):
        for lane in range(1, 7):
            db.upsert_entry(self.conn, {
                "race_id": race_id, "lane": lane,
                "reg_no": 4000 + lane, "racer_name": f"選手{lane}",
            })
        self.conn.commit()

    def test_races_without_entries_triggers_download(self):
        """レース枠だけあってentriesが空なら、番組表を取りに行くこと

        これが2026-07-24の障害。races件数で判定していた頃は
        ダウンロードされずTrueが返り、空の特徴量で予測に進んでいた。
        (2026-08-02以降は上流404で公式サイトフォールバックが続くため、
        テストではそれも空応答にスタブする=実ネットワークに出さない)
        """
        import official_programs

        self._add_race()

        with patch.object(predict, "download_day",
                          return_value={"program": None, "result": None}) as m, \
                patch.object(official_programs, "fetch_official_programs",
                             return_value={"programs": []}):
            ok = predict._ensure_program(self.conn, D)

        m.assert_called_once()  # 素通りしていないこと
        self.assertFalse(ok)    # 番組表が取れなければ開催なし扱い

    def test_entries_present_skips_download(self):
        """出走表が揃っていれば再ダウンロードしないこと"""
        self._add_entries(self._add_race())

        with patch.object(predict, "download_day") as m:
            ok = predict._ensure_program(self.conn, D)

        m.assert_not_called()
        self.assertTrue(ok)

    def test_entries_of_other_day_do_not_count(self):
        """別日のentriesを当日分と数えないこと"""
        other = date(2026, 7, 23)
        race_id = db.make_race_id(other.isoformat(), 22, 1)
        db.upsert_race(self.conn, {
            "race_id": race_id, "date": other.isoformat(),
            "venue_code": 22, "race_no": 1,
        })
        self._add_entries(race_id)
        self._add_race()  # 当日はレース枠のみ

        import official_programs

        with patch.object(predict, "download_day",
                          return_value={"program": None, "result": None}) as m, \
                patch.object(official_programs, "fetch_official_programs",
                             return_value={"programs": []}):
            predict._ensure_program(self.conn, D)

        m.assert_called_once()


class TestCollectDayReturn(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmpdir.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def test_returns_program_availability(self):
        """collect_dayが(データ有無, 番組表有無)を返すこと"""
        import collect

        with patch.object(collect, "download_day",
                          return_value={"program": None, "result": None}):
            self.assertEqual(collect.collect_day(self.conn, D), (False, False))

    def test_result_only_reports_missing_program(self):
        """結果JSONだけ取れた日を「番組表なし」と報告すること

        2026-07-24の障害そのもの。当時は単にTrueを返すだけで、
        番組表が欠けたままログに「OK」と記録されていた。
        """
        import collect

        empty = {"races": [], "results": [], "payouts": []}
        with patch.object(collect, "download_day",
                          return_value={"program": None, "result": Path("dummy.json")}), \
             patch.object(collect, "_load_json", return_value={}), \
             patch.object(collect, "parse_result", return_value=empty):
            found, has_program = collect.collect_day(self.conn, D)

        self.assertTrue(found)        # 結果は取れている
        self.assertFalse(has_program)  # 番組表の欠落を報告する


if __name__ == "__main__":
    unittest.main()


class TestPartialProgramSkip(unittest.TestCase):
    """一部レースだけ出走表が空のとき、そのレースをスキップすること
    (2026-07-30の障害の回帰テスト)

    上流が当日データを約7時間遅延で部分公開し、「レース枠はあるが
    出走表特徴量が0件」のレースが混在した。当時はrankedが空のまま
    ranked[0]を参照してIndexErrorでクラッシュした(hotfix 544440f)。
    7/24のガード(_ensure_program)は日単位の欠落しか見ないため、
    レース単位の欠落はpredict_day側で防ぐ必要がある。
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        self.conn = db.connect(self.db_path)
        for race_no in (1, 2):   # 1Rは出走表あり、2Rはレース枠のみ
            db.upsert_race(self.conn, {
                "race_id": db.make_race_id(D.isoformat(), 22, race_no),
                "date": D.isoformat(), "venue_code": 22, "race_no": race_no,
                "deadline_time": f"{D.isoformat()} 1{race_no}:00:00",
            })
        self.conn.commit()
        self.conn.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_race_without_features_is_skipped(self):
        import pandas as pd

        from features import FEATURE_COLUMNS

        rid1 = db.make_race_id(D.isoformat(), 22, 1)
        rows = []
        for lane in range(1, 7):
            row = {c: 1.0 for c in FEATURE_COLUMNS}
            row.update({"race_id": rid1, "lane": lane,
                        "racer_name": f"選手{lane}", "racer_class": "A1"})
            rows.append(row)
        feats = pd.DataFrame(rows)   # 2Rの行は無い=特徴量0件

        class FakeBooster:
            def predict(self, X):
                return [0.5, 0.2, 0.12, 0.08, 0.06, 0.04][: len(X)]

        with patch.object(predict, "DB_PATH", self.db_path), \
                patch.object(predict, "_ensure_program", return_value=True), \
                patch.object(predict, "build_program_features",
                             return_value=feats), \
                patch.object(predict, "_fetch_weather_by_race",
                             return_value={}), \
                patch.object(predict.lgb, "Booster",
                             return_value=FakeBooster()), \
                patch.object(predict.MODEL_PATH.__class__, "read_text",
                             lambda self, **kw: "dummy", ):
            races = predict.predict_day(D)   # ここでIndexErrorにならないこと

        self.assertEqual([r["race_no"] for r in races], [1])
        self.assertEqual(len(races[0]["ranked"]), 6)


class TestOfficialFallback(unittest.TestCase):
    """上流404時に公式サイト直取りフォールバックが発動すること
    (2026-08-02の上流遅延4回目を受けた恒久対策の回帰テスト)"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmpdir.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def _fake_official(self):
        return {"programs": [{
            "date": D.isoformat(), "stadium_number": 22, "number": 1,
            "title": None, "subtitle": None, "grade_label": None,
            "day_label": None, "distance": 1800,
            "closed_at": f"{D.isoformat()} 10:38:00",
            "boats": [{"racer_boat_number": i, "racer_number": 4000 + i}
                      for i in range(1, 7)],
        }]}

    def test_fallback_fills_entries_when_upstream_404(self):
        import official_programs

        with patch.object(predict, "download_day",
                          return_value={"program": None, "result": None}), \
                patch.object(official_programs, "fetch_official_programs",
                             return_value=self._fake_official()):
            ok = predict._ensure_program(self.conn, D)

        self.assertTrue(ok)
        n = self.conn.execute(
            "SELECT COUNT(*) FROM entries e JOIN races r ON e.race_id = r.race_id "
            "WHERE r.date = ?", (D.isoformat(),)).fetchone()[0]
        self.assertEqual(n, 6)

    def test_fallback_error_returns_false(self):
        import official_programs

        with patch.object(predict, "download_day",
                          return_value={"program": None, "result": None}), \
                patch.object(official_programs, "fetch_official_programs",
                             side_effect=RuntimeError("公式サイト到達不可")):
            self.assertFalse(predict._ensure_program(self.conn, D))


class TestKonsenBandPlanOverride(unittest.TestCase):
    """5場で本命表示に吸われた20%未満のレースにも⑬構成が適用されること
    (2026-08-04・検証⑮の回帰テスト。従来は本命構成1,000円のままの適用漏れ)"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test.db"
        conn = db.connect(self.db_path)
        db.upsert_race(conn, {
            "race_id": db.make_race_id(D.isoformat(), 4, 1),   # 平和島=対象5場
            "date": D.isoformat(), "venue_code": 4, "race_no": 1,
            "deadline_time": f"{D.isoformat()} 11:00:00",
        })
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_absorbed_konsen_band_gets_plan13(self):
        import pandas as pd

        from features import FEATURE_COLUMNS

        rid = db.make_race_id(D.isoformat(), 4, 1)
        rows = []
        for lane in range(1, 7):
            row = {c: 1.0 for c in FEATURE_COLUMNS}
            row.update({"race_id": rid, "lane": lane,
                        "racer_name": f"選手{lane}", "racer_class": "A1"})
            rows.append(row)
        feats = pd.DataFrame(rows)

        class FakeBooster:
            def predict(self, X):
                # 1位生値0.19=超混戦帯。5場なので本命に吸われるケース
                return [0.19, 0.18, 0.17, 0.16, 0.15, 0.14][: len(X)]

        with patch.object(predict, "DB_PATH", self.db_path), \
                patch.object(predict, "_ensure_program", return_value=True), \
                patch.object(predict, "build_program_features",
                             return_value=feats), \
                patch.object(predict, "_fetch_weather_by_race",
                             return_value={}), \
                patch.object(predict, "_rising_lanes", return_value={}), \
                patch.object(predict.lgb, "Booster",
                             return_value=FakeBooster()), \
                patch.object(predict.MODEL_PATH.__class__, "read_text",
                             lambda self, **kw: "dummy", ):
            races = predict.predict_day(D)

        race = races[0]
        self.assertEqual(race["shobusho"], "本命")   # 表示は本命のまま
        plan = race["bets"]["plan"]
        self.assertEqual(sum(y for _b, _c, y, _s in plan), 2000)   # ⑬構成
        self.assertTrue(any(s == "深い波乱" for _b, _c, _y, s in plan))
        self.assertTrue(any(s == "差され追加" for _b, _c, _y, s in plan))
