# -*- coding: utf-8 -*-
"""v2.2特徴量(EXTRA_COLUMNS)のリーク安全性テスト(2026-08-31追加)

KR指数・期内走数・通算成績は「当該レースより前の情報だけ」から
計算されなければならない(walk-forward安全)。また結果未確定の
未来レース(当日朝の予測対象)にも値が付くこと。
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
from features import (EXTRA_COLUMNS, FEATURE_COLUMNS, compute_kr_features,
                      compute_stat_robust_features)


class FeatureDBTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.conn = db.connect(Path(self.tmpdir.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def add_race(self, date_str, race_no=1, venue=22, lanes=6,
                 arrivals=None):
        """1レース追加。arrivals={lane: 着順}を渡すと結果も入る"""
        rid = db.make_race_id(date_str, venue, race_no)
        db.upsert_race(self.conn, {
            "race_id": rid, "date": date_str,
            "venue_code": venue, "race_no": race_no,
        })
        for lane in range(1, lanes + 1):
            db.upsert_entry(self.conn, {
                "race_id": rid, "lane": lane,
                "reg_no": 4000 + lane, "racer_name": f"選手{lane}",
            })
        if arrivals:
            for lane, ao in arrivals.items():
                db.upsert_result(self.conn, {
                    "race_id": rid, "lane": lane, "arrival_order": ao,
                })
        self.conn.commit()
        return rid


class TestKrFeatures(FeatureDBTestCase):
    def test_pre_race_rating_is_leak_safe(self):
        """最初のレースは全艇1500。勝者は次のレースで1500超、敗者は未満"""
        r1 = self.add_race("2026-01-01",
                           arrivals={l: l for l in range(1, 7)})
        r2 = self.add_race("2026-01-02",
                           arrivals={l: 7 - l for l in range(1, 7)})
        kr = compute_kr_features(self.conn).set_index(["race_id", "lane"])["kr"]
        for lane in range(1, 7):
            self.assertEqual(kr.loc[(r1, lane)], 1500.0)
        # r1で1着だった枠1(reg 4001)はr2の直前レートが上がっている
        self.assertGreater(kr.loc[(r2, 1)], 1500.0)
        # r1で6着だった枠6は下がっている
        self.assertLess(kr.loc[(r2, 6)], 1500.0)

    def test_future_race_gets_latest_rating(self):
        """結果未確定のレース(当日朝)にも最新レートが付く"""
        self.add_race("2026-01-01", arrivals={l: l for l in range(1, 7)})
        future = self.add_race("2026-01-03")   # 結果なし
        kr = compute_kr_features(self.conn).set_index(["race_id", "lane"])["kr"]
        self.assertGreater(kr.loc[(future, 1)], 1500.0)
        self.assertLess(kr.loc[(future, 6)], 1500.0)


class TestStatRobustFeatures(FeatureDBTestCase):
    def test_n_starts_period_resets_on_july(self):
        """期内走数は7/1でリセットされる(級別期替わりに対応)"""
        self.add_race("2026-06-28", arrivals={l: l for l in range(1, 7)})
        self.add_race("2026-06-29", race_no=2,
                      arrivals={l: l for l in range(1, 7)})
        r_july = self.add_race("2026-07-02")
        f = compute_stat_robust_features(self.conn).set_index(
            ["race_id", "lane"])
        # 7/2時点: 前期の2走はカウントされず0
        self.assertEqual(f.loc[(r_july, 1), "n_starts_period"], 0)

    def test_n_starts_period_counts_prior_starts_only(self):
        r1 = self.add_race("2026-07-05", arrivals={l: l for l in range(1, 7)})
        r2 = self.add_race("2026-07-06", race_no=2)
        f = compute_stat_robust_features(self.conn).set_index(
            ["race_id", "lane"])
        self.assertEqual(f.loc[(r1, 1), "n_starts_period"], 0)
        self.assertEqual(f.loc[(r2, 1), "n_starts_period"], 1)

    def test_career_top2_needs_min_history(self):
        """10走未満の選手のcareer_top2はNaN(新人を無理に評価しない)"""
        self.add_race("2026-01-01", arrivals={l: l for l in range(1, 7)})
        r2 = self.add_race("2026-01-02", race_no=2)
        f = compute_stat_robust_features(self.conn).set_index(
            ["race_id", "lane"])
        self.assertTrue(f.loc[(r2, 1)].isna()["career_top2"])


class TestFeatureColumns(unittest.TestCase):
    def test_extra_columns_registered(self):
        for col in ("kr", "kr_rank", "nobi_gap", "n_starts_period",
                    "career_top2"):
            self.assertIn(col, EXTRA_COLUMNS)
            self.assertIn(col, FEATURE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
