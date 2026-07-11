import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

import db
from analyze_market import (
    calibration_rows,
    classify_miss,
    extract_r1,
    normalize_implied,
    pct_change,
    perm_shares,
    pick_manshu,
    pick_ninki,
)
from collect_final_odds import pick_targets


class TestOddsMath(unittest.TestCase):
    def test_normalize_implied_sums_to_one(self):
        implied = normalize_implied({"1-2-3": 2.0, "1-3-2": 4.0, "2-1-3": 4.0})
        self.assertAlmostEqual(sum(implied.values()), 1.0)
        # オッズ2倍の目はオッズ4倍の目の2倍の含意確率
        self.assertAlmostEqual(implied["1-2-3"], 0.5)
        self.assertAlmostEqual(implied["1-3-2"], 0.25)

    def test_normalize_implied_skips_zero_odds(self):
        implied = normalize_implied({"1-2-3": 2.0, "4-5-6": 0})
        self.assertEqual(list(implied), ["1-2-3"])

    def test_pct_change_sign(self):
        # 正=最終オッズの方が高い(実払戻が有利)
        self.assertAlmostEqual(pct_change(10.0, 12.0), 0.2)
        self.assertAlmostEqual(pct_change(10.0, 8.0), -0.2)

    def test_calibration_rows_bins_and_ratio(self):
        # 含意0.1のビンに10観測・実際の的中1回 → 実際/含意=1.0
        obs = [(0.1, i == 0) for i in range(10)]
        rows = calibration_rows(obs, [0.0, 0.05, 0.2, 1.01])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n"], 10)
        self.assertAlmostEqual(rows[0]["ratio"], 1.0)


class TestMissBreakdown(unittest.TestCase):
    # 荒れ注意の検証済み構成(r1=1, r2=2, r3=3, r4=4)
    PLAN = [
        ["3連複", "1=2=3", 200, "検証済み"],
        ["3連複", "1=2=4", 200, "検証済み"],
        ["3連複", "1=3=4", 100, "検証済み"],
        ["3連単", "3-1-2", 200, "検証済み"],
        ["3連単", "4-1-2", 200, "検証済み"],
        ["3連単", "5-6-4", 100, "勝万舟"],
    ]

    def test_extract_r1_from_validated_plan(self):
        # 3連単2点(3-1-2, 4-1-2)の共通2着=予測1位
        self.assertEqual(extract_r1(self.PLAN), 1)

    def test_extract_r1_unrecoverable_returns_none(self):
        # 標準プラン(検証済み3連単がない)からは復元しない
        plan = [["3連複", "1=2=3", 300, "本線"], ["3連単", "1-2-3", 200, "山田"]]
        self.assertIsNone(extract_r1(plan))

    def test_classify_miss(self):
        self.assertEqual(classify_miss(1, {2, 3, 4}), "軸飛び")
        self.assertEqual(classify_miss(1, {1, 5, 6}), "ヒモ抜け")


class TestPaperPicks(unittest.TestCase):
    ODDS = {"1-2-3": 5.8, "1-3-2": 9.9, "4-5-6": 250.0, "5-4-6": 120.5}

    def test_ninki_picks_lowest_odds(self):
        self.assertEqual(pick_ninki(self.ODDS), ("1-2-3", 5.8))

    def test_manshu_picks_lowest_at_or_above_100(self):
        # 100倍以上の中で最も低いオッズ(=最も当たりやすい万舟候補)
        self.assertEqual(pick_manshu(self.ODDS), ("5-4-6", 120.5))

    def test_manshu_none_when_no_candidate(self):
        self.assertIsNone(pick_manshu({"1-2-3": 5.8, "1-3-2": 99.9}))

    def test_ninki_none_when_empty(self):
        self.assertIsNone(pick_ninki({}))


class TestPermShares(unittest.TestCase):
    def test_shares_normalized_within_set(self):
        tri = {"1-2-3": 0.10, "1-3-2": 0.06, "2-1-3": 0.04,
               "2-3-1": 0.0, "3-1-2": 0.0, "3-2-1": 0.0,
               "4-5-6": 0.50}  # 別の組は無視される
        shares = perm_shares(tri, (1, 2, 3))
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares["1-2-3"], 0.5)

    def test_empty_when_set_missing(self):
        self.assertEqual(perm_shares({"1-2-3": 0.1}, (4, 5, 6)), {})


class TestFinalOddsCollection(unittest.TestCase):
    """odds_finalテーブルと遡及取得の対象選定"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.conn = db.connect(Path(self.tmpdir.name) / "test.db")
        self.addCleanup(self.conn.close)

    def _add_race(self, rid, d, snap=False, final=False, backfill=False):
        db.upsert_race(self.conn, {
            "race_id": rid, "date": d,
            "venue_code": int(rid.split("_")[1]), "race_no": int(rid.split("_")[2]),
        })
        if snap:
            db.upsert_odds(self.conn, {
                "race_id": rid, "bet_type": "3連単", "combination": "1-2-3",
                "odds": 5.0, "fetched_at": f"{d}T10:00:00",
            })
        if backfill:
            db.upsert_odds(self.conn, {
                "race_id": rid, "bet_type": "3連単", "combination": "1-2-3",
                "odds": 5.0, "fetched_at": "final-backfill",
            })
        if final:
            db.upsert_odds_final(self.conn, {
                "race_id": rid, "bet_type": "3連単", "combination": "1-2-3",
                "odds": 6.0, "fetched_at": "2026-07-12T09:00:00",
            })

    def test_upsert_odds_final_is_separate_from_odds(self):
        # 最終オッズを入れてもスナップショット(odds)には影響しない
        self._add_race("20260708_04_01", "2026-07-08", snap=True, final=True)
        snap = self.conn.execute("SELECT odds FROM odds").fetchone()[0]
        final = self.conn.execute("SELECT odds FROM odds_final").fetchone()[0]
        self.assertEqual((snap, final), (5.0, 6.0))

    def test_pick_targets_only_past_snapshot_races_without_final(self):
        from datetime import date
        today = date(2026, 7, 12)
        self._add_race("20260708_04_01", "2026-07-08", snap=True)              # 対象
        self._add_race("20260708_04_02", "2026-07-08", snap=True, final=True)  # 取得済み
        self._add_race("20260501_04_01", "2026-05-01", backfill=True)          # 比較対象なし
        self._add_race("20260712_04_01", "2026-07-12", snap=True)              # 当日(未確定)

        targets = pick_targets(self.conn, today)
        self.assertEqual([t[0] for t in targets], ["20260708_04_01"])


if __name__ == "__main__":
    unittest.main()
