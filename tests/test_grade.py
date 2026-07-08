import json
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
import grade_predictions as G


def _picks(shobusho="本命"):
    return {
        "date": "2026-07-07",
        "races": [{
            "race_id": "20260707_04_01",
            "venue_code": 4,
            "race_no": 1,
            "confidence": "荒れ注意",
            "shobusho": shobusho,
            "a": [["2連複", "1=2", 0.3]],
            "b": [["3連単", "1-2-3", 0.1]],
            "c": [["3連単", "4-1-2", 0.004]],
            "ken": [["3連複", "1=2=3", 200, "検証済み"], ["3連単", "4-1-2", 100, "勝万舟"]],
        }],
    }


class TestGradeDay(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(db.SCHEMA)
        db.upsert_race(self.conn, {
            "race_id": "20260707_04_01", "date": "2026-07-07", "venue_code": 4, "race_no": 1})

    def test_grades_hits_and_misses(self):
        # 3連複1=2=3(500円)と3連単4-1-2(12000円)が的中した想定
        db.upsert_payout(self.conn, {"race_id": "20260707_04_01", "bet_type": "3連複",
                                     "combination": "1=2=3", "amount_yen": 500})
        db.upsert_payout(self.conn, {"race_id": "20260707_04_01", "bet_type": "3連単",
                                     "combination": "4-1-2", "amount_yen": 12000})
        day = G.grade_day(_picks(), self.conn)

        self.assertEqual(day["a"], {"stake": 100, "ret": 0, "races": 1, "hits": 0})
        self.assertEqual(day["c"], {"stake": 100, "ret": 12000, "races": 1, "hits": 1})
        # ken: 3連複200円->1000円、3連単100円->12000円
        self.assertEqual(day["ken"]["ret"], 1000 + 12000)
        self.assertEqual(day["ken_hon"]["ret"], 13000)  # 本命勝負所として集計
        self.assertEqual(day["ken_jun"]["races"], 0)

    def test_returns_none_when_no_payouts(self):
        self.assertIsNone(G.grade_day(_picks(), self.conn))


class TestCollectHits(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(db.SCHEMA)
        db.upsert_race(self.conn, {
            "race_id": "20260707_04_01", "date": "2026-07-07", "venue_code": 4, "race_no": 1})

    def test_collects_only_hit_races(self):
        # C(3連単4-1-2)とken(3連複1=2=3・3連単4-1-2)が的中、A・Bは外れ
        db.upsert_payout(self.conn, {"race_id": "20260707_04_01", "bet_type": "3連複",
                                     "combination": "1=2=3", "amount_yen": 500})
        db.upsert_payout(self.conn, {"race_id": "20260707_04_01", "bet_type": "3連単",
                                     "combination": "4-1-2", "amount_yen": 12000})
        hits = G.collect_hits(_picks(), self.conn, "2026-07-07")

        self.assertEqual(hits["a"], [])  # 2連複1=2は非的中
        self.assertEqual(len(hits["c"]), 1)
        self.assertEqual(hits["c"][0]["ret"], 12000)
        self.assertEqual(hits["c"][0]["venue"], "平和島")
        self.assertEqual(hits["c"][0]["chaku"], "4-1-2")  # 3連単キーから決着を復元
        # ken: 3連複200円->1000円 + 3連単100円->12000円 = 13000円、本命勝負所にも入る
        self.assertEqual(hits["ken"][0]["ret"], 13000)
        self.assertEqual(len(hits["ken"][0]["lines"]), 2)
        self.assertEqual(len(hits["ken_hon"]), 1)
        self.assertEqual(hits["ken_jun"], [])

    def test_no_payouts_means_no_hits(self):
        hits = G.collect_hits(_picks(), self.conn, "2026-07-07")
        self.assertTrue(all(v == [] for v in hits.values()))


class TestLedgerAndStats(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        tmp_path = Path(self.tmp.name)
        patcher1 = patch.object(G, "DATA_DIR", tmp_path / "data")
        patcher2 = patch.object(G, "SITE_DIR", tmp_path)
        patcher1.start(); patcher2.start()
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_ledger_roundtrip_and_overwrite(self):
        G.save_ledger([{"date": "2026-07-07", "stats": {"ken": G._zero()}}])
        ledger = G.load_ledger()
        self.assertEqual(len(ledger), 1)

        # 同日を上書き(冪等)
        ledger = [e for e in ledger if e["date"] != "2026-07-07"]
        ledger.append({"date": "2026-07-07", "stats": {"ken": {"stake": 1, "ret": 2, "races": 1, "hits": 1}}})
        G.save_ledger(ledger)
        self.assertEqual(G.load_ledger()[0]["stats"]["ken"]["ret"], 2)

    def test_render_stats_contains_predictors(self):
        stats = {k: {"stake": 1000, "ret": 1200, "races": 2, "hits": 1} for k in G.PREDICTOR_LABELS}
        html = G.render_stats([{"date": "2026-07-07", "stats": stats}])
        self.assertIn("A 石橋渡", html)
        self.assertIn("予想屋ken(全レース)", html)
        self.assertIn("ken 本命勝負所", html)
        self.assertIn("120.0%", html)
        self.assertIn("viewport", html)

    def test_render_stats_embeds_clickable_hits(self):
        stats = {k: {"stake": 1000, "ret": 1200, "races": 2, "hits": 1} for k in G.PREDICTOR_LABELS}
        hit = {"date": "2026-07-07", "venue": "常滑", "race_no": 2, "chaku": "3-2-1",
               "stake": 1000, "ret": 20880, "lines": [{"label": "3連単 3-2-1", "payout": 19560}]}
        html = G.render_stats([{"date": "2026-07-07", "stats": stats,
                                "hits": {"ken_hon": [hit]}}])
        self.assertIn("data-key='ken_hon'", html)   # 行がクリック可能
        self.assertIn("const HITS =", html)          # 明細データ埋め込み
        self.assertIn("const DAYS =", html)          # 日付一覧データ埋め込み
        self.assertIn("3-2-1", html)                 # 的中買い目が含まれる
        self.assertIn("showDays", html)              # 人→日付の第1階層
        self.assertIn("showDayHits", html)           # 日付→履歴の第2階層


if __name__ == "__main__":
    unittest.main()
