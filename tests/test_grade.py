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
        self.assertIn("ken 本命", html)
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
        self.assertIn("showDays('ken_hon', false)", html)  # 初期表示で自動オープン

    def test_viewer_shows_only_recent_days_but_totals_all(self):
        # 6日分のledger(的中つき)→ ビューワーには新しい4日だけ、通算は全期間
        stats = {k: {"stake": 1000, "ret": 1100, "races": 1, "hits": 1} for k in G.PREDICTOR_LABELS}
        ledger = []
        for i in range(1, 7):  # 07-01〜07-06
            d = f"2026-07-{i:02d}"
            hit = {"date": d, "venue": "常滑", "race_no": 1, "chaku": "1-2-3",
                   "stake": 1000, "ret": 1100, "lines": [{"label": "3連複 1=2=3", "payout": 1100}]}
            ledger.append({"date": d, "stats": stats, "hits": {"ken_hon": [hit]}})
        html = G.render_stats(ledger)

        for d in ("2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06"):
            self.assertIn(d, html)                   # 直近4日は表示
        self.assertNotIn("2026-07-02", html)         # 5日前は非表示
        # 初日は「採点開始 2026-07-01〜」の見出しにだけ現れる(日別・履歴には出ない)
        self.assertEqual(html.count("2026-07-01"), 1)
        # 通算は全6日分(1,000円×6日=6,000円が母数: 回収率110.0%)
        self.assertIn("110.0%", html)
        self.assertIn("日別(直近4日)", html)


class TestStatsPageChrome(unittest.TestCase):
    """更新時刻と手動更新ボタン(2026-07-14ユーザー要望・表示のみの凍結例外)"""

    def _html(self):
        stats = {k: {"stake": 1000, "ret": 500, "races": 1, "hits": 1}
                 for k in G.PREDICTOR_LABELS}
        return G.render_stats([{"date": "2026-07-07", "stats": stats}])

    def test_shows_last_updated(self):
        self.assertIn("最終更新:", self._html())

    def test_has_manual_run_button(self):
        html = self._html()
        self.assertIn("手動で採点を更新", html)
        self.assertIn("actions/workflows/grade.yml", html)


class TestKonsenBucket(unittest.TestCase):
    def test_konsen_goes_to_own_bucket(self):
        import tempfile
        from pathlib import Path as P_
        import db
        picks = _picks(shobusho="超混戦")
        with tempfile.TemporaryDirectory() as tmp:
            conn = db.connect(P_(tmp) / "t.db")
            rid = picks["races"][0]["race_id"]
            db.upsert_payout(conn, {"race_id": rid, "bet_type": "3連複",
                                    "combination": "1=2=3", "amount_yen": 500})
            day = G.grade_day(picks, conn)
            conn.close()
        self.assertEqual(day["ken_konsen"]["races"], 1)
        self.assertEqual(day["ken_hon"]["races"], 0)   # 本命には入らない
        self.assertEqual(day["ken"]["races"], 1)


if __name__ == "__main__":
    unittest.main()
