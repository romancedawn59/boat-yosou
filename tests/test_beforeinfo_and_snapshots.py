"""直前情報の拡張(スタート展示の進入・ST・安定板)、オッズ時系列、確定オッズの全レース取得のテスト"""
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "test"))

import db
import exhibition


def _start_row(lane: int, st: str) -> str:
    return (f'<div class="table1_boatImage1"><span class="table1_boatImage1Number is-type{lane}">{lane}</span>'
            f'<span class="table1_boatImage1Line"></span>'
            f'<span class="table1_boatImage1Time is-fBold">{st}</span></div>')


class StartExhibitionTest(unittest.TestCase):
    def test_course_is_row_order_and_flying_is_negative(self):
        html = "".join(_start_row(l, s) for l, s in
                       ((1, "F.01"), (2, ".27"), (3, ".12"), (4, ".04"), (6, ".03"), (5, "L")))
        got = exhibition.parse_start_exhibition(html)
        self.assertEqual(got[1], {"ex_course": 1, "ex_st": -0.01})
        self.assertEqual(got[2], {"ex_course": 2, "ex_st": 0.27})
        self.assertEqual(got[6]["ex_course"], 5)      # 6号艇が5コースに前づけ
        self.assertEqual(got[5], {"ex_course": 6, "ex_st": None})   # 出遅れは数値なし

    def test_stabilizer_flag(self):
        self.assertEqual(exhibition.parse_exhibition_html("安定板使用"), [])
        row = ('toban=1234"></a><td rowspan="2">52.0kg</td> <td rowspan="4">6.80</td> '
               '<td rowspan="4">-0.5</td>')
        self.assertEqual(exhibition.parse_exhibition_html(row + "安定板使用")[0]["stabilizer"], 1)
        self.assertEqual(exhibition.parse_exhibition_html(row)[0]["stabilizer"], 0)


class SchemaTest(unittest.TestCase):
    def test_migrate_adds_columns_to_old_exhibition_table(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.db"
            old = sqlite3.connect(path)
            old.execute("CREATE TABLE exhibition (race_id TEXT, lane INTEGER, reg_no INTEGER, "
                        "weight_kg REAL, exhibition_time REAL, tilt REAL, PRIMARY KEY (race_id, lane))")
            old.commit()
            old.close()
            conn = db.connect(path)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(exhibition)")}
            conn.close()
            self.assertTrue({"ex_course", "ex_st", "stabilizer"} <= cols)

    def test_odds_snapshots_keep_every_fetch(self):
        with tempfile.TemporaryDirectory() as d:
            conn = db.connect(Path(d) / "t.db")
            for t, o in (("2026-09-22T10:00:00", 12.3), ("2026-09-22T10:10:00", 10.1)):
                row = {"race_id": "20260922_20_01", "bet_type": "3連単", "combination": "1-2-3",
                       "odds": o, "fetched_at": t}
                db.upsert_odds(conn, row)
                db.upsert_odds_snapshot(conn, row)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0], 2)
            conn.close()


class FinalOddsAllTargetsTest(unittest.TestCase):
    def test_all_mode_ignores_snapshot_requirement(self):
        import collect_final_odds as C
        with tempfile.TemporaryDirectory() as d:
            conn = db.connect(Path(d) / "t.db")
            for rid, day in (("20260901_20_01", "2026-09-01"), ("20260902_20_01", "2026-09-02")):
                db.upsert_race(conn, {"race_id": rid, "date": day, "venue_code": 20, "race_no": 1})
                db.upsert_payout(conn, {"race_id": rid, "bet_type": "3連単",
                                        "combination": "1-2-3", "amount_yen": 1000})
            db.upsert_odds_final(conn, {"race_id": "20260901_20_01", "bet_type": "3連単",
                                        "combination": "1-2-3", "odds": 10.0, "fetched_at": "x"})
            conn.commit()
            self.assertEqual(C.pick_targets(conn, date(2026, 9, 21)), [])
            got = C.pick_all_targets(conn, date(2026, 9, 21), "2026-09-01", "2026-09-30")
            self.assertEqual([r[0] for r in got], ["20260902_20_01"])
            conn.close()


if __name__ == "__main__":
    unittest.main()
