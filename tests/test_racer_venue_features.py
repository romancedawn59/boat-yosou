# -*- coding: utf-8 -*-
"""検証⑫(test/verify_racer_venue.py)の新特徴量のリーク防止と計算の正しさ"""
import sqlite3
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

import db
from verify_racer_venue import band_of, compute_new_features


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.executescript(db.SCHEMA)
    return conn


def _add_race(conn, rid: str, date: str, venue: int, entries: list[dict]):
    """entries: [{lane, reg_no, arrival, motor_no?, boat_no?, racer_class?, wr?}]"""
    db.upsert_race(conn, {"race_id": rid, "date": date, "venue_code": venue,
                          "race_no": int(rid[-2:]), "grade": "一般"})
    for e in entries:
        db.upsert_entry(conn, {
            "race_id": rid, "lane": e["lane"], "reg_no": e["reg_no"],
            "motor_no": e.get("motor_no"), "boat_no": e.get("boat_no"),
            "racer_class": e.get("racer_class"),
            "national_win_rate": e.get("wr"),
        })
        if e.get("arrival") is not None:
            db.upsert_result(conn, {"race_id": rid, "lane": e["lane"],
                                    "arrival_order": e["arrival"]})


class TestRacerVenueFeatures(unittest.TestCase):
    """選手×場ローリングが「当該レースより前」だけから計算されること"""

    def setUp(self):
        self.conn = _conn()
        self.addCleanup(self.conn.close)
        # 選手9999が場2で7走(着順 1,3,1,2,4,1 + 結果未確定)、間に場3で1走
        arrivals = [1, 3, 1, 2, 4, 1, None]
        for i, a in enumerate(arrivals):
            _add_race(self.conn, f"2025070{i+1}_02_01", f"2025-07-0{i+1}", 2,
                      [{"lane": 1, "reg_no": 9999, "arrival": a}])
        _add_race(self.conn, "20250704_03_01", "2025-07-04", 3,
                  [{"lane": 1, "reg_no": 9999, "arrival": 1}])
        self.conn.commit()
        self.feat = compute_new_features(self.conn)

    def _row(self, rid):
        return self.feat[self.feat["race_id"] == rid].iloc[0]

    def test_rolling_excludes_current_race_and_other_venues(self):
        # 7走目(場2): 場2の過去6走(1,3,1,2,4,1着)のみから計算。
        # 場3の1着や当該レースは含まれない
        last = self._row("20250707_02_01")
        self.assertAlmostEqual(last["rv_win_rate"], 3 / 6)
        self.assertAlmostEqual(last["rv_top3_rate"], 5 / 6)  # 4着の1走のみ圏外

    def test_min_periods_guard(self):
        # 5走目: 場2の過去は4走のみ → min_periods=5未満でNaN
        self.assertTrue(pd.isna(self._row("20250705_02_01")["rv_win_rate"]))
        # 6走目: 過去5走(1,3,1,2,4着)で初めて値が出る
        self.assertAlmostEqual(self._row("20250706_02_01")["rv_win_rate"], 2 / 5)

    def test_other_venue_starts_fresh(self):
        # 場3の初出走: 場2の実績は持ち込まれずNaN
        self.assertTrue(pd.isna(self._row("20250704_03_01")["rv_win_rate"]))


class TestVenueLaneFeatures(unittest.TestCase):
    def test_expanding_and_dev_zero_when_single_venue(self):
        conn = _conn()
        self.addCleanup(conn.close)
        # 場2×枠1で31走(勝ち20/負け11)。場が1つだけなら場×枠=全場同枠なのでvl_dev=0
        for i in range(31):
            _add_race(conn, f"2025{i+101:04d}_02_01", f"2025-{i//28+1:02d}-{i%28+1:02d}",
                      2, [{"lane": 1, "reg_no": 100 + i,
                           "arrival": 1 if i < 20 else 2}])
        conn.commit()
        feat = compute_new_features(conn)
        last = feat.sort_values("race_id").iloc[-1]
        self.assertAlmostEqual(last["vl_win_rate"], 20 / 30)  # 過去30走のみ(当該除外)
        self.assertAlmostEqual(last["vl_dev"], 0.0)
        # 30走目まではmin_periods=30未満でNaN
        prev = feat.sort_values("race_id").iloc[-2]
        self.assertTrue(pd.isna(prev["vl_win_rate"]))


class TestMotorBoatFeatures(unittest.TestCase):
    def test_motor_rolling_by_venue(self):
        conn = _conn()
        self.addCleanup(conn.close)
        # 場2のモーター15が4走: 2連対(1,2着)→2回、圏外(5着)→1回、4走目は評価対象
        arrivals = [1, 5, 2, 3]
        for i, a in enumerate(arrivals):
            _add_race(conn, f"2025070{i+1}_02_01", f"2025-07-0{i+1}", 2,
                      [{"lane": 1, "reg_no": 200 + i, "motor_no": 15,
                        "boat_no": 40, "arrival": a}])
        conn.commit()
        feat = compute_new_features(conn)
        rows = feat.sort_values("race_id")
        # 3走目: 過去2走のみ → min_periods=3未満でNaN
        self.assertTrue(pd.isna(rows.iloc[2]["motor_recent_top2"]))
        # 4走目: 過去3走(1,5,2着)→2連対率2/3。ボートも同じ履歴なので同値
        self.assertAlmostEqual(rows.iloc[3]["motor_recent_top2"], 2 / 3)
        self.assertAlmostEqual(rows.iloc[3]["boat_recent_top2"], 2 / 3)


class TestRaceQualityFeatures(unittest.TestCase):
    def test_edges_and_a1_count(self):
        conn = _conn()
        self.addCleanup(conn.close)
        _add_race(conn, "20250701_02_01", "2025-07-01", 2, [
            {"lane": 1, "reg_no": 1, "racer_class": "A1", "wr": 7.0, "arrival": 1},
            {"lane": 2, "reg_no": 2, "racer_class": "A1", "wr": 6.0, "arrival": 2},
            {"lane": 3, "reg_no": 3, "racer_class": "B1", "wr": 5.0, "arrival": 3},
        ])
        conn.commit()
        feat = compute_new_features(conn).set_index("lane")
        # 全国勝率の他艇最大との差: トップは2番手との差、他はトップとの差
        self.assertAlmostEqual(feat.loc[1, "rq_wr_edge"], 1.0)
        self.assertAlmostEqual(feat.loc[2, "rq_wr_edge"], -1.0)
        self.assertAlmostEqual(feat.loc[3, "rq_wr_edge"], -2.0)
        # 級別ord(A1=3, B1=1): A1同士のトップ2は差0、B1は-2
        self.assertAlmostEqual(feat.loc[1, "rq_class_edge"], 0.0)
        self.assertAlmostEqual(feat.loc[2, "rq_class_edge"], 0.0)
        self.assertAlmostEqual(feat.loc[3, "rq_class_edge"], -2.0)
        self.assertEqual(feat["rq_n_a1"].tolist(), [2.0, 2.0, 2.0])

    def test_tied_top_wr_edge_is_zero(self):
        conn = _conn()
        self.addCleanup(conn.close)
        _add_race(conn, "20250701_02_01", "2025-07-01", 2, [
            {"lane": 1, "reg_no": 1, "wr": 7.0, "arrival": 1},
            {"lane": 2, "reg_no": 2, "wr": 7.0, "arrival": 2},
        ])
        conn.commit()
        feat = compute_new_features(conn).set_index("lane")
        # 同値トップが2艇 → 双方とも他艇最大=7.0で差0
        self.assertAlmostEqual(feat.loc[1, "rq_wr_edge"], 0.0)
        self.assertAlmostEqual(feat.loc[2, "rq_wr_edge"], 0.0)


class TestBandOf(unittest.TestCase):
    def test_fixed_boundaries(self):
        self.assertEqual(band_of(0.10), "〜20%")
        self.assertEqual(band_of(0.20), "20〜25%")
        self.assertEqual(band_of(0.2999), "25〜30%")
        self.assertEqual(band_of(0.34), "30〜35%")
        self.assertEqual(band_of(0.35), "35%〜")


if __name__ == "__main__":
    unittest.main()
