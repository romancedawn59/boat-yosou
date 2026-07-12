import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test"))

from tax_summary import collect_hits, ichiji_shotoku, parse_line

LEDGER_DAY = {
    "date": "2026-07-07",
    "stats": {"ken_hon": {"stake": 10000, "ret": 30100, "races": 10, "hits": 5}},
    "hits": {
        "ken_hon": [
            {"date": "2026-07-07", "venue": "平和島", "race_no": 2, "chaku": "1-5-2",
             "stake": 1000, "ret": 2580,
             "lines": [{"label": "3連複 1=2=5（200円）", "payout": 2580}]},
            {"date": "2026-07-07", "venue": "常滑", "race_no": 9, "chaku": "6-1-2",
             "stake": 1000, "ret": 74510,
             "lines": [{"label": "3連単 6-1-2（100円）", "payout": 74510}]},
        ],
        "ken_jun": [],
    },
}


class TestParseLine(unittest.TestCase):
    def test_parses_bt_comb_stake(self):
        self.assertEqual(parse_line("3連複 1=2=5（200円）"), ("3連複", "1=2=5", 200))
        self.assertEqual(parse_line("3連単 6-1-2（100円）"), ("3連単", "6-1-2", 100))

    def test_unknown_format_returns_none(self):
        self.assertIsNone(parse_line("3連複 1=2=5"))  # 購入額なしの旧形式は要人間確認


class TestCollectHits(unittest.TestCase):
    def test_expands_lines_with_stake_and_payout(self):
        rows = collect_hits([LEDGER_DAY], ["ken_hon"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["line_stake"], 200)
        self.assertEqual(rows[0]["line_payout"], 2580)
        self.assertEqual(rows[1]["line_stake"], 100)
        self.assertEqual(rows[1]["line_payout"], 74510)

    def test_scope_keys_are_respected(self):
        self.assertEqual(collect_hits([LEDGER_DAY], ["ken_jun"]), [])


class TestIchijiShotoku(unittest.TestCase):
    def test_below_deduction_is_zero(self):
        # 払戻45万・当たり券費1万 → 50万控除以下なので一時所得0
        self.assertEqual(ichiji_shotoku(450_000, 10_000), (0, 0))

    def test_above_deduction_halved(self):
        # 払戻223万・当たり券費15万 → (223-15-50)=158万、課税対象はその半分
        base, taxable = ichiji_shotoku(2_230_000, 150_000)
        self.assertEqual(base, 1_580_000)
        self.assertEqual(taxable, 790_000)

    def test_loss_is_not_negative(self):
        # 外れ年でも他所得との損益通算はできない(0止まり)
        self.assertEqual(ichiji_shotoku(0, 0), (0, 0))


if __name__ == "__main__":
    unittest.main()
