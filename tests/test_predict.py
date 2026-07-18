import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import predictors as P
from predict import build_notify_text, render_venue_page, shobu_summary


def _ranked(probs):
    lanes = [1, 2, 3, 4, 5, 6]
    return [{"lane": lanes[i], "name": f"選手{i}", "racer_class": "A1", "prob": p}
            for i, p in enumerate(probs[: len(lanes)])]


def _race(probs, venue_code=4, race_no=1, shobusho=None, wx=None):
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
        "deadline": "2026-07-05 10:47:00",
        "weather": wx,
        "ranked": ranked,
        "picks_a": a,
        "picks_b": b,
        "picks_c": c,
        "bets": {"confidence": conf, "plan": P.ken_portfolio(conf, ranked, b, c)},
        "shobusho": shobusho,
    }


ARERU = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
KATAME = [0.60, 0.2, 0.1, 0.05, 0.03, 0.02]


class TestShobuSummary(unittest.TestCase):
    def test_lists_and_budget_v2(self):
        races = [
            _race(ARERU, 4, 5, shobusho="本命"),
            _race(ARERU, 1, 2, shobusho="超混戦"),   # 全場対象(桐生)
            _race(ARERU, 13, 3, shobusho="要注目"),  # 観測のみ・予算に入らない
            _race(KATAME, 20, 1),
        ]
        honmei, konsen, attention, budget = shobu_summary(races)
        self.assertEqual(honmei, ["平和島5R"])
        self.assertEqual(konsen, ["桐生2R"])
        self.assertEqual(attention, ["尼崎3R"])
        self.assertEqual(budget, 2000)  # 購入=本命+超混戦の2レース×1000円

    def test_empty(self):
        honmei, konsen, attention, budget = shobu_summary([_race(KATAME)])
        self.assertEqual((honmei, konsen, attention, budget), ([], [], [], 0))


class TestBuildNotifyText(unittest.TestCase):
    def test_includes_shobusho_and_url(self):
        races = [_race(ARERU, 4, 5, shobusho="本命"), _race(ARERU, 1, 2, shobusho="超混戦"),
                 _race(ARERU, 13, 3, shobusho="要注目")]
        text = build_notify_text(date(2026, 7, 5), races)
        self.assertIn("本命: 平和島5R", text)
        self.assertIn("超混戦: 桐生2R", text)
        self.assertIn("購入予算: 2,000円(1レース1,000円)", text)  # 要注目は予算外
        self.assertNotIn("要注目", text)  # 要注目は通知しない(ユーザー指示)
        self.assertIn("https://", text)

    def test_no_shobusho_day(self):
        text = build_notify_text(date(2026, 7, 5), [_race(KATAME)])
        self.assertIn("本日は購入対象なし", text)


class TestRenderVenuePage(unittest.TestCase):
    def test_page_has_nav_picks_and_ken_box(self):
        races = [_race(ARERU, 4, 5, shobusho="本命"), _race(KATAME, 13, 1)]
        html = render_venue_page(date(2026, 7, 5), 4, races)
        # ナビゲーション(v2: トップ=買い目一覧、平和島は自分のページを持つ)
        self.assertIn('href="index.html"', html)      # 本日の買い目
        self.assertIn('href="heiwajima.html"', html)
        self.assertIn('href="amagasaki.html"', html)
        self.assertIn('href="stats.html"', html)
        # A/B/Cの順で掲載
        pos_a = html.find("A 石橋渡")
        pos_b = html.find("B 山田三連単")
        pos_c = html.find("C 勝万舟")
        pos_ken = html.find("予想屋ken のポートフォリオ")
        self.assertTrue(0 < pos_a < pos_b < pos_c < pos_ken)
        # kenは水色ボックス+金額
        self.assertIn("class='ken'", html)
        self.assertIn("計1,000円", html)
        self.assertIn(">本命</span>", html)  # v2バッジ
        self.assertIn("viewport", html)

    def test_non_racing_venue_page(self):
        races = [_race(ARERU, 4, 5)]  # 平和島のみ開催
        html = render_venue_page(date(2026, 7, 5), 20, races)  # 若松ページ
        self.assertIn("非開催", html)
        self.assertIn("若松・休", html) if "若松・休" in html else self.assertIn("休", html)

    def test_weather_display(self):
        wx = {"wind_speed_m": 3.5, "wind_dir": "南東", "wave_height_cm": 2.4, "temperature": 28.0}
        html = render_venue_page(date(2026, 7, 5), 4, [_race(ARERU, 4, 1, wx=wx)])
        self.assertIn("風速3.5m/s(南東の風)", html)
        self.assertIn("予測には未使用", html)


class TestShoppingPage(unittest.TestCase):
    def test_sections_and_order(self):
        from predict import render_shopping_page
        races = [
            _race(ARERU, 1, 2, shobusho="超混戦"),
            _race(ARERU, 4, 5, shobusho="本命"),
            _race(ARERU, 13, 3, shobusho="要注目"),
            _race(KATAME, 20, 1),  # 選外は載らない
        ]
        html = render_shopping_page(date(2026, 7, 5), races)
        self.assertIn("本日の買い目", html)
        pos_hon = html.find("🔴 本命")
        pos_kon = html.find("🟣 超混戦")
        pos_att = html.find("👀 要注目")
        self.assertTrue(0 < pos_hon < pos_kon < pos_att)
        self.assertIn("venue-tag", html)     # 一覧では場名を表示
        self.assertIn("桐生", html)           # 他19場のレースも載る
        self.assertNotIn("若松1R", html)      # 選外レースは載らない
        self.assertIn("購入予算 2,000円", html)
        self.assertNotIn("要注目(観測のみ・購入なし): ", html)  # サマリーには載せない

    def test_empty_day(self):
        from predict import render_shopping_page
        html = render_shopping_page(date(2026, 7, 5), [_race(KATAME, 20, 1)])
        self.assertIn("購入対象なし", html)


if __name__ == "__main__":
    unittest.main()
