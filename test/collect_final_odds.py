# -*- coding: utf-8 -*-
"""確定最終オッズの遡及取得CLI(市場分析基盤・約束①用)

    py -X utf8 test/collect_final_odds.py
    py -X utf8 test/collect_final_odds.py --all --from 2026-09-01 --to 2026-09-20 [--limit 500]

--all を付けると、15分前スナップショットの有無に関係なく期間内の全レースを対象にする
(2026-09-21: 公式サイトは2025-10の過去日でも全120通り+3連複20通りの確定オッズを返すと確認。
 市場分析の網羅性はこれで確保できる。1レース2リクエスト・間隔はREQUEST_INTERVAL_SEC)。

既定の対象: 「15分前スナップショット(oddsテーブル、fetched_at≠'final-backfill')があり、
かつ odds_final にまだ無い」過去日のレースのみ。
公式サイトの過去日付オッズページは最終オッズを表示し続ける(HANDOVERの裏技)ため、
既存 odds.fetch_odds() をそのまま過去日付で呼べば確定最終オッズが取れる。

方針:
- oddsテーブル(15分前スナップショットの純度が資産)には一切書かない。
  最終オッズは odds_final テーブルへ分離保存する
- 当日(JST)のレースは未確定のため対象外
- 冪等: odds_finalに1件でもあるレースはスキップ(再実行で差分のみ取得)
- REQUEST_INTERVAL_SEC を厳守してサーバー負荷に配慮する
"""
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
import odds as odds_mod
from config import DB_PATH, JST, REQUEST_INTERVAL_SEC, jst_today


def pick_targets(conn, today: date) -> list[tuple[str, int, int, str]]:
    """取得対象の(race_id, venue_code, race_no, date)。

    final-backfill行はスナップショットではないため対象にしない
    (比較対象の「15分前」が存在しないレースの最終オッズは分析に使えない)。
    """
    return conn.execute(
        """
        SELECT DISTINCT o.race_id, r.venue_code, r.race_no, r.date
        FROM odds o
        JOIN races r ON r.race_id = o.race_id
        WHERE o.fetched_at != 'final-backfill'
          AND r.date < ?
          AND NOT EXISTS (SELECT 1 FROM odds_final f WHERE f.race_id = o.race_id)
        ORDER BY r.date, r.venue_code, r.race_no
        """,
        (today.isoformat(),),
    ).fetchall()


MIN_INTERVAL_SEC = 2.0     # 2本同時実行でも合計1秒1リクエスト以下にするための下限
COMMIT_EVERY = 20          # 同期ドライブ上のDBはコミットが遅いのでまとめて書く
STOP_STATUS = (403, 429)   # 拒否・過負荷の応答が来たら即中止する(サイト側の意思表示を尊重)
MAX_CONSECUTIVE_FAIL = 5


def collect(conn, targets: list[tuple[str, int, int, str]], tri_only: bool = False) -> int:
    """1本の接続で順番に取得する(並列化しない)。

    2026-09-21の実測で公式サイトは1リクエストあたり約8〜10秒かけて応答する(接続は0.04秒で
    サーバー側の待ち)。これはサイト側のペース指定とみなし、応答が速い場合だけ
    REQUEST_INTERVAL_SEC の待機を足す。tri_only=True は3連単だけ(リクエスト半分)。
    """
    import urllib.error
    ok = fails = 0
    started = time.time()
    fetched_at = datetime.now(JST).isoformat(timespec="seconds")
    for i, (race_id, venue_code, race_no, d_str) in enumerate(targets, 1):
        t0 = time.time()
        try:
            d = date.fromisoformat(d_str)
            if tri_only:
                ymd = d.strftime("%Y%m%d")
                o = {"3連単": odds_mod.parse_odds3t(odds_mod._fetch(
                    odds_mod._URL_3T.format(rno=race_no, jcd=venue_code, ymd=ymd))), "3連複": {}}
            else:
                o = odds_mod.fetch_odds(venue_code, race_no, d)
            fails = 0
        except urllib.error.HTTPError as e:
            print(f"{race_id}: HTTP {e.code}", flush=True)
            if e.code in STOP_STATUS:
                print("拒否または過負荷の応答のため中止します", flush=True)
                break
            fails += 1
            o = None
        except Exception as e:
            print(f"{race_id}: 取得失敗 ({e})", flush=True)
            fails += 1
            o = None
        if fails >= MAX_CONSECUTIVE_FAIL:
            print(f"{MAX_CONSECUTIVE_FAIL}回連続で失敗したため中止します", flush=True)
            break
        n = 0
        if o:
            for bt_name, sep in (("3連単", "-"), ("3連複", "=")):
                for key, val in o[bt_name].items():
                    db.upsert_odds_final(conn, {
                        "race_id": race_id, "bet_type": bt_name,
                        "combination": sep.join(map(str, key)),
                        "odds": val, "fetched_at": fetched_at,
                    })
                    n += 1
            ok += n > 0
        if i % COMMIT_EVERY == 0:
            conn.commit()
            el = time.time() - started
            print(f"{i:,}/{len(targets):,} 完了{ok:,} 直近{race_id} "
                  f"{i / el * 3600:.0f}レース/時 経過{el / 3600:.1f}時間 残り見込み{el / i * (len(targets) - i) / 3600:.0f}時間",
                  flush=True)
        # 1リクエストあたり MIN_INTERVAL_SEC 以上あける(展示の遡及と同時に走らせるための取り決め)
        wait = MIN_INTERVAL_SEC * (1 if tri_only else 2) - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
    conn.commit()
    return ok


def pick_all_targets(conn, today: date, date_from: str, date_to: str,
                     limit: int | None = None) -> list[tuple[str, int, int, str]]:
    """期間内で結果が確定していて odds_final に無い全レース(新しい日付から)"""
    rows = conn.execute(
        """
        SELECT r.race_id, r.venue_code, r.race_no, r.date
        FROM races r
        WHERE r.date >= ? AND r.date <= ? AND r.date < ?
          AND EXISTS (SELECT 1 FROM payouts p WHERE p.race_id = r.race_id)
          AND NOT EXISTS (SELECT 1 FROM odds_final f WHERE f.race_id = r.race_id)
        ORDER BY r.date DESC, r.venue_code, r.race_no
        """,
        (date_from, date_to, today.isoformat()),
    ).fetchall()
    return rows[:limit] if limit else rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--from", dest="date_from", default="2025-07-15")
    ap.add_argument("--to", dest="date_to", default="2099-12-31")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tri-only", action="store_true", help="3連単だけ取る(リクエスト半分)")
    args = ap.parse_args()
    conn = db.connect(DB_PATH)
    targets = (pick_all_targets(conn, jst_today(), args.date_from, args.date_to, args.limit)
               if args.all else pick_targets(conn, jst_today()))
    print(f"取得対象: {len(targets)}レース")
    done = collect(conn, targets, tri_only=args.tri_only)
    conn.close()
    print(f"完了: {done}/{len(targets)}レース")
