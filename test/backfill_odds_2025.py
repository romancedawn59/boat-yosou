# -*- coding: utf-8 -*-
"""2025年8-10月の確定最終オッズ遡及取得(ケン理論のアウトオブサンプル検証用)

    py -X utf8 test/backfill_odds_2025.py

- 書き込み先は本体DBではなく odds_backfill.db(別ファイル)。
  明朝6:00の日次ジョブとのSQLiteロック衝突を避けるため
- 冪等: 取得済みrace_idはスキップ(中断→再実行で続きから)
- REQUEST_INTERVAL_SEC を厳守
"""
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
import odds as odds_mod
from config import DB_PATH, PROJECT_DIR, REQUEST_INTERVAL_SEC

START, END = "2025-08-01", "2025-10-31"
OUT_PATH = PROJECT_DIR / "odds_backfill.db"


def main():
    # 対象レース一覧を本体DBから読み、すぐ閉じる(ロック回避)
    conn = db.connect(DB_PATH)
    targets = conn.execute(
        "SELECT race_id, venue_code, race_no, date FROM races "
        "WHERE date BETWEEN ? AND ? "
        "AND EXISTS (SELECT 1 FROM results s WHERE s.race_id = races.race_id "
        "            AND s.arrival_order = 1) "
        "ORDER BY date, venue_code, race_no", (START, END)).fetchall()
    conn.close()

    out = sqlite3.connect(OUT_PATH)
    out.execute("""CREATE TABLE IF NOT EXISTS odds_final (
        race_id TEXT NOT NULL, bet_type TEXT NOT NULL,
        combination TEXT NOT NULL, odds REAL, fetched_at TEXT,
        PRIMARY KEY (race_id, bet_type, combination))""")
    done = {r[0] for r in out.execute("SELECT DISTINCT race_id FROM odds_final")}
    todo = [t for t in targets if t[0] not in done]
    print(f"対象{len(targets):,}R 取得済{len(done):,}R 残り{len(todo):,}R", flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, (race_id, vc, rno, d_str) in enumerate(todo, 1):
        try:
            o = odds_mod.fetch_odds(vc, rno, date.fromisoformat(d_str))
        except Exception as e:
            fail += 1
            if fail <= 20:
                print(f"{race_id}: 失敗 ({e})", flush=True)
            time.sleep(REQUEST_INTERVAL_SEC)
            continue
        fetched_at = datetime.now().isoformat(timespec="seconds")
        n = 0
        for bt_name, sep in (("3連単", "-"), ("3連複", "=")):
            for key, val in o[bt_name].items():
                out.execute(
                    "INSERT OR REPLACE INTO odds_final VALUES (?,?,?,?,?)",
                    (race_id, bt_name, sep.join(map(str, key)), val, fetched_at))
                n += 1
        out.commit()
        if n:
            ok += 1
        if i % 200 == 0:
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"{i:,}/{len(todo):,}R 完了 成功{ok:,} 失敗{fail} "
                  f"経過{el/3600:.1f}h 残り目安{eta/3600:.1f}h", flush=True)
        time.sleep(REQUEST_INTERVAL_SEC)

    out.close()
    print(f"完了: 成功{ok:,}R 失敗{fail}R → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
