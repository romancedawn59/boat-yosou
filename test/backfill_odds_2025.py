# -*- coding: utf-8 -*-
"""2025年8-10月の確定最終オッズ遡及取得(ケン理論のアウトオブサンプル検証用)

    py -X utf8 test/backfill_odds_2025.py [START END] [--out PATH] [--interval SEC]

引数省略時は2025-08-01〜2025-10-31を対象に odds_backfill.db へ書き込む。
月ごとに並列実行する場合は、日付範囲と出力DBをストリームごとに分けて
別プロセスで起動し、--interval を並列本数倍に伸ばすことでサーバーへの
合計リクエストレートを単一実行時と同じに保つ(例: 3並列なら各3.0秒)。
完了後は複数の odds_backfill_*.db を merge_backfill.py 等で本体に統合するか、
検証スクリプト側でATTACHして横断クエリする。

- 書き込み先は本体DBではなく別ファイル(既定 odds_backfill.db)。
  明朝6:00の日次ジョブとのSQLiteロック衝突を避けるため
- 冪等: 取得済みrace_idはスキップ(中断→再実行で続きから)
- REQUEST_INTERVAL_SEC(既定値)を基準に --interval で調整
"""
import argparse
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
import odds as odds_mod
from config import DB_PATH, PROJECT_DIR, REQUEST_INTERVAL_SEC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("start", nargs="?", default="2025-08-01")
    p.add_argument("end", nargs="?", default="2025-10-31")
    p.add_argument("--out", default=None, help="出力DBファイル名(既定: odds_backfill.db)")
    p.add_argument("--interval", type=float, default=REQUEST_INTERVAL_SEC,
                   help="1レースごとの待機秒数(並列本数倍に伸ばして総負荷を一定に保つ)")
    return p.parse_args()


def main():
    args = parse_args()
    START, END = args.start, args.end
    OUT_PATH = PROJECT_DIR / (args.out or "odds_backfill.db")
    interval = args.interval

    # 対象レース一覧を本体DBから読み、すぐ閉じる(ロック回避)
    conn = db.connect(DB_PATH)
    targets = conn.execute(
        "SELECT race_id, venue_code, race_no, date FROM races "
        "WHERE date BETWEEN ? AND ? "
        "AND EXISTS (SELECT 1 FROM results s WHERE s.race_id = races.race_id "
        "            AND s.arrival_order = 1) "
        "ORDER BY date, venue_code, race_no", (START, END)).fetchall()
    conn.close()

    out = sqlite3.connect(OUT_PATH, timeout=30)
    out.execute("PRAGMA journal_mode=WAL")  # 読み取り専用アクセスと共存するため
    out.execute("""CREATE TABLE IF NOT EXISTS odds_final (
        race_id TEXT NOT NULL, bet_type TEXT NOT NULL,
        combination TEXT NOT NULL, odds REAL, fetched_at TEXT,
        PRIMARY KEY (race_id, bet_type, combination))""")
    done = {r[0] for r in out.execute("SELECT DISTINCT race_id FROM odds_final")}
    todo = [t for t in targets if t[0] not in done]
    print(f"[{OUT_PATH.name}] {START}〜{END} 対象{len(targets):,}R "
          f"取得済{len(done):,}R 残り{len(todo):,}R interval={interval}s", flush=True)

    ok = fail = 0
    t0 = time.time()
    for i, (race_id, vc, rno, d_str) in enumerate(todo, 1):
        try:
            o = odds_mod.fetch_odds(vc, rno, date.fromisoformat(d_str))
        except Exception as e:
            fail += 1
            if fail <= 20:
                print(f"{race_id}: 失敗 ({e})", flush=True)
            time.sleep(interval)
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
        time.sleep(interval)

    out.close()
    print(f"完了: 成功{ok:,}R 失敗{fail}R → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
