# -*- coding: utf-8 -*-
"""3並列取得したodds_backfill*.dbを1本(odds_backfill.db)に統合する

    py -X utf8 test/merge_backfill.py

odds_backfill_b.db・odds_backfill_c.dbの内容をodds_backfill.dbへ
INSERT OR REPLACEし、統合後に_b/_cは削除せず残す(再実行時の冪等性のため)。
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PROJECT_DIR

MAIN = PROJECT_DIR / "odds_backfill.db"
PARTS = ["odds_backfill_b.db", "odds_backfill_c.db"]


def main():
    out = sqlite3.connect(MAIN)
    out.execute("""CREATE TABLE IF NOT EXISTS odds_final (
        race_id TEXT NOT NULL, bet_type TEXT NOT NULL,
        combination TEXT NOT NULL, odds REAL, fetched_at TEXT,
        PRIMARY KEY (race_id, bet_type, combination))""")

    for name in PARTS:
        path = PROJECT_DIR / name
        if not path.exists():
            print(f"{name}: ファイルなし、スキップ")
            continue
        src = sqlite3.connect(path)
        rows = src.execute(
            "SELECT race_id, bet_type, combination, odds, fetched_at "
            "FROM odds_final").fetchall()
        src.close()
        out.executemany(
            "INSERT OR REPLACE INTO odds_final VALUES (?,?,?,?,?)", rows)
        out.commit()
        print(f"{name}: {len(rows):,}行を統合")

    n = out.execute("SELECT COUNT(DISTINCT race_id) FROM odds_final").fetchone()[0]
    d = out.execute(
        "SELECT MIN(substr(race_id,1,8)), MAX(substr(race_id,1,8)) "
        "FROM odds_final").fetchone()
    out.close()
    print(f"\n統合完了: {MAIN.name} 合計{n:,}R (期間 {d[0]}〜{d[1]})")


if __name__ == "__main__":
    main()
