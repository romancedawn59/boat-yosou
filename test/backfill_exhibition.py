# -*- coding: utf-8 -*-
"""展示データ(スタート展示の進入・ST・安定板)の遡及取得(2026-09-21ケンさん指示)

    py -X utf8 test/backfill_exhibition.py            # 取得(別ファイルのSQLiteへ書く)
    py -X utf8 test/backfill_exhibition.py --merge    # 本番DBの exhibition へ統合
    py -X utf8 test/backfill_exhibition.py --chain    # 取得→統合→第1段階の集計まで続けて実行

対象: 本番DBの exhibition に exhibition_time があり ex_st / ex_course が空のレース。
順序: ① odds_final があるレース → ② odds(締切15分前)があるレース → ③ それ以外(各段階は新しい日付から)。
守ること:
 - 本番DBには書かない(確定オッズの遡及と同時書き込みを避ける)。data_raw/exhibition_backfill.db へ。
 - リクエストの間隔は MIN_INTERVAL_SEC 以上。1本の接続で順番に取る。
 - 応答時間とエラーを記録し、悪化したら自動で止まる(直近20件でエラー5件以上、
   または平均応答が最初の20件の2倍超かつ20秒超)。403/429は即中止。
 - 取得済み(fetch_logにある)レースは飛ばすので、再実行で続きから取れる。
統合: ex_st / ex_course / stabilizer だけをUPDATE。exhibition_time 等の既存列は上書きしない。
"""
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import exhibition as ex_mod
from config import DB_PATH, JST, USER_AGENT

BACKFILL_DB = Path(DB_PATH).parent / "data_raw" / "exhibition_backfill.db"
MIN_INTERVAL_SEC = 2.0
STOP_STATUS = (403, 429)


def open_backfill():
    conn = sqlite3.connect(BACKFILL_DB, timeout=60)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS exhibition (
            race_id TEXT NOT NULL, lane INTEGER NOT NULL, reg_no INTEGER, weight_kg REAL,
            exhibition_time REAL, tilt REAL, ex_course INTEGER, ex_st REAL, stabilizer INTEGER,
            PRIMARY KEY (race_id, lane));
        CREATE TABLE IF NOT EXISTS fetch_log (
            race_id TEXT PRIMARY KEY, stage INTEGER, status TEXT, latency_sec REAL, fetched_at TEXT);
    """)
    return conn


def pick_targets(done: set[str]) -> list[tuple[str, int, int, str, int]]:
    main = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=120)
    rows = main.execute("""
        SELECT r.race_id, r.venue_code, r.race_no, r.date,
               CASE WHEN EXISTS(SELECT 1 FROM odds_final f WHERE f.race_id=r.race_id) THEN 1
                    WHEN EXISTS(SELECT 1 FROM odds o WHERE o.race_id=r.race_id) THEN 2 ELSE 3 END AS stage
        FROM races r
        WHERE EXISTS(SELECT 1 FROM exhibition e WHERE e.race_id=r.race_id
                     AND e.exhibition_time IS NOT NULL AND e.ex_st IS NULL AND e.ex_course IS NULL)
          AND r.date < date('now', 'localtime')
        ORDER BY stage, r.date DESC, r.venue_code, r.race_no
    """).fetchall()
    main.close()
    return [r for r in rows if r[0] not in done]


def fetch(venue_code: int, race_no: int, ymd: str) -> tuple[list[dict], float]:
    url = ex_mod._URL_TMPL.format(race_no=race_no, venue=venue_code, yyyymmdd=ymd)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=40) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    return ex_mod.parse_exhibition_html(html), time.time() - t0


def run() -> bool:
    """全段階を取り切ったらTrueを返す(途中停止はFalse)"""
    bf = open_backfill()
    done = {r[0] for r in bf.execute("SELECT race_id FROM fetch_log")}
    targets = pick_targets(done)
    per_stage = {s: sum(1 for t in targets if t[4] == s) for s in (1, 2, 3)}
    print(f"取得対象: {len(targets):,}レース(①確定オッズあり {per_stage[1]:,} / ②15分前オッズあり "
          f"{per_stage[2]:,} / ③それ以外 {per_stage[3]:,}) 取得済み {len(done):,}", flush=True)
    started = time.time()
    recent: list[tuple[float, bool]] = []   # (応答秒, エラーか)
    baseline = None
    cur_stage = None
    n_ok = n_err = 0
    for i, (race_id, venue, race_no, d_str, stage) in enumerate(targets, 1):
        if cur_stage is not None and stage != cur_stage:
            c = bf.execute("SELECT COUNT(*) FROM fetch_log WHERE stage=? AND status='ok'", (cur_stage,)).fetchone()[0]
            print(f"===== 段階{cur_stage} 完了: 取得成功 {c:,}レース =====", flush=True)
        cur_stage = stage
        t0 = time.time()
        status, latency = "ok", None
        try:
            rows, latency = fetch(venue, race_no, d_str.replace("-", ""))
            if rows and any(r["ex_course"] is not None for r in rows):
                for r in rows:
                    bf.execute("INSERT OR REPLACE INTO exhibition VALUES (?,?,?,?,?,?,?,?,?)",
                               (race_id, r["lane"], r["reg_no"], r["weight_kg"], r["exhibition_time"],
                                r["tilt"], r["ex_course"], r["ex_st"], r["stabilizer"]))
                n_ok += 1
            else:
                status = "empty"
        except urllib.error.HTTPError as e:
            status = f"http{e.code}"
            if e.code in STOP_STATUS:
                print(f"{race_id}: HTTP {e.code} 拒否または過負荷の応答のため中止します", flush=True)
                break
        except Exception as e:
            status = f"error:{type(e).__name__}"
        is_err = status.startswith(("http", "error"))
        n_err += is_err
        latency = latency if latency is not None else time.time() - t0
        recent = (recent + [(latency, is_err)])[-20:]
        if not is_err:                       # エラーのレースは記録しない=次回の再実行で取り直す
            bf.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?)",
                       (race_id, stage, status, latency, datetime.now(JST).isoformat(timespec="seconds")))
        if i % 20 == 0:
            bf.commit()
            avg = sum(x[0] for x in recent) / len(recent)
            errs = sum(1 for x in recent if x[1])
            baseline = baseline or avg
            el = time.time() - started
            print(f"{i:,}/{len(targets):,} 成功{n_ok:,} エラー{n_err} 直近20件: 平均応答{avg:.1f}秒・エラー{errs}件 "
                  f"/ {i / el * 3600:.0f}レース/時 残り見込み{el / i * (len(targets) - i) / 3600:.1f}時間 直近{race_id}",
                  flush=True)
            if errs >= 5 or (avg > baseline * 2 and avg > 20):
                print(f"応答が悪化したため一時停止します(基準{baseline:.1f}秒 → 直近{avg:.1f}秒・エラー{errs}件)。"
                      f"同じコマンドの再実行で続きから取れます", flush=True)
                break
        wait = MIN_INTERVAL_SEC - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
    else:
        if cur_stage is not None:
            c = bf.execute("SELECT COUNT(*) FROM fetch_log WHERE stage=? AND status='ok'", (cur_stage,)).fetchone()[0]
            print(f"===== 段階{cur_stage} 完了: 取得成功 {c:,}レース =====", flush=True)
        print("全段階の取得が完了しました", flush=True)
        bf.commit()
        bf.close()
        return True
    bf.commit()
    bf.close()
    return False


def merge() -> None:
    main = sqlite3.connect(DB_PATH, timeout=300)
    count = lambda: main.execute(
        "SELECT COUNT(DISTINCT CASE WHEN ex_st IS NOT NULL THEN race_id END), "
        "COUNT(DISTINCT CASE WHEN ex_course IS NOT NULL THEN race_id END), "
        "COUNT(DISTINCT CASE WHEN exhibition_time IS NOT NULL THEN race_id END), "
        "COALESCE(SUM(exhibition_time), 0) FROM exhibition").fetchone()
    before = count()
    main.execute("ATTACH DATABASE ? AS bf", (str(BACKFILL_DB),))
    main.execute("""
        UPDATE exhibition SET
            ex_st = (SELECT b.ex_st FROM bf.exhibition b WHERE b.race_id=exhibition.race_id AND b.lane=exhibition.lane),
            ex_course = (SELECT b.ex_course FROM bf.exhibition b WHERE b.race_id=exhibition.race_id AND b.lane=exhibition.lane),
            stabilizer = (SELECT b.stabilizer FROM bf.exhibition b WHERE b.race_id=exhibition.race_id AND b.lane=exhibition.lane)
        WHERE ex_course IS NULL
          AND EXISTS (SELECT 1 FROM bf.exhibition b WHERE b.race_id=exhibition.race_id AND b.lane=exhibition.lane)
    """)
    main.commit()
    after = count()
    main.close()
    print(f"統合前: ex_stあり {before[0]:,}R / ex_courseあり {before[1]:,}R / 展示タイムあり {before[2]:,}R")
    print(f"統合後: ex_stあり {after[0]:,}R / ex_courseあり {after[1]:,}R / 展示タイムあり {after[2]:,}R")
    print(f"既存列の保全: 展示タイムの合計 {before[3]:.2f} → {after[3]:.2f}({'変化なし' if abs(before[3] - after[3]) < 1e-6 else '変化あり!'})")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    elif "--chain" in sys.argv:       # 取得 → 統合 → 第1段階の集計まで続けて行う
        if run():
            merge()
            import subprocess
            here = Path(__file__).resolve().parent
            subprocess.run([sys.executable, "-X", "utf8", str(here / "sim_eliminate_by_exhibition.py")])
    else:
        run()
