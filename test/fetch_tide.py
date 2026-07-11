# -*- coding: utf-8 -*-
"""気象庁の潮位表(天文潮汐の推算値)を取得しtideテーブルへ格納するCLI(検証⑩用)

    py -X utf8 test/fetch_tide.py              # 2025・2026年の4地点を取得
    py -X utf8 test/fetch_tide.py 2024         # 年を指定して追加取得

出典・利用条件:
- 気象庁「潮位表」 https://www.data.jma.go.jp/kaiyou/db/tide/suisan/
  テキストデータ: https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{年}/{地点}.txt
- 気象庁ホームページのコンテンツは出典明記で利用可(政府標準利用規約準拠・CC BY 4.0互換)
- 潮汐は天文計算による決定論的な推算値のため、日次収集は不要。年単位で一括取得し、
  生データを data_raw/tide/ に、1時間刻みの潮位を tide テーブルに保存する

地点の選定(場→気象庁の潮位表地点):
- 江戸川(3)・平和島(4) → 東京(TK)。両場とも東京湾奥で検潮所が至近
- 常滑(8) → 名古屋(NG)。伊勢湾内で最寄り。※常滑は水門で干満の影響なし=対照群
- 尼崎(13) → 大阪(OS)。大阪湾奥で最寄り(神戸KBより尼崎に近い)。※淡水プール=対照群
- 若松(20) → 門司(MO)。洞海湾は関門海峡に直結しており、気象庁の推算地点では
  門司(33°57'N 130°57'E)が最も近い(戸畑は推算地点一覧に存在しない)

テキスト形式(気象庁readmeより):
- 1行=1日。1〜72桁が毎時潮位(0時〜23時、3桁×24個、cm)、
  73-74=年(下2桁)、75-76=月、77-78=日、79-80=地点コード。以降は満潮・干潮情報(未使用)
"""
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
from config import DATA_RAW_DIR, DB_PATH, REQUEST_INTERVAL_SEC, USER_AGENT

TIDE_URL_TMPL = "https://www.data.jma.go.jp/kaiyou/data/db/tide/suisan/txt/{year}/{station}.txt"
TIDE_RAW_DIR = DATA_RAW_DIR / "tide"

# 場コード -> 気象庁地点コード(選定理由はdocstring参照)
VENUE_STATION = {3: "TK", 4: "TK", 8: "NG", 13: "OS", 20: "MO"}
STATIONS = sorted(set(VENUE_STATION.values()))

DEFAULT_YEARS = (2025, 2026)


def parse_tide_text(text: str) -> list[tuple[str, str, float]]:
    """潮位表テキスト -> [(station, 'YYYY-MM-DD HH:00:00', level_cm)]"""
    rows = []
    for line in text.splitlines():
        if len(line) < 80:
            continue
        try:
            yy = int(line[72:74])
            mm = int(line[74:76])
            dd = int(line[76:78])
        except ValueError:
            continue
        station = line[78:80].strip()
        year = 2000 + yy  # 潮位表の提供範囲は2000年以降のため下2桁+2000で確定
        for h in range(24):
            cell = line[h * 3:(h + 1) * 3].strip()
            if not cell:
                continue
            try:
                level = float(cell)
            except ValueError:
                continue  # 欠測表現はスキップ
            rows.append((station, f"{year:04d}-{mm:02d}-{dd:02d} {h:02d}:00:00", level))
    return rows


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_year(conn, station: str, year: int) -> int:
    """1地点1年分を取得して格納。取得済み(365日×24時間の9割以上)ならスキップ(冪等)"""
    have = conn.execute(
        "SELECT COUNT(*) FROM tide WHERE station = ? AND datetime LIKE ?",
        (station, f"{year:04d}-%"),
    ).fetchone()[0]
    if have >= 365 * 24 * 0.9:
        print(f"{station} {year}: 取得済み({have}件)のためスキップ")
        return 0

    url = TIDE_URL_TMPL.format(year=year, station=station)
    text = _fetch(url)
    TIDE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    (TIDE_RAW_DIR / f"{station}_{year}.txt").write_text(text, encoding="utf-8")

    rows = parse_tide_text(text)
    for station_code, dt, level in rows:
        db.upsert_tide(conn, {"station": station_code, "datetime": dt, "level_cm": level})
    conn.commit()
    print(f"{station} {year}: {len(rows):,}件を格納")
    return len(rows)


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or list(DEFAULT_YEARS)
    conn = db.connect(DB_PATH)
    total = 0
    for year in years:
        for station in STATIONS:
            try:
                total += fetch_year(conn, station, year)
            except Exception as e:
                print(f"{station} {year}: 取得失敗 ({e})")
            time.sleep(REQUEST_INTERVAL_SEC)  # サーバー負荷配慮
    conn.close()
    print(f"完了: {total:,}件")
