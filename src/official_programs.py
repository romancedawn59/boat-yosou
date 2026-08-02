"""公式サイト(boatrace.jp)から番組表を直接取得するフォールバックエンジン

上流BoatraceOpenAPIの当日データ公開遅延(2026-07-19/20/30・08-02で4回)への
恒久対策(2026-08-02ケンさん指示「PCの電源が点いてなくても発動するように」)。
predict._ensure_program が上流404を検知したとき自動で発動し、公式の出走表ページを
パースして上流互換のprograms JSON(parser_bがそのまま読める形)を組み立てる。

- 開催場は公式の本日レース一覧ページから検出(1リクエスト)
- 出走表ページは1レース1リクエスト・REQUEST_INTERVAL_SEC間隔(公式への負荷配慮)。
  全場で約150リクエスト=5〜8分。発動は上流404時のみ
- 取得結果は data_raw/programs_YYYYMMDD.json にも保存(上流と同じ置き場・
  後続のcollectがキャッシュとして再利用できる)
"""
import json
import re
import time
from datetime import date
from pathlib import Path

import requests

from config import DATA_RAW_DIR, REQUEST_INTERVAL_SEC, USER_AGENT

DAY_INDEX_URL = "https://www.boatrace.jp/owpc/pc/race/index?hd={ymd}"
RACELIST_URL = "https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd:02d}&hd={ymd}"
CLASS_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def _floats(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def parse_racelist(html: str) -> tuple[list[dict], dict, str | None, int | None]:
    """出走表1ページから (boats, 締切{race_no: 'HH:MM'}, タイトル, 距離m) を返す"""
    deadlines = {}
    m = re.search(r"締切予定時刻</td>(.*?)</tr>", html, re.S)
    if m:
        for i, t in enumerate(re.findall(r">(\d{1,2}:\d{2})<", m.group(1)), 1):
            deadlines[i] = t
    title, dist = None, None
    m = re.search(r"title16_titleDetail__add2020\">\s*(.*?)\s*(\d{3,4})m", html, re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip() or None
        dist = int(m.group(2))

    boats = []
    blocks = re.split(r'<td class="is-boatColor(\d) is-fs14" rowspan="4">', html)
    for i in range(1, len(blocks) - 1, 2):
        lane = int(blocks[i])
        block = blocks[i + 1]
        tm = re.search(r"toban=(\d+)", block)
        if not tm:
            continue  # 欠場等
        cm = re.search(r"/\s*<span class=\"[^\"]*\">(A1|A2|B1|B2)</span>", block)
        nm = re.search(r'profile\?toban=\d+">([^<]+)</a>', block)
        aw = re.search(r"(\d+)歳/([\d.]+)kg", block)
        tds = re.findall(r'<td[^>]*rowspan="4"[^>]*>(.*?)</td>', block, re.S)
        fls = nat = loc = mot = boa = None
        numeric_tds = []
        for td in tds:
            if re.match(r"\s*F\d+", td):
                fm = re.search(r"F(\d+).*?L(\d+).*?([\d.]+)", td, re.S)
                if fm:
                    fls = (int(fm.group(1)), int(fm.group(2)), float(fm.group(3)))
            else:
                vals = _floats(re.sub(r"<[^>]+>", " ", td))
                if len(vals) == 3:
                    numeric_tds.append(vals)
        if len(numeric_tds) >= 4:
            nat, loc, mot, boa = numeric_tds[:4]
        boats.append({
            "racer_boat_number": lane,
            "racer_number": int(tm.group(1)),
            "racer_name": re.sub(r"\s+", " ", nm.group(1)).strip() if nm else None,
            "racer_class_number": CLASS_NUM.get(cm.group(1)) if cm else None,
            "racer_branch_number": None,
            "racer_birthplace_number": None,
            "racer_age": int(aw.group(1)) if aw else None,
            "racer_weight": float(aw.group(2)) if aw else None,
            "racer_flying_count": fls[0] if fls else None,
            "racer_late_count": fls[1] if fls else None,
            "racer_average_start_timing": fls[2] if fls else None,
            "racer_national_top_1_percent": nat[0] if nat else None,
            "racer_national_top_2_percent": nat[1] if nat else None,
            "racer_national_top_3_percent": nat[2] if nat else None,
            "racer_local_top_1_percent": loc[0] if loc else None,
            "racer_local_top_2_percent": loc[1] if loc else None,
            "racer_local_top_3_percent": loc[2] if loc else None,
            "racer_assigned_motor_number": int(mot[0]) if mot else None,
            "racer_assigned_motor_top_2_percent": mot[1] if mot else None,
            "racer_assigned_motor_top_3_percent": mot[2] if mot else None,
            "racer_assigned_boat_number": int(boa[0]) if boa else None,
            "racer_assigned_boat_top_2_percent": boa[1] if boa else None,
            "racer_assigned_boat_top_3_percent": boa[2] if boa else None,
        })
    return boats, deadlines, title, dist


def discover_venues(d: date) -> list[int]:
    """本日レース一覧ページから開催場コードを検出する"""
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    resp = session.get(DAY_INDEX_URL.format(ymd=ymd), timeout=30)
    time.sleep(REQUEST_INTERVAL_SEC)
    if resp.status_code != 200:
        return []
    return sorted(set(
        int(x) for x in re.findall(
            r"racelist\?rno=\d+&(?:amp;)?jcd=(\d+)&(?:amp;)?hd=" + ymd, resp.text)))


def fetch_official_programs(d: date, venues: list[int] | None = None,
                            log=print) -> dict:
    """公式サイトから1日分の番組表を取得し、上流v3互換のdictを返す。

    戻り値: {"programs": [...]}(parser_b.parse_programにそのまま渡せる)。
    data_raw/programs_YYYYMMDD.json にも保存する。
    """
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    if venues is None:
        venues = discover_venues(d)
    if not venues:
        log(f"{d}: 公式サイトから開催場を検出できませんでした")
        return {"programs": []}
    log(f"{d}: 公式サイト直取りフォールバック開始({len(venues)}場)")

    programs = []
    for vc in venues:
        deadlines_v: dict[int, str] = {}
        got = 0
        for rno in range(1, 13):
            resp = session.get(RACELIST_URL.format(rno=rno, jcd=vc, ymd=ymd),
                               timeout=30)
            time.sleep(REQUEST_INTERVAL_SEC)
            if resp.status_code != 200:
                continue
            boats, deadlines, title, dist = parse_racelist(resp.text)
            if len(boats) < 2:
                continue  # レースが存在しない/中止
            deadlines_v.update(
                {k: v for k, v in deadlines.items() if k not in deadlines_v})
            closed = deadlines_v.get(rno)
            programs.append({
                "date": d.isoformat(),
                "stadium_number": vc,
                "number": rno,
                "title": title,
                "subtitle": None,
                "grade_label": None,
                "day_label": None,
                "distance": dist,
                "closed_at": f"{d.isoformat()} {closed}:00" if closed else None,
                "boats": boats,
            })
            got += 1
        log(f"  場{vc}: {got}R取得")

    data = {"programs": programs}
    out = Path(DATA_RAW_DIR) / f"programs_{ymd}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    n_boats = sum(len(p["boats"]) for p in programs)
    log(f"{d}: フォールバック完了 {len(programs)}R・{n_boats}艇 -> {out.name}")
    return data
