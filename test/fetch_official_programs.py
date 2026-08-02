# -*- coding: utf-8 -*-
"""公式サイト(boatrace.jp)から番組表を直接取得するフォールバックエンジン

    py -X utf8 test/fetch_official_programs.py [YYYY-MM-DD]

上流BoatraceOpenAPIの当日データ公開遅延(7/19・7/20・7/30・8/2で4回目)への
非常用エンジン。公式の出走表ページ(racelist)を全レース分パースし、
上流と互換のprograms JSON(v3形式・parser_bがそのまま読める)を
data_raw/programs_YYYYMMDD.json に書き出す。
その後は通常どおり: py collect.py → py predict.py today で復旧できる。

- 開催場はDBのracesテーブル(結果APIは正常なのでレース枠が入っている)から取る
- 約144リクエスト×1秒間隔=3分弱。公式サイトへの負荷は最小限に
- 欠場等で選手が空の枠はスキップ(parser_bの既存仕様と同じ扱いになる)
"""
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import db
from config import DB_PATH, DATA_RAW_DIR, REQUEST_INTERVAL_SEC, USER_AGENT, jst_today

BASE = "https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd:02d}&hd={ymd}"
CLASS_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def _floats(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]


def parse_racelist(html: str) -> tuple[list[dict], dict, str | None, int | None]:
    """1ページから (boats, 締切{race_no: 'HH:MM'}, タイトル, 距離m) を返す"""
    # 締切予定時刻の行(そのページの場の全12R分が載っている)
    deadlines = {}
    m = re.search(r"締切予定時刻</td>(.*?)</tr>", html, re.S)
    if m:
        for i, t in enumerate(re.findall(r">(\d{1,2}:\d{2})<", m.group(1)), 1):
            deadlines[i] = t
    # タイトルと距離
    title, dist = None, None
    m = re.search(r"title16_titleDetail__add2020\">\s*(.*?)\s*(\d{3,4})m", html, re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip() or None
        dist = int(m.group(2))

    boats = []
    # 枠ごとのブロックに分割(先頭は枠番セル)
    blocks = re.split(r'<td class="is-boatColor(\d) is-fs14" rowspan="4">', html)
    for i in range(1, len(blocks) - 1, 2):
        lane = int(blocks[i])
        block = blocks[i + 1]
        tm = re.search(r"toban=(\d+)", block)
        if not tm:
            continue  # 欠場等
        reg_no = int(tm.group(1))
        cm = re.search(r"/\s*<span class=\"[^\"]*\">(A1|A2|B1|B2)</span>", block)
        nm = re.search(r'profile\?toban=\d+">([^<]+)</a>', block)
        aw = re.search(r"(\d+)歳/([\d.]+)kg", block)
        tds = re.findall(r'<td[^>]*rowspan="4"[^>]*>(.*?)</td>', block, re.S)
        fls = nat = loc = mot = boa = None
        numeric_tds = []
        for td in tds:
            if re.search(r">F\d+", "> " + td) or re.match(r"\s*F\d+", td):
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
            "racer_number": reg_no,
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


def main() -> None:
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else jst_today()
    ymd = f"{d.year:04d}{d.month:02d}{d.day:02d}"

    conn = db.connect(DB_PATH)
    rows = conn.execute(
        "SELECT venue_code, race_no FROM races WHERE date = ? "
        "ORDER BY venue_code, race_no", (d.isoformat(),)).fetchall()
    conn.close()
    if not rows:
        print(f"{d}: racesテーブルにレース枠がありません(結果APIも未公開?)")
        return
    by_venue: dict[int, list[int]] = {}
    for vc, rno in rows:
        by_venue.setdefault(vc, []).append(rno)
    print(f"{d}: {len(by_venue)}場 {len(rows)}レースを公式サイトから取得します")

    programs = []
    for vc, rnos in by_venue.items():
        deadlines_v: dict[int, str] = {}
        for rno in rnos:
            url = BASE.format(rno=rno, jcd=vc, ymd=ymd)
            resp = session.get(url, timeout=30)
            time.sleep(REQUEST_INTERVAL_SEC)
            if resp.status_code != 200:
                print(f"  場{vc} {rno}R: HTTP {resp.status_code} スキップ")
                continue
            boats, deadlines, title, dist = parse_racelist(resp.text)
            deadlines_v.update({k: v for k, v in deadlines.items() if k not in deadlines_v})
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
        got = [p for p in programs if p["stadium_number"] == vc]
        n_boats = sum(len(p["boats"]) for p in got)
        print(f"  場{vc}: {len(got)}R {n_boats}艇 取得")

    out = DATA_RAW_DIR / f"programs_{ymd}.json"
    out.write_text(json.dumps({"programs": programs}, ensure_ascii=False),
                   encoding="utf-8")
    total_boats = sum(len(p["boats"]) for p in programs)
    print(f"書き出し: {out} ({len(programs)}R・{total_boats}艇)")
    print("次の手順: cd src && py -X utf8 collect.py && py -X utf8 predict.py today")


if __name__ == "__main__":
    main()
