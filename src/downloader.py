"""BoatraceOpenAPIの日次JSON(番組表・競走成績)のダウンロード"""
import time
from datetime import date
from pathlib import Path

import requests

from config import DATA_RAW_DIR, PROGRAMS_URL_TMPL, RESULTS_URL_TMPL, REQUEST_INTERVAL_SEC, USER_AGENT

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def _urls_for(d: date) -> tuple[str, str]:
    yyyy = f"{d.year:04d}"
    yyyymmdd = f"{d.year:04d}{d.month:02d}{d.day:02d}"
    program_url = PROGRAMS_URL_TMPL.format(yyyy=yyyy, yyyymmdd=yyyymmdd)
    result_url = RESULTS_URL_TMPL.format(yyyy=yyyy, yyyymmdd=yyyymmdd)
    return program_url, result_url


def _dest_path(kind: str, d: date) -> Path:
    return DATA_RAW_DIR / f"{kind}_{d.year:04d}{d.month:02d}{d.day:02d}.json"


def _download_one(url: str, dest: Path, force: bool = False) -> Path | None:
    if dest.exists() and not force:
        return dest

    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None  # その日は開催なし、または保持期間外・未公開
    resp.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    time.sleep(REQUEST_INTERVAL_SEC)
    return dest


def download_day(d: date, force: bool = False) -> dict[str, Path | None]:
    """指定日の番組表・成績JSONをダウンロードしローカルパスを返す"""
    program_url, result_url = _urls_for(d)
    program_path = _download_one(program_url, _dest_path("programs", d), force=force)
    result_path = _download_one(result_url, _dest_path("results", d), force=force)
    return {"program": program_path, "result": result_path}
