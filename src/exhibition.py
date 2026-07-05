"""直前情報(展示タイム・チルト)の取得

BOATRACE公式サイト(boatrace.jp)の直前情報ページをスクレイピングする。
展示は締切の約20分前に実施されるため、その時間にならないとデータは存在しない。

現時点では予測モデルの特徴量には使わず、将来の分析用にDBへ保存するだけ
(collect_exhibition.py参照)。robots.txt確認済み(Disallow指定なし)。
"""
import re
import urllib.request
from datetime import date

from config import USER_AGENT

_URL_TMPL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={venue:02d}&hd={yyyymmdd}"

# レーサー詳細リンクのtoban(登録番号)→体重→展示タイム→チルトの並びで
# 1艇ぶんずつ出現する。マッチ順=枠番1〜6の順。
_ROW_PATTERN = re.compile(
    r'toban=(\d+)\">.*?<td rowspan="2">([\d.]+)kg</td>\s*'
    r'<td rowspan="4">([\d.]+)</td>\s*<td rowspan="4">(-?[\d.]+)</td>',
    re.S,
)


def parse_exhibition_html(html: str) -> list[dict]:
    """直前情報ページのHTMLから艇ごとの展示データを抽出する(枠番1〜6の順)"""
    return [
        {
            "lane": lane,
            "reg_no": int(reg_no),
            "weight_kg": float(weight),
            "exhibition_time": float(ex_time),
            "tilt": float(tilt),
        }
        for lane, (reg_no, weight, ex_time, tilt) in enumerate(_ROW_PATTERN.findall(html), 1)
    ]


def fetch_exhibition(venue_code: int, race_no: int, d: date) -> list[dict]:
    """指定レースの直前情報を取得する。展示未実施・非開催なら空リストを返す"""
    url = _URL_TMPL.format(race_no=race_no, venue=venue_code, yyyymmdd=d.strftime("%Y%m%d"))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    return parse_exhibition_html(html)
