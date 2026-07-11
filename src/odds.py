"""直前オッズ(3連単・3連複)の取得

boatrace.jp公式のオッズページをスクレイピングする。オッズは締切直前まで
変動するため、collect_exhibition.py(10分おき)から締切前に取得して保存する。
過去のオッズはどこにも配布されていないため、自前で貯めることに価値がある
(将来のバリューベッティング=モデル確率×オッズ>1のときだけ買う判定に使う)。

過去日付のページは最終オッズを表示し続けるので、パーサーの検証にも使える。
"""
import re
import urllib.request
from datetime import date

from config import USER_AGENT

_URL_3T = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={ymd}"
_URL_3F = "https://www.boatrace.jp/owpc/pc/race/odds3f?rno={rno}&jcd={jcd:02d}&hd={ymd}"


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_rows(html: str) -> list[list[tuple[bool, str]]]:
    """オッズ表のtbody(oddsPointセルを含むもの)の各行を [(rowspan有無, セル文字列), ...] にする

    ページには締切時刻表など複数のtbodyがあるため、オッズセルを含むtbodyだけを対象にする。
    """
    rows = []
    for chunk in html.split("<tbody")[1:]:
        body = chunk.split("</tbody>", 1)[0]
        if "oddsPoint" not in body:
            continue
        for tr in body.split("<tr")[1:]:
            tr = tr.split("</tr>", 1)[0]
            cells = []
            for m in re.finditer(r"<td([^>]*)>(.*?)</td>", tr, re.S):
                attrs, content = m.group(1), m.group(2)
                text = re.sub(r"<[^>]+>", "", content).strip()
                cells.append(("rowspan" in attrs, text))
            if cells:
                rows.append(cells)
    return rows


def _walk_groups(rows: list[list[tuple[bool, str]]]):
    """(グループ番号, 2番手艇, 3番手艇, オッズ文字列) を順に生成する。

    行は6グループ×k セルの均一構造。k=3なら各グループ[2番手, 3番手, オッズ]、
    k=2なら[3番手, オッズ](2番手は前の行から継続)。
    ブロック最終行の2番手セルはrowspanなしで現れることがあるため、
    rowspan属性ではなくセル数kで判定する。
    """
    second = {}
    for row in rows:
        if len(row) % 6 != 0:
            continue
        k = len(row) // 6
        if k not in (2, 3):
            continue
        for group in range(6):
            base = group * k
            if k == 3:
                sec_text = row[base][1]
                if sec_text.isdigit():
                    second[group] = int(sec_text)
                third_text, odds_text = row[base + 1][1], row[base + 2][1]
            else:
                third_text, odds_text = row[base][1], row[base + 1][1]
            sec = second.get(group)
            if sec is not None and third_text.isdigit():
                yield group, sec, int(third_text), odds_text


def parse_odds3t(html: str) -> dict[tuple[int, int, int], float]:
    """3連単オッズページ -> {(1着,2着,3着): オッズ}。グループ=1着艇(1..6)"""
    out = {}
    for group, sec, third, odds_text in _walk_groups(_parse_rows(html)):
        try:
            out[(group + 1, sec, third)] = float(odds_text)
        except ValueError:
            pass  # 欠場等
    return out


def parse_odds3f(html: str) -> dict[tuple[int, int, int], float]:
    """3連複オッズページ -> {(a,b,c)昇順: オッズ}。グループ=最小艇番(1..4)"""
    out = {}
    for group, sec, third, odds_text in _walk_groups(_parse_rows(html)):
        try:
            out[tuple(sorted((group + 1, sec, third)))] = float(odds_text)
        except ValueError:
            pass
    return out


def fetch_odds(venue_code: int, race_no: int, d: date) -> dict[str, dict]:
    """{'3連単': {(a,b,c): odds}, '3連複': {(a,b,c): odds}} を返す"""
    ymd = d.strftime("%Y%m%d")
    tri = parse_odds3t(_fetch(_URL_3T.format(rno=race_no, jcd=venue_code, ymd=ymd)))
    trio = parse_odds3f(_fetch(_URL_3F.format(rno=race_no, jcd=venue_code, ymd=ymd)))
    return {"3連単": tri, "3連複": trio}
