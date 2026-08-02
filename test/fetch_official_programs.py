# -*- coding: utf-8 -*-
"""公式サイトから番組表を直接取得する手動CLI(本体は src/official_programs.py)

    py -X utf8 test/fetch_official_programs.py [YYYY-MM-DD]

通常は predict._ensure_program が上流404時に自動発動するため手動実行は不要。
デバッグ・過去日の取り直し用に残している。
実行後は: cd src && py -X utf8 collect.py YYYY-MM-DD YYYY-MM-DD && py -X utf8 predict.py today
(当日をcollectする場合は日付明示で。引数なしのcollectは当日をforce再取得するため
手作りキャッシュを素通りする)
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import jst_today
from official_programs import fetch_official_programs

if __name__ == "__main__":
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else jst_today()
    fetch_official_programs(d)
