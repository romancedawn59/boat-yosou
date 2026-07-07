"""場コード・パス・URL設定"""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_DIR / "data_raw"
DB_PATH = PROJECT_DIR / "boat.db"
MODEL_PATH = PROJECT_DIR / "models" / "lgbm_win.txt"

# 予測対象場(分析用データ・学習は全24場で行い、当日予測・買い目提示はこの5場のみ)。
# 場別ウォークフォワード検証(2026-07)で「最大1発を除いても回収率100%超・
# 5期間中3期間以上黒字・荒れ注意の出現率が十分」を満たした場。
# 戸田(2)は回収率79.5%・黒字1/5期間で対象から除外した。
TARGET_VENUE_CODES = [3, 4, 8, 13, 20]  # 江戸川・平和島・常滑・尼崎・若松

# 予測対象場の座標(Open-Meteoの気象予報取得用。数kmの誤差は許容)
VENUE_COORDS = {
    3: (35.688, 139.883),   # 江戸川
    4: (35.578, 139.735),   # 平和島
    8: (34.887, 136.832),   # 常滑
    13: (34.717, 135.427),  # 尼崎
    20: (33.906, 130.813),  # 若松
}

# GitHub Pagesの公開先(LINE通知の本文にレポートURLを載せるため)
PAGES_URL = "https://romancedawn59.github.io/boat-yosou"

VENUE_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

# データ源: BoatraceOpenAPI (非公式・MITライセンス・GitHub Pages配信)
# 旧公式オープンデータ(www1.mbrace.or.jp/od2)は2025-03-05でサービス終了。
# programs(番組表)v3は2023-05-01以降の全アーカイブあり。
# results(競走成績)v2は約1年分のローリング保持のみ。
BASE_URL = "https://boatraceopenapi.github.io"
PROGRAMS_URL_TMPL = BASE_URL + "/programs/v3/{yyyy}/{yyyymmdd}.json"
RESULTS_URL_TMPL = BASE_URL + "/results/v2/{yyyy}/{yyyymmdd}.json"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

REQUEST_INTERVAL_SEC = 1.0  # サーバー負荷配慮のための待機

# collect.pyを引数なしで実行した際、DBが空なら今日からこの日数分だけ遡って収集する。
# resultsの保持期間(約1年)に合わせている。
DEFAULT_LOOKBACK_DAYS = 350
