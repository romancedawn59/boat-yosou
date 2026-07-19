"""場コード・パス・URL設定"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

# クラウド(GitHub Actions)のランナーはUTCで動くため、「今日」は必ずJSTで判定する
JST = timezone(timedelta(hours=9))


def jst_today():
    return datetime.now(JST).date()

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_DIR / "data_raw"
DB_PATH = PROJECT_DIR / "boat.db"
MODEL_PATH = PROJECT_DIR / "models" / "lgbm_win.txt"

# 本命勝負所の対象場(場別ウォークフォワード検証(2026-07)で「最大1発を除いても
# 回収率100%超・5期間中3期間以上黒字・荒れ注意の出現率が十分」を満たした場。
# 戸田(2)は回収率79.5%・黒字1/5期間で対象から除外した)。
TARGET_VENUE_CODES = [3, 4, 8, 13, 20]  # 江戸川・平和島・常滑・尼崎・若松

# === v2(ケンさん案・2026-07-18ユーザー決定/検証はtest/verify_ken_v2*.py) ===
# 予測は全24場で行い、購入対象は
#   本命   = 5場×1位勝率35%未満のうち1位勝率が低い順に最大HONMEI_CAPレース
#   超混戦 = 全場×1位勝率がKONSEN_PROB_MAX未満(市場も予測できない本物の混戦。
#            エッジの本体。walk-forwardで回収率387%/最大1発除き312%)
# の和集合。旧・準勝負所は「要注目」(観測専用・購入なし)に再定義する。
KONSEN_PROB_MAX = 0.20  # 超混戦の閾値(帯検証: 20〜25%帯は他19場で回収率92%のため20%固定)
HONMEI_CAP = 6          # 本命の1日上限(検証: cap6はcap10より回収率・DDとも改善)
ATTENTION_CAP = 4       # 要注目(観測枠)の1日上限(本命6+要注目4=従来の表示規模10を維持)

# 舟券が買えない時間帯(公式メンテナンス等)。締切がこの窓[start, end)にかかるレースは
# 購入対象(本命/超混戦)から外して要注目(観測・購入0点)へ回す。これにより実購入・採点・
# 税集計・成績まとめから自動的に除外される(「買えなかったレース」を成績に混ぜない)。
# 文字列は "YYYY-MM-DD HH:MM:SS"(races.deadline_timeと同形式・辞書順比較で判定)。
PURCHASE_BLACKOUTS = [
    # 2026-07-19 21:00〜7/20 12:20 システムメンテナンス。7/20は12:30以降に締切の
    # あるレース(=12:30以降に買えるレース)だけを購入対象にする(ユーザー指定)。
    ("2026-07-19 21:00:00", "2026-07-20 12:30:00"),
]


def is_buyable(deadline_time) -> bool:
    """締切がメンテナンス等の購入不可窓に入っていないか。締切不明は安全側でFalse"""
    if not deadline_time:
        return False
    return not any(start <= deadline_time < end for start, end in PURCHASE_BLACKOUTS)

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
