"""タスクスケジューラから呼び出す各ジョブのエントリポイント

対話画面のないスケジュール実行では標準出力が見えないため、
logs/{job}.log に標準出力ごと追記して後から確認できるようにする。

    python scheduled_runner.py daily        # 全場データの収集(collect.py相当)
    python scheduled_runner.py predict      # 当日の対象5場の予測レポート出力
    python scheduled_runner.py exhibition   # 直前情報の収集(締切前のみ実データ取得)
    python scheduled_runner.py retrain      # モデル再学習+バックテスト
"""
import contextlib
import sys
import traceback
from datetime import date, datetime

from config import PROJECT_DIR

LOG_DIR = PROJECT_DIR / "logs"


def run_daily():
    import collect
    start, end, force_first = collect.auto_range()
    collect.collect_range(start, end, force_first=force_first)


def run_predict():
    import predict
    predict.run(date.today())


def run_exhibition():
    import collect_exhibition
    collect_exhibition.collect_today(date.today(), datetime.now())


def run_retrain():
    import backtest
    import train_model
    train_model.main()
    backtest.main()


JOBS = {
    "daily": run_daily,
    "predict": run_predict,
    "exhibition": run_exhibition,
    "retrain": run_retrain,
}


def run_logged(name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{name}.log"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        try:
            with contextlib.redirect_stdout(f):
                JOBS[name]()
            f.write("OK\n")
        except Exception:
            f.write("失敗\n" + traceback.format_exc() + "\n")
            raise


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in JOBS:
        print(f"usage: python scheduled_runner.py {{{'|'.join(JOBS)}}}")
        sys.exit(1)

    run_logged(sys.argv[1])
