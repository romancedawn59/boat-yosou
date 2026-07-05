import sys
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import scheduled_runner


class TestRunLogged(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        patcher = patch.object(scheduled_runner, "LOG_DIR", Path(self.tmpdir.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmpdir.cleanup)

    def test_success_logs_ok_and_captures_stdout(self):
        with patch.dict(scheduled_runner.JOBS, {"daily": lambda: print("進捗メッセージ")}):
            scheduled_runner.run_logged("daily")

        log = (Path(self.tmpdir.name) / "daily.log").read_text(encoding="utf-8")
        self.assertIn("進捗メッセージ", log)
        self.assertIn("OK", log)

    def test_failure_logs_traceback_and_reraises(self):
        def boom():
            raise RuntimeError("テスト用エラー")

        with patch.dict(scheduled_runner.JOBS, {"predict": boom}):
            with self.assertRaises(RuntimeError):
                scheduled_runner.run_logged("predict")

        log = (Path(self.tmpdir.name) / "predict.log").read_text(encoding="utf-8")
        self.assertIn("失敗", log)
        self.assertIn("RuntimeError", log)
        self.assertIn("テスト用エラー", log)

    def test_appends_across_multiple_runs(self):
        with patch.dict(scheduled_runner.JOBS, {"exhibition": lambda: print("run")}):
            scheduled_runner.run_logged("exhibition")
            scheduled_runner.run_logged("exhibition")

        log = (Path(self.tmpdir.name) / "exhibition.log").read_text(encoding="utf-8")
        self.assertEqual(log.count("run"), 2)
        self.assertEqual(log.count("OK"), 2)


class TestJobImplementations(unittest.TestCase):
    """run_predict/run_retrain が正しいモジュール関数を正しい引数で呼ぶことを確認する。

    関数内でimportしているため、sys.modulesにモックを差し込んでから呼び出す。
    """

    def test_predict_runs_for_today(self):
        mock_predict = MagicMock()
        fixed_today = date(2026, 7, 20)
        with patch.dict(sys.modules, {"predict": mock_predict}), \
             patch("scheduled_runner.date") as mock_date:
            mock_date.today.return_value = fixed_today
            scheduled_runner.run_predict()

        mock_predict.run.assert_called_once_with(fixed_today)

    def test_retrain_runs_train_and_backtest(self):
        mock_train = MagicMock()
        mock_backtest = MagicMock()
        with patch.dict(sys.modules, {"train_model": mock_train, "backtest": mock_backtest}):
            scheduled_runner.run_retrain()

        mock_train.main.assert_called_once()
        mock_backtest.main.assert_called_once()


if __name__ == "__main__":
    unittest.main()
