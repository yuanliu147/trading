import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trend_system.config import ResearchConfig
from trend_system.metrics import run_baseline


DESIGN_DOCUMENTS = [
    "00_设计_主板趋势波段交易系统研究.md",
    "01_实施计划_主板趋势波段交易系统.md",
    "02_预注册_极简趋势消融.md",
]


class BaselineIntegrationTest(unittest.TestCase):
    def test_synthetic_baseline_is_reproducible_and_executes_after_signal_at_open(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            panel, config = self._write_inputs(root)
            with patch("trend_system.metrics.PROJECT_ROOT", root):
                first = run_baseline(config, config_path=root / "config" / "baseline.json")
                first_hashes = self._output_hashes(first.output_dir)
                second = run_baseline(config, config_path=root / "config" / "baseline.json")
                second_hashes = self._output_hashes(second.output_dir)

            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(first.summary, second.summary)
            self.assertEqual(first.manifest, second.manifest)
            pd.testing.assert_frame_equal(first.backtest.daily_account, second.backtest.daily_account)
            pd.testing.assert_frame_equal(first.backtest.trades, second.backtest.trades)
            pd.testing.assert_frame_equal(first.annual_rolling, second.annual_rolling)
            pd.testing.assert_frame_equal(first.benchmarks, second.benchmarks)

            trades = first.backtest.trades
            self.assertEqual(trades["side"].tolist(), ["buy", "sell"])
            self.assertTrue((trades["date"] > trades["signal_date"]).all())
            executions = trades.merge(
                panel[["date", "code", "open", "close"]],
                on=["date", "code"],
                how="left",
                validate="one_to_one",
            )
            for trade in executions.itertuples(index=False):
                expected_price = trade.open * (1.001 if trade.side == "buy" else 0.999)
                self.assertAlmostEqual(trade.price, expected_price)
                self.assertNotAlmostEqual(trade.price, trade.close)

            self.assertEqual(
                set(first.benchmarks["benchmark"]),
                {"现金", "上证综指", "沪深300", "主板等权"},
            )
            self.assertFalse(first.benchmark_metadata["上证综指"]["dividends_included"])
            self.assertFalse(first.benchmark_metadata["沪深300"]["dividends_included"])
            self.assertIn("不可由2万元账户直接复制", first.benchmark_metadata["主板等权"]["note"])
            self.assertEqual(first.manifest["inputs"]["main_panel"]["rows"], len(panel))
            self.assertGreater(first.manifest["inputs"]["backtest_panel"]["rows"], 0)
            self.assertEqual(first.manifest["config"]["values"], asdict(config))
            self.assertEqual(first.manifest["config"]["source"], "file")
            self.assertEqual(
                first.manifest["config"]["path"],
                str((root / "config" / "baseline.json").resolve()),
            )
            saved_summary = json.loads((first.output_dir / "performance_summary.json").read_text())
            self.assertIn("closed_round_trip_turnover", saved_summary)
            self.assertIn("realized_annual_contribution", saved_summary)
            self.assertIn("realized_stock_contribution", saved_summary)
            self.assertNotIn("turnover", saved_summary)
            markdown = (first.output_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("不含未闭合交易、未实现盈亏和分红", markdown)
            self.assertIn("不与账户年度权益盈亏直接相等", markdown)
            self.assertEqual(len(first.manifest["design_documents"]), 3)
            self.assertTrue(first.manifest["code_files"])
            expected_outputs = {
                "daily_account.parquet",
                "daily_positions.parquet",
                "trades.parquet",
                "orders.parquet",
                "round_trips.parquet",
                "action_log.parquet",
                "annual_and_rolling.csv",
                "benchmarks.csv",
                "performance_summary.json",
                "run_manifest.json",
                "summary.md",
            }
            self.assertEqual(set(first_hashes), expected_outputs)

    def test_direct_run_without_config_path_hashes_canonical_values_not_default_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _, config = self._write_inputs(root)
            (root / "config" / "baseline.json").unlink()
            with patch("trend_system.metrics.PROJECT_ROOT", root):
                result = run_baseline(config)

        canonical = json.dumps(
            asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(result.manifest["config"]["source"], "canonical_config_values")
        self.assertIsNone(result.manifest["config"]["path"])
        self.assertEqual(result.manifest["config"]["sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(result.manifest["config"]["values"], asdict(config))

    @staticmethod
    def _write_inputs(root):
        data_directory = root / "data"
        config_directory = root / "config"
        data_directory.mkdir(parents=True)
        config_directory.mkdir()
        dates = pd.bdate_range("2023-01-02", periods=236).astype(str).tolist()
        rows = []
        for index, day in enumerate(dates):
            bar = {
                "date": day,
                "code": "600000.SH",
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": 10.0,
                "prev_close": 10.0,
                "volume": 1000000.0,
                "amount": 10000000.0,
                "trade_status": 1,
                "is_st": 0,
                "ipo_date": "2020-01-01",
                "out_date": pd.NA,
                "data_valid": True,
                "execution_conflict": False,
                "adjustment_source_conflict": False,
                "source_conflict": False,
            }
            if index == 230:
                bar.update(open=11.0, high=11.2, low=10.8, close=11.0, prev_close=10.0)
            elif index == 231:
                bar.update(open=11.2, high=14.2, low=11.0, close=14.0, prev_close=11.0)
            elif index == 232:
                bar.update(open=13.0, high=13.2, low=9.3, close=9.4, prev_close=14.0)
            elif index >= 233:
                bar.update(open=9.4, high=9.8, low=9.3, close=9.5, prev_close=9.4)
            for field in ("open", "high", "low", "close"):
                bar[f"signal_{field}"] = bar[field]
            rows.append(bar)
        panel = pd.DataFrame(rows)
        panel.to_parquet(data_directory / "main_panel.parquet", index=False)

        index_rows = []
        for index, day in enumerate(dates):
            for code, base in (("000001.SH", 3000.0), ("000300.SH", 4000.0)):
                index_rows.append(
                    {
                        "date": day,
                        "code": code,
                        "open": base + index,
                        "high": base + index + 10,
                        "low": base + index - 10,
                        "close": base + index,
                        "prev_close": base + index - 1,
                        "volume": 1.0,
                        "amount": 1.0,
                    }
                )
        pd.DataFrame(index_rows).to_parquet(data_directory / "index_panel.parquet", index=False)
        pd.DataFrame(
            columns=[
                "date",
                "code",
                "dividend_per_share",
                "per_share_bonus",
                "rights_ratio",
                "rights_price",
            ]
        ).to_parquet(data_directory / "corporate_actions.parquet", index=False)

        config = ResearchConfig(
            hithink_db_path="unused",
            baostock_db_path="unused",
            start_date=dates[230],
            end_date=dates[235],
        )
        (config_directory / "baseline.json").write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for name in DESIGN_DOCUMENTS:
            (root / name).write_text(f"synthetic copy of {name}\n", encoding="utf-8")
        return panel, config

    @staticmethod
    def _output_hashes(output_directory):
        hashes = {}
        for path in sorted(output_directory.iterdir()):
            if path.is_file():
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes


if __name__ == "__main__":
    unittest.main()
