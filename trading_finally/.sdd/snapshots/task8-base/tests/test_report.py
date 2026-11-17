import tempfile
import unittest
from pathlib import Path

from trend_system.report import render_final_report


class ReportTest(unittest.TestCase):
    def test_final_report_discloses_gates_negative_results_and_descriptive_limits(self):
        summary = {
            "config_sha256": "abc123",
            "config": {"start_date": "2016-10-13", "end_date": "2026-10-29"},
            "baseline": {
                "start_date": "2016-10-13",
                "end_date": "2026-10-29",
                "total_return": -0.5,
                "max_drawdown": 0.6,
                "final_equity": 10000.0,
            },
            "max_drawdown": 0.6,
            "gates": [
                {
                    "id": "main_window_positive",
                    "passed": False,
                    "actual": -0.5,
                    "requirement": "> 0",
                },
                {
                    "id": "drawdown_boundary",
                    "passed": False,
                    "actual": 0.6,
                    "requirement": "<= 0.10; <= 0.08 required for the strongest label",
                },
            ],
            "verdict": "REJECTED",
            "variants": [
                {
                    "family": "neighborhood",
                    "name": "entry_lookback_90",
                    "status": "completed",
                    "total_return": -0.4,
                    "max_drawdown": 0.5,
                    "output_dir": "results/robustness/neighborhood/entry_lookback_90",
                }
            ],
            "annual_equity": [
                {"year": "2019", "equity_pnl": 100.0, "is_top_contributor": True},
                {"year": "2020", "equity_pnl": -200.0, "is_top_contributor": False},
            ],
            "profit_concentration": {"top_year": "2019", "remaining_equity_pnl": -200.0},
            "common_window": {
                "start_date": "2017-08-01",
                "end_date": "2026-10-29",
                "warmup_source_start": "2016-10-13",
                "baostock_total_return": -0.4,
                "baostock_max_drawdown": 0.5,
                "hithink_total_return": -0.3,
                "hithink_max_drawdown": 0.4,
                "entry_signal_differences": 12,
                "exit_signal_differences": 20,
            },
            "bootstrap": {
                "iterations": 1000,
                "block_size": 20,
                "total_return_quantiles": {"p02_5": -0.8, "p50": -0.5, "p97_5": -0.1},
                "exploratory_only": True,
            },
            "descriptive": {
                "formulas": {
                    "post_hoc_only": "事后描述，不参与门槛，不形成交易闸门。",
                    "market_regime": "沪深300：200日均线与20日斜率。",
                    "size_style": "中证1000与沪深300的60日收益差。",
                },
                "groups": [
                    {"group_type": "market_regime", "group": "down", "days": 10, "compound_return": -0.1}
                ],
            },
            "stress": {
                "status": "completed",
                "source_mode": "baostock_single_source",
                "start_date": "2010-01-04",
                "end_date": "2015-12-31",
                "total_return": 0.2,
                "max_drawdown": 0.3,
            },
            "outputs": {"robustness": "results/robustness", "stress": "results/stress_2010_2015"},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = render_final_report(summary, Path(temporary_directory) / "90_研究结论.md")
            text = path.read_text(encoding="utf-8")
        self.assertIn("REJECTED", text)
        self.assertIn("main_window_positive", text)
        self.assertIn("-0.500000", text)
        self.assertIn("账户年度权益盈亏", text)
        self.assertIn("共同窗口不能视为完整主窗口", text)
        self.assertIn("不参与门槛", text)
        self.assertIn("单源", text)
        self.assertIn("entry_lookback_90", text)


if __name__ == "__main__":
    unittest.main()
