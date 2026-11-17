import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from trend_system.broker import BacktestResult
from trend_system.config import ResearchConfig


DATES = ["2024-01-02", "2024-01-03", "2024-01-04"]
CODE = "600000.SH"


class RiskReportTest(unittest.TestCase):
    def test_entry_risk_gap_minimum_commission_and_same_day_exit_are_quantified(self):
        from trend_system.risk import build_risk_report

        panel = self._panel()
        panel.loc[panel["date"] == DATES[1], ["close", "prev_close"]] = [9.5, 10.0]
        panel.loc[panel["date"] == DATES[2], ["open", "high", "low", "close", "prev_close"]] = [9.0, 9.5, 8.9, 9.0, 9.5]
        signals = panel[["date", "code"]].copy()
        signals["channel_low_120"] = 9.0
        trades = pd.DataFrame(
            [
                {
                    "order_id": 1,
                    "signal_date": DATES[0],
                    "date": DATES[1],
                    "code": CODE,
                    "side": "buy",
                    "quantity": 400,
                    "price": 10.0,
                    "amount": 4000.0,
                    "commission": 5.0,
                    "stamp_duty": 0.0,
                    "transfer_fee": 0.04,
                    "costs": 5.04,
                    "cash_delta": -4005.04,
                },
                {
                    "order_id": 2,
                    "signal_date": DATES[1],
                    "date": DATES[2],
                    "code": CODE,
                    "side": "sell",
                    "quantity": 400,
                    "price": 9.0,
                    "amount": 3600.0,
                    "commission": 5.0,
                    "stamp_duty": 3.6,
                    "transfer_fee": 0.036,
                    "costs": 8.636,
                    "cash_delta": 3591.364,
                },
            ]
        )
        round_trips = pd.DataFrame(
            [{
                "code": CODE,
                "entry_date": DATES[1],
                "exit_date": DATES[2],
                "entry_amount": 4000.0,
                "exit_amount": 3600.0,
                "net_pnl": -413.676,
            }]
        )
        orders = pd.DataFrame(
            [
                {"order_id": 1, "signal_date": DATES[0], "date": DATES[1], "code": CODE, "side": "buy", "quantity": 400, "status": "filled", "reason": ""},
                {"order_id": 2, "signal_date": DATES[1], "date": DATES[2], "code": CODE, "side": "sell", "quantity": 400, "status": "filled", "reason": "exit_signal"},
            ]
        )
        backtest = BacktestResult(
            trades=trades,
            orders=orders,
            round_trips=round_trips,
            daily_account=pd.DataFrame({
                "date": DATES,
                "cash": [10000.0, 5994.96, 9586.324],
                "receivables": [0.0] * 3,
                "market_value": [0.0, 4000.0, 0.0],
                "equity": [10000.0, 9994.96, 9586.324],
            }),
            daily_positions=pd.DataFrame({
                "date": [DATES[1]],
                "code": [CODE],
                "entry_date": [DATES[1]],
                "quantity": [400],
                "close": [9.5],
                "market_value": [3800.0],
                "stale_valuation": [False],
            }),
            action_log=pd.DataFrame(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            risk = build_risk_report(panel, signals, backtest, self._config(), Path(temporary_directory))

        entry = risk["tables"]["entry_risk"][0]
        self.assertEqual(entry["initial_exit_20d"], 9.0)
        self.assertEqual(entry["initial_risk_amount"], 400.0)
        self.assertEqual(entry["risk_account_share"], 0.04)
        self.assertEqual(risk["exit_execution"]["overnight_gap_extra_loss"], 200.0)
        self.assertEqual(risk["minimum_commission"]["minimum_applied_trades"], 2)
        self.assertAlmostEqual(risk["minimum_commission"]["max_commission_to_amount"], 5.0 / 3600.0)
        self.assertEqual(risk["same_day_exits"]["max_exits"], 1)

    def test_holding_returns_reset_at_each_entry_episode(self):
        from trend_system.risk import _holding_correlations

        dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        positions = pd.DataFrame({
            "date": [dates[0], dates[1], dates[3], dates[4]],
            "code": [CODE] * 4,
            "entry_date": [dates[0], dates[0], dates[3], dates[3]],
        })
        panel = pd.DataFrame({
            "date": dates,
            "code": [CODE] * 5,
            "signal_close": [10.0, 11.0, 50.0, 30.0, 33.0],
        })
        original_pivot = pd.DataFrame.pivot
        captured = {}

        def capture_returns(frame, *args, **kwargs):
            if "stock_return" in frame:
                captured["held"] = frame.copy()
            return original_pivot(frame, *args, **kwargs)

        with patch.object(pd.DataFrame, "pivot", capture_returns):
            _holding_correlations(positions, panel)

        returns = captured["held"].set_index(["date", "code"])["stock_return"]
        self.assertAlmostEqual(returns.loc[(dates[1], CODE)], 0.1)
        self.assertTrue(pd.isna(returns.loc[(dates[3], CODE)]))
        self.assertAlmostEqual(returns.loc[(dates[4], CODE)], 0.1)

    def test_stale_valuation_discount_is_accounting_only_and_reports_equity_mdd(self):
        from trend_system.risk import build_risk_report

        panel = self._panel()
        signals = panel[["date", "code"]].copy()
        signals["channel_low_120"] = 9.0
        backtest = BacktestResult(
            trades=pd.DataFrame(),
            orders=pd.DataFrame(),
            round_trips=pd.DataFrame(),
            daily_account=pd.DataFrame({
                "date": DATES,
                "cash": [6000.0, 6000.0, 5000.0],
                "receivables": [0.0] * 3,
                "market_value": [4000.0] * 3,
                "equity": [10000.0, 10000.0, 9000.0],
            }),
            daily_positions=pd.DataFrame({
                "date": DATES,
                "code": [CODE] * 3,
                "entry_date": [DATES[0]] * 3,
                "quantity": [400] * 3,
                "close": [10.0] * 3,
                "market_value": [4000.0] * 3,
                "stale_valuation": [False, True, True],
            }),
            action_log=pd.DataFrame(),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            risk = build_risk_report(panel, signals, backtest, self._config(), output)
            self.assertTrue((output / "risk_report.json").is_file())
            self.assertTrue((output / "risk_report.md").is_file())
            self.assertFalse((output / "orders.parquet").exists())

        ten = risk["stale_valuation_stress"]["10pct"]
        twenty = risk["stale_valuation_stress"]["20pct"]
        self.assertEqual(ten["final_equity"], 8600.0)
        self.assertAlmostEqual(ten["max_drawdown"], 0.14)
        self.assertEqual(twenty["final_equity"], 8200.0)
        self.assertAlmostEqual(twenty["max_drawdown"], 0.18)
        self.assertTrue(risk["diagnostic_only"])
        self.assertFalse(risk["orders_generated"])
        self.assertFalse(risk["included_in_mechanical_gates"])

    @staticmethod
    def _panel():
        return pd.DataFrame({
            "date": DATES,
            "code": [CODE] * 3,
            "open": [10.0] * 3,
            "high": [10.5] * 3,
            "low": [9.5] * 3,
            "close": [10.0] * 3,
            "prev_close": [10.0] * 3,
            "signal_close": [10.0] * 3,
            "is_st": [0] * 3,
        })

    @staticmethod
    def _config():
        return ResearchConfig(
            hithink_db_path="unused",
            baostock_db_path="unused",
            start_date=DATES[0],
            end_date=DATES[-1],
        )


if __name__ == "__main__":
    unittest.main()
