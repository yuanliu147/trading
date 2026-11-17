import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from trend_system.legacy_config import load_legacy_config
from trend_system.legacy_runner import (
    prepare_legacy_frames,
    run_legacy_frames,
    validate_legacy_status,
)


CODE_A = "600000.SH"
CODE_B = "000001.SZ"


class LegacyRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).resolve().parents[1] / "config" / "legacy_candidates.json"
        cls.frozen_config = load_legacy_config(config_path)

    def test_prepare_adds_factor_selects_window_and_rejects_key_drift(self):
        panel = self._panel(self._dates(4), [CODE_A])
        signals = self._signals(panel)

        prepared_panel, prepared_signals = prepare_legacy_frames(
            panel, signals, replace(self.frozen_config, start_date="2024-01-03", end_date="2024-01-05")
        )

        self.assertEqual(prepared_panel["date"].min(), "2024-01-03")
        self.assertEqual(prepared_panel["date"].max(), "2024-01-05")
        self.assertAlmostEqual(prepared_panel["adjustment_factor"].iloc[0], 1.0)
        self.assertEqual(
            set(zip(prepared_panel["date"], prepared_panel["code"])),
            set(zip(prepared_signals["date"], prepared_signals["code"])),
        )

        drifted = prepared_signals.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            prepare_legacy_frames(prepared_panel, drifted, self.frozen_config)

    def test_repeated_runs_are_identical_and_never_trade_on_signal_day(self):
        panel = self._panel(self._dates(4), [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, "2024-01-02", CODE_A, entry=True)

        outputs = []
        for _ in range(2):
            result = run_legacy_frames(
                panel,
                signals,
                self._empty_actions(),
                replace(
                    self.frozen_config,
                    start_date="2024-01-02",
                    end_date="2024-01-05",
                    initial_cash=20000,
                ),
            )
            outputs.append(
                (
                    result.daily_account.to_csv(index=False),
                    result.trades.to_csv(index=False),
                    result.orders.to_csv(index=False),
                    result.round_trips.to_csv(index=False),
                )
            )
            self.assertTrue((result.trades["date"] > result.trades["signal_date"]).all())

        self.assertEqual(outputs[0], outputs[1])

    def test_not_evaluable_status_aborts_before_engine(self):
        with self.assertRaises(ValueError):
            validate_legacy_status({"candidate_a_status": "NOT_EVALUABLE"}, "three_of_two_low_vol")
        with self.assertRaises(ValueError):
            validate_legacy_status({"candidate_a_status": "EVALUABLE"}, "early_trend_breadth")
        validate_legacy_status(
            {"candidate_a_status": "EVALUABLE", "account_backtest_allowed": True},
            "three_of_two_low_vol",
        )

    @staticmethod
    def _dates(count):
        return pd.bdate_range("2024-01-02", periods=count).astype(str).tolist()

    @staticmethod
    def _panel(dates, codes):
        rows = []
        for day in dates:
            for code in codes:
                rows.append({
                    "date": day,
                    "code": code,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "prev_close": 10.0,
                    "signal_open": 10.0,
                    "signal_high": 10.5,
                    "signal_low": 9.5,
                    "signal_close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "trade_status": 1,
                    "is_st": 0,
                    "out_date": pd.NA,
                    "data_valid": True,
                    "execution_conflict": False,
                    "adjustment_source_conflict": False,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _signals(panel):
        signals = panel[["date", "code", "signal_close"]].copy()
        signals["entry_signal"] = False
        signals["legacy_rank_score"] = 0.0
        signals["rank_return_120"] = 0.0
        signals["stop_ref"] = 9.0
        signals["ma5"] = 2.0
        signals["ma20"] = 1.0
        signals["whist"] = 1.0
        return signals

    @staticmethod
    def _signal(signals, day, code, entry=None):
        selected = (signals["date"] == day) & (signals["code"] == code)
        if entry is not None:
            signals.loc[selected, "entry_signal"] = entry

    @staticmethod
    def _empty_actions():
        return pd.DataFrame(
            {
                "date": pd.Series(dtype=str),
                "code": pd.Series(dtype=str),
                "dividend_per_share": pd.Series(dtype=float),
                "per_share_bonus": pd.Series(dtype=float),
                "rights_ratio": pd.Series(dtype=float),
                "rights_price": pd.Series(dtype=float),
            }
        )


if __name__ == "__main__":
    unittest.main()
