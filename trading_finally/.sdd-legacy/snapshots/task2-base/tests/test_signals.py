import unittest

import numpy as np
import pandas as pd

from trend_system.config import ResearchConfig
from trend_system.signals import compute_signals


SIGNAL_COLUMNS = [
    "ma200",
    "channel_high_120",
    "channel_low_120",
    "rank_return_120",
    "entry_signal",
    "exit_signal",
    "st_exit",
]


class SignalsTest(unittest.TestCase):
    def test_channels_use_only_prior_rows_and_windows_are_complete(self):
        panel = self._panel()
        dates = panel["date"]

        result = compute_signals(panel, self._config(dates.iloc[0], dates.iloc[-1]))

        self.assertTrue(pd.isna(result.loc[119, "channel_high_120"]))
        self.assertEqual(result.loc[120, "channel_high_120"], panel.loc[119, "signal_high"])
        self.assertEqual(result.loc[239, "channel_high_120"], panel.loc[238, "signal_high"])
        self.assertTrue(pd.isna(result.loc[19, "channel_low_120"]))
        self.assertEqual(result.loc[20, "channel_low_120"], panel.loc[0, "signal_low"])
        self.assertEqual(result.loc[239, "channel_low_120"], panel.loc[219, "signal_low"])
        self.assertTrue(pd.isna(result.loc[198, "ma200"]))
        self.assertAlmostEqual(result.loc[199, "ma200"], panel.loc[:199, "signal_close"].mean())
        self.assertTrue(pd.isna(result.loc[119, "rank_return_120"]))
        self.assertAlmostEqual(
            result.loc[120, "rank_return_120"],
            panel.loc[120, "signal_close"] / panel.loc[0, "signal_close"] - 1,
        )
        self.assertFalse(result.loc[:218, "entry_signal"].any())
        self.assertTrue(result.loc[219, "entry_signal"])

    def test_future_ex_right_rows_do_not_change_prior_signals(self):
        panel = self._panel(260)
        cutoff = 230
        full = panel.copy()
        full.loc[cutoff:, ["signal_open", "signal_high", "signal_low", "signal_close"]] *= 0.5
        full.loc[cutoff:, "is_st"] = 1
        config = self._config(panel.loc[0, "date"], panel.loc[259, "date"])

        truncated_result = compute_signals(full.loc[: cutoff - 1].copy(), config)
        full_result = compute_signals(full, config)

        pd.testing.assert_frame_equal(
            full_result.loc[: cutoff - 1, SIGNAL_COLUMNS].reset_index(drop=True),
            truncated_result[SIGNAL_COLUMNS].reset_index(drop=True),
            check_exact=True,
        )

    def test_current_record_does_not_rewrite_yesterday(self):
        panel = self._panel()
        config = self._config(panel.loc[0, "date"], panel.loc[239, "date"])
        baseline = compute_signals(panel, config)
        changed = panel.copy()
        changed.loc[239, ["signal_high", "signal_close"]] = [1000.0, 999.0]

        actual = compute_signals(changed, config)

        pd.testing.assert_frame_equal(
            baseline.loc[:238, SIGNAL_COLUMNS], actual.loc[:238, SIGNAL_COLUMNS], check_exact=True
        )
        self.assertNotEqual(baseline.loc[239, "rank_return_120"], actual.loc[239, "rank_return_120"])

    def test_st_is_not_backfilled_to_history(self):
        panel = self._panel()
        panel.loc[150:, "is_st"] = 1
        config = self._config(panel.loc[0, "date"], panel.loc[239, "date"])

        result = compute_signals(panel, config)

        self.assertFalse(result.loc[:149, "st_exit"].any())
        self.assertTrue(result.loc[150:, "st_exit"].all())
        self.assertFalse(result.loc[150:, "entry_signal"].any())

    def test_entry_gates_and_adjustment_conflict_rule(self):
        panel = self._panel()
        config = self._config(panel.loc[0, "date"], panel.loc[239, "date"])
        row = 230
        for column, value in (
            ("is_st", 1),
            ("trade_status", 0),
            ("data_valid", False),
            ("execution_conflict", True),
        ):
            with self.subTest(column=column):
                changed = panel.copy()
                changed.loc[row, column] = value
                result = compute_signals(changed, config)
                self.assertFalse(result.loc[row, "entry_signal"])

        changed = panel.copy()
        changed.loc[row, "adjustment_source_conflict"] = True
        changed.loc[row, "source_conflict"] = True
        result = compute_signals(changed, config)
        self.assertTrue(result.loc[row, "entry_signal"])
        self.assertTrue(result.loc[row, "adjustment_source_conflict"])

    def test_price_ma_slope_and_channel_each_have_independent_negative_cases(self):
        config = self._config(self._panel().loc[0, "date"], self._panel().loc[239, "date"])
        row = 230

        price = self._panel()
        price.loc[:109, ["signal_open", "signal_high", "signal_low", "signal_close"]] = 1000.0
        price_result = compute_signals(price, config)
        self.assertLessEqual(price_result.loc[row, "signal_close"], price_result.loc[row, "ma200"])
        self.assertGreater(price_result.loc[row, "signal_close"], price_result.loc[row, "channel_high_120"])
        self.assertFalse(price_result.loc[row, "entry_signal"])

        slope = self._panel()
        slope.loc[191:210, ["signal_open", "signal_high", "signal_low", "signal_close"]] = 100.0
        slope.loc[211:229, ["signal_open", "signal_high", "signal_low", "signal_close"]] = 1.0
        slope.loc[row, ["signal_open", "signal_high", "signal_low", "signal_close"]] = 102.0
        slope_result = compute_signals(slope, config)
        self.assertGreater(slope_result.loc[row, "signal_close"], slope_result.loc[row, "ma200"])
        self.assertLessEqual(slope_result.loc[row, "ma200"], slope_result.loc[row - 20, "ma200"])
        self.assertGreater(slope_result.loc[row, "signal_close"], slope_result.loc[row, "channel_high_120"])
        self.assertFalse(slope_result.loc[row, "entry_signal"])

        channel = self._panel()
        channel.loc[row - 1, "signal_high"] = 1000.0
        channel_result = compute_signals(channel, config)
        self.assertGreater(channel_result.loc[row, "signal_close"], channel_result.loc[row, "ma200"])
        self.assertGreater(channel_result.loc[row, "ma200"], channel_result.loc[row - 20, "ma200"])
        self.assertLessEqual(channel_result.loc[row, "signal_close"], channel_result.loc[row, "channel_high_120"])
        self.assertFalse(channel_result.loc[row, "entry_signal"])

    def test_missing_entry_state_fields_fail_closed(self):
        panel = self._panel()
        config = self._config(panel.loc[0, "date"], panel.loc[239, "date"])
        for column in ("is_st", "trade_status", "data_valid", "execution_conflict"):
            with self.subTest(column=column):
                changed = panel.copy()
                changed[column] = changed[column].astype(object)
                changed.loc[230, column] = np.nan
                result = compute_signals(changed, config)
                self.assertFalse(result.loc[230, "entry_signal"])

    def test_entry_is_limited_to_main_window_and_exit_is_separate(self):
        panel = self._panel()
        config = self._config(panel.loc[230, "date"], panel.loc[235, "date"])

        result = compute_signals(panel, config)

        self.assertFalse(result.loc[:229, "entry_signal"].any())
        self.assertTrue(result.loc[230:235, "entry_signal"].all())
        self.assertFalse(result.loc[236:, "entry_signal"].any())

        changed = panel.copy()
        changed.loc[230, "signal_close"] = 0.1
        changed.loc[231, "is_st"] = 1
        result = compute_signals(changed, config)
        self.assertTrue(result.loc[230, "exit_signal"])
        self.assertFalse(result.loc[230, "st_exit"])
        self.assertTrue(result.loc[231, "st_exit"])

    def test_stocks_are_sorted_and_calculated_independently(self):
        first = self._panel()
        second = self._panel()
        second["code"] = "000001.SZ"
        second[["signal_open", "signal_high", "signal_low", "signal_close"]] *= 3
        panel = pd.concat([first, second], ignore_index=True).sample(frac=1, random_state=7)
        config = self._config(first.loc[0, "date"], first.loc[239, "date"])

        result = compute_signals(panel, config)

        self.assertEqual(result[["date", "code"]].duplicated().sum(), 0)
        self.assertEqual(result["code"].tolist(), ["000001.SZ"] * 240 + ["600000.SH"] * 240)
        for code, scale in (("000001.SZ", 3), ("600000.SH", 1)):
            stock = result.loc[result["code"] == code].reset_index(drop=True)
            self.assertAlmostEqual(stock.loc[199, "ma200"], first.loc[:199, "signal_close"].mean() * scale)
            self.assertTrue(pd.isna(stock.loc[119, "rank_return_120"]))
            self.assertTrue(stock.loc[219, "entry_signal"])

    @staticmethod
    def _panel(rows=240):
        dates = pd.bdate_range("2020-01-01", periods=rows).strftime("%Y-%m-%d")
        close = 10.0 + np.arange(rows, dtype=float) * 0.1
        return pd.DataFrame(
            {
                "date": dates,
                "code": "600000.SH",
                "signal_open": close,
                "signal_high": close + 0.05,
                "signal_low": close - 0.05,
                "signal_close": close,
                "trade_status": 1,
                "is_st": 0,
                "data_valid": True,
                "execution_conflict": False,
                "adjustment_source_conflict": False,
                "source_conflict": False,
            }
        )

    @staticmethod
    def _config(start_date, end_date):
        return ResearchConfig(
            hithink_db_path="unused",
            baostock_db_path="unused",
            start_date=start_date,
            end_date=end_date,
        )


if __name__ == "__main__":
    unittest.main()
