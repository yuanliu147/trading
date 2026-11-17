import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trend_system.data import (
    _assemble_panel,
    _compare_overlap,
    _conflict_exclusions,
    _hithink_signal_panel,
    _summarize_adjustment_diagnostics,
    effective_corporate_actions,
    load_panel,
    normalize_code,
)


class DataPanelTest(unittest.TestCase):
    def test_normalize_code(self):
        self.assertEqual(normalize_code("sh.600000"), "600000.SH")
        self.assertEqual(normalize_code("sz.000001"), "000001.SZ")
        self.assertEqual(normalize_code("600000.SH"), "600000.SH")

    def test_unconvertible_board_code_is_rejected(self):
        for code in ("bj.830799", "sh.6000000", "600000", "600000.BJ"):
            with self.subTest(code=code), self.assertRaises(ValueError):
                normalize_code(code)

    def test_missing_hfq_keeps_raw_row_without_future_backfill(self):
        raw = self._raw("2024-01-02", "2024-01-03")
        hfq = self._hfq("2024-01-03")
        hfq.loc[0, ["open", "high", "low", "close"]] = [22.0, 23.0, 21.0, 22.5]

        panel = _assemble_panel(raw, hfq, self._basic())

        self.assertEqual(len(panel), 2)
        first = panel.loc[panel["date"] == "2024-01-02"].iloc[0]
        second = panel.loc[panel["date"] == "2024-01-03"].iloc[0]
        self.assertFalse(first["data_valid"])
        self.assertTrue(pd.isna(first["signal_close"]))
        self.assertTrue(second["data_valid"])
        self.assertEqual(second["signal_close"], 22.5)

    def test_duplicate_key_is_rejected(self):
        raw = self._raw("2024-01-02")
        duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            _assemble_panel(duplicated, self._hfq("2024-01-02"), self._basic())

    def test_negative_price_is_rejected(self):
        raw = self._raw("2024-01-02")
        raw.loc[0, "open"] = -0.01
        with self.assertRaisesRegex(ValueError, "negative"):
            _assemble_panel(raw, self._hfq("2024-01-02"), self._basic())

        raw = self._raw("2024-01-02")
        hfq = self._hfq("2024-01-02")
        hfq.loc[0, "close"] = -20.0
        with self.assertRaisesRegex(ValueError, "negative"):
            _assemble_panel(raw, hfq, self._basic())

    def test_raw_price_threshold_is_strict(self):
        panel = _assemble_panel(self._raw("2024-01-02"), self._hfq("2024-01-02"), self._basic())
        hithink = self._hithink("2024-01-02")
        hithink.loc[0, "hithink_close"] = 10.511
        hithink.loc[0, "hithink_signal_close"] = 21.022

        compared, conflicts, _ = _compare_overlap(panel, hithink)
        self.assertFalse(compared.loc[0, "source_conflict"])
        self.assertTrue(conflicts.empty)

        hithink.loc[0, "hithink_close"] = 10.512
        hithink.loc[0, "hithink_signal_close"] = 21.024
        compared, conflicts, _ = _compare_overlap(panel, hithink)
        self.assertTrue(compared.loc[0, "source_conflict"])
        self.assertIn("close", conflicts.loc[0, "conflict_fields"])

    def test_volume_and_amount_relative_thresholds(self):
        panel = _assemble_panel(self._raw("2024-01-02"), self._hfq("2024-01-02"), self._basic())
        hithink = self._hithink("2024-01-02")
        hithink.loc[0, "hithink_volume"] = 1000.011
        hithink.loc[0, "hithink_amount"] = 10000.02

        compared, conflicts, _ = _compare_overlap(panel, hithink)

        self.assertTrue(compared.loc[0, "source_conflict"])
        self.assertFalse(compared.loc[0, "execution_conflict"])
        self.assertFalse(compared.loc[0, "adjustment_source_conflict"])
        self.assertIn("volume", conflicts.loc[0, "conflict_fields"])
        self.assertIn("amount", conflicts.loc[0, "conflict_fields"])

    def test_adjusted_series_are_compared_after_per_stock_normalization(self):
        raw = self._raw("2024-01-02", "2024-01-03")
        hfq = self._hfq("2024-01-02", "2024-01-03")
        panel = _assemble_panel(raw, hfq, self._basic())
        hithink = self._hithink("2024-01-02", "2024-01-03")
        hithink[["hithink_signal_open", "hithink_signal_high", "hithink_signal_low", "hithink_signal_close"]] *= 10

        compared, conflicts, _ = _compare_overlap(panel, hithink)
        self.assertFalse(compared["source_conflict"].any())
        self.assertTrue(conflicts.empty)

        hithink.loc[1, "hithink_signal_close"] *= 1.02
        compared, conflicts, _ = _compare_overlap(panel, hithink)
        self.assertFalse(compared.loc[0, "source_conflict"])
        self.assertTrue(compared.loc[1, "source_conflict"])
        self.assertIn("signal_close", conflicts.loc[0, "conflict_fields"])

    def test_execution_conflict_is_limited_to_raw_ohlc(self):
        panel = _assemble_panel(self._raw("2024-01-02"), self._hfq("2024-01-02"), self._basic())
        hithink = self._hithink("2024-01-02")
        hithink.loc[0, "hithink_open"] = 10.512

        compared, _, _ = _compare_overlap(panel, hithink)

        self.assertTrue(compared.loc[0, "source_conflict"])
        self.assertTrue(compared.loc[0, "execution_conflict"])
        self.assertFalse(compared.loc[0, "adjustment_source_conflict"])

    def test_adjustment_source_conflict_is_not_execution_conflict(self):
        panel = _assemble_panel(
            self._raw("2024-01-02", "2024-01-03"),
            self._hfq("2024-01-02", "2024-01-03"),
            self._basic(),
        )
        hithink = self._hithink("2024-01-02", "2024-01-03")
        hithink[["hithink_signal_open", "hithink_signal_high", "hithink_signal_low", "hithink_signal_close"]] *= 10
        hithink.loc[1, "hithink_signal_close"] *= 1.02

        compared, _, _ = _compare_overlap(panel, hithink)

        self.assertTrue(compared.loc[1, "source_conflict"])
        self.assertFalse(compared.loc[1, "execution_conflict"])
        self.assertTrue(compared.loc[1, "adjustment_source_conflict"])

    def test_hithink_signal_panel_preserves_all_rows(self):
        hithink = self._hithink("2024-01-02", "2024-01-03")

        result = _hithink_signal_panel(hithink)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["date"].tolist(), ["2024-01-02", "2024-01-03"])
        self.assertEqual(result["signal_close"].tolist(), [21.0, 21.0])
        self.assertEqual(result["adjustment_factor"].tolist(), [2.0, 2.0])
        self.assertTrue(result["data_valid"].all())

    def test_adjustment_diagnostics_separate_scale_precision_and_material(self):
        diagnostics = pd.DataFrame(
            {
                "normalized_close_relative_diff": [0.0, 0.0005, 0.001217, 0.02],
                "absolute_factor_scale_relative_diff": [0.0, 2.0, 2.0, 2.0],
                "adjustment_source_conflict": [False, True, True, True],
            }
        )

        result = _summarize_adjustment_diagnostics(diagnostics)

        self.assertEqual(result["rows"], 4)
        self.assertAlmostEqual(result["quantiles"]["p50"], 0.0008585)
        self.assertEqual(
            result["relative_difference_bands"],
            {"eq_0": 1, "gt_0_le_0.001": 1, "gt_0.001_le_0.01": 1, "gt_0.01": 1},
        )
        self.assertEqual(result["absolute_factor_scale"]["different_scale_rows"], 3)
        self.assertEqual(result["adjustment_conflict_subset"]["rows"], 3)
        self.assertAlmostEqual(
            result["adjustment_conflict_subset"]["quantiles"]["p50"], 0.001217
        )
        self.assertEqual(
            result["adjustment_conflict_subset"]["relative_difference_bands"]["gt_0.01"],
            1,
        )
        self.assertEqual(
            result["absolute_factor_scale"]["different_scale_with_normalized_close_le_0.001"],
            1,
        )

    def test_exclusions_separate_execution_blocking_from_audit_flags(self):
        panel = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "code": ["600000.SH"] * 3,
                "execution_conflict": [True, False, False],
                "adjustment_source_conflict": [True, True, False],
                "source_conflict": [True, True, True],
            }
        )

        result = _conflict_exclusions(panel)

        self.assertEqual(
            result["reason"].tolist(),
            ["execution_conflict", "adjustment_source_conflict", "source_conflict"],
        )
        self.assertIn("ineligible", result.loc[0, "detail"])
        self.assertIn("sensitivity", result.loc[1, "detail"])
        self.assertIn("audit only", result.loc[2, "detail"])
        self.assertNotIn("ineligible", result.loc[1, "detail"])
        self.assertNotIn("ineligible", result.loc[2, "detail"])

    def test_corporate_actions_include_ex_date_but_not_future_events(self):
        events = pd.DataFrame(
            {
                "code": ["600000.SH", "600000.SH", "600000.SH"],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "dividend_per_share": [0.1, 0.2, 0.3],
                "per_share_bonus": [0.0, 0.0, 0.0],
                "rights_ratio": [0.0, 0.0, 0.0],
                "rights_price": [0.0, 0.0, 0.0],
                "source_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        )

        effective = effective_corporate_actions(events, "2024-01-02")

        self.assertEqual(effective["date"].tolist(), ["2024-01-01", "2024-01-02"])

    def test_load_panel_round_trip(self):
        expected = _assemble_panel(self._raw("2024-01-02"), self._hfq("2024-01-02"), self._basic())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.parquet"
            expected.to_parquet(path, index=False)

            actual = load_panel(path)

        pd.testing.assert_frame_equal(actual, expected)

    @staticmethod
    def _raw(*dates):
        rows = []
        for date in dates:
            rows.append(
                {
                    "code": "sh.600000",
                    "date": date,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.5,
                    "preclose": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "trade_status": 1,
                    "is_st": 0,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _hfq(*dates):
        return pd.DataFrame(
            {
                "code": ["sh.600000"] * len(dates),
                "date": list(dates),
                "open": [20.0] * len(dates),
                "high": [22.0] * len(dates),
                "low": [18.0] * len(dates),
                "close": [21.0] * len(dates),
            }
        )

    @staticmethod
    def _basic():
        return pd.DataFrame(
            {"code": ["sh.600000"], "ipo_date": ["1999-11-10"], "out_date": [None]}
        )

    @staticmethod
    def _hithink(*dates):
        count = len(dates)
        return pd.DataFrame(
            {
                "code": ["600000.SH"] * count,
                "date": list(dates),
                "hithink_open": [10.0] * count,
                "hithink_signal_open": [20.0] * count,
                "hithink_high": [11.0] * count,
                "hithink_signal_high": [22.0] * count,
                "hithink_low": [9.0] * count,
                "hithink_signal_low": [18.0] * count,
                "hithink_close": [10.5] * count,
                "hithink_signal_close": [21.0] * count,
                "hithink_volume": [1000.0] * count,
                "hithink_amount": [10000.0] * count,
            }
        )


if __name__ == "__main__":
    unittest.main()
