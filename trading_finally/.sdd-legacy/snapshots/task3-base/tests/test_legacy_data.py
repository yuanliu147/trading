import unittest
from pathlib import Path

import pandas as pd

from trend_system.legacy_config import load_legacy_config
from trend_system.legacy_data import (
    assert_expected_trading_dates,
    candidate_entry_valid,
    complete_hfq_signals,
    coverage_gate_passes,
    daily_coverage,
    evaluate_candidate_statuses,
    point_in_time_industry,
    refresh_context_flags,
    required_field_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "legacy_candidates.json"


class PointInTimeIndustryTest(unittest.TestCase):
    def test_future_industry_record_is_not_visible_before_valid_from(self):
        records = pd.DataFrame(
            {
                "code": ["600000.SH", "600000.SH"],
                "industry": ["银行", "未来行业"],
                "valid_from": ["2020-01-01", "2024-01-03"],
                "valid_to": [None, None],
            }
        )

        actual = point_in_time_industry(records, "2024-01-02")

        self.assertEqual(actual["industry"].tolist(), ["银行"])

    def test_current_snapshot_without_valid_from_cannot_backfill_history(self):
        records = pd.DataFrame(
            {
                "code": ["600000.SH"],
                "industry": ["当前行业快照"],
                "valid_from": [pd.NA],
                "valid_to": [pd.NA],
            }
        )

        actual = point_in_time_industry(records, "2020-01-02")

        self.assertTrue(actual.empty)


class CoverageTest(unittest.TestCase):
    def test_daily_denominator_is_fixed_by_point_in_time_active_universe(self):
        active = pd.DataFrame(
            {
                "date": ["2024-01-02"] * 2 + ["2024-01-03"] * 2,
                "code": ["600000.SH", "000001.SZ"] * 2,
            }
        )
        panel = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"],
                "code": ["600000.SH", "600000.SH", "000001.SZ", "300001.SZ"],
                "close": [10.0, 10.0, 20.0, 30.0],
                "volume": [100.0, 100.0, 200.0, 300.0],
            }
        )

        actual = daily_coverage(active, panel, ("close", "volume"))

        self.assertEqual(actual["denominator"].tolist(), [2, 2])
        self.assertEqual(actual["numerator"].tolist(), [1, 2])
        self.assertEqual(actual["coverage"].tolist(), [0.5, 1.0])

    def test_98_99_percent_fails_and_exact_99_percent_passes(self):
        below = pd.DataFrame(
            {"date": ["2024-01-02"], "denominator": [10000], "coverage": [0.9899]}
        )
        exact = pd.DataFrame(
            {"date": ["2024-01-02"], "denominator": [100], "coverage": [0.99]}
        )

        self.assertFalse(coverage_gate_passes(below, 0.99))
        self.assertTrue(coverage_gate_passes(exact, 0.99))

    def test_missing_required_entry_field_has_zero_coverage(self):
        panel = pd.DataFrame({"close": [10.0, 11.0]})

        self.assertEqual(required_field_coverage(panel, ("close", "amount")), 0.0)

    def test_partially_missing_required_entry_field_is_not_complete(self):
        panel = pd.DataFrame({"close": [10.0, None], "amount": [100.0, 200.0]})

        self.assertEqual(required_field_coverage(panel, ("close", "amount")), 0.5)

    def test_whole_missing_trading_day_fails_against_independent_calendar(self):
        with self.assertRaisesRegex(ValueError, "Missing expected trading dates"):
            assert_expected_trading_dates(
                {"2024-01-02"}, {"2024-01-02", "2024-01-03"}
            )

    def test_exact_expected_trading_date_set_passes(self):
        assert_expected_trading_dates(
            {"2024-01-03", "2024-01-02"}, {"2024-01-02", "2024-01-03"}
        )


class ContextSemanticsTest(unittest.TestCase):
    def test_normal_suspension_is_complete_but_not_rank_eligible(self):
        row = self._row(trade_status=0, volume=None, amount=None)

        actual = refresh_context_flags(row)

        self.assertTrue(actual.loc[0, "context_complete"])
        self.assertFalse(actual.loc[0, "rank_eligible"])

    def test_normal_trading_null_amount_is_not_complete(self):
        row = self._row(trade_status=1, amount=None)

        actual = refresh_context_flags(row)

        self.assertFalse(actual.loc[0, "context_complete"])
        self.assertFalse(actual.loc[0, "rank_eligible"])

    def test_exact_hfq_completion_fills_only_null_fields(self):
        panel = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "code": ["600000.SH", "600000.SH"],
                "trade_status": [1, 1],
                "signal_open": [None, None],
                "signal_high": [11.0, None],
                "signal_low": [None, None],
                "signal_close": [None, None],
            }
        )
        hithink = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "code": ["600000.SH"],
                "signal_open": [20.0],
                "signal_high": [21.0],
                "signal_low": [19.0],
                "signal_close": [22.0],
            }
        )

        actual, audit = complete_hfq_signals(panel, hithink)

        self.assertEqual(
            actual.loc[0, ["signal_open", "signal_high", "signal_low", "signal_close"]].tolist(),
            [20.0, 11.0, 19.0, 22.0],
        )
        self.assertTrue(actual.loc[1, list(("signal_open", "signal_high", "signal_low", "signal_close"))].isna().all())
        self.assertEqual(audit.loc[0, "completed_fields"], "signal_open|signal_low|signal_close")

    @staticmethod
    def _row(**overrides):
        values = {
            "date": "2024-01-02",
            "code": "600000.SH",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "prev_close": 10.0,
            "signal_open": 20.0,
            "signal_high": 21.0,
            "signal_low": 19.0,
            "signal_close": 20.5,
            "volume": 1000.0,
            "amount": 10000.0,
            "trade_status": 1,
            "is_st": 0,
            "ipo_date": "2020-01-01",
            "out_date": None,
            "is_main_board": True,
            "execution_conflict": False,
        }
        values.update(overrides)
        return pd.DataFrame([values])


class CandidateEntryValidityTest(unittest.TestCase):
    def test_complete_normally_trading_positive_nonconflict_row_passes(self):
        row = ContextSemanticsTest._row()

        self.assertTrue(candidate_entry_valid(row).iloc[0])

    def test_suspension_fails_closed(self):
        row = ContextSemanticsTest._row(trade_status=0)
        self.assertFalse(candidate_entry_valid(row).iloc[0])

    def test_zero_volume_fails_closed(self):
        row = ContextSemanticsTest._row(volume=0.0)
        self.assertFalse(candidate_entry_valid(row).iloc[0])

    def test_null_amount_fails_closed(self):
        row = ContextSemanticsTest._row(amount=None)
        self.assertFalse(candidate_entry_valid(row).iloc[0])

    def test_execution_conflict_fails_closed(self):
        row = ContextSemanticsTest._row(execution_conflict=True)
        self.assertFalse(candidate_entry_valid(row).iloc[0])


class CandidateGateTest(unittest.TestCase):
    def setUp(self):
        self.config = load_legacy_config(CONFIG_PATH)

    def test_candidate_a_full_a_pass_waits_for_task3_actual_entry_gate(self):
        gates = {
            "full_a_point_in_time": True,
            "a_entry_fields": False,
            "industry_point_in_time": True,
            "b_entry_fields": True,
        }

        actual = evaluate_candidate_statuses(self.config, gates)

        self.assertEqual(actual["three_of_two_low_vol"], "PENDING_DATA_GATE")

    def test_candidate_a_fails_when_full_a_coverage_fails(self):
        gates = {
            "full_a_point_in_time": False,
            "a_entry_fields": True,
            "industry_point_in_time": True,
            "b_entry_fields": True,
        }

        actual = evaluate_candidate_statuses(self.config, gates)

        self.assertEqual(actual["three_of_two_low_vol"], "NOT_EVALUABLE")

    def test_candidate_b_fails_without_dated_industry_source(self):
        gates = {
            "full_a_point_in_time": True,
            "a_entry_fields": True,
            "industry_point_in_time": False,
            "b_entry_fields": True,
        }

        actual = evaluate_candidate_statuses(self.config, gates)

        self.assertEqual(actual["early_trend_breadth"], "NOT_EVALUABLE")

    def test_candidate_b_freeze_cannot_be_restored_by_complete_data(self):
        gates = {
            "full_a_point_in_time": True,
            "a_entry_fields": True,
            "industry_point_in_time": True,
            "b_entry_fields": True,
        }

        actual = evaluate_candidate_statuses(self.config, gates)

        self.assertEqual(actual["early_trend_breadth"], "NOT_EVALUABLE")
        self.assertEqual(len(self.config.candidates[1].not_evaluable_fields), 12)


if __name__ == "__main__":
    unittest.main()
