import json
import os
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from trend_system.legacy_config import LegacyResearchConfig, load_legacy_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "legacy_candidates.json"
PREREG_PATH = PROJECT_ROOT / "13_预注册_旧候选统一复核.md"


class LegacyResearchConfigTest(unittest.TestCase):
    def test_config_loads_frozen_research_values(self):
        config = load_legacy_config(CONFIG_PATH)

        self.assertIsInstance(config, LegacyResearchConfig)
        self.assertEqual(config.start_date, "2016-10-13")
        self.assertEqual(config.end_date, "2026-10-29")
        self.assertEqual(config.initial_cash, 20000)
        self.assertEqual(config.random_seed, 20261101)
        self.assertEqual(config.commission_rate, 0.00025)
        self.assertEqual(config.min_commission, 5)
        self.assertEqual(config.slippage_bps, 10)
        self.assertEqual(config.point_in_time_coverage_min, 0.99)
        self.assertEqual(config.industry_coverage_min, 0.99)
        self.assertEqual(config.required_entry_coverage, 1.0)
        self.assertEqual(config.max_main_drawdown, 0.20)
        self.assertEqual(config.unrejected_max_drawdown, 0.10)
        self.assertEqual(config.random_paths, 20)
        self.assertEqual(config.random_positive_min, 16)
        self.assertEqual(config.stress_cost_multiplier, 2)
        self.assertEqual(config.stress_slippage_bps, 20)
        self.assertEqual(config.rolling_three_year_positive_threshold, 0.5)

    def test_config_contains_exactly_two_frozen_candidates(self):
        config = load_legacy_config(CONFIG_PATH)

        self.assertEqual(
            tuple(candidate.name for candidate in config.candidates),
            ("three_of_two_low_vol", "early_trend_breadth"),
        )
        self.assertEqual(len(config.candidates), 2)
        self.assertEqual(len({candidate.name for candidate in config.candidates}), 2)

    def test_candidate_a_rule_and_source_freeze(self):
        candidate = load_legacy_config(CONFIG_PATH).candidates[0]

        self.assertEqual(candidate.evaluation_status, "PENDING_DATA_GATE")
        self.assertTrue(candidate.low10_includes_signal_day)
        self.assertTrue(candidate.entry_day_counts_toward_hold)
        self.assertFalse(candidate.suspension_counts_toward_hold)
        self.assertEqual(candidate.max_slots, 4)
        self.assertEqual(candidate.max_new_per_day, 2)
        self.assertEqual(candidate.max_weight, 0.20)
        self.assertEqual(candidate.max_hold, 40)
        self.assertEqual(candidate.entry_stop_pct, 0.08)
        self.assertEqual(candidate.amount_ma20_min, 30000000)
        self.assertEqual(candidate.s_vol_max, 0.5)
        self.assertEqual(candidate.s_heat_min, 0.5)
        self.assertEqual(candidate.stock_history_min, 260)
        self.assertEqual(candidate.stock_amount_ma60_min, 20000000)
        self.assertEqual(
            candidate.exit_priority,
            ("stop", "time_exit", "ma5_below_ma20", "whist_negative", "close_below_ma20"),
        )
        self.assertEqual(
            tuple(source.sha256 for source in candidate.sources),
            (
                "73b5eb05cba0f7090386f2d07de471bbff75b37c68c6151bc7f201635253c8a3",
                "52400a17160771a7846cf8f58a9e4899cdc2be5c3f541769dcbf32ace7ad2406",
            ),
        )

    def test_candidate_a_entry_veto_matches_preregistration(self):
        candidate = load_legacy_config(CONFIG_PATH).candidates[0]
        preregistration = PREREG_PATH.read_text(encoding="utf-8")

        self.assertIn("skip_if_stop_at_or_above_entry=true", preregistration)
        self.assertIn("`backtest.py` 616—618", preregistration)
        self.assertIs(candidate.skip_if_stop_at_or_above_entry, True)

    def test_candidate_b_rule_freeze_and_not_evaluable_fields(self):
        candidate = load_legacy_config(CONFIG_PATH).candidates[1]

        self.assertEqual(candidate.evaluation_status, "NOT_EVALUABLE")
        self.assertEqual(candidate.eligibility_start, "2017-10-13")
        self.assertEqual(candidate.max_slots, 6)
        self.assertEqual(candidate.max_weight, 0.20)
        self.assertEqual(candidate.risk_per_trade, 0.005)
        self.assertEqual(candidate.heat_reject_threshold, 0.0301)
        self.assertEqual(candidate.max_hold, 20)
        self.assertTrue(candidate.entry_day_counts_toward_hold)
        self.assertTrue(candidate.suspension_counts_toward_hold)
        self.assertEqual(candidate.market_breadth_threshold, 0.50)
        self.assertEqual(candidate.trend_age_min, 5)
        self.assertEqual(candidate.trend_age_max, 30)
        self.assertEqual(candidate.actual_close_min, 2)
        self.assertEqual(candidate.atr_pct_max, 0.044)
        self.assertEqual(candidate.dd60_threshold, -0.08)
        self.assertEqual(candidate.sector_breadth_min, 0.60)
        self.assertEqual(candidate.sector_ret20_threshold, 0.05)
        self.assertEqual(candidate.catastrophe_stop_pct, 0.20)
        self.assertEqual(candidate.risk_atr_multiple, 2)
        self.assertEqual(candidate.risk_size_min, 0.04)
        self.assertEqual(candidate.risk_size_max, 0.12)
        self.assertEqual(candidate.min_position_weight, 0.01)
        self.assertEqual(candidate.exit_priority, ("catastrophe_stop", "time_exit"))
        self.assertEqual(
            candidate.sources[0].sha256,
            "8406120aedd2ad21cbbca35e87fb784262434730b3399697b7afbe822b4686e8",
        )
        self.assertTrue(candidate.lu_filter)
        self.assertTrue(candidate.gap_filter)
        self.assertEqual(candidate.gap_max, 0.05)
        self.assertEqual(
            candidate.not_evaluable_fields,
            (
                "up_struct_formula",
                "trend_age_formula",
                "atr_pct_formula",
                "dd60_formula",
                "lu_count20_formula",
                "market_breadth20_formula",
                "sector_breadth20_formula",
                "sector_ret20_formula",
                "amt20_formula",
                "historical_industry_membership",
                "next_gap_formula_and_timestamp",
                "amount_ranking_tie_break",
            ),
        )

    def test_config_and_candidates_are_immutable(self):
        config = load_legacy_config(CONFIG_PATH)

        with self.assertRaises(FrozenInstanceError):
            config.initial_cash = 30000
        with self.assertRaises(FrozenInstanceError):
            config.candidates[0].name = "changed"
        with self.assertRaises(FrozenInstanceError):
            config.candidates[0].sources[0].path = "changed"

    def test_unknown_top_level_field_is_rejected(self):
        data = self._config_data()
        data["unknown_field"] = 1

        with self.assertRaisesRegex(ValueError, "Unknown config fields"):
            self._load(data)

    def test_additional_candidate_parameter_is_rejected(self):
        data = self._config_data()
        data["candidates"][0]["stop_loss"] = 0.08

        with self.assertRaisesRegex(ValueError, "Unknown candidate fields"):
            self._load(data)

    def test_nonstandard_json_constants_are_rejected(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "config.json"
                    path.write_text(
                        '{"start_date":"2016-10-13","initial_cash":' + constant + "}",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "Invalid JSON constant"):
                        load_legacy_config(path)

    def test_environment_override_is_rejected(self):
        data = self._config_data()
        with patch.dict(os.environ, {"LEGACY_INITIAL_CASH": "30000"}):
            config = self._load(data)

        self.assertEqual(config.initial_cash, 20000)

    def test_cli_style_override_field_is_rejected(self):
        data = self._config_data()
        data["initial_cash_override"] = 30000

        with self.assertRaisesRegex(ValueError, "Unknown config fields"):
            self._load(data)

    def test_missing_file_and_malformed_json_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unable to load legacy config"):
            load_legacy_config(CONFIG_PATH.with_name("does-not-exist.json"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unable to load legacy config"):
                load_legacy_config(path)

    def test_actual_cli_arguments_cannot_override_config(self):
        data = self._config_data()
        with patch.object(sys, "argv", ["prog", "--initial-cash=30000"]):
            config = self._load(data)

        self.assertEqual(config.initial_cash, 20000)

    def test_candidate_set_is_exact(self):
        for mutation in ("missing", "duplicate", "extra", "unknown"):
            with self.subTest(mutation=mutation):
                data = self._config_data()
                if mutation == "missing":
                    data["candidates"] = data["candidates"][:1]
                elif mutation == "duplicate":
                    data["candidates"][1] = data["candidates"][0]
                elif mutation == "extra":
                    data["candidates"].append(data["candidates"][0])
                else:
                    data["candidates"][1]["name"] = "unknown"
                with self.assertRaises(ValueError):
                    self._load(data)

    def test_duplicate_json_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"initial_cash":20000,"initial_cash":30000}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
                load_legacy_config(path)

    def test_non_object_root_and_invalid_scalar_types_are_rejected(self):
        for value in ([], "config", 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "root must be a JSON object"):
                    self._load(value)

        data = self._config_data()
        data["initial_cash"] = True
        with self.assertRaisesRegex(ValueError, "Invalid field type"):
            self._load(data)

    def test_unknown_source_field_is_rejected(self):
        data = self._config_data()
        data["candidates"][0]["sources"][0]["version"] = "other"

        with self.assertRaisesRegex(ValueError, "Unknown source fields"):
            self._load(data)

    def test_missing_required_field_is_rejected(self):
        data = self._config_data()
        del data["random_seed"]

        with self.assertRaisesRegex(ValueError, "Missing config fields"):
            self._load(data)

    def _load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_legacy_config(path)

    @staticmethod
    def _config_data():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
