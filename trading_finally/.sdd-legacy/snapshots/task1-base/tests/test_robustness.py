import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from trend_system.config import ResearchConfig
from trend_system.robustness import (
    _common_source_inputs,
    annual_equity_contributions,
    block_bootstrap,
    evaluate_gates,
    neighborhood_configs,
    random_entry_signals,
    random_order_signals,
    run_robustness,
)


def _config(**changes):
    values = {
        "hithink_db_path": "unused",
        "baostock_db_path": "unused",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
    }
    values.update(changes)
    return ResearchConfig(**values)


FROZEN_GATE_IDS = (
    "data_execution_integrity",
    "main_window_positive",
    "rolling_3y_majority_positive",
    "double_cost_positive_and_executable",
    "neighborhood_at_least_6_positive",
    "random_order_at_least_16_positive",
    "remaining_years_equity_pnl_positive",
    "fully_after_close_reproducible",
    "hithink_common_positive_mdd_and_direction",
    "drawdown_boundary",
)


def _passing_summary(max_drawdown):
    return {
        "max_drawdown": max_drawdown,
        "gates": [
            {"id": identifier, "passed": True}
            for identifier in FROZEN_GATE_IDS
        ],
    }


class RobustnessUnitTest(unittest.TestCase):
    def test_eight_neighborhoods_change_exactly_one_frozen_field(self):
        config = _config()
        variants = neighborhood_configs(config)
        expected = {
            ("entry_lookback", 90),
            ("entry_lookback", 150),
            ("trend_window", 180),
            ("trend_window", 220),
            ("trend_slope_window", 10),
            ("trend_slope_window", 30),
            ("exit_lookback", 15),
            ("exit_lookback", 25),
        }
        self.assertEqual(len(variants), 8)
        observed = set()
        baseline = asdict(config)
        for name, variant in variants:
            values = asdict(variant)
            changed = {
                key: (baseline[key], values[key])
                for key in baseline
                if values[key] != baseline[key]
            }
            self.assertEqual(len(changed), 1, name)
            field, (_, new_value) = next(iter(changed.items()))
            observed.add((field, new_value))
        self.assertEqual(observed, expected)
        self.assertEqual(asdict(config), baseline)

    def test_mechanical_verdict_boundaries_and_other_gate_failure(self):
        self.assertEqual(evaluate_gates(_passing_summary(0.08)), "UNREJECTED")
        self.assertEqual(evaluate_gates(_passing_summary(0.10)), "UNREJECTED")
        self.assertEqual(evaluate_gates(_passing_summary(0.100001)), "BOUNDARY")
        self.assertEqual(evaluate_gates(_passing_summary(0.20)), "BOUNDARY")
        self.assertEqual(evaluate_gates(_passing_summary(0.200001)), "REJECTED")
        failed = _passing_summary(0.02)
        failed["gates"][5]["passed"] = False
        self.assertEqual(evaluate_gates(failed), "REJECTED")
        self.assertEqual(
            {evaluate_gates(_passing_summary(value)) for value in (0.01, 0.15, 0.21)},
            {"UNREJECTED", "BOUNDARY", "REJECTED"},
        )

    def test_gate_ids_must_be_exactly_the_ten_frozen_ids(self):
        unrelated_only = {"max_drawdown": 0.01, "gates": [{"id": "unrelated", "passed": True}]}
        self.assertEqual(evaluate_gates(unrelated_only), "REJECTED")
        for mutation in ("missing", "duplicate", "unknown"):
            with self.subTest(mutation=mutation):
                summary = _passing_summary(0.01)
                if mutation == "missing":
                    summary["gates"] = summary["gates"][:-1]
                elif mutation == "duplicate":
                    summary["gates"][-1] = summary["gates"][0].copy()
                else:
                    summary["gates"][-1]["id"] = "unrelated"
                self.assertEqual(evaluate_gates(summary), "REJECTED")

    def test_gate_passed_values_must_be_actual_booleans(self):
        string_false = _passing_summary(0.05)
        for gate in string_false["gates"]:
            gate["passed"] = "False"
        self.assertEqual(evaluate_gates(string_false), "REJECTED")
        for malformed in ("False", "True", 0, 1, None, np.bool_(True)):
            with self.subTest(malformed=repr(malformed)):
                summary = _passing_summary(0.05)
                summary["gates"][0]["passed"] = malformed
                self.assertEqual(evaluate_gates(summary), "REJECTED")
        self.assertEqual(evaluate_gates(_passing_summary(0.05)), "UNREJECTED")
        legal_false = _passing_summary(0.05)
        legal_false["gates"][0]["passed"] = False
        self.assertEqual(evaluate_gates(legal_false), "REJECTED")

    def test_profit_concentration_uses_annual_account_equity_pnl(self):
        account = pd.DataFrame(
            {
                "date": [
                    "2020-01-02",
                    "2020-12-31",
                    "2021-12-31",
                    "2022-12-30",
                ],
                "equity": [10000.0, 16000.0, 13000.0, 14000.0],
            }
        )
        annual = annual_equity_contributions(account, initial_cash=10000)
        self.assertEqual(annual["equity_pnl"].tolist(), [6000.0, -3000.0, 1000.0])
        top_year = annual.loc[annual["is_top_contributor"], "year"].item()
        remaining = annual.loc[~annual["is_top_contributor"], "equity_pnl"].sum()
        self.assertEqual(top_year, "2020")
        self.assertEqual(remaining, -2000.0)

    def test_random_order_is_reproducible_and_preserves_signals(self):
        signals = self._random_signal_fixture()
        first, first_log = random_order_signals(signals, _config(), path_id=3)
        second, second_log = random_order_signals(signals, _config(), path_id=3)
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(first_log, second_log)
        self.assertEqual(
            first_log.columns.tolist(),
            ["path_id", "date", "code", "random_rank"],
        )
        self.assertEqual(len(first_log), int(signals["entry_signal"].sum()))
        expected_log = first.loc[first["entry_signal"], ["date", "code", "rank_return_120"]].reset_index(drop=True)
        expected_log.columns = ["date", "code", "random_rank"]
        pd.testing.assert_frame_equal(
            first_log[["date", "code", "random_rank"]].reset_index(drop=True),
            expected_log,
        )
        self.assertTrue(first["entry_signal"].equals(signals["entry_signal"]))
        self.assertTrue(first["exit_signal"].equals(signals["exit_signal"]))
        entry = signals["entry_signal"]
        self.assertFalse(first.loc[entry, "rank_return_120"].equals(signals.loc[entry, "rank_return_120"]))
        self.assertTrue(first.loc[~entry, "rank_return_120"].equals(signals.loc[~entry, "rank_return_120"]))

    def test_random_entry_replaces_without_replacement_and_keeps_real_rank(self):
        signals = self._random_signal_fixture()
        changed, log = random_entry_signals(signals, _config(), path_id=2)
        day = signals["date"].iloc[0]
        original = set(signals.loc[signals["entry_signal"], "code"])
        replacement = set(changed.loc[changed["entry_signal"], "code"])
        self.assertEqual(len(replacement), len(original))
        self.assertTrue(original.isdisjoint(replacement))
        self.assertEqual(log["path_id"].tolist(), [2])
        self.assertEqual(log["date"].tolist(), [day])
        self.assertEqual(log["required"].item(), len(original))
        ranks = signals.set_index("code")["rank_return_120"]
        for code in replacement:
            self.assertEqual(changed.loc[changed["code"] == code, "rank_return_120"].item(), ranks[code])

    def test_random_entry_fails_explicitly_when_pool_is_insufficient(self):
        signals = self._random_signal_fixture()
        signals.loc[~signals["entry_signal"], "data_valid"] = False
        with self.assertRaisesRegex(RuntimeError, "random entry pool insufficient"):
            random_entry_signals(signals, _config(), path_id=0)

    def test_block_bootstrap_only_resamples_return_series(self):
        account = pd.DataFrame(
            {
                "date": pd.bdate_range("2020-01-01", periods=50).astype(str),
                "equity": np.linspace(10000.0, 11000.0, 50),
            }
        )
        first = block_bootstrap(account, seed=20261101, iterations=10, block_size=20)
        second = block_bootstrap(account, seed=20261101, iterations=10, block_size=20)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(first["replicate"].tolist(), list(range(10)))

    def test_dual_source_common_window_aligns_date_major_multi_stock_keys(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_project_inputs(root)
            first = pd.read_parquet(root / "data" / "main_panel.parquet")
            second = first.copy()
            second["code"] = "000001.SZ"
            panel = (
                pd.concat([first, second], ignore_index=True)
                .sort_values(["date", "code"], kind="stable")
                .reset_index(drop=True)
            )
            hithink = panel[["date", "code", "signal_open", "signal_high", "signal_low", "signal_close"]].copy()
            config = _config(start_date=panel["date"].min(), end_date=panel["date"].max())

            (
                baostock_panel,
                hithink_panel,
                baostock_signals,
                hithink_signals,
                _,
                common_start,
            ) = _common_source_inputs(panel, hithink, config)

            keys = ["date", "code"]
            expected = panel.loc[panel["date"].ge(common_start), keys]
            expected_keys = set(expected.itertuples(index=False, name=None))
            for name, frame in (
                ("baostock_panel", baostock_panel),
                ("hithink_panel", hithink_panel),
                ("baostock_signals", baostock_signals),
                ("hithink_signals", hithink_signals),
            ):
                with self.subTest(frame=name):
                    self.assertEqual(len(frame), len(expected))
                    self.assertEqual(
                        set(frame[keys].itertuples(index=False, name=None)),
                        expected_keys,
                    )
                    self.assertEqual(frame["date"].min(), common_start)
                    self.assertEqual(frame["date"].max(), config.end_date)

    def test_synthetic_full_matrix_executes_all_variants_and_persists_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_project_inputs(root)
            from unittest.mock import patch

            with patch("trend_system.robustness.PROJECT_ROOT", root):
                result = run_robustness(_config(start_date="2010-01-04", end_date="2010-11-18"))

            self.assertEqual(len(result.variants), 53)
            counts = result.variants.groupby("family").size().to_dict()
            self.assertEqual(
                counts,
                {
                    "neighborhood": 8,
                    "dual_source": 2,
                    "double_cost": 1,
                    "random_order": 20,
                    "random_entry": 20,
                    "portfolio": 2,
                },
            )
            self.assertTrue((result.variants["status"] == "completed").all())
            self.assertEqual(len(result.bootstrap), 1000)
            self.assertIn(result.verdict, {"UNREJECTED", "BOUNDARY", "REJECTED"})
            self.assertTrue((root / "results" / "robustness" / "main_freeze.json").is_file())
            self.assertTrue((root / "results" / "stress_2010_2015" / "performance_summary.json").is_file())
            self.assertTrue(result.report_path.is_file())
            saved = json.loads((root / "results" / "robustness" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["verdict"], result.verdict)
            self.assertEqual(len(saved["gates"]), 10)
            self.assertIsInstance(saved["execution_integrity"]["baseline"], dict)
            self.assertIn("cash_nonnegative", saved["execution_integrity"]["baseline"])
            fully = next(gate for gate in saved["gates"] if gate["id"] == "fully_after_close_reproducible")
            self.assertIsInstance(fully["actual"], dict)
            self.assertIn("legal_fill_prices", fully["actual"])
            self.assertIn("risk_report", saved)
            self.assertEqual(len(saved["input_manifest"]["inputs"]), 5)
            for path_id in range(20):
                log_path = root / "results" / "robustness" / "random_order" / f"path_{path_id:02d}" / "random_order_log.csv"
                self.assertTrue(log_path.is_file())
                log = pd.read_csv(log_path)
                self.assertEqual(
                    log.columns.tolist(), ["path_id", "date", "code", "random_rank"]
                )
            self.assertIn("post_hoc_only", saved["descriptive"]["formulas"])
            freeze = json.loads((root / "results" / "robustness" / "main_freeze.json").read_text(encoding="utf-8"))
            self.assertNotIn("stress_2010_2015", freeze["files"])

    @staticmethod
    def _random_signal_fixture():
        return pd.DataFrame(
            {
                "date": ["2020-01-02"] * 4,
                "code": ["600000.SH", "000001.SZ", "000002.SZ", "000003.SZ"],
                "is_st": [0, 0, 0, 0],
                "trade_status": [1, 1, 1, 1],
                "data_valid": [True] * 4,
                "execution_conflict": [False] * 4,
                "entry_signal": [True, False, False, False],
                "exit_signal": [False] * 4,
                "st_exit": [False] * 4,
                "ma200": [10.0] * 4,
                "channel_high_120": [9.0] * 4,
                "channel_low_120": [8.0] * 4,
                "rank_return_120": [0.4, 0.3, 0.2, 0.1],
            }
        )

    @staticmethod
    def _write_project_inputs(root):
        data = root / "data"
        data.mkdir(parents=True)
        dates = pd.bdate_range("2010-01-04", periods=225).astype(str).tolist()
        rows = []
        for day in dates:
            rows.append(
                {
                    "date": day,
                    "code": "600000.SH",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "prev_close": 10.0,
                    "signal_open": 10.0,
                    "signal_high": 10.5,
                    "signal_low": 9.5,
                    "signal_close": 10.0,
                    "volume": 1000000.0,
                    "amount": 10000000.0,
                    "trade_status": 1,
                    "is_st": 0,
                    "ipo_date": "2000-01-01",
                    "out_date": pd.NA,
                    "data_valid": True,
                    "execution_conflict": False,
                    "adjustment_source_conflict": False,
                    "source_conflict": False,
                }
            )
        panel = pd.DataFrame(rows)
        panel.to_parquet(data / "main_panel.parquet", index=False)
        panel.to_parquet(data / "stress_panel.parquet", index=False)
        panel[["date", "code", "signal_open", "signal_high", "signal_low", "signal_close", "data_valid"]].to_parquet(
            data / "hithink_signal_panel.parquet", index=False
        )
        index_rows = []
        for day in dates:
            for code in ("000300.SH", "000852.SH"):
                index_rows.append({"date": day, "code": code, "close": 1000.0})
        pd.DataFrame(index_rows).to_parquet(data / "index_panel.parquet", index=False)
        pd.DataFrame(
            columns=[
                "date",
                "code",
                "dividend_per_share",
                "per_share_bonus",
                "rights_ratio",
                "rights_price",
            ]
        ).to_parquet(data / "corporate_actions.parquet", index=False)


if __name__ == "__main__":
    unittest.main()
