import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from trend_system.config import ResearchConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "config" / "baseline.json"


class ResearchConfigTest(unittest.TestCase):
    def test_baseline_loads_frozen_values(self):
        config = load_config(BASELINE_PATH)

        self.assertIsInstance(config, ResearchConfig)
        self.assertEqual(config.initial_cash, 20000)
        self.assertEqual(config.slots, 3)
        self.assertEqual(config.max_weight, 0.20)
        self.assertEqual(config.max_exposure, 0.60)
        self.assertEqual(config.entry_lookback, 120)
        self.assertEqual(config.trend_window, 200)
        self.assertEqual(config.trend_slope_window, 20)
        self.assertEqual(config.exit_lookback, 20)
        self.assertEqual(config.start_date, "2016-10-13")
        self.assertEqual(config.end_date, "2026-10-29")
        self.assertEqual(config.commission_rate, 0.00025)
        self.assertEqual(config.min_commission, 5)
        self.assertEqual(config.slippage_bps, 10)
        self.assertEqual(config.random_seed, 20261101)
        self.assertEqual(
            config.hithink_db_path,
            "/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb",
        )
        self.assertEqual(
            config.baostock_db_path,
            "/Users/admin/WorkSpace/baostock_data/data/baostock.db",
        )

    def test_config_is_immutable(self):
        config = load_config(BASELINE_PATH)

        with self.assertRaises(FrozenInstanceError):
            config.slots = 4

    def test_unknown_field_is_rejected(self):
        data = self._baseline_data()
        data["unknown_field"] = 1

        with self.assertRaises(ValueError):
            self._load(data)

    def test_additional_strategy_parameter_is_rejected(self):
        data = self._baseline_data()
        data["stop_loss"] = 0.08

        with self.assertRaises(ValueError):
            self._load(data)

    def test_each_missing_database_path_is_rejected(self):
        for field in ("hithink_db_path", "baostock_db_path"):
            with self.subTest(field=field):
                data = self._baseline_data()
                del data[field]

                with self.assertRaises(ValueError):
                    self._load(data)

    def _load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_config(path)

    @staticmethod
    def _baseline_data():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
