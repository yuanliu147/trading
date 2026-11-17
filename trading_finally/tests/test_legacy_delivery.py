import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NegativeDeliveryTest(unittest.TestCase):
    def test_stop_boundary_document_exists_and_has_both_candidates(self):
        report = (ROOT / "97_停止边界与后续.md").read_text(encoding="utf-8")
        self.assertIn("候选 A", report)
        self.assertIn("候选 B", report)
        self.assertIn("-79.200%", report)
        self.assertIn("-12.115%", report)
        self.assertIn("停止", report)
        self.assertIn("4,526.32", report)

    def test_no_simulation_or_live_delivery_files_exist(self):
        forbidden = [
            ROOT / "97_纯模拟观察协议.md",
            ROOT / "src" / "trend_system" / "legacy_daily.py",
            ROOT / "scripts" / "legacy_daily_plan.py",
            ROOT / "state" / "legacy_forward_ledger.csv",
        ]
        for path in forbidden:
            self.assertFalse(path.exists(), f"unexpected delivery file: {path}")

    def test_no_candidate_exposes_orders_or_positions_for_live_use(self):
        for candidate in ("three_of_two_low_vol", "early_trend_breadth"):
            directory = ROOT / "results" / "legacy_candidates" / candidate
            self.assertFalse((directory / "daily_plan.csv").exists())
            self.assertFalse((directory / "live_orders.csv").exists())


if __name__ == "__main__":
    unittest.main()
