import unittest

import pandas as pd

from trend_system import broker, execution
from trend_system.legacy_config import load_legacy_config
from trend_system.legacy_robustness import (
    GATE_IDS,
    RandomOrderLegacyPolicy,
    derive_entries_for_window,
    doubled_cost_context,
    evaluate_legacy_gates,
)
from trend_system.legacy_policy import LegacyPolicy


class LegacyRobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        cls.config = load_legacy_config(
            Path(__file__).resolve().parents[1] / "config" / "legacy_candidates.json"
        )

    def _summary(self, **overrides):
        summary = {
            "candidate_status": "EVALUABLE",
            "main_max_drawdown": 0.05,
            "gates": {gate_id: {"passed": True} for gate_id in GATE_IDS},
        }
        summary.update(overrides)
        return summary

    def test_gate_ids_fail_closed_and_mdd_boundaries(self):
        self.assertEqual(evaluate_legacy_gates(self._summary()), "UNREJECTED")
        self.assertEqual(
            evaluate_legacy_gates(self._summary(main_max_drawdown=0.10)), "UNREJECTED"
        )
        self.assertEqual(
            evaluate_legacy_gates(self._summary(main_max_drawdown=0.100001)), "BOUNDARY"
        )
        self.assertEqual(
            evaluate_legacy_gates(self._summary(main_max_drawdown=0.20)), "BOUNDARY"
        )
        self.assertEqual(
            evaluate_legacy_gates(self._summary(main_max_drawdown=0.200001)), "REJECTED"
        )
        self.assertEqual(
            evaluate_legacy_gates(self._summary(candidate_status="NOT_EVALUABLE")),
            "NOT_EVALUABLE",
        )

        missing = self._summary()
        del missing["gates"][GATE_IDS[0]]
        self.assertEqual(evaluate_legacy_gates(missing), "REJECTED")
        extra = self._summary()
        extra["gates"]["unknown"] = {"passed": True}
        self.assertEqual(evaluate_legacy_gates(extra), "REJECTED")
        for bad in ("False", 0, 1, None):
            broken = self._summary()
            broken["gates"][GATE_IDS[0]] = {"passed": bad}
            self.assertEqual(evaluate_legacy_gates(broken), "REJECTED")

    def test_random_order_is_seeded_and_logs_a_permutation(self):
        rows = [
            {"date": "2024-01-02", "code": f"60000{i}.SH", "legacy_rank_score": 5.0 - i}
            for i in range(5)
        ]
        first = RandomOrderLegacyPolicy(self.config, 20261101, 0)
        second = RandomOrderLegacyPolicy(self.config, 20261101, 0)
        other = RandomOrderLegacyPolicy(self.config, 20261101, 1)
        order_a = first.order_candidates(list(rows))
        order_b = second.order_candidates(list(rows))
        order_c = other.order_candidates(list(rows))
        self.assertEqual(order_a, order_b)
        self.assertEqual(sorted(order_a), sorted(row["code"] for row in rows))
        self.assertEqual(
            sorted(entry["random_rank"] for entry in first.audit), list(range(len(rows)))
        )
        self.assertEqual(len(order_c), len(rows))
        self.assertIsInstance(LegacyPolicy(self.config), LegacyPolicy)

    def test_doubled_costs_double_every_fee_component_for_sizing_and_fill(self):
        args = ("2024-06-03", 100000.0, "sell")
        baseline = execution.transaction_costs(*args)["total"]
        with doubled_cost_context():
            patched_execution = execution.transaction_costs(*args)["total"]
            patched_broker = broker.transaction_costs(*args)["total"]
        self.assertAlmostEqual(patched_execution, baseline * 2)
        self.assertAlmostEqual(patched_broker, baseline * 2)
        self.assertAlmostEqual(execution.transaction_costs(*args)["total"], baseline)

    def test_stress_entries_are_derived_with_the_same_frozen_validity(self):
        frame = pd.DataFrame({
            "date": ["2015-06-01", "2015-06-02", "2016-10-18", "2016-10-19"],
            "code": ["600000.SH"] * 4,
            "accepted_signal": [True, True, True, True],
            "is_main_board": [True, True, True, True],
            "is_st": [0, 1, 0, 0],
            "open": [10.0] * 4, "high": [10.5] * 4, "low": [9.5] * 4,
            "close": [10.0] * 4, "prev_close": [10.0] * 4,
            "signal_open": [10.0] * 4, "signal_high": [10.5] * 4,
            "signal_low": [9.5] * 4, "signal_close": [10.0] * 4,
            "volume": [1000.0] * 4, "amount": [10000.0] * 4,
            "trade_status": [1] * 4,
            "data_valid": [True] * 4,
            "execution_conflict": [False] * 4,
            "ipo_date": ["2000-01-01"] * 4,
            "entry_signal": [False, False, True, True],
        })
        derived = derive_entries_for_window(frame, "2015-01-01", "2015-12-31")
        self.assertEqual(derived["entry_signal"].tolist(), [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
