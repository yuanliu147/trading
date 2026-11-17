import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from trend_system.broker import run_backtest
from trend_system.legacy_b_policy import LegacyBPolicy
from trend_system.legacy_config import load_legacy_config


CODE = "600000.SH"


class LegacyBPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_legacy_config(
            Path(__file__).resolve().parents[1] / "config" / "legacy_candidates.json"
        )

    def _bar(self, **changes):
        bar = {
            "date": "2024-01-03", "code": CODE, "open": 10.0, "high": 10.5,
            "low": 9.5, "close": 10.0, "prev_close": 10.0, "is_st": 0,
            "trade_status": 1, "adjustment_factor": 1.0,
        }
        bar.update(changes)
        return bar

    def test_risk_sizing_uses_atr_clip_heat_and_whole_lots(self):
        policy = LegacyBPolicy(self.config)
        policy.pending[CODE] = {"atr_pct": 0.02}
        quantity = policy.size_order(
            self._bar(), 100_000.0, 100_000.0, 0.0, self.config, positions={}
        )
        self.assertEqual(quantity, 1200)

    def test_heat_cap_rejects_new_position(self):
        policy = LegacyBPolicy(self.config)
        policy.pending[CODE] = {"atr_pct": 0.02}
        state = SimpleNamespace(risk_ps=0.52)
        positions = {"600001.SH": SimpleNamespace(quantity=5000, policy_state=state)}
        quantity = policy.size_order(
            self._bar(), 100_000.0, 100_000.0, 50_000.0, self.config, positions=positions
        )
        self.assertEqual(quantity, 0)

    def test_gap_filter_rejects_and_reports_reason(self):
        policy = LegacyBPolicy(self.config)
        fill = SimpleNamespace(price=10.6)
        state = policy.entry_state(
            {"atr_pct": 0.02}, self._bar(open=10.6, prev_close=10.0), fill, date_index=3
        )
        self.assertIsNone(state)
        self.assertEqual(policy.last_rejection_reason, "gap_filter")

    def test_twenty_day_global_index_exit_and_catastrophe(self):
        policy = LegacyBPolicy(self.config)
        fill = SimpleNamespace(price=10.0)
        state = policy.entry_state(
            {"atr_pct": 0.02}, self._bar(), fill, date_index=5
        )
        self.assertEqual(
            policy.exit_reason(state, self._bar(close=9.9), {}, date_index=23), None
        )
        self.assertEqual(
            policy.exit_reason(state, self._bar(close=9.9), {}, date_index=24),
            "time_exit",
        )
        self.assertEqual(
            policy.exit_reason(state, self._bar(close=7.9), {}, date_index=23),
            "catastrophe_stop",
        )

    def test_corporate_action_anchor_direction_and_st(self):
        policy = LegacyBPolicy(self.config)
        fill = SimpleNamespace(price=10.0)
        state = policy.entry_state(
            {"atr_pct": 0.02}, self._bar(), fill, date_index=0
        )
        self.assertEqual(
            policy.exit_reason(
                state, self._bar(close=4.5, adjustment_factor=2.0), {}, date_index=1
            ),
            None,
        )
        self.assertEqual(
            policy.exit_reason(
                state, self._bar(close=4.0, adjustment_factor=2.0, is_st=1), {}, date_index=1
            ),
            "catastrophe_stop;st_exit",
        )

    def test_integration_buys_next_open_and_sells_after_twentieth_day(self):
        dates = pd.bdate_range("2024-01-02", periods=25).astype(str).tolist()
        panel = pd.DataFrame({
            "date": dates, "code": CODE,
            "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0,
            "prev_close": 10.0, "signal_open": 10.0, "signal_high": 10.5,
            "signal_low": 9.5, "signal_close": 10.0, "adjustment_factor": 1.0,
            "volume": 1000.0, "amount": 1e8, "trade_status": 1, "is_st": 0,
            "out_date": pd.NA, "data_valid": True, "execution_conflict": False,
            "adjustment_source_conflict": False,
        })
        signals = panel[["date", "code"]].copy()
        signals["entry_signal"] = False
        signals.loc[0, "entry_signal"] = True
        signals["atr_pct"] = 0.02
        signals["amt20"] = 1e8
        signals["legacy_rank_score"] = -1e8
        signals["rank_return_120"] = -1e8
        actions = pd.DataFrame({
            "date": [], "code": [], "dividend_per_share": [], "per_share_bonus": [],
            "rights_ratio": [], "rights_price": [],
        })
        config = replace(
            self.config, start_date=dates[0], end_date=dates[-1], initial_cash=20000
        )
        result = run_backtest(panel, signals, actions, config, policy=LegacyBPolicy(config))
        buys = result.trades.loc[result.trades["side"] == "buy"]
        sells = result.trades.loc[result.trades["side"] == "sell"]
        self.assertEqual(buys["date"].tolist(), [dates[1]])
        self.assertEqual(sells["signal_date"].tolist(), [dates[20]])
        self.assertEqual(sells["date"].tolist(), [dates[21]])


if __name__ == "__main__":
    unittest.main()
