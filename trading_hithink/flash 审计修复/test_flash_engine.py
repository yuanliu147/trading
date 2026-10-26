from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from flash_data import data_from_frames, load_config
from flash_engine import EngineConfig, run_backtest


class FlashEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strategy = load_config()

    def config(self, **changes):
        value = dict(self.strategy["execution"])
        value.update(changes)
        return EngineConfig.from_dict(value)

    def dataset(self, bars=None, candidates=None, events=None, start="2024-01-01", periods=25):
        dates = pd.bdate_range(start, periods=periods)
        bars = bars or {"600000.SH": {}}
        codes = sorted(set(bars) | {"000001.SZ"})
        rows = []
        for code in codes:
            for index, date in enumerate(dates):
                custom = bars.get(code, {}).get(index)
                if custom is None:
                    if code in bars and index in bars[code]:
                        continue
                    custom = {}
                price = custom.get("price", 10.0)
                factor = custom.get("factor", 1.0)
                row = {
                    "thscode": code, "date": date,
                    "open": custom.get("open", price),
                    "high": custom.get("high", price),
                    "low": custom.get("low", price),
                    "close": custom.get("close", price),
                    "volume": custom.get("volume", 1_000_000),
                    "reference_close": custom.get("reference_close", np.nan),
                    "previous_actual_close": custom.get("previous_actual_close", np.nan),
                    "backward_factor": factor,
                    "factor_only_adjustment": custom.get("factor_only_adjustment", False),
                }
                for name in ["open", "high", "low", "close"]:
                    row[f"hfq_{name}"] = custom.get(f"hfq_{name}", row[name] * factor)
                rows.append(row)
        candidate_rows = []
        for code, signal_index, rank in candidates or [("600000.SH", 0, 1)]:
            candidate_rows.append({
                "thscode": code, "date": dates[signal_index], "rank": rank,
                "score": float(rank), "amount20": 100_000_000,
            })
        event_rows = []
        for code, date_index, dividend, bonus in events or []:
            event_rows.append({
                "thscode": code, "ex_date": dates[date_index],
                "dividend_per_share": dividend, "per_share_bonus": bonus,
                "rights_ratio": 0.0, "rights_price": 0.0,
            })
        return data_from_frames(
            pd.DataFrame(rows), pd.DataFrame(candidate_rows), pd.DataFrame(event_rows), sessions=dates
        )

    def test_high_price_fee_inclusive_budget(self):
        data = self.dataset({"600000.SH": {i: {"price": 101.71} for i in range(25)}})
        small = run_backtest(data, self.config(initial_capital=20_000))
        self.assertEqual(len(small.trades), 0)
        self.assertTrue((small.daily_equity["cash"] == 20_000).all())
        self.assertEqual(small.orders.iloc[0]["status_reason"], "slot_budget_including_fees")
        larger = run_backtest(data, self.config(initial_capital=50_000))
        self.assertEqual(larger.orders.iloc[0]["shares"], 200)
        self.assertTrue((larger.daily_equity["cash"] >= 0).all())
        filled = larger.orders[larger.orders["status"] == "filled"].iloc[0]
        self.assertLessEqual(-filled["cash_delta"], filled["order_budget"] + 1e-9)

    def test_t_plus_one_profit_target(self):
        data = self.dataset({"600000.SH": {
            3: {"open": 10, "high": 12.5, "low": 9.9, "close": 11},
            4: {"open": 10.5, "high": 12.2, "low": 10.4, "close": 12},
        }})
        result = run_backtest(data, self.config(limit_rate=0.30))
        trade = result.trades.iloc[0]
        self.assertEqual(trade["buy_date"], data.dates[3])
        self.assertEqual(trade["sell_date"], data.dates[4])
        self.assertEqual(trade["exit_reason"], "profit_target")
        self.assertEqual(trade["sell_price"], 11.98)
        self.assertGreater(trade["sell_date"], trade["buy_date"])

    def test_missing_bar_defers_time_exit(self):
        data = self.dataset({"600000.SH": {4: None}})
        result = run_backtest(data, self.config(initial_capital=2200, holding_sessions=2))
        deferred = result.orders[(result.orders["side"] == "sell") & (result.orders["status"] == "deferred")]
        self.assertEqual(deferred.iloc[0]["status_reason"], "no_bar")
        self.assertEqual(deferred.iloc[0]["date"], data.dates[4])
        self.assertEqual(result.trades.iloc[0]["sell_date"], data.dates[5])
        stale = result.positions_daily[result.positions_daily["date"] == data.dates[4]]
        self.assertTrue(stale.iloc[0]["stale_valuation"])

    def test_one_price_limit_down_defers_time_exit(self):
        data = self.dataset({"600000.SH": {
            4: {"price": 9.0},
            5: {"open": 9.2, "high": 9.4, "low": 9.1, "close": 9.3},
        }})
        result = run_backtest(data, self.config(initial_capital=2200, holding_sessions=2))
        deferred = result.orders[(result.orders["side"] == "sell") & (result.orders["status"] == "deferred")]
        self.assertEqual(deferred.iloc[0]["status_reason"], "close_limit_down")
        self.assertEqual(result.trades.iloc[0]["sell_date"], data.dates[5])
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "max_holding")

    def test_explicit_bonus_dividend_and_tax(self):
        data = self.dataset(
            {"600000.SH": {
                4: {"price": 6.6, "factor": 10 / 6.6, "reference_close": 6.6},
                5: {"price": 6.6, "factor": 10 / 6.6},
            }},
            events=[("600000.SH", 4, 0.1, 0.5)],
        )
        result = run_backtest(data, self.config(initial_capital=2200, holding_sessions=2))
        trade = result.trades.iloc[0]
        self.assertEqual(trade["buy_shares"], 100)
        self.assertEqual(trade["bonus_shares"], 50)
        self.assertEqual(trade["sell_shares"], 150)
        self.assertAlmostEqual(trade["dividend_received_gross"], 10)
        self.assertAlmostEqual(trade["dividend_tax"], 2)
        deferred = result.orders[(result.orders["side"] == "sell") & (result.orders["status"] == "deferred")]
        self.assertEqual(deferred.iloc[0]["status_reason"], "shares_not_sellable")
        self.assertEqual(len(result.corporate_actions), 1)
        self.assertTrue((result.daily_equity["cash"] >= 0).all())

    def test_factor_only_receivable(self):
        data = self.dataset({"600000.SH": {
            4: {"price": 9.9, "factor": 10 / 9.9, "reference_close": 9.9,
                "previous_actual_close": 10.0, "factor_only_adjustment": True},
        }})
        result = run_backtest(data, self.config(initial_capital=2200, holding_sessions=2))
        event = result.corporate_actions.iloc[0]
        self.assertEqual(event["event_type"], "factor_only_cash_equivalent")
        self.assertAlmostEqual(event["dividend_added"], 10)
        self.assertAlmostEqual(result.trades.iloc[0]["dividend_tax"], 2)
        self.assertLess(result.daily_equity["receivable_identity_error"].abs().max(), 1e-8)

    def test_historical_fee_schedule(self):
        data = self.dataset({"600000.SH": {
            3: {"open": 10, "high": 11, "low": 10, "close": 11},
            4: {"open": 10.5, "high": 12.05, "low": 10.4, "close": 12},
        }}, start="2022-01-03")
        result = run_backtest(data, self.config(initial_capital=2200))
        trade = result.trades.iloc[0]
        self.assertAlmostEqual(trade["buy_transfer_fee"], 1001 * 0.00002)
        self.assertAlmostEqual(trade["sell_transfer_fee"], 1198 * 0.00002)
        self.assertAlmostEqual(trade["sell_stamp_duty"], 1198 * 0.001)
        self.assertAlmostEqual(trade["buy_commission"], 5)
        self.assertAlmostEqual(trade["sell_commission"], 5)

    def test_reference_close_controls_limit(self):
        data = self.dataset({"600000.SH": {
            3: {"price": 10.6, "reference_close": 9.5, "previous_actual_close": 10.0},
        }})
        result = run_backtest(data, self.config())
        self.assertEqual(result.orders.iloc[0]["status_reason"], "open_limit_up")
        self.assertEqual(len(result.trades), 0)

    def test_rejected_candidate_is_not_replaced(self):
        data = self.dataset(
            {
                "600000.SH": {3: {"price": 11, "reference_close": 10}},
                "600001.SH": {},
                "600002.SH": {},
            },
            candidates=[("600000.SH", 0, 1), ("600001.SH", 0, 2), ("600002.SH", 0, 3)],
        )
        result = run_backtest(data, self.config(max_positions=2))
        buys = result.orders[(result.orders["side"] == "buy") & (result.orders["status"] == "filled")]
        self.assertEqual(buys["thscode"].tolist(), ["600001.SH"])
        self.assertFalse((buys["thscode"] == "600002.SH").any())

    def test_same_open_sale_proceeds_are_not_reused(self):
        data = self.dataset(
            {
                "600000.SH": {
                    i: {"price": 140} for i in range(25)
                } | {2: {"price": 126}, 3: {"price": 126.5}},
                "600001.SH": {i: {"price": 200} for i in range(25)},
            },
            candidates=[("600000.SH", 0, 1), ("600001.SH", 2, 1)],
        )
        result = run_backtest(data, self.config(
            initial_capital=30_000, max_positions=1, entry_lag_sessions=1, holding_sessions=2,
        ))
        b = result.orders[(result.orders["thscode"] == "600001.SH") & (result.orders["side"] == "buy")].iloc[0]
        self.assertEqual(b["status"], "rejected")
        day = result.daily_equity[result.daily_equity["date"] == data.dates[3]].iloc[0]
        self.assertGreater(day["same_open_sale_proceeds"], 0)
        self.assertLessEqual(b["order_budget"], day["cash_start"] + 1e-9)
        self.assertTrue((result.daily_equity["cash"] >= 0).all())

    def test_next_open_profit_target(self):
        data = self.dataset({"600000.SH": {
            4: {"open": 10.5, "high": 12.2, "low": 10.4, "close": 12},
            5: {"price": 11.8},
        }})
        result = run_backtest(data, self.config(take_profit_mode="next_open", limit_rate=0.30))
        self.assertEqual(result.trades.iloc[0]["sell_date"], data.dates[5])
        self.assertTrue((result.orders["status"] == "pending").any())

    def test_open_positions_are_not_fake_trades(self):
        result = run_backtest(self.dataset(candidates=[("600000.SH", 20, 1)]), self.config())
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(len(result.open_positions), 1)
        self.assertEqual(result.metrics["ending_open_positions"], 1)
        self.assertTrue(result.metrics["final_reconciliation_passed"])

    def test_participation_cap(self):
        data = self.dataset()
        data.candidates[0] = [replace(data.candidates[0][0], amount20=10_000_000)]
        result = run_backtest(data, self.config(max_participation=0.001))
        order = result.orders.iloc[0]
        self.assertEqual(order["shares"], 900)
        self.assertLessEqual(order["gross"], order["amount20"] * 0.001)
        self.assertTrue((result.daily_equity["cash"] >= 0).all())

    def test_deterministic_account(self):
        data = self.dataset()
        first = run_backtest(data, self.config(initial_capital=20_000))
        second = run_backtest(data, self.config(initial_capital=20_000))
        pd.testing.assert_frame_equal(first.daily_equity, second.daily_equity)
        pd.testing.assert_frame_equal(first.trades, second.trades)
        pd.testing.assert_frame_equal(first.orders, second.orders)


if __name__ == "__main__":
    unittest.main(verbosity=2)
