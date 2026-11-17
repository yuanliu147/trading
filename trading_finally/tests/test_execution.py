import unittest
from dataclasses import replace
from unittest.mock import patch

from trend_system.config import ResearchConfig
from trend_system.execution import (
    Fill,
    Rejection,
    open_fill,
    price_limits,
    size_buy,
    transaction_costs,
)


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig(
            hithink_db_path="unused",
            baostock_db_path="unused",
        )

    def test_price_limits_use_main_board_and_st_rates(self):
        self.assertEqual(price_limits(10.0, False), (11.0, 9.0))
        self.assertEqual(price_limits(10.0, True), (10.5, 9.5))

    def test_price_limits_round_half_up_to_cent(self):
        self.assertEqual(price_limits(9.99, False), (10.99, 8.99))

    def test_buy_commission_has_five_yuan_minimum(self):
        costs = transaction_costs("2024-01-02", 4000.0, "buy")

        self.assertEqual(costs["commission"], 5.0)
        self.assertEqual(costs["stamp_duty"], 0.0)
        self.assertEqual(costs["transfer_fee"], 0.04)
        self.assertEqual(costs["total"], 5.04)

    def test_sell_stamp_duty_changes_on_2023_08_28(self):
        before = transaction_costs("2023-08-27", 4000.0, "sell")
        on_change = transaction_costs("2023-08-28", 4000.0, "sell")

        self.assertEqual(before["stamp_duty"], 4.0)
        self.assertEqual(before["total"], 9.04)
        self.assertEqual(on_change["stamp_duty"], 2.0)
        self.assertEqual(on_change["total"], 7.04)

    def test_transfer_fee_changes_on_2022_04_29(self):
        before = transaction_costs("2022-04-28", 100000.0, "buy")
        on_change = transaction_costs("2022-04-29", 100000.0, "buy")

        self.assertEqual(before["transfer_fee"], 2.0)
        self.assertEqual(before["total"], 27.0)
        self.assertEqual(on_change["transfer_fee"], 1.0)
        self.assertEqual(on_change["total"], 26.0)

    def test_cost_multiplier_doubles_minimum_commission_and_fees(self):
        costs = transaction_costs("2024-01-02", 4000.0, "buy", multiplier=2.0)

        self.assertEqual(costs["commission"], 10.0)
        self.assertEqual(costs["transfer_fee"], 0.08)
        self.assertEqual(costs["total"], 10.08)

    def test_double_cost_sizing_and_fill_include_doubled_minimum_and_20bp_slippage(self):
        run_config = replace(
            self.config,
            commission_rate=self.config.commission_rate * 2,
            min_commission=self.config.min_commission * 2,
            slippage_bps=20,
        )
        original_costs = transaction_costs

        def doubled_costs(date, amount, side, multiplier=1.0):
            return original_costs(date, amount, side, multiplier=2.0 * multiplier)

        bar = self._bar()
        order = {
            key: bar[key]
            for key in ("date", "open", "high", "low", "prev_close", "is_st")
        }
        with patch("trend_system.execution.transaction_costs", doubled_costs):
            quantity = size_buy(order, 20000.0, 4018.07, 0.0, run_config)
            fill = open_fill(
                {"side": "buy", "quantity": quantity}, bar, run_config
            )

        self.assertEqual(quantity, 300)
        self.assertIsInstance(fill, Fill)
        self.assertEqual(fill.price, 10.02)
        self.assertEqual(fill.amount, 3006.0)
        self.assertEqual(fill.costs["commission"], 10.0)
        self.assertEqual(fill.costs["transfer_fee"], 0.06)
        self.assertEqual(fill.costs["total"], 10.06)
        self.assertEqual(fill.cash_delta, -3016.06)
        self.assertLessEqual(-fill.cash_delta, 4018.07)

    def test_size_buy_uses_unslipped_open_for_single_stock_cap(self):
        order = {"date": "2024-01-02", "open": 10.0}

        quantity = size_buy(order, 20000.0, 20000.0, 0.0, self.config)
        fill = open_fill(
            {"side": "buy", "quantity": quantity}, self._bar(), self.config
        )

        self.assertEqual(quantity, 400)
        self.assertEqual(quantity % 100, 0)
        self.assertEqual(quantity * order["open"], 4000.0)
        self.assertLessEqual(quantity * order["open"], 20000.0 * self.config.max_weight)
        self.assertIsInstance(fill, Fill)
        self.assertEqual(fill.price, 10.01)
        self.assertEqual(fill.amount, 4004.0)
        self.assertEqual(fill.costs["total"], 5.04)
        self.assertEqual(fill.cash_delta, -4009.04)
        self.assertGreaterEqual(20000.0 + fill.cash_delta, 0.0)

    def test_size_buy_reduces_one_lot_when_rounded_cash_debit_exceeds_cash(self):
        order = {"date": "2024-01-02", "open": 10.0}

        quantity = size_buy(order, 20000.0, 4005.0, 0.0, self.config)

        self.assertEqual(quantity, 300)
        fill = open_fill(
            {"side": "buy", "quantity": quantity}, self._bar(), self.config
        )
        self.assertIsInstance(fill, Fill)
        self.assertLessEqual(-fill.cash_delta, 4005.0)

    def test_size_buy_uses_same_clamped_fill_price_and_cash_delta_as_open_fill(self):
        bar = self._bar(high=10.0)
        order = {
            key: bar[key]
            for key in ("date", "open", "high", "low", "prev_close", "is_st")
        }

        quantity = size_buy(order, 20000.0, 4006.0, 0.0, self.config)
        fill = open_fill(
            {"side": "buy", "quantity": quantity}, bar, self.config
        )

        self.assertEqual(quantity, 400)
        self.assertIsInstance(fill, Fill)
        self.assertEqual(fill.price, 10.0)
        self.assertEqual(fill.cash_delta, -4005.04)
        self.assertLessEqual(-fill.cash_delta, 4006.0)
        self.assertEqual(size_buy(order, 20000.0, 4005.03, 0.0, self.config), 300)

    def test_size_buy_respects_remaining_total_exposure(self):
        order = {"date": "2024-01-02", "open": 10.0}

        quantity = size_buy(order, 20000.0, 10000.0, 10000.0, self.config)

        self.assertEqual(quantity, 200)
        self.assertEqual(10000.0 + quantity * order["open"], 12000.0)

    def test_open_sell_fill_uses_adverse_slippage_and_exact_cash_credit(self):
        fill = open_fill(
            {"side": "sell", "quantity": 400}, self._bar(), self.config
        )

        self.assertIsInstance(fill, Fill)
        self.assertEqual(fill.price, 9.99)
        self.assertEqual(fill.quantity, 400)
        self.assertEqual(fill.amount, 3996.0)
        self.assertEqual(fill.costs["total"], 7.04)
        self.assertEqual(fill.cash_delta, 3988.96)

    def test_slipped_fill_is_clamped_to_daily_price_range(self):
        buy_bar = self._bar(high=10.0)
        sell_bar = self._bar(low=10.0)

        buy = open_fill({"side": "buy", "quantity": 100}, buy_bar, self.config)
        sell = open_fill({"side": "sell", "quantity": 100}, sell_bar, self.config)

        self.assertIsInstance(buy, Fill)
        self.assertIsInstance(sell, Fill)
        self.assertEqual(buy.price, 10.0)
        self.assertEqual(sell.price, 10.0)

    def test_buy_at_open_limit_up_is_cancelled(self):
        bar = self._bar(open=11.0, high=11.0, low=10.5)

        result = open_fill({"side": "buy", "quantity": 100}, bar, self.config)

        self.assertIsInstance(result, Rejection)
        self.assertEqual(result.reason, "buy_limit_up")
        self.assertEqual(result.disposition, "cancel")

    def test_sell_at_open_limit_down_is_kept(self):
        bar = self._bar(open=9.0, high=9.5, low=9.0)

        result = open_fill({"side": "sell", "quantity": 100}, bar, self.config)

        self.assertIsInstance(result, Rejection)
        self.assertEqual(result.reason, "sell_limit_down")
        self.assertEqual(result.disposition, "keep")

    def test_suspension_has_side_specific_disposition(self):
        bar = self._bar(trade_status=0)

        buy = open_fill({"side": "buy", "quantity": 100}, bar, self.config)
        sell = open_fill({"side": "sell", "quantity": 100}, bar, self.config)

        self.assertEqual((buy.reason, buy.disposition), ("suspended", "cancel"))
        self.assertEqual((sell.reason, sell.disposition), ("suspended", "keep"))

    def test_invalid_data_is_rejected_explicitly(self):
        bar = self._bar(data_valid=False)

        result = open_fill({"side": "buy", "quantity": 100}, bar, self.config)

        self.assertIsInstance(result, Rejection)
        self.assertEqual(result.reason, "invalid_bar")
        self.assertEqual(result.disposition, "cancel")

    def test_execution_conflict_is_rejected_explicitly(self):
        bar = self._bar(execution_conflict=True)

        result = open_fill({"side": "sell", "quantity": 100}, bar, self.config)

        self.assertIsInstance(result, Rejection)
        self.assertEqual(result.reason, "execution_conflict")
        self.assertEqual(result.disposition, "keep")

    def test_open_outside_legal_or_daily_range_is_rejected(self):
        for bar in (
            self._bar(open=11.01, high=11.01),
            self._bar(open=10.1, high=10.0),
        ):
            with self.subTest(bar=bar):
                result = open_fill(
                    {"side": "buy", "quantity": 100}, bar, self.config
                )
                self.assertIsInstance(result, Rejection)
                self.assertEqual(result.reason, "price_out_of_bounds")
                self.assertEqual(result.disposition, "cancel")

    def test_invalid_order_is_rejected_before_pricing(self):
        for order in (
            {"side": "buy", "quantity": 0},
            {"side": "buy", "quantity": 50},
            {"side": "hold", "quantity": 100},
        ):
            with self.subTest(order=order):
                result = open_fill(order, self._bar(), self.config)
                self.assertIsInstance(result, Rejection)
                self.assertEqual(result.reason, "invalid_order")

    @staticmethod
    def _bar(**changes):
        bar = {
            "date": "2024-01-02",
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "prev_close": 10.0,
            "trade_status": 1,
            "is_st": False,
            "data_valid": True,
            "execution_conflict": False,
        }
        bar.update(changes)
        return bar


if __name__ == "__main__":
    unittest.main()
