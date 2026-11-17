import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from trend_system.broker import BacktestResult, run_backtest
from trend_system.config import ResearchConfig


DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
CODE_A = "600000.SH"
CODE_B = "000001.SZ"


class BrokerTest(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig(
            hithink_db_path="unused",
            baostock_db_path="unused",
            start_date=DATES[0],
            end_date=DATES[-1],
        )

    def test_entry_executes_next_open_and_bought_shares_are_not_sellable_same_day(self):
        panel = self._panel(DATES[:3], [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)

        result = self._run(panel, signals)

        self.assertEqual(result.trades["side"].tolist(), ["buy", "sell"])
        self.assertEqual(result.trades["date"].tolist(), DATES[1:3])
        bought = result.daily_positions.loc[
            (result.daily_positions["date"] == DATES[1])
            & (result.daily_positions["code"] == CODE_A)
        ].iloc[0]
        self.assertEqual(bought["available_quantity"], 0)
        self.assertEqual(len(result.round_trips), 1)
        self.assertEqual(
            result.round_trips.iloc[0]["quantity"],
            result.trades.iloc[0]["quantity"],
        )
        self.assertAlmostEqual(
            result.round_trips.iloc[0]["net_pnl"],
            result.trades["cash_delta"].sum(),
        )

    def test_unfilled_buy_is_cancelled_and_requires_a_new_entry_signal(self):
        panel = self._panel(DATES, [CODE_A])
        self._bar(panel, DATES[1], CODE_A, open=11.0, high=11.0, low=10.5, close=11.0)
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[2], CODE_A, entry=True)

        result = self._run(panel, signals)

        self.assertEqual(result.trades["date"].tolist(), [DATES[3]])
        cancelled = result.orders.loc[
            (result.orders["date"] == DATES[1]) & (result.orders["side"] == "buy")
        ].iloc[0]
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["reason"], "buy_limit_up")
        self.assertFalse(
            ((result.orders["date"] == DATES[2]) & (result.orders["side"] == "buy")).any()
        )

    def test_sell_order_rolls_through_limit_down_days_and_uses_current_open(self):
        panel = self._panel(DATES + ["2024-01-08"], [CODE_A])
        for day in DATES[2:4]:
            self._bar(panel, day, CODE_A, open=9.0, high=9.5, low=9.0, close=9.0)
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)

        result = self._run(panel, signals)

        sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
        self.assertEqual(sell["date"], "2024-01-08")
        self.assertAlmostEqual(sell["price"], 10.0 * 0.999)
        rejected = result.orders.loc[
            (result.orders["side"] == "sell")
            & (result.orders["date"].isin(DATES[2:4]))
        ]
        self.assertEqual(rejected["status"].tolist(), ["pending", "pending"])
        self.assertEqual(rejected["reason"].tolist(), ["sell_limit_down"] * 2)

    def test_suspended_position_uses_last_actual_close_and_marks_stale(self):
        panel = self._panel(DATES[:3], [CODE_A])
        self._bar(panel, DATES[2], CODE_A, trade_status=0, close=12.0)
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)

        result = self._run(panel, signals)

        account = result.daily_account.set_index("date").loc[DATES[2]]
        quantity = result.daily_positions.iloc[-1]["quantity"]
        self.assertTrue(account["stale_valuation"])
        self.assertAlmostEqual(account["market_value"], quantity * 10.0)
        self.assertGreater(account["market_value"], 0)
        self.assertGreater(account["equity"], account["cash"])

    def test_missing_bar_uses_stale_position_value_not_cash(self):
        panel = pd.concat(
            [
                self._panel(DATES[:2], [CODE_A]),
                self._panel([DATES[2]], [CODE_B]),
            ],
            ignore_index=True,
        )
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)

        result = self._run(panel, signals)

        account = result.daily_account.set_index("date").loc[DATES[2]]
        quantity = result.daily_positions.iloc[-1]["quantity"]
        self.assertTrue(account["stale_valuation"])
        self.assertAlmostEqual(account["market_value"], quantity * 10.0)

    def test_pending_exit_keeps_its_slot_for_next_day_buy(self):
        panel = self._panel(DATES[:3], [CODE_A, CODE_B])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)
        self._signal(signals, DATES[1], CODE_B, entry=True)

        result = self._run(panel, signals, config=replace(self.config, slots=1))

        self.assertEqual(
            result.trades.loc[result.trades["side"] == "buy", "code"].tolist(),
            [CODE_A],
        )
        self.assertFalse((result.trades["code"] == CODE_B).any())

    def test_pending_exit_proceeds_do_not_fund_next_day_buy(self):
        panel = self._panel(DATES[:3], [CODE_A, CODE_B])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)
        self._signal(signals, DATES[1], CODE_B, entry=True)
        config = replace(
            self.config,
            initial_cash=10000,
            slots=2,
            max_weight=1.0,
            max_exposure=1.0,
        )

        result = self._run(panel, signals, config=config)

        self.assertEqual(result.trades.loc[result.trades["side"] == "sell"].iloc[0]["date"], DATES[2])
        self.assertFalse((result.trades["code"] == CODE_B).any())

    def test_execution_integrity_has_independent_key_cash_lot_price_and_equity_evidence(self):
        from trend_system.broker import execution_integrity

        panel = self._panel(DATES[:3], [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)
        result = run_backtest(panel, signals, self._actions(), self.config)

        evidence = execution_integrity(panel, signals, result, self.config)
        expected_checks = {
            "key_set_and_window",
            "cash_nonnegative",
            "whole_lot_buys",
            "legal_fill_prices",
            "equity_identity",
            "next_open_after_signal",
        }
        self.assertTrue(expected_checks.issubset(evidence))
        self.assertTrue(all(evidence[name] for name in expected_checks))
        self.assertTrue(evidence["all_passed"])

        account = result.daily_account.copy()
        account.loc[1, "cash"] = -0.01
        negative_cash = execution_integrity(
            panel,
            signals,
            BacktestResult(
                trades=result.trades,
                orders=result.orders,
                round_trips=result.round_trips,
                daily_account=account,
                daily_positions=result.daily_positions,
                action_log=result.action_log,
            ),
            self.config,
        )
        self.assertFalse(negative_cash["cash_nonnegative"])
        self.assertFalse(negative_cash["equity_identity"])
        self.assertTrue(negative_cash["next_open_after_signal"])
        self.assertFalse(negative_cash["all_passed"])

        missing_key = execution_integrity(panel, signals.iloc[:-1], result, self.config)
        self.assertFalse(missing_key["key_set_and_window"])
        self.assertTrue(missing_key["next_open_after_signal"])

        trades = result.trades.copy()
        trades.loc[trades["side"] == "buy", "quantity"] = 50
        bad_lot = execution_integrity(
            panel,
            signals,
            BacktestResult(
                trades=trades,
                orders=result.orders,
                round_trips=result.round_trips,
                daily_account=result.daily_account,
                daily_positions=result.daily_positions,
                action_log=result.action_log,
            ),
            self.config,
        )
        self.assertFalse(bad_lot["whole_lot_buys"])
        self.assertFalse(bad_lot["legal_fill_prices"])

    def test_execution_integrity_rejects_delayed_buy_non_open_fill_and_unproven_delayed_sell(self):
        from trend_system.broker import execution_integrity

        panel = self._panel(DATES, [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)
        result = run_backtest(panel, signals, self._actions(), self.config)
        self.assertTrue(execution_integrity(panel, signals, result, self.config)["all_passed"])

        with self.subTest("delayed_buy"):
            delayed_trades = result.trades.copy()
            delayed_trades.loc[delayed_trades["side"] == "buy", "date"] = DATES[2]
            delayed_buy = execution_integrity(
                panel, signals, replace(result, trades=delayed_trades), self.config
            )
            self.assertFalse(delayed_buy["next_open_after_signal"])
            self.assertFalse(delayed_buy["all_passed"])

        with self.subTest("non_open_fill"):
            non_open_trades = result.trades.copy()
            non_open_trades.loc[non_open_trades["side"] == "buy", "price"] = 10.02
            non_open = execution_integrity(
                panel, signals, replace(result, trades=non_open_trades), self.config
            )
            self.assertFalse(non_open["legal_fill_prices"])
            self.assertFalse(non_open["all_passed"])

        with self.subTest("delayed_sell_without_pending"):
            delayed_sell_trades = result.trades.copy()
            delayed_sell_trades.loc[delayed_sell_trades["side"] == "sell", "date"] = DATES[3]
            delayed_sell_orders = result.orders.copy()
            delayed_sell_orders.loc[
                (delayed_sell_orders["side"] == "sell")
                & (delayed_sell_orders["status"] == "filled"),
                "date",
            ] = DATES[3]
            delayed_sell = execution_integrity(
                panel,
                signals,
                replace(result, trades=delayed_sell_trades, orders=delayed_sell_orders),
                self.config,
            )
            self.assertTrue(delayed_sell["legal_fill_prices"])
            self.assertFalse(delayed_sell["next_open_after_signal"])
            self.assertFalse(delayed_sell["all_passed"])

    def test_execution_integrity_accepts_real_pending_delay_evidence(self):
        from trend_system.broker import execution_integrity

        for scenario in ("limit_down", "suspended", "missing_bar"):
            with self.subTest(scenario=scenario):
                dates = DATES + ["2024-01-08"] if scenario == "limit_down" else DATES
                if scenario == "missing_bar":
                    panel = pd.concat(
                        [
                            self._panel([DATES[0], DATES[1], DATES[3]], [CODE_A]),
                            self._panel([DATES[2]], [CODE_B]),
                        ],
                        ignore_index=True,
                    )
                else:
                    panel = self._panel(dates, [CODE_A])
                if scenario == "limit_down":
                    for day in DATES[2:4]:
                        self._bar(panel, day, CODE_A, open=9.0, high=9.5, low=9.0, close=9.0)
                elif scenario == "suspended":
                    self._bar(panel, DATES[2], CODE_A, trade_status=0)
                signals = self._signals(panel)
                self._signal(signals, DATES[0], CODE_A, entry=True)
                self._signal(signals, DATES[1], CODE_A, exit=True)
                config = replace(self.config, end_date=dates[-1])
                result = run_backtest(panel, signals, self._actions(), config)
                sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
                self.assertGreater(sell["date"], DATES[2] if scenario == "limit_down" else DATES[2])
                pending = result.orders.loc[result.orders["status"] == "pending"]
                self.assertFalse(pending.empty)
                evidence = execution_integrity(panel, signals, result, config)
                self.assertTrue(evidence["next_open_after_signal"])
                self.assertTrue(evidence["legal_fill_prices"])
                self.assertTrue(evidence["all_passed"])

    def test_account_identities_hold_on_every_day(self):
        panel = self._panel(DATES[:3], [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)

        result = run_backtest(panel, signals, self._actions(), self.config)

        for account in result.daily_account.itertuples(index=False):
            self.assertGreaterEqual(account.cash, 0)
            self.assertAlmostEqual(
                account.equity,
                account.cash + account.receivables + account.market_value,
            )
            self.assertGreaterEqual(account.market_value, 0)
        self.assertTrue((result.daily_positions["quantity"] >= 0).all())
        self.assertTrue((result.daily_positions["available_quantity"] >= 0).all())

    def test_dividend_settles_at_close_but_does_not_fund_frozen_buy(self):
        panel = self._panel(DATES[:3], [CODE_A, CODE_B])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_B, entry=True)
        actions = self._actions(
            [{"date": DATES[2], "code": CODE_A, "dividend_per_share": 1.0}]
        )
        config = replace(
            self.config,
            initial_cash=10000,
            slots=2,
            max_weight=1.0,
            max_exposure=1.0,
        )

        result = self._run(panel, signals, actions, config)

        previous = result.daily_account.set_index("date").loc[DATES[1]]
        current = result.daily_account.set_index("date").loc[DATES[2]]
        dividend_quantity = result.daily_positions.loc[
            (result.daily_positions["date"] == DATES[1])
            & (result.daily_positions["code"] == CODE_A),
            "quantity",
        ].iloc[0]
        self.assertAlmostEqual(current["cash"], previous["cash"] + dividend_quantity)
        self.assertEqual(current["receivables"], 0)
        self.assertFalse((result.trades["code"] == CODE_B).any())
        logged = set(result.action_log["action"])
        self.assertIn("dividend_receivable", logged)
        self.assertIn("dividend_settled", logged)

    def test_bonus_shares_become_sellable_on_next_trading_day(self):
        panel = self._panel(DATES, [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)
        actions = self._actions(
            [{"date": DATES[2], "code": CODE_A, "per_share_bonus": 0.5}]
        )

        result = self._run(panel, signals, actions)

        sells = result.trades.loc[result.trades["side"] == "sell"]
        self.assertEqual(sells["date"].tolist(), [DATES[2], DATES[3]])
        self.assertEqual(sells["quantity"].tolist(), [400, 200])
        remaining = result.daily_positions.loc[
            result.daily_positions["date"] == DATES[2]
        ].iloc[0]
        self.assertEqual(remaining["quantity"], 200)
        self.assertEqual(remaining["available_quantity"], 0)
        round_trip = result.round_trips.iloc[0]
        self.assertEqual(round_trip["entry_quantity"], 400)
        self.assertEqual(round_trip["exit_quantity"], 600)
        self.assertAlmostEqual(round_trip["net_pnl"], result.trades["cash_delta"].sum())

    def test_rights_issue_is_not_automatically_participated(self):
        panel = self._panel(DATES[:3], [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        actions = self._actions(
            [{
                "date": DATES[2],
                "code": CODE_A,
                "rights_ratio": 0.2,
                "rights_price": 8.0,
            }]
        )

        result = self._run(panel, signals, actions)

        accounts = result.daily_account.set_index("date")
        self.assertEqual(accounts.loc[DATES[2], "cash"], accounts.loc[DATES[1], "cash"])
        self.assertEqual(result.daily_positions.iloc[-1]["quantity"], 400)
        self.assertIn("rights_not_participated", set(result.action_log["action"]))

    def test_null_corporate_action_fields_are_treated_as_zero(self):
        panel = self._panel(DATES[:3], [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        actions = self._actions(
            [{
                "date": DATES[2],
                "code": CODE_A,
                "dividend_per_share": 1.0,
                "per_share_bonus": float("nan"),
                "rights_ratio": float("nan"),
                "rights_price": float("nan"),
            }]
        )
        actions["per_share_bonus"] = float("nan")
        actions["rights_ratio"] = float("nan")

        result = self._run(panel, signals, actions)

        self.assertEqual(result.daily_positions.iloc[-1]["quantity"], 400)
        self.assertIn("dividend_settled", set(result.action_log["action"]))

    def test_delisting_writeoff_is_after_out_date_and_never_an_early_sale(self):
        panel = pd.concat(
            [
                self._panel(DATES[:3], [CODE_A]),
                self._panel([DATES[3]], [CODE_B]),
            ],
            ignore_index=True,
        )
        panel.loc[panel["code"] == CODE_A, "out_date"] = DATES[2]
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)

        result = self._run(panel, signals)

        self.assertEqual(result.trades["side"].tolist(), ["buy"])
        writeoff = result.action_log.loc[
            result.action_log["action"] == "delisting_zero_residual"
        ].iloc[0]
        self.assertEqual(writeoff["date"], DATES[3])
        last = result.daily_account.iloc[-1]
        self.assertEqual(last["market_value"], 0)
        self.assertEqual(last["equity"], last["cash"] + last["receivables"])
        self.assertEqual(result.round_trips.iloc[0]["exit_reason"], "delisting_zero_residual")

    def test_st_exit_is_frozen_for_next_day(self):
        panel = self._panel(DATES[:3], [CODE_A])
        self._bar(panel, DATES[1], CODE_A, is_st=1)
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, st=True)

        result = self._run(panel, signals)

        self.assertEqual(result.trades["side"].tolist(), ["buy", "sell"])
        self.assertEqual(result.trades["date"].tolist(), DATES[1:3])
        self.assertEqual(result.round_trips.iloc[0]["exit_reason"], "st_exit")

    def test_terminal_delisting_suspension_is_stale_then_written_off_next_day(self):
        panel = pd.concat(
            [
                self._panel(DATES[:3], [CODE_A]),
                self._panel([DATES[3]], [CODE_B]),
            ],
            ignore_index=True,
        )
        panel.loc[panel["code"] == CODE_A, "out_date"] = DATES[2]
        terminal = (panel["code"] == CODE_A) & (panel["date"] == DATES[2])
        panel.loc[terminal, ["open", "high", "low", "close", "prev_close"]] = np.nan
        panel.loc[terminal, "trade_status"] = 0
        panel.loc[terminal, "data_valid"] = False
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)

        result = run_backtest(panel, signals, self._actions(), self.config)

        terminal_account = result.daily_account.set_index("date").loc[DATES[2]]
        terminal_position = result.daily_positions.loc[
            result.daily_positions["date"] == DATES[2]
        ].iloc[0]
        self.assertTrue(terminal_account["stale_valuation"])
        self.assertTrue(terminal_position["stale_valuation"])
        self.assertEqual(terminal_account["market_value"], terminal_position["market_value"])
        self.assertEqual(result.round_trips.iloc[0]["exit_reason"], "delisting_zero_residual")
        self.assertEqual(result.daily_account.iloc[-1]["market_value"], 0.0)

    def test_held_invalid_data_or_execution_conflict_aborts_with_diagnostic(self):
        for column, reason in (
            ("data_valid", "data_valid=False"),
            ("execution_conflict", "execution_conflict=True"),
        ):
            with self.subTest(column=column):
                panel = self._panel(DATES[:3], [CODE_A])
                self._bar(panel, DATES[2], CODE_A, **{column: False if column == "data_valid" else True})
                signals = self._signals(panel)
                self._signal(signals, DATES[0], CODE_A, entry=True)

                with self.assertRaises(RuntimeError) as caught:
                    run_backtest(panel, signals, self._actions(), self.config)

                message = str(caught.exception)
                self.assertIn(DATES[2], message)
                self.assertIn(CODE_A, message)
                self.assertIn(reason, message)

    def test_adjustment_conflict_is_logged_without_abort(self):
        panel = self._panel(DATES[:3], [CODE_A])
        self._bar(panel, DATES[2], CODE_A, adjustment_source_conflict=True)
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)

        result = self._run(panel, signals)

        logged = result.action_log.loc[
            result.action_log["action"] == "adjustment_source_conflict"
        ]
        self.assertEqual(logged[["date", "code"]].iloc[0].tolist(), [DATES[2], CODE_A])
        self.assertEqual(result.daily_positions.iloc[-1]["quantity"], 400)

    def test_end_of_panel_does_not_force_liquidation(self):
        panel = self._panel(DATES[:2], [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)

        result = self._run(panel, signals)

        self.assertEqual(result.trades["side"].tolist(), ["buy"])
        self.assertEqual(len(result.daily_positions), 1)
        self.assertGreater(result.daily_positions.iloc[0]["quantity"], 0)
        self.assertTrue(result.round_trips.empty)

    def test_entries_are_ranked_and_existing_positions_are_not_added(self):
        panel = self._panel(DATES[:3], [CODE_A, CODE_B])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True, rank=0.1)
        self._signal(signals, DATES[0], CODE_B, entry=True, rank=0.2)
        self._signal(signals, DATES[1], CODE_A, entry=True, rank=0.3)
        self._signal(signals, DATES[1], CODE_B, entry=True, rank=0.4)

        result = self._run(panel, signals, config=replace(self.config, slots=1))

        buys = result.trades.loc[result.trades["side"] == "buy"]
        self.assertEqual(buys[["date", "code"]].values.tolist(), [[DATES[1], CODE_B]])
        self.assertEqual(len(buys), 1)

    def test_failed_first_candidate_does_not_block_ordered_backfill(self):
        for scenario in ("limit_up", "missing", "no_size"):
            with self.subTest(scenario=scenario):
                panel = self._panel(DATES[:2], [CODE_A, CODE_B])
                if scenario == "limit_up":
                    self._bar(panel, DATES[1], CODE_A, open=11.0, high=11.0, low=10.5, close=11.0)
                elif scenario == "no_size":
                    self._bar(
                        panel, DATES[1], CODE_A, open=100.0, high=100.5,
                        low=99.5, close=100.0, prev_close=100.0,
                    )
                else:
                    panel = panel.loc[
                        ~((panel["date"] == DATES[1]) & (panel["code"] == CODE_A))
                    ].reset_index(drop=True)
                signals = self._signals(panel)
                self._signal(signals, DATES[0], CODE_A, entry=True, rank=0.2)
                self._signal(signals, DATES[0], CODE_B, entry=True, rank=0.1)

                result = self._run(panel, signals, config=replace(self.config, slots=1))

                buys = result.trades.loc[result.trades["side"] == "buy"]
                self.assertEqual(buys[["date", "code"]].values.tolist(), [[DATES[1], CODE_B]])
                first = result.orders.loc[result.orders["code"] == CODE_A].iloc[0]
                self.assertEqual(first["status"], "cancelled")

    def test_same_day_multiple_buys_deplete_frozen_cash_in_rank_order(self):
        code_c = "000002.SZ"
        codes = [CODE_A, CODE_B, code_c]
        panel = self._panel(DATES[:2], codes)
        for column in ("open", "high", "low", "close", "prev_close"):
            panel[column] = {"open": 1.0, "high": 1.05, "low": 0.95, "close": 1.0, "prev_close": 1.0}[column]
        signals = self._signals(panel)
        for rank, code in enumerate(codes, start=1):
            self._signal(signals, DATES[0], code, entry=True, rank=1.0 / rank)
        config = replace(
            self.config, initial_cash=250, slots=3, max_weight=0.5, max_exposure=1.0
        )

        result = self._run(panel, signals, config=config)

        buys = result.trades.loc[result.trades["side"] == "buy"]
        self.assertEqual(buys["code"].tolist(), [CODE_A, CODE_B])
        self.assertEqual(buys["quantity"].tolist(), [100, 100])
        cancelled = result.orders.loc[result.orders["code"] == code_c].iloc[0]
        self.assertEqual((cancelled["status"], cancelled["reason"]), ("cancelled", "no_size"))
        self.assertAlmostEqual(result.daily_account.iloc[-1]["cash"], 250 + buys["cash_delta"].sum())
        self.assertGreaterEqual(result.daily_account.iloc[-1]["cash"], 0)

    def test_negative_small_lot_sell_is_kept_when_it_would_overdraw_cash(self):
        panel = self._panel(DATES, [CODE_A])
        for column in ("open", "high", "low", "close", "prev_close"):
            panel[column] = {"open": 0.06, "high": 0.07, "low": 0.05, "close": 0.06, "prev_close": 0.06}[column]
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)
        actions = self._actions(
            [{"date": DATES[2], "code": CODE_A, "per_share_bonus": 0.5}]
        )
        config = replace(
            self.config, initial_cash=12, max_weight=0.5, max_exposure=1.0
        )

        result = self._run(panel, signals, actions, config)

        sells = result.trades.loc[result.trades["side"] == "sell"]
        self.assertEqual(sells[["date", "quantity"]].values.tolist(), [[DATES[2], 100]])
        attempt = result.orders.loc[
            (result.orders["date"] == DATES[3]) & (result.orders["side"] == "sell")
        ].iloc[0]
        self.assertEqual(attempt["status"], "pending")
        self.assertEqual(attempt["reason"], "insufficient_cash_for_sell")
        self.assertTrue((result.daily_account["cash"] >= 0).all())
        self.assertEqual(result.daily_positions.iloc[-1]["quantity"], 50)

    def test_negative_sell_reduces_cash_available_to_frozen_buy(self):
        panel = self._panel(DATES, [CODE_A, CODE_B])
        for column in ("open", "high", "low", "close", "prev_close"):
            panel[column] = {"open": 0.06, "high": 0.07, "low": 0.05, "close": 0.06, "prev_close": 0.06}[column]
        self._bar(
            panel, DATES[3], CODE_B, open=0.05, high=0.07, low=0.05,
            close=0.05, prev_close=0.06,
        )
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)
        self._signal(signals, DATES[2], CODE_B, entry=True)
        actions = self._actions(
            [{"date": DATES[2], "code": CODE_A, "per_share_bonus": 0.5}]
        )
        config = replace(
            self.config,
            initial_cash=20.52,
            slots=2,
            max_weight=0.5,
            max_exposure=1.0,
        )

        result = self._run(panel, signals, actions, config)

        self.assertFalse((result.trades["code"] == CODE_B).any())
        self.assertEqual(result.daily_account.iloc[-1]["cash"], 8.50)
        buy_attempt = result.orders.loc[result.orders["code"] == CODE_B].iloc[0]
        self.assertEqual((buy_attempt["status"], buy_attempt["reason"]), ("cancelled", "no_size"))
        self.assertTrue((result.daily_account["cash"] >= 0).all())

    def test_multi_position_abort_precedes_all_position_updates(self):
        panel = self._panel(DATES[:3], [CODE_A, CODE_B])
        self._bar(
            panel, DATES[2], CODE_A,
            adjustment_source_conflict=True, out_date=DATES[0],
        )
        self._bar(panel, DATES[2], CODE_B, execution_conflict=True)
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True, rank=0.2)
        self._signal(signals, DATES[0], CODE_B, entry=True, rank=0.1)

        with patch("trend_system.broker._out_date", wraps=lambda value: None if pd.isna(value) else str(value)) as mocked:
            with self.assertRaises(RuntimeError):
                run_backtest(panel, signals, self._actions(), self.config)

        inspected_values = [call.args[0] for call in mocked.call_args_list]
        self.assertTrue(all(pd.isna(value) for value in inspected_values))

    def test_missing_bar_sell_rolls_forward_and_account_calendar_is_complete(self):
        panel = pd.concat(
            [
                self._panel([DATES[0], DATES[1], DATES[3]], [CODE_A]),
                self._panel([DATES[2]], [CODE_B]),
            ],
            ignore_index=True,
        )
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)

        result = self._run(panel, signals)

        self.assertEqual(result.daily_account["date"].tolist(), DATES)
        missing = result.orders.loc[
            (result.orders["date"] == DATES[2]) & (result.orders["side"] == "sell")
        ].iloc[0]
        self.assertEqual((missing["status"], missing["reason"]), ("pending", "missing_bar"))
        sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
        self.assertEqual(sell["date"], DATES[3])

    def test_dividend_and_sale_on_same_day_use_pre_sale_quantity(self):
        panel = self._panel(DATES[:3], [CODE_A])
        signals = self._signals(panel)
        self._signal(signals, DATES[0], CODE_A, entry=True)
        self._signal(signals, DATES[1], CODE_A, exit=True)
        actions = self._actions(
            [{"date": DATES[2], "code": CODE_A, "dividend_per_share": 1.0}]
        )

        result = self._run(panel, signals, actions)

        sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
        accounts = result.daily_account.set_index("date")
        self.assertAlmostEqual(
            accounts.loc[DATES[2], "cash"],
            accounts.loc[DATES[1], "cash"] + sell["cash_delta"] + 400.0,
        )
        dividend = result.action_log.loc[result.action_log["action"] == "dividend_receivable"].iloc[0]
        self.assertEqual((dividend["quantity"], dividend["amount"]), (400, 400.0))
        self.assertEqual(accounts.loc[DATES[2], "receivables"], 0)

    def _run(self, panel, signals, actions=None, config=None):
        result = run_backtest(
            panel,
            signals,
            self._actions() if actions is None else actions,
            self.config if config is None else config,
        )
        for account in result.daily_account.itertuples(index=False):
            self.assertGreaterEqual(account.cash, -1e-8)
            self.assertAlmostEqual(
                account.equity,
                account.cash + account.receivables + account.market_value,
            )
        self.assertTrue((result.daily_positions["quantity"] >= 0).all())
        return result

    @staticmethod
    def _panel(dates, codes):
        rows = []
        for day in dates:
            for code in codes:
                rows.append(
                    {
                        "date": day,
                        "code": code,
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.5,
                        "close": 10.0,
                        "prev_close": 10.0,
                        "trade_status": 1,
                        "is_st": 0,
                        "out_date": pd.NA,
                        "data_valid": True,
                        "execution_conflict": False,
                        "adjustment_source_conflict": False,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _signals(panel):
        signals = panel[["date", "code"]].copy()
        signals["entry_signal"] = False
        signals["exit_signal"] = False
        signals["st_exit"] = False
        signals["rank_return_120"] = 0.0
        return signals

    @staticmethod
    def _signal(signals, day, code, entry=None, exit=None, st=None, rank=None):
        selected = (signals["date"] == day) & (signals["code"] == code)
        for column, value in (
            ("entry_signal", entry),
            ("exit_signal", exit),
            ("st_exit", st),
            ("rank_return_120", rank),
        ):
            if value is not None:
                signals.loc[selected, column] = value

    @staticmethod
    def _bar(panel, day, code, **changes):
        selected = (panel["date"] == day) & (panel["code"] == code)
        for column, value in changes.items():
            panel.loc[selected, column] = value

    @staticmethod
    def _actions(rows=None):
        columns = [
            "date",
            "code",
            "dividend_per_share",
            "per_share_bonus",
            "rights_ratio",
            "rights_price",
        ]
        actions = pd.DataFrame(rows or [], columns=columns)
        for column in columns[2:]:
            actions[column] = pd.to_numeric(actions[column], errors="coerce").fillna(0.0)
        return actions


if __name__ == "__main__":
    unittest.main()
