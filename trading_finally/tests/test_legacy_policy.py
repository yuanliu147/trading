import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from trend_system.broker import run_backtest
from trend_system.legacy_config import load_legacy_config
from trend_system.legacy_policy import LegacyPolicy


CODE_A = "600000.SH"
CODE_B = "000001.SZ"
CODE_C = "000002.SZ"
CODE_D = "000003.SZ"
CODE_E = "000004.SZ"
ALL_CODES = [CODE_A, CODE_B, CODE_C, CODE_D, CODE_E]


class LegacyPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).resolve().parents[1] / "config" / "legacy_candidates.json"
        cls.frozen_config = load_legacy_config(config_path)

    def test_authoritative_score_orders_ties_and_actual_daily_opens_are_two(self):
        dates = self._dates(3)
        codes = [CODE_A, CODE_B, CODE_C, CODE_D]
        panel = self._panel(dates, codes)
        signals = self._signals(panel)
        for code, score, donchian_alias in (
            (CODE_A, 4.0, 100.0),
            (CODE_B, 5.0, 0.0),
            (CODE_C, 5.0, 0.0),
            (CODE_D, 4.0, 99.0),
        ):
            self._signal(
                signals, dates[0], code, entry=True, score=score, donchian_alias=donchian_alias
            )

        result = self._run(panel, signals, initial_cash=100000)

        buys = result.trades.loc[result.trades["side"] == "buy"]
        self.assertEqual(buys["code"].tolist(), [CODE_B, CODE_C])
        self.assertEqual(buys["date"].tolist(), [dates[1], dates[1]])
        self.assertTrue((result.trades["date"] > result.trades["signal_date"]).all())
        self.assertFalse((result.trades["date"] == dates[0]).any())
        cancelled = result.orders.loc[
            (result.orders["side"] == "buy") & result.orders["code"].isin([CODE_A, CODE_D])
        ]
        self.assertEqual(cancelled["reason"].tolist(), ["no_slot", "no_slot"])

    def test_four_slots_twenty_percent_whole_lots_cash_and_no_adding(self):
        dates = self._dates(5)
        panel = self._panel(dates, ALL_CODES)
        signals = self._signals(panel)
        for code in (CODE_A, CODE_B):
            self._signal(signals, dates[0], code, entry=True, score=5.0)
        for code in (CODE_A, CODE_C, CODE_D):
            self._signal(signals, dates[1], code, entry=True, score=5.0)
        self._signal(signals, dates[2], CODE_E, entry=True, score=5.0)

        result = self._run(panel, signals, initial_cash=100000)

        buys = result.trades.loc[result.trades["side"] == "buy"]
        self.assertEqual(buys["code"].tolist(), [CODE_B, CODE_A, CODE_C, CODE_D])
        self.assertEqual(buys["quantity"].tolist(), [2000, 2000, 1900, 1900])
        self.assertEqual(buys["date"].tolist(), [dates[1], dates[1], dates[2], dates[2]])
        self.assertTrue((buys["quantity"] % 100 == 0).all())
        self.assertEqual(len(buys.loc[buys["code"] == CODE_A]), 1)
        self.assertFalse((buys["code"] == CODE_E).any())
        fifth = result.orders.loc[
            (result.orders["side"] == "buy") & (result.orders["code"] == CODE_E)
        ].iloc[0]
        self.assertEqual((fifth["status"], fifth["reason"]), ("cancelled", "no_slot"))
        self.assertEqual(result.daily_account.iloc[-1]["position_count"], 4)
        self.assertAlmostEqual(
            result.daily_account.iloc[-1]["cash"], 100000 + result.trades["cash_delta"].sum()
        )
        self.assertTrue((result.trades["costs"] > 0).all())
        self.assertTrue((result.daily_account["cash"] >= 0).all())

    def test_rejection_and_stop_at_entry_do_not_consume_daily_open_slots(self):
        dates = self._dates(3)
        codes = [CODE_A, CODE_B, CODE_C, CODE_D]
        panel = self._panel(dates, codes)
        self._bar(
            panel, dates[1], CODE_A,
            open=11.0, high=11.0, low=10.5, close=11.0, prev_close=10.0,
        )
        signals = self._signals(panel)
        for code, score in ((CODE_A, 4.0), (CODE_B, 3.0), (CODE_C, 2.0), (CODE_D, 1.0)):
            self._signal(signals, dates[0], code, entry=True, score=score)
        self._signal(signals, dates[0], CODE_B, stop=10.01)

        result = self._run(panel, signals)

        buys = result.trades.loc[result.trades["side"] == "buy"]
        self.assertEqual(buys["code"].tolist(), [CODE_C, CODE_D])
        attempts = result.orders.loc[result.orders["side"] == "buy"].set_index("code")
        self.assertEqual(attempts.loc[CODE_A, "reason"], "buy_limit_up")
        self.assertEqual(attempts.loc[CODE_B, "status"], "cancelled")
        self.assertEqual(attempts.loc[CODE_B, "reason"], "stop_at_or_above_entry")
        self.assertEqual(attempts.loc[[CODE_C, CODE_D], "status"].tolist(), ["filled", "filled"])

    def test_entry_stop_uses_fixed_signal_low10_not_later_low10(self):
        dates = self._dates(4)
        panel = self._panel(dates, [CODE_A])
        self._bar(panel, dates[2], CODE_A, high=10.1, low=9.4, close=9.5)
        signals = self._signals(panel)
        self._signal(signals, dates[0], CODE_A, entry=True, stop=9.5)
        for day in dates[1:]:
            self._signal(signals, day, CODE_A, stop=1.0)

        result = self._run(panel, signals)

        sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
        self.assertEqual((sell["signal_date"], sell["date"]), (dates[2], dates[3]))
        self.assertEqual(result.round_trips.iloc[0]["exit_reason"], "stop_loss")

    def test_corporate_action_adjusts_fixed_anchor_and_signal_factor_to_actual_scale(self):
        dates = self._dates(5)
        panel = self._panel(dates, [CODE_A])
        # Real A-share convention: adjustment_factor = signal_close / close is
        # cumulative post-adjustment and jumps UP at a bonus/split (e.g. 002695.SZ
        # 2016-09-22: factor 1.0294 -> 4.1255 while signal stays continuous).
        for day in dates[2:]:
            self._bar(panel, day, CODE_A, signal_close=10.0, adjustment_factor=2.0)
        self._bar(
            panel, dates[2], CODE_A,
            open=5.0, high=5.2, low=4.7, close=5.0, prev_close=5.0,
            signal_close=10.0, adjustment_factor=2.0,
        )
        self._bar(
            panel, dates[3], CODE_A,
            open=5.0, high=5.2, low=4.7, close=4.75, prev_close=5.0,
            signal_close=10.0, adjustment_factor=2.0,
        )
        self._bar(
            panel, dates[4], CODE_A,
            open=4.75, high=4.9, low=4.6, close=4.8, prev_close=4.75,
            signal_close=10.0, adjustment_factor=2.0,
        )
        signals = self._signals(panel)
        self._signal(signals, dates[0], CODE_A, entry=True, stop=9.5)
        actions = self._actions([
            {"date": dates[2], "code": CODE_A, "per_share_bonus": 1.0}
        ])

        result = self._run(panel, signals, actions=actions)

        self.assertEqual(result.trades["side"].tolist(), ["buy", "sell"])
        sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
        self.assertEqual((sell["signal_date"], sell["date"]), (dates[3], dates[4]))
        self.assertEqual(sell["quantity"], 800)
        self.assertEqual(result.round_trips.iloc[0]["exit_reason"], "stop_loss")

    def test_hold_40_counts_entry_day_but_not_suspension_zero_volume_or_missing_bar(self):
        dates = self._dates(45)
        panel = self._panel(dates, [CODE_A, CODE_B])
        panel = panel.loc[
            ~((panel["date"] == dates[3]) & (panel["code"] == CODE_A))
        ].reset_index(drop=True)
        self._bar(panel, dates[2], CODE_A, trade_status=0, volume=0.0)
        self._bar(panel, dates[4], CODE_A, volume=0.0)
        signals = self._signals(panel)
        self._signal(signals, dates[0], CODE_A, entry=True, stop=9.0)

        result = self._run(panel, signals)

        sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
        self.assertEqual((sell["signal_date"], sell["date"]), (dates[43], dates[44]))
        self.assertEqual(result.round_trips.iloc[0]["exit_reason"], "max_hold")

    def test_same_close_records_all_old_reasons_and_separate_st_with_one_sell(self):
        dates = self._dates(42)
        panel = self._panel(dates, [CODE_A])
        self._bar(
            panel, dates[40], CODE_A,
            open=10.0, high=10.0, low=8.0, close=8.0, prev_close=10.0, is_st=1,
        )
        self._bar(
            panel, dates[41], CODE_A,
            open=8.0, high=8.2, low=7.8, close=8.0, prev_close=8.0, is_st=1,
        )
        signals = self._signals(panel)
        self._signal(signals, dates[0], CODE_A, entry=True, stop=9.5)
        self._signal(
            signals, dates[40], CODE_A,
            signal_close=8.0, ma5=0.0, ma20=9.0, whist=-1.0,
        )

        result = self._run(panel, signals)

        sells = result.trades.loc[result.trades["side"] == "sell"]
        self.assertEqual(len(sells), 1)
        self.assertEqual((sells.iloc[0]["signal_date"], sells.iloc[0]["date"]), (dates[40], dates[41]))
        expected = (
            "stop_loss|max_hold|ma5_below_ma20|whist_negative|close_below_ma20;st_exit"
        )
        self.assertEqual(result.round_trips.iloc[0]["exit_reason"], expected)
        sell_orders = result.orders.loc[result.orders["side"] == "sell"]
        self.assertEqual(len(sell_orders), 1)
        self.assertEqual(sell_orders.iloc[0]["status"], "filled")

    def test_policy_stop_exit_rolls_through_limit_down_days(self):
        dates = self._dates(6)
        panel = self._panel(dates, [CODE_A])
        self._bar(panel, dates[2], CODE_A, high=10.0, low=8.9, close=9.0)
        self._bar(
            panel, dates[3], CODE_A,
            open=8.1, high=8.2, low=8.1, close=8.1, prev_close=9.0,
        )
        self._bar(
            panel, dates[4], CODE_A,
            open=7.29, high=7.4, low=7.29, close=7.3, prev_close=8.1,
        )
        self._bar(
            panel, dates[5], CODE_A,
            open=7.0, high=7.2, low=6.9, close=7.0, prev_close=7.29,
        )
        signals = self._signals(panel)
        self._signal(signals, dates[0], CODE_A, entry=True, stop=9.5)

        result = self._run(panel, signals)

        sell = result.trades.loc[result.trades["side"] == "sell"].iloc[0]
        self.assertEqual((sell["signal_date"], sell["date"]), (dates[2], dates[5]))
        self.assertAlmostEqual(sell["price"], 7.0 * 0.999)
        delayed = result.orders.loc[
            (result.orders["side"] == "sell") & result.orders["date"].isin(dates[3:5])
        ]
        self.assertEqual(delayed["status"].tolist(), ["pending", "pending"])
        self.assertEqual(delayed["reason"].tolist(), ["sell_limit_down", "sell_limit_down"])

    def test_explicit_none_matches_omitted_policy_without_legacy_fields(self):
        dates = self._dates(3)
        panel = self._panel(dates, [CODE_A, CODE_B])
        signals = panel[["date", "code"]].copy()
        signals["entry_signal"] = False
        signals["exit_signal"] = False
        signals["st_exit"] = False
        signals["rank_return_120"] = 0.0
        self._signal(signals, dates[0], CODE_A, entry=True, donchian_alias=0.1)
        self._signal(signals, dates[0], CODE_B, entry=True, donchian_alias=0.2)
        from trend_system.config import ResearchConfig
        default_config = ResearchConfig(
            hithink_db_path="unused", baostock_db_path="unused",
            start_date=dates[0], end_date=dates[-1], slots=1,
        )
        actions = self._actions()

        omitted = run_backtest(panel, signals, actions, default_config)
        explicit = run_backtest(panel, signals, actions, default_config, policy=None)

        for field in ("trades", "orders", "round_trips", "daily_account", "daily_positions", "action_log"):
            pd.testing.assert_frame_equal(getattr(omitted, field), getattr(explicit, field))
        self.assertEqual(omitted.trades["code"].tolist(), [CODE_B])

    def _run(self, panel, signals, actions=None, initial_cash=20000):
        config = self._account_config(panel, initial_cash)
        result = run_backtest(
            panel,
            signals,
            self._actions() if actions is None else actions,
            config,
            policy=LegacyPolicy(config),
        )
        for account in result.daily_account.itertuples(index=False):
            self.assertGreaterEqual(account.cash, -1e-8)
            self.assertAlmostEqual(account.equity, account.cash + account.receivables + account.market_value)
        return result

    def _account_config(self, panel, initial_cash=20000):
        return replace(
            self.frozen_config,
            start_date=str(panel["date"].min()),
            end_date=str(panel["date"].max()),
            initial_cash=initial_cash,
        )

    @staticmethod
    def _dates(count):
        return pd.bdate_range("2024-01-02", periods=count).astype(str).tolist()

    @staticmethod
    def _panel(dates, codes):
        rows = []
        for day in dates:
            for code in codes:
                rows.append({
                    "date": day,
                    "code": code,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.0,
                    "prev_close": 10.0,
                    "signal_open": 10.0,
                    "signal_high": 10.5,
                    "signal_low": 9.5,
                    "signal_close": 10.0,
                    "adjustment_factor": 1.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                    "trade_status": 1,
                    "is_st": 0,
                    "out_date": pd.NA,
                    "data_valid": True,
                    "execution_conflict": False,
                    "adjustment_source_conflict": False,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _signals(panel):
        signals = panel[["date", "code", "signal_close"]].copy()
        signals["entry_signal"] = False
        signals["legacy_rank_score"] = 0.0
        signals["rank_return_120"] = 0.0
        signals["stop_ref"] = 9.0
        signals["ma5"] = 2.0
        signals["ma20"] = 1.0
        signals["whist"] = 1.0
        return signals

    @staticmethod
    def _signal(
        signals, day, code, entry=None, score=None, donchian_alias=None,
        stop=None, signal_close=None, ma5=None, ma20=None, whist=None,
    ):
        selected = (signals["date"] == day) & (signals["code"] == code)
        for column, value in (
            ("entry_signal", entry),
            ("legacy_rank_score", score),
            ("rank_return_120", donchian_alias),
            ("stop_ref", stop),
            ("signal_close", signal_close),
            ("ma5", ma5),
            ("ma20", ma20),
            ("whist", whist),
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
            "date", "code", "dividend_per_share", "per_share_bonus", "rights_ratio", "rights_price",
        ]
        actions = pd.DataFrame(rows or [], columns=columns)
        for column in columns[2:]:
            actions[column] = pd.to_numeric(actions[column], errors="coerce").fillna(0.0)
        return actions


if __name__ == "__main__":
    unittest.main()
