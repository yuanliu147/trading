"""仅合成数据；python3 -B 本文件。不会读取真实行情或写出收益文件。"""

from dataclasses import replace
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from 策略注册 import Strategy
from 现金账本 import run_broker, transaction_costs


LOOSE = Strategy("合成宽松预算", slots=4, max_weight=1, risk_per_trade=1,
                 heat_cap=1, max_stop_distance=0.5)


def fixture(n=5, codes=("000001",), start="2024-01-02"):
    days = pd.bdate_range(start, periods=n)
    bars = pd.DataFrame([
        {"date": date, "code": code, "open": 10.0, "high": 10.1, "low": 9.9,
         "close": 10.0, "adj_factor": 1.0, "prev_amount20": 100_000_000.0,
         "limit_up": 20.0, "limit_down": 1.0, "minimum_buy_size": 100, "tradable": True}
        for date in days for code in codes
    ])
    return days, bars


def change(bars, day, code="000001", **values):
    mask = bars.date.eq(day) & bars.code.eq(code)
    for key, value in values.items():
        bars.loc[mask, key] = value


def order(date, code="000001", stop=9.0, risk=1.0, rank=1):
    return {"signal_date": date, "code": code, "stop_adj": stop, "initial_risk_adj": risk, "rank": rank}


def sells(result):
    return result["成交"].loc[result["成交"].方向.eq("卖出")]


def buys(result):
    return result["成交"].loc[result["成交"].方向.eq("买入")]


class CashBrokerTests(unittest.TestCase):
    def test_entry_day_stop_persists_even_if_next_open_recovers(self):
        days, bars = fixture(3)
        change(bars, days[1], low=9.0, close=10.0)
        result = self.run_case(days, bars, [order(days[0], stop=9.5, risk=.5)])
        self.assertEqual(sells(result)["日期"].tolist(), [days[2]])
        self.assertEqual(sells(result).iloc[0]["成交原价"], 10.0)
        pending = result["每日持仓"].query("日期 == @days[1]")["待卖股数"].iloc[0]
        self.assertGreater(pending, 0)

    def test_stop_fills_use_cent_ticks(self):
        days, bars = fixture(3)
        change(bars, days[2], low=9.5)
        result = self.run_case(days, bars, [order(days[0], stop=9.555, risk=.445)])
        self.assertEqual(buys(result).iloc[0]["初始止损原价"], 9.55)
        self.assertEqual(sells(result).iloc[0]["成交原价"], 9.55)
        self.assertTrue(((result["成交"]["成交原价"] * 100).round(8) % 1 == 0).all())

    def test_fixed_eight_percent_is_from_actual_open(self):
        days, bars = fixture(2)
        change(bars, days[1], open=11.0, high=11.1, low=10.9, close=11.0)
        result = self.run_case(days, bars, [order(days[0], stop=9.2, risk=.8)],
                               strategy=replace(LOOSE, stop="fixed8"))
        buy = buys(result).iloc[0]
        self.assertEqual(buy["初始止损原价"], 10.12)
        self.assertAlmostEqual(1 - buy["初始止损原价"] / buy["成交原价"], .08)

    def test_atr_distance_is_from_actual_open(self):
        days, bars = fixture(2)
        change(bars, days[1], open=11.0, high=11.1, low=10.9, close=11.0)
        result = self.run_case(days, bars, [order(days[0], stop=8.0, risk=2.0)],
                               strategy=replace(LOOSE, stop="atr2"))
        self.assertEqual(buys(result).iloc[0]["初始止损原价"], 9.0)

    def test_lazy_provider_matches_dataframe(self):
        days, bars = fixture(codes=("000001", "000002"))
        lookup = {d: g.set_index("code").to_dict("index") for d, g in bars.groupby("date")}

        class Provider:
            def get_day(self, date, codes):
                return {c: lookup[date][c] for c in codes if c in lookup[date]}

        orders = [order(days[0]), order(days[2], "000002")]
        expected = run_broker(bars, orders, days, LOOSE)
        actual = run_broker(Provider(), orders, days, LOOSE)
        for key in expected:
            assert_frame_equal(expected[key], actual[key])

    def test_star_partial_sale_below_two_hundred_is_not_filled(self):
        days, bars = fixture()
        bars["minimum_buy_size"] = 200
        instructions = [dict(date=days[1], code="000001", next_open_sell_fraction=.5),
                        dict(date=days[3], code="000001", next_open_sell_fraction=1.0)]
        result = self.run_case(days, bars, strategy=replace(LOOSE, max_weight=.105),
                               close_instructions=instructions)
        self.assertEqual(buys(result).iloc[0]["股数"], 210)
        self.assertEqual(sells(result)["日期"].tolist(), [days[4]])
        self.assertEqual(sells(result).iloc[0]["股数"], 210)

    def run_case(self, days, bars, orders=None, strategy=LOOSE, **kwargs):
        result = run_broker(bars, orders if orders is not None else [order(days[0])], days, strategy, **kwargs)
        account = result["每日账户"]
        self.assertTrue(account["可用现金"].ge(0).all())
        self.assertTrue(result["成交"]["股数"].mod(1).eq(0).all())
        for _, row in account.iterrows():
            self.assertAlmostEqual(row["账户权益"], row["可用现金"] + row["分红应收"] + row["持仓市值"], places=7)
            self.assertAlmostEqual(row["账户权益"] - kwargs.get("initial_cash", 20000),
                                   row["累计已实现盈亏"] + row["未实现盈亏"], places=7)
        holdings = result["每日持仓"]
        self.assertTrue(holdings["锁定股数"].ge(0).all())
        self.assertTrue(holdings["可卖股数"].ge(0).all())
        self.assertTrue((holdings["锁定股数"] + holdings["可卖股数"]).eq(holdings["股数"]).all())
        return result

    def test_first_fill_fees_reduce_cash_and_equity(self):
        days, bars = fixture()
        result = self.run_case(days, bars, strategy=Strategy("默认"))
        trade = buys(result).iloc[0]
        self.assertEqual(trade["日期"], days[1])
        self.assertEqual(trade["股数"], 200)
        self.assertEqual(trade["成交原价"], 10)
        self.assertAlmostEqual(trade["总费用"], 5 + 0.02 + 2)
        self.assertAlmostEqual(result["每日账户"].iloc[1]["可用现金"], 20000 - 2000 - 7.02)
        self.assertAlmostEqual(result["每日账户"].iloc[1]["账户权益"], 20000 - 7.02)

    def test_twenty_thousand_real_lots_and_no_overdraft(self):
        days, bars = fixture(codes=("000001", "000002"))
        result = self.run_case(days, bars, [order(days[0]), order(days[0], "000002", rank=0)])
        self.assertEqual(buys(result)["股数"].tolist(), [1900])
        self.assertEqual(buys(result).iloc[0]["股数"] % 100, 0)
        self.assertAlmostEqual(result["每日账户"].iloc[1]["可用现金"], 975.11)
        self.assertEqual(len(result["失败订单"]), 1)

    def test_raw_shares_not_adjusted_shares(self):
        days, bars = fixture()
        bars["adj_factor"] = 2.0
        result = self.run_case(days, bars, [order(days[0], stop=18, risk=2)], strategy=Strategy("默认"))
        self.assertEqual(buys(result).iloc[0]["股数"], 200)
        self.assertEqual(buys(result).iloc[0]["成交金额"], 2000)
        self.assertEqual(result["期末持仓"].iloc[0]["止损原价"], 9)

    def test_t_plus_one_no_entry_day_stop_sale(self):
        days, bars = fixture(3)
        change(bars, days[1], low=8.0, close=8.5)
        change(bars, days[2], open=8.0, high=8.5, low=7.8, close=8.2)
        result = self.run_case(days, bars)
        self.assertEqual(sells(result)["日期"].tolist(), [days[2]])
        self.assertEqual(sells(result).iloc[0]["时点"], "开盘")

    def test_gap_stop_executes_at_open_not_old_stop(self):
        days, bars = fixture(3)
        change(bars, days[2], open=8.0, high=8.3, low=7.8, close=8.1)
        result = self.run_case(days, bars)
        trade = sells(result).iloc[0]
        self.assertEqual(trade["成交原价"], 8.0)
        self.assertEqual(trade["原因"], "开盘穿止损")
        self.assertAlmostEqual(trade["滑点成本"], trade["成交金额"] * 0.001)

    def test_consecutive_limit_down_persists_even_after_price_recovers(self):
        days, bars = fixture(5)
        change(bars, days[2], open=9.0, limit_down=9.0, high=9.5, low=9.0, close=9.1)
        change(bars, days[3], open=8.1, limit_down=8.1, high=8.6, low=8.1, close=8.2)
        result = self.run_case(days, bars)
        self.assertEqual(sells(result)["日期"].tolist(), [days[4]])
        self.assertEqual(sells(result).iloc[0]["成交原价"], 10)
        self.assertEqual(result["每日持仓"].query("日期 == @days[3]")["待卖股数"].iloc[0], 1900)

    def test_intraday_proceeds_cannot_fund_same_open_buys(self):
        days, bars = fixture(4, codes=("000001", "000002"))
        change(bars, days[2], low=8.8)
        result = self.run_case(days, bars, [order(days[0]), order(days[1], "000002")])
        self.assertEqual(buys(result)["代码"].tolist(), ["000001"])
        self.assertEqual(sells(result)["时点"].tolist(), ["盘中"])
        self.assertEqual(sells(result).iloc[0]["成交原价"], 9)
        self.assertGreater(result["每日账户"].iloc[2]["可用现金"], 17000)
        # 新的隔夜信号次日可以使用这笔钱，原失败买单本身不会自动顺延。
        again = self.run_case(days, bars, [order(days[0]), order(days[1], "000002"), order(days[2], "000002")])
        self.assertEqual(buys(again)["日期"].tolist(), [days[1], days[3]])

    def test_open_exit_proceeds_can_fund_open_buy(self):
        days, bars = fixture(3, codes=("000001", "000002"))
        result = self.run_case(days, bars, [order(days[0]), order(days[1], "000002")],
                               close_instructions=[{"date": days[1], "code": "000001", "next_open_sell_fraction": 1.0}])
        self.assertEqual(result["成交"]["方向"].tolist(), ["买入", "卖出", "买入"])
        self.assertEqual(buys(result)["代码"].tolist(), ["000001", "000002"])

    def test_close_stop_raise_effective_next_day_only(self):
        days, bars = fixture(4)
        change(bars, days[2], low=9.2, close=10.0)
        change(bars, days[3], open=9.4, high=9.6, low=9.3, close=9.4)
        result = self.run_case(days, bars, close_instructions=[{"date": days[2], "code": "000001", "stop_adj": 9.5}])
        self.assertEqual(sells(result)["日期"].tolist(), [days[3]])
        self.assertEqual(sells(result).iloc[0]["成交原价"], 9.4)
        self.assertEqual(result["收盘指令"].iloc[0]["生效日期"], days[3])

    def test_limit_up_rejects_regardless_of_daily_high_low(self):
        days, bars = fixture(3)
        change(bars, days[1], limit_up=10.0, high=10.0, low=9.2)
        baseline = self.run_case(days, bars)
        self.assertTrue(buys(baseline).empty)
        changed = bars.copy()
        change(changed, days[1], high=100.0, low=0.1)
        actual = self.run_case(days, changed)
        assert_frame_equal(actual["成交"], baseline["成交"])
        assert_frame_equal(actual["失败订单"], baseline["失败订单"])
        # 反过来，开盘未涨停、盘中触涨停不能导致拒买。
        change(bars, days[1], open=9.9, high=10.0)
        self.assertEqual(len(buys(self.run_case(days, bars))), 1)

    def test_commission_minimum_per_order_both_sides(self):
        for side in ("买入", "卖出"):
            self.assertEqual(transaction_costs("2024-01-02", 1000, side)["佣金"], 5)
            self.assertAlmostEqual(transaction_costs("2024-01-02", 20000, side)["佣金"], 6)

    def test_stamp_tax_change_boundary(self):
        self.assertEqual(transaction_costs("2023-08-27", 10000, "卖出")["印花税"], 10)
        self.assertEqual(transaction_costs("2023-08-28", 10000, "卖出")["印花税"], 5)
        self.assertEqual(transaction_costs("2023-08-28", 10000, "买入")["印花税"], 0)
        days, bars = fixture(4, start="2023-08-24")
        result = self.run_case(days, bars, close_instructions=[{"date": days[1], "code": "000001", "next_open_sell_fraction": 1}])
        trade = sells(result).iloc[0]
        self.assertEqual(trade["日期"], pd.Timestamp("2023-08-28"))
        self.assertAlmostEqual(trade["印花税"], trade["成交金额"] * 0.0005)

    def test_transfer_tax_change_and_slippage_stress(self):
        self.assertEqual(transaction_costs("2022-04-28", 10000, "买入")["过户费"], 0.2)
        self.assertEqual(transaction_costs("2022-04-29", 10000, "卖出")["过户费"], 0.1)
        days, bars = fixture(2)
        for bps in (10, 20, 30):
            result = self.run_case(days, bars, slippage_bps=bps)
            trade = buys(result).iloc[0]
            self.assertEqual(trade["成交原价"], 10)
            self.assertAlmostEqual(trade["滑点成本"], trade["成交金额"] * bps / 10000)

    def test_star_minimum_200_increment_one(self):
        days, bars = fixture(2, codes=("688001",))
        bars["minimum_buy_size"] = 200
        for weight, expected in ((0.1005, 201), (0.1, 200), (0.0995, None)):
            result = self.run_case(days, bars, [order(days[0], "688001")], strategy=replace(LOOSE, max_weight=weight))
            if expected is None:
                self.assertTrue(buys(result).empty)
            else:
                self.assertEqual(buys(result).iloc[0]["股数"], expected)

    def test_dividend_receivable_payment_and_split(self):
        days, bars = fixture(5)
        bars.loc[bars.date >= days[2], ["open", "high", "low", "close"]] = [4.5, 4.6, 4.4, 4.5]
        bars.loc[bars.date >= days[2], "adj_factor"] = 2.0
        actions = [{"ex_date": days[2], "pay_date": days[4], "code": "000001", "share_multiplier": 2,
                    "cash_dividend_per_old_share": 1.0}]
        result = self.run_case(days, bars, [order(days[0], stop=8, risk=2)], corp_actions=actions)
        account = result["每日账户"]
        self.assertEqual(result["期末持仓"].iloc[0]["股数"], 3800)
        self.assertEqual(result["期末持仓"].iloc[0]["止损原价"], 4)
        self.assertEqual(account.iloc[2]["分红应收"], 1900)
        self.assertEqual(account.iloc[2]["可用现金"], account.iloc[1]["可用现金"])
        self.assertAlmostEqual(account.iloc[2]["账户权益"], account.iloc[1]["账户权益"])
        self.assertEqual(account.iloc[4]["分红应收"], 0)
        self.assertAlmostEqual(account.iloc[4]["可用现金"], account.iloc[1]["可用现金"] + 1900)
        self.assertEqual(result["分红应收"].iloc[0]["实际入账日期"], days[4])

    def test_unpaid_dividend_cannot_fund_open_buy(self):
        days, bars = fixture(5, codes=("000001", "000002"))
        actions = [{"ex_date": days[2], "pay_date": days[4], "code": "000001", "share_multiplier": 1,
                    "cash_dividend_per_old_share": 1}]
        for day in days[2:]:
            change(bars, day, open=9.0, close=9.0, low=8.9, high=9.1)
        result = self.run_case(days, bars, [order(days[0], stop=8, risk=2), order(days[1], "000002"), order(days[3], "000002")], corp_actions=actions)
        self.assertEqual(buys(result)["日期"].tolist(), [days[1], days[4]])

    def test_ex_date_buyer_has_no_entitlement_and_absent_actions_not_invented(self):
        days, bars = fixture(3)
        actions = [{"ex_date": days[1], "pay_date": days[2], "code": "000001", "share_multiplier": 2,
                    "cash_dividend_per_old_share": 1}]
        result = self.run_case(days, bars, corp_actions=actions)
        self.assertTrue(result["分红应收"].empty)
        self.assertTrue(result["公司行动"].empty)
        bars.loc[bars.date >= days[2], "adj_factor"] = 2
        result = self.run_case(days, bars)
        self.assertEqual(result["期末持仓"].iloc[0]["股数"], 1900)
        self.assertTrue(result["公司行动"].empty)

    def test_fractional_split_half_sale_and_final_odd_lot(self):
        days, bars = fixture(5)
        bars.loc[bars.date >= days[2], "adj_factor"] = 1.0005
        change(bars, days[2], tradable=False)
        actions = [{"ex_date": days[2], "pay_date": days[2], "code": "000001", "share_multiplier": 1.0005,
                    "cash_dividend_per_old_share": 0}]
        instructions = [{"date": days[2], "code": "000001", "next_open_sell_fraction": 0.5},
                        {"date": days[3], "code": "000001", "next_open_sell_fraction": 1}]
        result = self.run_case(days, bars, corp_actions=actions, close_instructions=instructions)
        self.assertEqual(sells(result)["股数"].tolist(), [900, 1000])
        action = result["公司行动"].iloc[0]
        self.assertEqual(action["新股数"], 1900)
        self.assertAlmostEqual(action["未计价零碎权益"], .95)
        self.assertIn("保守分配假设", action["处理说明"])
        self.assertIn("不假定现金补偿", action["处理说明"])
        holdings = result["每日持仓"].set_index("日期")
        self.assertEqual(holdings.loc[days[2], "持仓成本"], holdings.loc[days[1], "持仓成本"])
        self.assertAlmostEqual(holdings.loc[days[2], "持仓市值"], 19000 / 1.0005)
        self.assertAlmostEqual(holdings.loc[days[2], "未实现盈亏"] - holdings.loc[days[1], "未实现盈亏"],
                               -.95 * 10 / 1.0005)
        self.assertEqual(result["每日账户"].iloc[2]["可用现金"], result["每日账户"].iloc[1]["可用现金"])
        self.assertTrue(result["期末持仓"].empty)

    def test_cash_and_stock_payment_dates_and_gross_ex_price_are_independent(self):
        days, bars = fixture(5)
        bars.loc[bars.date >= days[2], "tradable"] = False
        for cash_day, stock_day in ((days[3], days[4]), (days[4], days[3])):
            for gross in (None, 1.0):
                with self.subTest(cash_day=cash_day, stock_day=stock_day, gross=gross):
                    action = dict(ex_date=days[2], pay_date=cash_day, stock_pay_date=stock_day,
                                  code="000001", share_multiplier=2, cash_dividend_per_old_share=.8)
                    if gross is not None:
                        action["cash_dividend_gross_per_old_share"] = gross
                    result = self.run_case(days, bars, corp_actions=[action])
                    holdings = result["每日持仓"].set_index("日期")
                    account = result["每日账户"].set_index("日期")
                    theoretical = (10 - (.8 if gross is None else gross)) / 2
                    self.assertEqual(holdings.loc[days[1], "锁定股数"], 1900)
                    self.assertEqual(holdings.loc[days[1], "可卖股数"], 0)
                    for day in days[2:]:
                        self.assertEqual(holdings.loc[day, "股数"], 3800)
                        self.assertEqual(holdings.loc[day, "锁定股数"], 1900 if day < stock_day else 0)
                        self.assertEqual(holdings.loc[day, "最新原价"], theoretical)
                        self.assertEqual(account.loc[day, "分红应收"], 1520 if day < cash_day else 0)
                        self.assertEqual(account.loc[day, "可用现金"],
                                         account.loc[days[1], "可用现金"] + (1520 if day >= cash_day else 0))
                    self.assertFalse(result["公司行动"].iloc[0]["可售日期使用默认"])
                    self.assertTrue(sells(result).empty)

    def test_ex_date_only_old_shares_sell_and_terminal_bonus_stays_locked(self):
        days, original = fixture(4)
        for exit_mode in ("预定", "跳空", "盘中"):
            with self.subTest(exit_mode=exit_mode):
                bars = original.copy()
                change(bars, days[2], adj_factor=1.01)
                instructions = []
                if exit_mode == "预定":
                    instructions = [dict(date=days[1], code="000001", next_open_sell_fraction=1)]
                elif exit_mode == "跳空":
                    change(bars, days[2], open=8, high=8.2, low=7.9, close=8)
                else:
                    change(bars, days[2], low=8)
                action = dict(ex_date=days[2], pay_date=days[2], stock_pay_date=days[2] + pd.Timedelta(days=5),
                              code="000001", share_multiplier=1.01, cash_dividend_per_old_share=0)
                result = self.run_case(days, bars, corp_actions=[action], close_instructions=instructions)
                self.assertEqual(sells(result)["股数"].tolist(), [1900])
                self.assertEqual(sells(result).iloc[0]["时点"], "盘中" if exit_mode == "盘中" else "开盘")
                final = result["期末持仓"].iloc[0]
                self.assertEqual(final["股数"], 19)
                self.assertEqual(final["锁定股数"], 19)
                self.assertEqual(final["可卖股数"], 0)
                self.assertEqual(final["待卖股数"], 19)
                self.assertGreater(final["持仓市值"], 0)

    def test_next_day_unlock_fills_pending_final_odd_lot(self):
        days, bars = fixture(4)
        bars.loc[bars.date >= days[2], "adj_factor"] = 1.01
        action = dict(ex_date=days[2], pay_date=days[2], stock_pay_date=days[3],
                      code="000001", share_multiplier=1.01, cash_dividend_per_old_share=0)
        result = self.run_case(days, bars, corp_actions=[action],
                               close_instructions=[dict(date=days[1], code="000001", next_open_sell_fraction=1)])
        self.assertEqual(sells(result)["日期"].tolist(), list(days[2:]))
        self.assertEqual(sells(result)["股数"].tolist(), [1900, 19])
        self.assertEqual(result["每日持仓"].iloc[-1]["待卖股数"], 19)
        self.assertTrue(result["期末持仓"].empty)

    def test_split_sales_are_integer_with_near_integer_tolerance(self):
        days, original = fixture(3)
        for multiplier, whole, fractional in ((1.0005, 1900, .95), (1.0105, 1919, .95),
                                               (1.01 - 5e-8 / 1900, 1919, 0),
                                               (1.01 + 5e-8 / 1900, 1919, 0),
                                               (1.01 - 2e-7 / 1900, 1918, 1 - 2e-7)):
            with self.subTest(multiplier=multiplier):
                bars = original.copy()
                change(bars, days[2], adj_factor=multiplier)
                action = dict(ex_date=days[2], pay_date=days[2], code="000001",
                              share_multiplier=multiplier, cash_dividend_per_old_share=0)
                result = self.run_case(days, bars, corp_actions=[action],
                                       close_instructions=[dict(date=days[1], code="000001", next_open_sell_fraction=1)])
                self.assertEqual(sells(result)["股数"].tolist(), [whole])
                logged = result["公司行动"].iloc[0]
                self.assertAlmostEqual(logged["未计价零碎权益"], fractional)
                self.assertEqual(logged["送转可售日期"], days[2])
                self.assertTrue(logged["可售日期使用默认"])
                self.assertIn("不代表日期已核验完整", logged["处理说明"])
                self.assertTrue(result["期末持仓"].empty)

    def test_source_actions_not_mutated_or_previous_bonus_unlocked(self):
        days, bars = fixture(5)
        actions = [dict(ex_date=days[2], pay_date=days[4], stock_pay_date=days[4], code="000001",
                        share_multiplier=1.01, cash_dividend_per_old_share=.08,
                        cash_dividend_gross_per_old_share=.1),
                   dict(ex_date=days[3], pay_date=days[4], code="000001",
                        share_multiplier=1.01, cash_dividend_per_old_share=0)]
        before = [a.copy() for a in actions]
        frame = pd.DataFrame(actions)
        frame_before = frame.copy(deep=True)
        first = self.run_case(days, bars, corp_actions=actions)
        second = self.run_case(days, bars, corp_actions=frame)
        self.assertEqual(actions, before)
        assert_frame_equal(frame, frame_before)
        for key in first:
            assert_frame_equal(first[key], second[key])
        holdings = first["每日持仓"].set_index("日期")
        self.assertEqual(holdings.loc[days[3], "股数"], 1938)
        self.assertEqual(holdings.loc[days[3], "锁定股数"], 19)
        self.assertEqual(holdings.loc[days[3], "可卖股数"], 1919)
        self.assertEqual(holdings.loc[days[4], "锁定股数"], 0)
        self.assertEqual(first["公司行动"]["可售日期使用默认"].tolist(), [False, True])

    def test_terminal_locked_asset_not_forced_to_cash(self):
        days, bars = fixture(3)
        change(bars, days[2], open=9, low=9, high=9, close=9, limit_down=9)
        result = self.run_case(days, bars)
        self.assertTrue(sells(result).empty)
        final = result["期末持仓"].iloc[0]
        self.assertEqual(final["待卖股数"], 1900)
        self.assertEqual(final["持仓市值"], 17100)
        self.assertLess(final["未实现盈亏"], 0)
        self.assertAlmostEqual(result["每日账户"].iloc[-1]["可用现金"], 975.11)

    def test_intraday_stop_at_limit_down_queues_then_sells_on_recovery(self):
        days, bars = fixture(4)
        change(bars, days[2], low=9, limit_down=9)
        result = self.run_case(days, bars)
        self.assertEqual(sells(result)["日期"].tolist(), [days[3]])
        self.assertEqual(sells(result).iloc[0]["成交原价"], 10)
        self.assertTrue(result["失败订单"]["原因"].str.contains("止损不可确认").any())

    def test_missing_limits_are_not_inferred(self):
        days, bars = fixture(4)
        change(bars, days[1], limit_up=float("nan"))
        self.assertTrue(buys(self.run_case(days, bars)).empty)
        change(bars, days[1], limit_up=20)
        change(bars, days[2], open=8, low=7.9, high=8.1, close=8, limit_down=float("nan"))
        result = self.run_case(days, bars)
        self.assertEqual(sells(result)["日期"].tolist(), [days[3]])

    def test_tradable_adapter_flag_blocks_all_execution(self):
        days, bars = fixture(4)
        change(bars, days[1], tradable=False)
        self.assertTrue(buys(self.run_case(days, bars)).empty)
        change(bars, days[1], tradable=True)
        change(bars, days[2], tradable=False, open=8, low=7, close=8)
        result = self.run_case(days, bars, close_instructions=[{"date": days[1], "code": "000001", "next_open_sell_fraction": 1}])
        self.assertEqual(sells(result)["日期"].tolist(), [days[3]])
        self.assertEqual(result["每日账户"].iloc[2]["持仓市值"], 19000)

    def test_missing_price_haircut_only_after_observed_days_no_cash(self):
        days, bars = fixture(63)
        bars = bars.loc[bars.date <= days[1]]
        for haircut in (0, 0.5, 1):
            result = self.run_case(days, bars, haircut=haircut)
            account = result["每日账户"]
            self.assertEqual(account.iloc[60]["持仓市值"], 19000)  # 缺价59日。
            self.assertEqual(account.iloc[61]["持仓市值"], 19000 * (1 - haircut))
            self.assertEqual(account.iloc[62]["持仓市值"], 19000 * (1 - haircut))
            self.assertEqual(account.iloc[62]["可用现金"], account.iloc[1]["可用现金"])
            self.assertTrue(sells(result).empty)

    def test_open_risk_mark_uses_open_not_close(self):
        days, bars = fixture(3, codes=("000001", "000002"))
        strategy = replace(LOOSE, max_weight=0.25, heat_cap=0.06)
        baseline = self.run_case(days, bars, [order(days[0]), order(days[1], "000002")], strategy=strategy)
        change(bars, days[2], close=1000, high=1001)
        changed = self.run_case(days, bars, [order(days[0]), order(days[1], "000002")], strategy=strategy)
        assert_frame_equal(changed["成交"], baseline["成交"])
        self.assertEqual(buys(changed)["股数"].tolist(), [500, 400])

    def test_missing_open_uses_previous_valuation_not_today_close(self):
        days, bars = fixture(3, codes=("000001", "000002"))
        change(bars, days[2], open=float("nan"), close=1000, high=1001)
        result = self.run_case(days, bars, [order(days[0]), order(days[1], "000002")], strategy=replace(LOOSE, max_weight=0.25))
        self.assertEqual(buys(result)["股数"].tolist(), [500, 400])

    def test_weight_drift_allowed_and_heat_uses_current_marks(self):
        days, bars = fixture(3, codes=("000001", "000002"))
        change(bars, days[2], open=20, low=19, high=20, close=20)
        result = self.run_case(days, bars, [order(days[0]), order(days[1], "000002")], strategy=replace(LOOSE, max_weight=0.25, heat_cap=0.04))
        self.assertEqual(buys(result)["代码"].tolist(), ["000001"])
        final = result["期末持仓"].iloc[0]
        self.assertGreater(final["持仓市值"] / result["每日账户"].iloc[-1]["账户权益"], 0.25)
        self.assertTrue(sells(result).empty)

    def test_initial_risk_basis_and_stop_distance_caps(self):
        days, bars = fixture(2)
        result = self.run_case(days, bars, [order(days[0], risk=2)], strategy=Strategy("默认"))
        self.assertEqual(buys(result).iloc[0]["股数"], 100)
        for stop in (10, 10.1, 7.9):
            self.assertTrue(buys(self.run_case(days, bars, [order(days[0], stop=stop)], strategy=Strategy("默认"))).empty)

    def test_yesterday_amount_limits_buys_and_partial_exits_persist(self):
        days, bars = fixture(5)
        change(bars, days[1], prev_amount20=3_000_000)
        change(bars, days[2], prev_amount20=1_000_000)
        change(bars, days[3], prev_amount20=1_000_000)
        result = self.run_case(days, bars, close_instructions=[{"date": days[1], "code": "000001", "next_open_sell_fraction": 1}])
        self.assertEqual(buys(result)["股数"].tolist(), [300])
        self.assertEqual(sells(result)["股数"].tolist(), [100, 100, 100])
        self.assertEqual(sells(result)["日期"].tolist(), list(days[2:]))
        for _, trade in result["成交"].iterrows():
            amount20 = bars.loc[bars.date.eq(trade["日期"]), "prev_amount20"].iloc[0]
            self.assertLessEqual(trade["成交金额"], amount20 * 0.001)

    def test_callback_gets_copy_and_only_current_date_state(self):
        days, bars = fixture(4)
        observed = []

        def callback(date, positions):
            self.assertTrue(positions["日期"].eq(date).all())
            self.assertFalse(any(name in positions for name in ("high", "low", "daily", "future")))
            observed.append(date)
            positions.loc[:, "股数"] = 999999  # 不得修改broker内部股数。
            if date == days[1]:
                return [{"code": "000001", "next_open_sell_fraction": 0.5, "stop_adj": 9.2}]
            return []

        result = self.run_case(days, bars, after_close=callback)
        self.assertEqual(observed, list(days))
        self.assertEqual(sells(result)["日期"].tolist(), [days[2]])
        self.assertEqual(sells(result)["股数"].tolist(), [900])
        self.assertEqual(result["期末持仓"].iloc[0]["股数"], 1000)

    def test_future_changes_and_truncation_leave_past_identical(self):
        days, bars = fixture(7, codes=("000001", "000002"))
        orders = [order(days[0]), order(days[3], "000002")]
        instructions = [{"date": days[2], "code": "000001", "next_open_sell_fraction": 0.5}]
        full = self.run_case(days, bars, orders, close_instructions=instructions)
        cutoff = days[3]
        altered = bars.copy()
        altered.loc[altered.date > cutoff, ["open", "high", "low", "close"]] *= 0.2
        changed = self.run_case(days, altered, orders, close_instructions=instructions)
        for name in ("成交", "每日账户", "每日持仓"):
            assert_frame_equal(full[name].loc[full[name]["日期"] <= cutoff].reset_index(drop=True),
                               changed[name].loc[changed[name]["日期"] <= cutoff].reset_index(drop=True))
        short = self.run_case(days[:4], bars.loc[bars.date <= cutoff], orders, close_instructions=instructions)
        for name in ("成交", "每日账户", "每日持仓"):
            assert_frame_equal(short[name], full[name].loc[full[name]["日期"] <= cutoff].reset_index(drop=True))

    def test_rank_ties_code_order_and_input_not_mutated(self):
        days, bars = fixture(2, codes=("000002", "000001"))
        before = bars.copy(deep=True)
        result = self.run_case(days, bars.iloc[::-1], [order(days[0], "000002"), order(days[0])])
        self.assertEqual(buys(result)["代码"].tolist(), ["000001"])
        assert_frame_equal(bars, before)

    def test_empty_outputs_keep_chinese_csv_schema(self):
        days, bars = fixture(2)
        result = self.run_case(days, bars, [])
        for name, frame in result.items():
            self.assertGreater(len(frame.columns), 0)
            self.assertTrue(all(any("\u4e00" <= char <= "\u9fff" for char in column) for column in frame.columns))
            self.assertIsInstance(frame.to_csv(index=False), str)
        self.assertEqual(result["每日账户"]["账户权益"].tolist(), [20000, 20000])

    def test_invalid_calendar_duplicate_rows_and_lower_stop_rejected(self):
        days, bars = fixture(3)
        with self.assertRaises(ValueError):
            run_broker(bars, [], days[::-1], LOOSE)
        with self.assertRaises(ValueError):
            run_broker(pd.concat([bars, bars.iloc[:1]]), [], days, LOOSE)
        with self.assertRaises(ValueError):
            self.run_case(days, bars, close_instructions=[{"date": days[1], "code": "000001", "stop_adj": 8}])
        action = dict(ex_date=days[2], pay_date=days[2], code="000001",
                      share_multiplier=1.01, cash_dividend_per_old_share=.8)
        for invalid in ({"stock_pay_date": days[1]},
                        {"stock_pay_date": days[2] + pd.Timedelta(hours=1)},
                        {"stock_pay_date": days[2].tz_localize("UTC")},
                        {"cash_dividend_gross_per_old_share": -1},
                        {"cash_dividend_gross_per_old_share": float("inf")}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.run_case(days, bars, corp_actions=[{**action, **invalid}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
