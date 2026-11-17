import unittest

import pandas as pd

from trend_system.metrics import annual_and_rolling, performance_summary


class MetricsTest(unittest.TestCase):
    def test_known_equity_path_and_two_round_trips(self):
        account = pd.DataFrame(
            {
                "date": ["2016-10-13", "2016-12-30", "2017-12-29", "2018-12-31"],
                "equity": [10000.0, 12000.0, 9000.0, 10800.0],
                "exposure": [0.0, 0.5, 0.25, 0.75],
            }
        )
        round_trips = pd.DataFrame(
            {
                "code": ["600000.SH", "000001.SZ"],
                "entry_date": ["2017-01-03", "2017-02-03"],
                "exit_date": ["2017-03-03", "2017-04-03"],
                "entry_amount": [1000.0, 800.0],
                "exit_amount": [1140.0, 740.0],
                "net_pnl": [120.0, -40.0],
            }
        )

        summary = performance_summary(account, round_trips)

        self.assertAlmostEqual(summary["total_return"], 0.08)
        expected_cagr = 1.08 ** (365.25 / 809) - 1
        self.assertAlmostEqual(summary["cagr"], expected_cagr)
        self.assertAlmostEqual(summary["max_drawdown"], 0.25)
        self.assertAlmostEqual(summary["annual_returns"]["2016"], 0.20)
        self.assertAlmostEqual(summary["annual_returns"]["2017"], -0.25)
        self.assertAlmostEqual(summary["annual_returns"]["2018"], 0.20)
        self.assertEqual(summary["partial_years"], ["2016"])
        self.assertAlmostEqual(summary["average_exposure"], 0.375)
        self.assertAlmostEqual(summary["max_exposure"], 0.75)
        self.assertAlmostEqual(summary["closed_round_trip_turnover"], 3680.0 / 10450.0)
        self.assertAlmostEqual(summary["net_expectancy"], 40.0)
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertAlmostEqual(summary["average_win"], 120.0)
        self.assertAlmostEqual(summary["average_loss"], -40.0)
        self.assertEqual(summary["longest_losing_streak"], 1)
        self.assertEqual(summary["realized_annual_contribution"], {"2017": 80.0})
        self.assertEqual(
            summary["realized_stock_contribution"],
            {"000001.SZ": -40.0, "600000.SH": 120.0},
        )

    def test_longest_losing_streak_uses_exit_order(self):
        account = pd.DataFrame(
            {"date": ["2020-01-02", "2021-01-04"], "equity": [100.0, 80.0], "exposure": [0.0, 0.0]}
        )
        round_trips = pd.DataFrame(
            {
                "code": list("ABCDEF"),
                "entry_date": ["2020-01-02"] * 6,
                "exit_date": [
                    "2020-02-03",
                    "2020-01-06",
                    "2020-03-03",
                    "2020-01-07",
                    "2020-04-03",
                    "2020-05-03",
                ],
                "entry_amount": [100.0] * 6,
                "exit_amount": [90.0, 90.0, 90.0, 110.0, 90.0, 110.0],
                "net_pnl": [-10.0, -10.0, -10.0, 10.0, -10.0, 10.0],
            }
        )

        summary = performance_summary(account, round_trips)

        self.assertEqual(summary["longest_losing_streak"], 3)

    def test_top_five_profitable_trade_contribution(self):
        account = pd.DataFrame(
            {"date": ["2020-01-02", "2020-12-31"], "equity": [100.0, 200.0], "exposure": [0.0, 0.0]}
        )
        profits = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, -20.0]
        round_trips = pd.DataFrame(
            {
                "code": [f"60000{index}.SH" for index in range(len(profits))],
                "entry_date": ["2020-01-02"] * len(profits),
                "exit_date": pd.date_range("2020-02-03", periods=len(profits), freq="D").astype(str),
                "entry_amount": [100.0] * len(profits),
                "exit_amount": [100.0 + value for value in profits],
                "net_pnl": profits,
            }
        )

        summary = performance_summary(account, round_trips)

        self.assertAlmostEqual(summary["top5_closed_profitable_trades_net_pnl"], 400.0)
        self.assertAlmostEqual(summary["top5_closed_profitable_trades_share_of_gross_profit"], 400.0 / 450.0)

    def test_annual_rows_mark_partial_2016_and_use_previous_year_end(self):
        account = pd.DataFrame(
            {
                "date": ["2016-10-13", "2016-12-30", "2017-12-29", "2018-12-31"],
                "equity": [100.0, 120.0, 90.0, 108.0],
            }
        )

        result = annual_and_rolling(account)
        annual = result.loc[result["period_type"].isin(["annual", "annual_partial"])]

        self.assertEqual(annual["period"].tolist(), ["2016", "2017", "2018"])
        self.assertEqual(
            annual["period_type"].tolist(),
            ["annual_partial", "annual", "annual"],
        )
        self.assertEqual(annual["is_partial"].tolist(), [True, False, False])
        self.assertAlmostEqual(annual.iloc[0]["return_pct"], 0.20)
        self.assertAlmostEqual(annual.iloc[1]["return_pct"], -0.25)
        self.assertAlmostEqual(annual.iloc[2]["return_pct"], 0.20)

    def test_month_end_rolling_three_year_uses_actual_dates_not_row_counts(self):
        dates = pd.bdate_range("2016-10-13", "2020-11-30")
        dates = dates[~((dates.year == 2018) & (dates.month == 2))]
        equities = 100.0 * 1.001 ** pd.Series(range(len(dates)), index=dates)
        account = pd.DataFrame(
            {"date": dates.astype(str), "equity": equities.to_numpy()}
        )

        result = annual_and_rolling(account)
        rolling = result.loc[result["period_type"] == "rolling_3y"].reset_index(drop=True)

        self.assertEqual(rolling.iloc[0]["period"], "2019-10")
        self.assertEqual(rolling.iloc[0]["start_date"], "2016-10-31")
        self.assertEqual(rolling.iloc[0]["end_date"], "2019-10-31")
        expected_start = equities.loc[pd.Timestamp("2016-10-31")]
        expected_end = equities.loc[pd.Timestamp("2019-10-31")]
        self.assertAlmostEqual(rolling.iloc[0]["return_pct"], expected_end / expected_start - 1)
        self.assertEqual(rolling.iloc[-1]["period"], "2020-11")
        self.assertEqual(rolling.iloc[-1]["start_date"], "2017-11-30")
        self.assertTrue((pd.to_datetime(rolling["end_date"]) - pd.to_datetime(rolling["start_date"])).dt.days.ge(1094).all())

    def test_rolling_three_year_skips_asof_start_beyond_31_day_gap(self):
        dates = pd.DatetimeIndex(
            ["2016-01-15", "2017-01-15", "2017-03-15", "2020-02-29", "2020-03-31"]
        )
        account = pd.DataFrame(
            {"date": dates.astype(str), "equity": [100.0, 110.0, 120.0, 130.0, 140.0]}
        )

        result = annual_and_rolling(account)
        rolling = result.loc[result["period_type"] == "rolling_3y"]

        # 2020-02 的目标日为 2017-02-28；asof 只能回到 2017-01-15，缺口 44 天。
        self.assertNotIn("2020-02", rolling["period"].tolist())
        march = rolling.loc[rolling["period"] == "2020-03"].iloc[0]
        self.assertEqual(march["start_date"], "2017-03-15")
        self.assertAlmostEqual(march["return_pct"], 140.0 / 120.0 - 1)

    def test_rolling_three_year_accepts_asof_gap_of_exactly_31_days(self):
        dates = pd.DatetimeIndex(
            ["2016-01-15", "2017-02-28", "2017-04-15", "2020-02-29", "2020-03-31"]
        )
        account = pd.DataFrame(
            {"date": dates.astype(str), "equity": [100.0, 110.0, 120.0, 130.0, 140.0]}
        )

        result = annual_and_rolling(account)
        rolling = result.loc[result["period_type"] == "rolling_3y"]

        march = rolling.loc[rolling["period"] == "2020-03"].iloc[0]
        self.assertEqual(march["start_date"], "2017-02-28")
        self.assertEqual((pd.Timestamp("2017-03-31") - pd.Timestamp("2017-02-28")).days, 31)

    def test_empty_round_trips_have_neutral_trade_metrics(self):
        account = pd.DataFrame(
            {"date": ["2020-01-02", "2020-12-31"], "equity": [100.0, 100.0], "exposure": [0.0, 0.0]}
        )
        round_trips = pd.DataFrame(
            columns=["code", "entry_date", "exit_date", "entry_amount", "exit_amount", "net_pnl"]
        )

        summary = performance_summary(account, round_trips)

        self.assertEqual(summary["round_trip_count"], 0)
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertEqual(summary["net_expectancy"], 0.0)
        self.assertEqual(summary["longest_losing_streak"], 0)
        self.assertEqual(summary["top5_closed_profitable_trades_net_pnl"], 0.0)


if __name__ == "__main__":
    unittest.main()
