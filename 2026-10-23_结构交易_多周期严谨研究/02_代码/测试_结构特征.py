"""纯合成行情单元测试；直接 python3 -B 本文件，不读真实数据、不回测收益。"""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from 结构特征 import build_features


BASE = (
    "open", "high", "low", "close", "ema6", "ema10", "up6", "up10",
    "pivot_high", "pivot_low", "previous_pivot_high", "previous_pivot_low",
    "higher_low", "up_pivot", "breakout", "pullback_trigger", "rsi14",
    "bear_divergence", "bull_low", "line_dist",
)
DAILY = (
    "ret20", "ret60", "atr14", "amount20", "volume20_prior",
    "volume5_prior", "volatility20", "age",
)
EVENTS = ("breakout", "pullback_trigger", "bear_divergence")
BOOLS = ("up6", "up10", "higher_low", "up_pivot") + EVENTS
STRUCTURE = [10, 11, 14, 12, 11, 12, 15, 13, 12, 13, 16, 17, 18]
DIVERGENCE = list(range(100, 121)) + list(range(119, 112, -1)) + list(range(114, 122)) + [120, 119, 118]


def make_bars(close, dates=None):
    close = np.asarray(close, dtype=float)
    if dates is None:
        dates = pd.bdate_range("2024-01-02", periods=len(close))
    return pd.DataFrame(
        {"open": close - 0.5, "high": close + 1, "low": close - 1,
         "close": close, "volume": np.arange(len(close), dtype=float) + 100,
         "amount": np.arange(len(close), dtype=float) * 1000 + 1_000_000},
        index=pd.DatetimeIndex(dates, name="date"),
    )


def periodic_bars(close, frequency):
    periods = pd.period_range("1960-01-01", periods=len(close), freq=frequency)
    calendar = pd.bdate_range(periods[0].start_time, periods[-1].end_time.normalize() + pd.Timedelta(days=14))
    availability = pd.Series(calendar, index=calendar.to_period(frequency)).groupby(level=0).max()
    dates = pd.DatetimeIndex(availability.reindex(periods))
    return make_bars(close, dates), calendar


class StructureFeatureTests(unittest.TestCase):
    def test_exact_schema_and_types(self):
        bars = make_bars(STRUCTURE)
        result = build_features(bars, bars.index)
        expected = [f"d_{column}" for column in BASE + DAILY]
        expected += [f"{prefix}_{column}" for prefix in ("w", "m", "q", "y") for column in BASE]
        self.assertEqual(list(result.columns), expected)
        self.assertEqual(len(expected), 108)
        self.assertEqual(result.index.name, "date")
        self.assertEqual(result.d_age.dtype, np.dtype("int64"))
        for prefix in ("d", "w", "m", "q", "y"):
            for column in BOOLS:
                self.assertEqual(result[f"{prefix}_{column}"].dtype, np.dtype(bool))
        self.assertFalse(result.isna().loc[:, [f"d_{column}" for column in BOOLS]].any().any())

    def test_pivots_confirm_two_bars_later_and_persist(self):
        bars = make_bars(STRUCTURE)
        result = build_features(bars, bars.index)
        self.assertTrue(result.d_pivot_high.iloc[:4].isna().all())
        self.assertEqual(result.d_pivot_high.iloc[4], 15)
        self.assertTrue(result.d_pivot_high.iloc[4:8].eq(15).all())
        self.assertTrue(result.d_previous_pivot_high.iloc[:8].isna().all())
        self.assertEqual(result.d_pivot_high.iloc[8], 16)
        self.assertEqual(result.d_previous_pivot_high.iloc[8], 15)
        self.assertTrue(result.d_pivot_low.iloc[:6].isna().all())
        self.assertEqual(result.d_pivot_low.iloc[6], 10)
        self.assertEqual(result.d_pivot_low.iloc[10], 11)
        self.assertEqual(result.d_previous_pivot_low.iloc[10], 10)
        self.assertFalse(result.d_higher_low.iloc[:10].any())
        self.assertTrue(result.d_higher_low.iloc[10])
        self.assertTrue(result.d_up_pivot.iloc[10])

    def test_equal_neighbor_extrema_are_not_pivots(self):
        bars = make_bars([10, 11, 14, 14, 11, 10])
        result = build_features(bars, bars.index)
        self.assertTrue(result.d_pivot_high.isna().all())
        bars = make_bars([14, 13, 10, 10, 13, 14])
        result = build_features(bars, bars.index)
        self.assertTrue(result.d_pivot_low.isna().all())

    def test_breakout_first_cross_and_pullback(self):
        bars = make_bars(STRUCTURE)
        result = build_features(bars, bars.index)
        self.assertFalse(result.d_breakout.iloc[10])  # 收盘等于高点不算上穿。
        self.assertTrue(result.d_breakout.iloc[11])
        self.assertTrue(result.d_pullback_trigger.iloc[11])
        self.assertFalse(result.d_breakout.iloc[12])
        self.assertFalse(result.d_pullback_trigger.iloc[12])
        self.assertEqual(result.d_breakout.sum(), 1)

    def test_breakout_uses_previous_known_high_on_confirmation_day(self):
        bars = make_bars([17, 18, 19, 18, 17, 20, 29, 19, 22, 24])
        result = build_features(bars, bars.index)
        self.assertEqual(result.d_pivot_high.iloc[7], 20)
        self.assertEqual(result.d_pivot_high.iloc[8], 30)
        self.assertTrue(result.d_breakout.iloc[8])  # 19→22 上穿昨天已知的 20，不是新 30。
        self.assertFalse(result.d_breakout.iloc[9])

    def test_ema_warmup_and_non_decreasing_line(self):
        bars = make_bars(np.arange(10, 30))
        result = build_features(bars, bars.index)
        for span in (6, 10):
            expected = bars.close.ewm(span=span, adjust=False, min_periods=span).mean()
            assert_series_equal(result[f"d_ema{span}"], expected, check_names=False, check_exact=True)
            self.assertTrue(result[f"d_ema{span}"].iloc[:span - 1].isna().all())
            self.assertFalse(result[f"d_up{span}"].iloc[:span].any())
            self.assertTrue(result[f"d_up{span}"].iloc[span:].all())
        np.testing.assert_allclose(result.d_line_dist.iloc[9:], bars.close.iloc[9:] / result.d_ema10.iloc[9:] - 1)

    def test_rsi_rolling_definition_and_zero_cases(self):
        for close, expected in ((np.arange(100, 120), 100), (np.arange(120, 100, -1), 0), (np.full(20, 100), 50)):
            with self.subTest(expected=expected):
                bars = make_bars(close)
                result = build_features(bars, bars.index)
                self.assertTrue(result.d_rsi14.iloc[:14].isna().all())
                self.assertTrue(result.d_rsi14.iloc[14:].eq(expected).all())

    def test_bear_divergence_uses_peak_rsi_and_confirmation_date(self):
        bars = make_bars(DIVERGENCE)
        result = build_features(bars, bars.index)
        first_peak, second_peak = 20, 35
        self.assertEqual(result.d_rsi14.iloc[first_peak], 100)
        delta = bars.close.diff().iloc[second_peak - 13:second_peak + 1]
        expected_rsi = 100 * delta.clip(lower=0).sum() / delta.abs().sum()
        self.assertAlmostEqual(result.d_rsi14.iloc[second_peak], expected_rsi)
        self.assertLess(expected_rsi, 100)
        self.assertEqual(result.index[result.d_bear_divergence].tolist(), [bars.index[second_peak + 2]])
        self.assertFalse(result.d_bear_divergence.iloc[second_peak])
        self.assertFalse(result.d_bear_divergence.iloc[second_peak + 1])

    def test_bull_low_is_last_completed_bullish_candle(self):
        bars = make_bars([10, 12, 11, 11, 13])
        bars.loc[bars.index[2:4], "open"] = [12, 11]
        result = build_features(bars, bars.index)
        np.testing.assert_array_equal(result.d_bull_low, [9, 11, 11, 11, 12])

    def test_daily_returns_atr_liquidity_and_volatility(self):
        bars = make_bars(np.arange(100, 170))
        result = build_features(bars, bars.index)
        self.assertTrue(result.d_ret20.iloc[:20].isna().all())
        self.assertTrue(result.d_ret60.iloc[:60].isna().all())
        self.assertAlmostEqual(result.d_ret20.iloc[20], 120 / 100 - 1)
        self.assertAlmostEqual(result.d_ret60.iloc[60], 160 / 100 - 1)
        self.assertTrue(result.d_atr14.iloc[:13].isna().all())
        self.assertEqual(result.d_atr14.iloc[13], 2)  # 价格单位，不是 ATR/close。
        self.assertEqual(result.d_amount20.iloc[19], bars.amount.iloc[:20].mean())
        self.assertTrue(result.d_volume20_prior.iloc[:20].isna().all())
        self.assertEqual(result.d_volume20_prior.iloc[20], bars.volume.iloc[:20].mean())
        self.assertEqual(result.d_volume5_prior.iloc[20], bars.volume.iloc[15:20].mean())
        self.assertAlmostEqual(result.d_volatility20.iloc[20], bars.close.pct_change(fill_method=None).iloc[1:21].std(ddof=1))
        changed = bars.copy()
        changed.loc[changed.index[20], "volume"] *= 100
        after = build_features(changed, bars.index)
        self.assertEqual(after.d_volume5_prior.iloc[20], result.d_volume5_prior.iloc[20])
        self.assertEqual(after.d_volume20_prior.iloc[20], result.d_volume20_prior.iloc[20])

    def test_future_price_volume_and_nan_changes_leave_past_identical(self):
        calendar = pd.bdate_range("2023-01-02", "2025-01-10")
        bars = make_bars(100 + np.sin(np.arange(len(calendar)) / 7) * 10, calendar)
        cutoff = pd.Timestamp("2023-08-16")
        original = build_features(bars, calendar).loc[:cutoff]
        for invalidate in (False, True):
            with self.subTest(invalidate=invalidate):
                changed = bars.copy()
                changed.loc[changed.index > cutoff] = np.nan if invalidate else changed.loc[changed.index > cutoff] * 13
                actual = build_features(changed, calendar).loc[:cutoff]
                assert_frame_equal(actual, original, check_exact=True)

    def test_every_cutoff_recompute_matches_full_sequence_exactly(self):
        calendar = pd.bdate_range("2023-11-01", "2024-04-10").drop(pd.Timestamp("2023-12-29"))
        rng = np.random.default_rng(17)
        bars = make_bars(100 + rng.normal(size=len(calendar)).cumsum(), calendar)
        bars.iloc[[12, 40, 73]] = np.nan
        bars = bars.drop(bars.index[[20, 21]])
        full = build_features(bars, calendar)
        for cutoff in calendar:
            with self.subTest(cutoff=cutoff):
                expected = full.loc[:cutoff]
                assert_frame_equal(build_features(bars, calendar, cutoff), expected, check_exact=True)
                assert_frame_equal(build_features(bars.loc[:cutoff], calendar, cutoff), expected, check_exact=True)

    def test_long_history_cutoffs_include_quarter_and_year_warmup(self):
        calendar = pd.bdate_range("2008-01-01", "2025-01-10")
        bars = make_bars(100 + np.arange(len(calendar)) / 200 + np.sin(np.arange(len(calendar)) / 23), calendar)
        full = build_features(bars, calendar)
        self.assertTrue(full.y_ema10.notna().any())
        for year in range(2018, 2025):
            for month, day in ((3, 29), (12, 30), (12, 31)):
                cutoff = pd.Timestamp(year, month, day)
                with self.subTest(cutoff=cutoff):
                    assert_frame_equal(build_features(bars.loc[:cutoff], calendar, cutoff), full.loc[:cutoff], check_exact=True)

    def test_friday_month_end_and_quarter_end_holiday(self):
        # 合成新股在3/25开始交易；日历覆盖完整季度，3/29为合成市场假期。
        calendar = pd.bdate_range("2024-01-01", "2024-04-05").drop(pd.Timestamp("2024-03-29"))
        listed_days = calendar[calendar >= "2024-03-25"]
        bars = make_bars(np.arange(10, 10 + len(listed_days)), listed_days)
        result = build_features(bars, calendar)
        for prefix in ("w", "m", "q"):
            with self.subTest(prefix=prefix):
                self.assertTrue(result.loc[:"2024-03-27", f"{prefix}_close"].isna().all())
                self.assertEqual(result.loc["2024-03-28", f"{prefix}_close"], bars.loc["2024-03-28", "close"])
                self.assertEqual(result.loc["2024-04-01", f"{prefix}_close"], bars.loc["2024-03-28", "close"])
                self.assertEqual(result.loc["2024-03-28", f"{prefix}_open"], bars.iloc[0].open)
        self.assertNotIn(pd.Timestamp("2024-03-29"), result.index)

    def test_year_end_holiday_uses_calendar_availability(self):
        calendar = pd.bdate_range("2021-01-01", "2022-01-07").drop(pd.Timestamp("2021-12-31"))
        listed_days = calendar[calendar >= "2021-12-27"]
        bars = make_bars(np.arange(10, 10 + len(listed_days)), listed_days)
        result = build_features(bars, calendar)
        self.assertTrue(result.y_close.loc[:"2021-12-29"].isna().all())
        self.assertEqual(result.y_close.loc["2021-12-30"], bars.close.loc["2021-12-30"])
        self.assertEqual(result.y_close.loc["2022-01-03"], bars.close.loc["2021-12-30"])

    def test_incomplete_initial_year_and_month_are_not_full_bars(self):
        calendar = pd.bdate_range("2016-10-13", "2018-01-05")
        bars = make_bars(np.arange(100, 100 + len(calendar)), calendar)
        result = build_features(bars, calendar)
        self.assertTrue(result.loc[:"2016-10-31", "m_close"].isna().all())
        self.assertTrue(result.loc[:"2017-12-28", "y_close"].isna().all())
        self.assertEqual(result.loc["2017-12-29", "y_open"], bars.loc["2017"].open.iloc[0])

    def test_unfinished_month_never_leaks_month_ohlc(self):
        calendar = pd.bdate_range("2024-01-01", "2024-03-08")
        bars = make_bars(np.arange(100, 100 + len(calendar)), calendar)
        bars.loc["2024-02-29", ["open", "high", "low", "close"]] = [1000, 1100, 900, 1050]
        full = build_features(bars, calendar)
        january = bars.loc["2024-01"]
        expected = {"open": january.open.iloc[0], "high": january.high.max(), "low": january.low.min(), "close": january.close.iloc[-1]}
        for column, value in expected.items():
            self.assertTrue(full.loc["2024-02-01":"2024-02-28", f"m_{column}"].eq(value).all())
        partial = build_features(bars, calendar, "2024-02-14")
        assert_frame_equal(partial, full.loc[:"2024-02-14"], check_exact=True)
        self.assertEqual(full.loc["2024-02-29", "m_high"], 1100)
        self.assertEqual(full.loc["2024-02-29", "m_close"], 1050)

    def test_incomplete_terminal_calendar_periods_are_conservatively_dropped(self):
        for end in ("2024-07-17", "2024-07-19"):
            calendar = pd.bdate_range("2024-07-15", end)
            bars = make_bars(np.arange(10, 10 + len(calendar)), calendar)
            result = build_features(bars, calendar)
            for prefix in ("w", "m", "q", "y"):
                self.assertTrue(result[f"{prefix}_close"].isna().all())
        calendar = pd.bdate_range("2024-07-01", "2024-07-31")
        bars = make_bars(np.arange(10, 10 + len(calendar)), calendar)
        result = build_features(bars, calendar)
        self.assertEqual(result.m_close.iloc[-1], bars.close.iloc[-1])
        self.assertTrue(result.q_close.isna().all())
        self.assertTrue(result.y_close.isna().all())

    def test_all_high_period_events_only_exist_on_availability_not_ffill(self):
        close = STRUCTURE + DIVERGENCE
        reference_bars = make_bars(close)
        reference = build_features(reference_bars, reference_bars.index)
        for event in EVENTS:
            self.assertGreater(reference[f"d_{event}"].sum(), 0)
        for prefix, frequency in (("w", "W-SUN"), ("m", "M"), ("q", "Q-DEC"), ("y", "Y-DEC")):
            with self.subTest(prefix=prefix):
                bars, calendar = periodic_bars(close, frequency)
                result = build_features(bars, calendar, calendar[-1])
                for event in EVENTS:
                    actual = result[f"{prefix}_{event}"]
                    self.assertEqual(actual.sum(), reference[f"d_{event}"].sum())
                    np.testing.assert_array_equal(actual.loc[bars.index], reference[f"d_{event}"])
                    self.assertFalse(actual.loc[~actual.index.isin(bars.index)].any())
                next_day = calendar[calendar > bars.index[11]][0]
                self.assertEqual(result.loc[next_day, f"{prefix}_close"], close[11])

    def test_suspended_high_period_availability_updates_state_but_not_trigger(self):
        sparse, calendar = periodic_bars(STRUCTURE, "W-SUN")
        period_prices = pd.Series(STRUCTURE, index=sparse.index.to_period("W-SUN"))
        dates = calendar[calendar <= sparse.index[-1]]
        close = period_prices.reindex(dates.to_period("W-SUN")).to_numpy()
        bars = make_bars(close, dates)
        event_date = sparse.index[11]
        baseline = build_features(bars, calendar)
        self.assertTrue(baseline.loc[event_date, "w_breakout"])
        bars.loc[event_date] = np.nan
        result = build_features(bars, calendar)
        self.assertEqual(result.loc[event_date, "w_close"], STRUCTURE[11])
        for event in EVENTS:
            self.assertFalse(result.loc[event_date, f"w_{event}"])
        next_day = calendar[calendar > event_date][0]
        self.assertFalse(result.loc[next_day, "w_breakout"])

    def test_nan_zero_volume_and_missing_rows_do_not_create_candles_or_events(self):
        bars = make_bars(STRUCTURE + [19, 20, 21])
        calendar = bars.index
        bars.iloc[12] = np.nan
        bars.loc[bars.index[13], "volume"] = 0
        bars = bars.drop(bars.index[14])
        result = build_features(bars, calendar)
        self.assertTrue(result.d_breakout.iloc[11])
        for position in (12, 13, 14):
            self.assertTrue(result.iloc[position][["d_open", "d_high", "d_low", "d_close"]].isna().all())
            self.assertEqual(result.d_age.iloc[position], 12)
            self.assertEqual(result.d_ema6.iloc[position], result.d_ema6.iloc[11])
            for prefix in ("d", "w", "m", "q", "y"):
                for event in EVENTS:
                    self.assertFalse(result.iloc[position][f"{prefix}_{event}"])
        self.assertEqual(result.d_age.iloc[-1], 13)
        self.assertTrue(pd.isna(result.d_volume5_prior.iloc[-1]))
        self.assertTrue(pd.isna(result.d_atr14.iloc[-1]))

    def test_suspension_does_not_count_as_pivot_confirmation_bar(self):
        bars = make_bars([10, 11, 14, 12, 11, 10])
        calendar = bars.index
        bars.iloc[3] = np.nan
        result = build_features(bars, calendar)
        self.assertTrue(result.d_pivot_high.iloc[:5].isna().all())
        self.assertEqual(result.d_pivot_high.iloc[5], 15)

    def test_new_listing_and_empty_periods_do_not_satisfy_warmup(self):
        calendar = pd.bdate_range("2024-01-01", "2024-03-08")
        bars = make_bars([10, 11, 12, 13, 14], calendar[20:25])
        result = build_features(bars, calendar, calendar[-1])
        self.assertTrue(result.d_age.iloc[:20].eq(0).all())
        self.assertTrue(result.d_age.iloc[24:].eq(5).all())
        for prefix in ("d", "w", "m", "q", "y"):
            self.assertTrue(result[f"{prefix}_ema6"].isna().all())
            self.assertTrue(result[f"{prefix}_ema10"].isna().all())
            self.assertFalse(result[f"{prefix}_up6"].any())
            self.assertFalse(result[f"{prefix}_up_pivot"].any())
        self.assertTrue(result.d_ret20.isna().all())
        self.assertTrue(result.d_rsi14.isna().all())
        self.assertTrue(result.d_amount20.isna().all())

    def test_data_end_not_delisting_and_default_cutoff_not_last_valid_row(self):
        calendar = pd.bdate_range("2024-01-01", "2024-03-08")
        bars = make_bars(STRUCTURE, calendar[:len(STRUCTURE)])
        bars.iloc[-1] = np.nan
        default = build_features(bars, calendar)
        self.assertEqual(default.index[-1], bars.index[-1])
        extended = build_features(bars, calendar, calendar[-1])
        assert_frame_equal(default, extended.loc[:bars.index[-1]], check_exact=True)
        self.assertEqual(extended.d_age.iloc[-1], len(bars) - 1)
        self.assertTrue(extended.d_close.iloc[len(bars) - 1:].isna().all())
        self.assertEqual(extended.w_close.iloc[-1], bars.close.iloc[-2])
        for prefix in ("d", "w", "m", "q", "y"):
            for event in EVENTS:
                self.assertFalse(extended[f"{prefix}_{event}"].iloc[len(bars) - 1:].any())

    def test_empty_input_and_all_invalid_keep_complete_schema(self):
        calendar = pd.bdate_range("2024-01-01", periods=6)
        empty = make_bars([], calendar[:0])
        no_cutoff = build_features(empty, calendar)
        self.assertEqual(no_cutoff.shape, (0, 108))
        result = build_features(empty, calendar, calendar[-1])
        self.assertEqual(result.shape, (6, 108))
        self.assertTrue(result.d_age.eq(0).all())
        invalid = make_bars([10] * len(calendar), calendar)
        invalid.iloc[:] = np.nan
        assert_frame_equal(build_features(invalid, calendar), result, check_exact=True)
        self.assertEqual(build_features(empty, calendar[:0]).shape, (0, 108))

    def test_unsorted_inputs_are_sorted_without_mutation(self):
        bars = make_bars(STRUCTURE)
        original = bars.copy(deep=True)
        calendar = bars.index
        result = build_features(bars.iloc[::-1], calendar[::-1])
        assert_frame_equal(result, build_features(bars, calendar), check_exact=True)
        assert_frame_equal(bars, original, check_exact=True)

    def test_mixed_datetime_resolutions_are_compatible(self):
        for bars_unit, calendar_unit in (("ns", "us"), ("us", "ns")):
            with self.subTest(bars_unit=bars_unit, calendar_unit=calendar_unit):
                bars = make_bars(STRUCTURE)
                calendar = bars.index.as_unit(calendar_unit)
                bars.index = bars.index.as_unit(bars_unit)
                reference = bars.copy()
                reference.index = reference.index.as_unit(calendar_unit)
                assert_frame_equal(build_features(bars, calendar), build_features(reference, calendar), check_exact=True)

    def test_duplicate_dates_missing_columns_and_off_calendar_dates_rejected(self):
        bars = make_bars(STRUCTURE)
        with self.assertRaises(ValueError):
            build_features(pd.concat([bars, bars.iloc[:1]]), bars.index)
        with self.assertRaises(ValueError):
            build_features(bars, bars.index.append(bars.index[:1]))
        with self.assertRaises(ValueError):
            build_features(bars.drop(columns="amount"), bars.index)
        with self.assertRaises(ValueError):
            build_features(bars, bars.index[1:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
