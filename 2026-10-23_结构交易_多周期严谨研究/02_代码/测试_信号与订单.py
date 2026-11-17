"""信号层纯合成回归测试；不测真实收益。"""
from dataclasses import replace
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from 结构特征 import build_features
from 信号与订单 import make_orders, close_instruction
from 策略注册 import registry, Strategy


class SignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.days = pd.bdate_range("2017-01-02", periods=1250)
        n = np.arange(len(cls.days))
        close = 10 + n * .005 + np.sin(n / 6) * .6 + np.sin(n / 25) * .8
        cls.raw = pd.DataFrame({"open": close - .02, "high": close + .08, "low": close - .08,
                                "close": close, "volume": 1e7, "amount": 1e8}, index=cls.days)
        cls.f = build_features(cls.raw, cls.days, cls.days[-1])
        cls.context = {"market_breadth60": pd.Series(.6, index=cls.days),
                       "industry_breadth60": pd.Series(.6, index=cls.days),
                       "industry_ret20": pd.Series(.1, index=cls.days)}

    def test_inputs_unchanged_and_strategy_order_independent(self):
        original = self.f.copy(deep=True)
        results = {s.name: make_orders("000001.SZ", self.raw, self.f, s, self.context) for s in registry()}
        assert_frame_equal(original, self.f)
        for s in reversed(registry()):
            assert_frame_equal(results[s.name], make_orders("000001.SZ", self.raw, self.f, s, self.context))
        self.assertGreater(len(results["日突破_无上级过滤"]), 0)

    def test_prefix_matches_truncated_features(self):
        cutoff = self.days[900]
        partial = build_features(self.raw.loc[:cutoff], self.days, cutoff)
        for s in registry()[:5]:
            full = make_orders("000001.SZ", self.raw, self.f, s, self.context)
            truncated = make_orders("000001.SZ", self.raw.loc[:cutoff], partial, s, self.context)
            assert_frame_equal(full[full.signal_date <= cutoff].reset_index(drop=True), truncated)

    def test_mainboard_only_user_permission(self):
        s = registry()[0]
        self.assertEqual(len(make_orders("688001.SH", self.raw, self.f, s, self.context)), 0)
        self.assertEqual(len(make_orders("300001.SZ", self.raw, self.f, s, self.context)), 0)
        unrestricted = dict(self.context, mainboard_only=False)
        self.assertGreater(len(make_orders("688001.SH", self.raw, self.f, s, unrestricted)), 0)

    def test_known_st_excluded_unknown_not_mislabelled(self):
        s = registry()[0]
        unknown = make_orders("000001.SZ", self.raw, self.f, s, self.context)
        self.assertTrue((~unknown.st_status_known).all())
        known = self.raw.assign(isST=1)
        self.assertTrue(make_orders("000001.SZ", known, self.f, s, self.context).empty)
        nonst = make_orders("000001.SZ", self.raw.assign(isST=0), self.f, s, self.context)
        self.assertEqual(len(nonst), len(unknown))
        self.assertTrue(nonst.st_status_known.all())

    def test_risk_distance_and_source_dates(self):
        result = make_orders("000001.SZ", self.raw, self.f, registry()[0], self.context)
        self.assertTrue((result.signal_date >= "2018-01-01").all())
        self.assertTrue(result.stop_distance.between(0, .20, inclusive="right").all())
        np.testing.assert_allclose(result.initial_risk_adj, result.signal_close_adj - result.stop_adj)

    def test_missing_context_is_explicit_error(self):
        with self.assertRaises(ValueError):
            make_orders("000001.SZ", self.raw, self.f, Strategy("市场", market_gate=True), {})

    def test_pivot_episode_key_uses_pivot_not_ema_state(self):
        base = registry()[0]
        available = make_orders("000001.SZ", self.raw, self.f, base, self.context)
        chosen = available.signal_date.iloc[[0, 2]].tolist()
        f = self.f.copy(deep=True)
        f["w_up10"] = False
        f["w_up_pivot"] = False
        f["d_breakout"] = False
        for d in chosen:
            f.loc[d, ["w_up_pivot", "d_breakout"]] = True
        actual = make_orders("000001.SZ", self.raw, f, replace(base, line="pivot"), self.context)
        self.assertEqual(actual.episode_key.nunique(), 2)
        self.assertEqual(actual.episode_key.tolist(), ["000001.SZ_" + d.strftime("%Y%m%d") for d in chosen])

    def test_close_partial_is_not_repeated_and_stop_does_not_lower(self):
        pos = {"代码": "000001.SZ", "止损后复权": 8.0}
        row = {"w_pivot_low": 7.0, "d_rsi14": 80, "d_line_dist": .12, "w_bull_low": 9.0}
        s = Strategy("部分", exit="partial")
        first = close_instruction(s, row, pos, 5)
        self.assertEqual(first["next_open_sell_fraction"], .5)
        self.assertEqual(first["stop_adj"], 9.0)
        following = close_instruction(s, row, pos, 6, True)
        self.assertNotIn("next_open_sell_fraction", following)
        self.assertEqual(following["stop_adj"], 9.0)

    def test_twenty_day_exit_is_next_open_instruction(self):
        pos = {"代码": "000001.SZ", "止损后复权": 8.0}
        row = {"w_pivot_low": 7.0}
        s = Strategy("时间", exit="time20")
        self.assertIsNone(close_instruction(s, row, pos, 19))
        self.assertEqual(close_instruction(s, row, pos, 20)["next_open_sell_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
