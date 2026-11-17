import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from trend_system.legacy_a import (
    _method_hits,
    audit_legacy_a_entries,
    compute_legacy_a_signals,
    order_legacy_a_candidates,
)
from trend_system.legacy_config import load_legacy_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEYS = ["date", "code"]
DIAGNOSTICS = [
    "ma5",
    "ma20",
    "amount_ma20",
    "volume_ma20",
    "high20_prev",
    "low10",
    "wdif",
    "wdea",
    "whist",
    "m1",
    "m2",
    "m3",
    "hit_count",
    "market_ok",
    "vol120",
    "range250",
    "act",
    "limfrac",
    "gapfrac",
    "s_vol",
    "s_heat",
    "raw_signal",
    "accepted_signal",
    "entry_signal",
    "rank_return_120",
]


class LegacyATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_legacy_config(PROJECT_ROOT / "config" / "legacy_candidates.json")
        cls.dates = pd.bdate_range("2015-09-01", periods=330).strftime("%Y-%m-%d")
        cls.target_row = 300
        cls.target_date = cls.dates[cls.target_row]
        cls.target_code = "600000.SH"

    def _panels(self):
        rows = []
        codes = [self.target_code, "000001.SZ", "300001.SZ"]
        for t, date in enumerate(self.dates):
            base = 10.0 * 1.001**t
            for code in codes:
                if code == self.target_code:
                    close = base * (1.10 if t == 100 else 1.0)
                    previous_close = base / 1.001 * (1.10 if t == 101 else 1.0)
                    amount = 40_000_000.0 if t < 270 else 100_000_000.0
                elif code == "000001.SZ":
                    close = base * (1.03 if t % 2 else 0.97)
                    previous_close = base / 1.001 * (1.03 if (t - 1) % 2 else 0.97)
                    amount = 50_000_000.0
                else:
                    close = base * (1.0 + 0.03 * np.sin(t / 5.0))
                    previous_close = base / 1.001 * (
                        1.0 + 0.03 * np.sin((t - 1) / 5.0)
                    )
                    amount = 80_000_000.0 if t < 270 else 40_000_000.0
                open_ = previous_close
                if code == self.target_code and t % 10 == 0 and t < 270:
                    open_ = previous_close * 1.03
                high = max(open_, close) + 0.001
                low = min(open_, close) - 0.001
                rows.append(
                    {
                        "date": date,
                        "code": code,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "prev_close": previous_close,
                        "signal_open": open_,
                        "signal_high": high,
                        "signal_low": low,
                        "signal_close": close,
                        "volume": 1_000_000.0,
                        "amount": amount,
                        "trade_status": 1,
                        "is_st": 0,
                        "ipo_date": "2010-01-01",
                        "out_date": None,
                        "is_main_board": code != "300001.SZ",
                        "data_valid": True,
                        "context_complete": True,
                        "execution_conflict": False,
                        "adjustment_source_conflict": False,
                        "source_conflict": False,
                        "rank_eligible": True,
                    }
                )
        context = pd.DataFrame(rows)
        context["volume"] = np.where(
            context["code"].eq(self.target_code)
            & context["date"].eq(self.target_date),
            2_000_000.0,
            context["volume"],
        )
        main = context.loc[context["is_main_board"]].copy().reset_index(drop=True)
        index = pd.DataFrame(
            {
                "code": "000001.SH",
                "date": self.dates,
                "close": 100.0 + np.arange(len(self.dates), dtype=float),
            }
        )
        return context, main, index

    def _compute(self, context=None, main=None, index=None, config=None):
        default_context, default_main, default_index = self._panels()
        return compute_legacy_a_signals(
            default_context if context is None else context,
            default_main if main is None else main,
            default_index if index is None else index,
            self.config if config is None else config,
        )

    def _row(self, result, date=None, code=None):
        selected = result.loc[
            result["date"].eq(date or self.target_date)
            & result["code"].eq(code or self.target_code)
        ]
        self.assertEqual(len(selected), 1)
        return selected.iloc[0]

    def test_formulas_and_two_of_three_methods_match_preregistration(self):
        context, _, _ = self._panels()
        result = self._compute(context=context)
        row = self._row(result)
        stock = context.loc[context["code"].eq(self.target_code)].sort_values("date")
        close = stock["signal_close"].reset_index(drop=True)
        high = stock["signal_high"].reset_index(drop=True)
        low = stock["signal_low"].reset_index(drop=True)
        volume = stock["volume"].reset_index(drop=True)
        amount = stock["amount"].reset_index(drop=True)
        t = self.target_row
        wdif = close.ewm(span=34, adjust=False).mean() - close.ewm(
            span=200, adjust=False
        ).mean()
        wdea = wdif.ewm(span=13, adjust=False).mean()

        self.assertAlmostEqual(row["ma5"], close.iloc[t - 4 : t + 1].mean())
        self.assertAlmostEqual(row["ma20"], close.iloc[t - 19 : t + 1].mean())
        self.assertAlmostEqual(row["amount_ma20"], amount.iloc[t - 19 : t + 1].mean())
        self.assertAlmostEqual(row["volume_ma20"], volume.iloc[t - 19 : t + 1].mean())
        self.assertEqual(row["high20_prev"], high.iloc[t - 20 : t].max())
        self.assertEqual(row["low10"], low.iloc[t - 9 : t + 1].min())
        self.assertAlmostEqual(row["wdif"], wdif.iloc[t])
        self.assertAlmostEqual(row["wdea"], wdea.iloc[t])
        self.assertAlmostEqual(row["whist"], 2 * (wdif.iloc[t] - wdea.iloc[t]))
        self.assertFalse(row["m1"])
        self.assertTrue(row["m2"])
        self.assertTrue(row["m3"])
        self.assertEqual(row["hit_count"], 2)
        self.assertTrue(row["market_ok"])
        self.assertTrue(row["raw_signal"])
        expected_score = 4 + 0.2 * volume.iloc[t] / volume.iloc[t - 19 : t + 1].mean()
        self.assertAlmostEqual(row["legacy_rank_score"], expected_score)
        self.assertAlmostEqual(row["rank_return_120"], expected_score)
        donchian_return_120 = close.iloc[t] / close.iloc[t - 120] - 1
        self.assertNotAlmostEqual(row["legacy_rank_score"], donchian_return_120)
        self.assertEqual(row["source_row_index"], t)

    def test_original_score_orders_descending_with_code_tie_break(self):
        candidates = pd.DataFrame(
            {
                "date": ["2016-10-25"] * 3,
                "code": ["000002.SZ", "000001.SZ", "600000.SH"],
                "legacy_rank_score": [4.2, 4.2, 4.4],
                "rank_return_120": [4.2, 4.2, 4.4],
                "signal_close": [1000.0, 500.0, 0.1],
            }
        )
        ordered = order_legacy_a_candidates(candidates)
        self.assertEqual(
            ordered["code"].tolist(), ["600000.SH", "000001.SZ", "000002.SZ"]
        )
        self.assertEqual(ordered["legacy_rank_score"].tolist(), [4.4, 4.2, 4.2])

    def test_any_two_methods_are_required_and_missing_boolean_fails_closed(self):
        m1 = pd.Series([True, True, False, True, pd.NA], dtype="boolean")
        m2 = pd.Series([True, False, True, True, pd.NA], dtype="boolean")
        m3 = pd.Series([False, True, True, False, pd.NA], dtype="boolean")
        hits, enough = _method_hits(m1, m2, m3)
        self.assertEqual(hits.tolist(), [2, 2, 2, 2, 0])
        self.assertEqual(enough.tolist(), [True, True, True, True, False])

    def test_cross_section_uses_all_a_percentiles_and_strict_boundaries(self):
        context, main, index = self._panels()
        result = self._compute(context=context, main=main, index=index)
        row = self._row(result)
        self.assertAlmostEqual(row["s_vol"], 1 / 3)
        self.assertAlmostEqual(row["s_heat"], 1.0)
        self.assertTrue(row["accepted_signal"])
        self.assertTrue(row["entry_signal"])

        without_non_main = context.loc[context["is_main_board"]].copy()
        reduced = self._compute(context=without_non_main, main=main, index=index)
        reduced_row = self._row(reduced)
        self.assertAlmostEqual(reduced_row["s_vol"], 0.5)
        self.assertFalse(reduced_row["raw_signal"])

        no_cross_section = context.copy()
        no_cross_section["rank_eligible"] = False
        empty = self._compute(context=no_cross_section, main=main, index=index)
        self.assertFalse(empty["raw_signal"].any())

    def test_future_industry_and_ex_right_rows_do_not_change_history(self):
        context, main, index = self._panels()
        future = context["date"].gt(self.target_date)
        context["industry"] = "historical"
        context.loc[future, "industry"] = "future-industry"
        context.loc[
            future, ["signal_open", "signal_high", "signal_low", "signal_close"]
        ] *= 0.5
        context.loc[future, ["volume", "amount"]] = 0.0
        full = self._compute(context=context, main=main, index=index)

        truncated = self._compute(
            context=context.loc[context["date"].le(self.target_date)].copy(),
            main=main.loc[main["date"].le(self.target_date)].copy(),
            index=index.loc[index["date"].le(self.target_date)].copy(),
        )
        historical = full.loc[full["date"].le(self.target_date)]
        compared = historical[KEYS + DIAGNOSTICS].merge(
            truncated[KEYS + DIAGNOSTICS],
            on=KEYS,
            how="outer",
            validate="one_to_one",
            suffixes=("_full", "_truncated"),
        )
        self.assertEqual(len(compared), len(truncated))
        for column in DIAGNOSTICS:
            pd.testing.assert_series_equal(
                compared[f"{column}_full"],
                compared[f"{column}_truncated"],
                check_names=False,
                check_exact=True,
            )

    def test_cooldown_uses_accepted_original_row_indices(self):
        context, main, index = self._panels()
        spike_dates = self.dates[self.target_row : self.target_row + 6]
        spike = context["code"].eq(self.target_code) & context["date"].isin(spike_dates)
        context.loc[spike, "volume"] = 100_000_000.0
        main_spike = main["code"].eq(self.target_code) & main["date"].isin(spike_dates)
        main.loc[main_spike, "volume"] = 100_000_000.0
        result = self._compute(context=context, main=main, index=index)
        stock = result.loc[
            result["code"].eq(self.target_code) & result["date"].isin(spike_dates)
        ]

        self.assertEqual(stock.loc[stock["raw_signal"], "source_row_index"].tolist(), list(range(300, 306)))
        self.assertEqual(stock.loc[stock["accepted_signal"], "source_row_index"].tolist(), [300, 305])
        self.assertEqual(stock["cooldown_rejected"].tolist(), [False, True, True, True, True, False])
        accepted = stock.loc[stock["accepted_signal"]]
        self.assertTrue(accepted["source_row_index"].diff().dropna().ge(5).all())
        self.assertEqual(self._row(stock, spike_dates[-1])["previous_accepted_row_index"], 300)

    def test_low10_includes_signal_day_and_retains_stop_metadata(self):
        context, main, index = self._panels()
        target = context["code"].eq(self.target_code) & context["date"].eq(self.target_date)
        stock = context.loc[context["code"].eq(self.target_code)].sort_values("date")
        previous_low = stock["signal_low"].iloc[
            self.target_row - 9 : self.target_row
        ].min()
        changed_low = previous_low - 0.001
        context.loc[target, "signal_low"] = changed_low
        main_target = main["code"].eq(self.target_code) & main["date"].eq(self.target_date)
        main.loc[main_target, "signal_low"] = changed_low
        row = self._row(self._compute(context=context, main=main, index=index))
        self.assertEqual(row["low10"], changed_low)
        self.assertEqual(row["stop_ref"], changed_low)
        self.assertEqual(row["stop_pct"], 0.08)
        self.assertTrue(row["entry_signal"])

    def test_st_conflict_non_main_and_out_of_window_permissions_fail_closed(self):
        for field, value in (("is_st", 1), ("execution_conflict", True)):
            with self.subTest(field=field):
                context, main, index = self._panels()
                for frame in (context, main):
                    target = frame["code"].eq(self.target_code) & frame["date"].eq(self.target_date)
                    frame.loc[target, field] = value
                row = self._row(self._compute(context=context, main=main, index=index))
                self.assertTrue(row["accepted_signal"])
                self.assertTrue(row["proposed_entry"])
                self.assertFalse(row["entry_signal"])

        context, main, index = self._panels()
        context["is_main_board"] = False
        non_main = main.copy()
        non_main["is_main_board"] = False
        result = self._compute(context=context, main=non_main, index=index)
        self.assertFalse(result["entry_signal"].any())

        window_config = replace(self.config, start_date="2016-12-01")
        row = self._row(
            self._compute(context=context, main=main, index=index, config=window_config)
        )
        self.assertTrue(row["accepted_signal"])
        self.assertFalse(row["proposed_entry"])
        self.assertFalse(row["entry_signal"])

    def test_suspension_and_zero_volume_cannot_enter(self):
        for field, value in (("trade_status", 0), ("volume", 0.0)):
            with self.subTest(field=field):
                context, main, index = self._panels()
                for frame in (context, main):
                    target = frame["code"].eq(self.target_code) & frame["date"].eq(self.target_date)
                    frame.loc[target, field] = value
                    if field == "trade_status":
                        frame.loc[target, "rank_eligible"] = False
                row = self._row(self._compute(context=context, main=main, index=index))
                self.assertFalse(row["entry_signal"])

    def test_main_key_outside_point_in_time_context_is_retained_without_signal(self):
        context, main, index = self._panels()
        target = context["code"].eq(self.target_code) & context["date"].eq(self.target_date)
        context = context.loc[~target].copy()
        result = self._compute(context=context, main=main, index=index)
        row = self._row(result)
        self.assertFalse(row["context_key_present"])
        self.assertFalse(row["raw_signal"])
        self.assertFalse(row["entry_signal"])
        self.assertEqual(len(result), len(main))

    def test_completed_context_hfq_is_aligned_by_key_for_entry(self):
        context, main, index = self._panels()
        target = main["code"].eq(self.target_code) & main["date"].eq(self.target_date)
        main.loc[target, "signal_open"] = np.nan
        result = self._compute(context=context, main=main, index=index)
        row = self._row(result)
        self.assertTrue(row["proposed_entry"])
        self.assertTrue(np.isfinite(row["signal_open"]))
        self.assertTrue(row["entry_signal"])

    def test_entry_audit_covers_only_proposed_keys_and_fails_on_one_invalid(self):
        context, main, index = self._panels()
        signals = self._compute(context=context, main=main, index=index)
        audit, summary = audit_legacy_a_entries(signals, context, self.config)
        self.assertEqual(len(audit), int(signals["proposed_entry"].sum()))
        self.assertEqual(summary["valid_numerator"], len(audit))
        self.assertEqual(summary["coverage"], 1.0)
        self.assertTrue(summary["passes"])

        for frame in (context, main):
            target = frame["code"].eq(self.target_code) & frame["date"].eq(self.target_date)
            frame.loc[target, "signal_open"] = np.nan
        invalid_signals = self._compute(context=context, main=main, index=index)
        invalid_audit, invalid_summary = audit_legacy_a_entries(
            invalid_signals, context, self.config
        )
        self.assertTrue(self._row(invalid_signals)["proposed_entry"])
        self.assertFalse(self._row(invalid_signals)["entry_signal"])
        self.assertEqual(invalid_summary["coverage"], 0.0)
        self.assertFalse(invalid_summary["passes"])
        self.assertEqual(invalid_summary["candidate_a_status"], "NOT_EVALUABLE")
        self.assertIn("signal_open", invalid_audit.iloc[0]["missing_fields"])
        self.assertFalse(invalid_audit.iloc[0]["candidate_entry_valid"])


if __name__ == "__main__":
    unittest.main()
