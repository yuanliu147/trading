import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from trend_system.legacy_b import compute_legacy_b_signals, load_industry_csv
from trend_system.legacy_config import load_legacy_config


MAIN_A = "600000.SH"
MAIN_B = "000001.SZ"
NON_MAIN = "300001.SZ"


class LegacyBSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).resolve().parents[1] / "config" / "legacy_candidates.json"
        cls.config = load_legacy_config(config_path)

    def _panel(self, dates, future_override=None):
        rows = []
        for i, day in enumerate(dates):
            for code, base, growth in (
                (MAIN_A, 10.0, 0.01),
                (MAIN_B, 20.0, 0.01),
                (NON_MAIN, 5.0, 0.01),
            ):
                close = base * (1 + growth) ** i
                if future_override and day > future_override[0]:
                    close = future_override[1]
                prev = base * (1 + growth) ** max(i - 1, 0)
                rows.append({
                    "date": day,
                    "code": code,
                    "open": close * 0.999,
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "prev_close": prev,
                    "signal_open": close * 0.999,
                    "signal_high": close * 1.005,
                    "signal_low": close * 0.995,
                    "signal_close": close,
                    "volume": 1_000_000.0,
                    "amount": 100_000_000.0,
                    "trade_status": 1,
                    "is_st": 0,
                    "ipo_date": "2000-01-01",
                    "execution_conflict": False,
                    "data_valid": True,
                    "is_main_board": code != NON_MAIN,
                })
        return pd.DataFrame(rows)

    def _industry(self):
        return pd.DataFrame({
            "code": [MAIN_A, MAIN_B, NON_MAIN],
            "start_date": ["2000-01-01"] * 3,
            "industry": ["银行"] * 3,
        })

    def test_entry_after_warmup_and_eligibility_with_pit_industry(self):
        dates = pd.bdate_range("2017-06-01", periods=120)
        panel = self._panel(dates)
        main = panel.loc[panel["is_main_board"]].copy()
        result = compute_legacy_b_signals(panel, main, self._industry(), self.config)
        entries = result.signals.loc[result.signals["entry_signal"]]
        self.assertGreater(len(entries), 0)
        self.assertGreaterEqual(entries["date"].min(), "2017-10-13")
        self.assertTrue(entries["code"].isin([MAIN_A, MAIN_B]).all())
        self.assertEqual(result.audit["industry_entry_coverage"], 1.0)
        self.assertEqual(result.audit["required_entry_coverage"], 1.0)

    def test_future_prices_do_not_change_past_entries(self):
        dates = pd.bdate_range("2017-06-01", periods=120)
        base = self._panel(dates)
        changed = self._panel(dates, future_override=(dates[100], 1.0))
        main_base = base.loc[base["is_main_board"]].copy()
        main_changed = changed.loc[changed["is_main_board"]].copy()
        first = compute_legacy_b_signals(base, main_base, self._industry(), self.config)
        second = compute_legacy_b_signals(changed, main_changed, self._industry(), self.config)
        left = first.signals.loc[first.signals["date"] <= str(dates[100].date())]
        right = second.signals.loc[second.signals["date"] <= str(dates[100].date())]
        self.assertEqual(left["entry_signal"].tolist(), right["entry_signal"].tolist())

    def test_stock_already_st_is_not_bought(self):
        dates = pd.bdate_range("2017-06-01", periods=120)
        panel = self._panel(dates)
        panel.loc[panel["code"] == MAIN_A, "is_st"] = 1
        main = panel.loc[panel["is_main_board"]].copy()
        result = compute_legacy_b_signals(panel, main, self._industry(), self.config)
        entries = result.signals.loc[result.signals["entry_signal"]]
        self.assertTrue(entries["code"].isin([MAIN_B]).all())
        self.assertGreater(result.audit["st_rejected_proposed"], 0)

    def test_missing_industry_blocks_entry(self):
        dates = pd.bdate_range("2017-06-01", periods=120)
        panel = self._panel(dates)
        main = panel.loc[panel["is_main_board"]].copy()
        industry = self._industry().iloc[0:0]
        result = compute_legacy_b_signals(panel, main, industry, self.config)
        self.assertEqual(int(result.signals["entry_signal"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
