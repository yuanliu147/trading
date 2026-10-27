#!/usr/bin/env python3
"""风险叠加：止损与仓位缩放（在“排除ST”基础上）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategy_lib import (Config, build_signals, load_matrices, load_panel, load_stocks,
                          metrics, run, save_result)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "03_结果" / "风险叠加"
PERIODS = [
    ("2005-2016", "2005-01-01", "2016-12-31"),
    ("2017-2026", "2017-10-19", "2026-10-23"),
    ("2005-2026", "2005-01-01", "2026-10-23"),
]
GRID = [
    ("C_基础_无止损", None, 0.50),
    ("C_止损15", 0.15, 0.50),
    ("C_止损20", 0.20, 0.50),
    ("C_止损25", 0.25, 0.50),
    ("C_止损15_仓位35", 0.15, 0.35),
    ("C_止损25_仓位35", 0.25, 0.35),
    ("C_止损15_仓位25", 0.15, 0.25),
    ("C_止损25_仓位25", 0.25, 0.25),
]


def main():
    panel = load_panel()
    stocks = load_stocks()
    M = load_matrices()
    rows = []
    for period, start, end in PERIODS:
        for label, stop, frac in GRID:
            if period == "2005-2016" and (stop is not None or frac != 0.50):
                continue
            cfg = Config(start=start, end=end, universe="all", gate="all",
                         extra_filter="is_st == 0", stop_loss=stop,
                         position_fraction=frac, label=f"{period}_{label}")
            sig = build_signals(panel, stocks, cfg)
            res = run(sig, M, stocks, cfg)
            m = metrics(res.equity, res.trades)
            save_result(res, cfg, OUT / period, label)
            rows.append(dict(期间=period, 标签=label, 止损=stop, 单票仓位=frac,
                             信号日=sig.date.nunique(), **m))
            print(period, label, {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in m.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "03_结果" / "风险叠加汇总.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
