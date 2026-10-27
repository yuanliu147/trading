#!/usr/bin/env python3
"""指标优化候选：每个过滤单独测试 + 组合，分三个样本期。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from strategy_lib import (Config, build_signals, load_matrices, load_panel, load_stocks,
                          metrics, run, save_result)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "03_结果" / "指标优化"
PERIODS = [
    ("2005-2016", "2005-01-01", "2016-12-31"),
    ("2017-2026", "2017-10-19", "2026-10-23"),
    ("2005-2026", "2005-01-01", "2026-10-23"),
]
VARIANTS = [
    ("C_排除ST", "is_st == 0"),
    ("M1_市场跌超10", "is_st == 0 and market_return20_all < -0.10"),
    ("M2_涨幅超20", "is_st == 0 and return_250d > 0.20"),
    ("M3_涨幅超50", "is_st == 0 and return_250d > 0.50"),
    ("M4_波动超45", "is_st == 0 and volatility20 > 0.45"),
    ("M5_波动超60", "is_st == 0 and volatility20 > 0.60"),
    ("M6_回撤不超75", "is_st == 0 and drawdown250 > -0.75"),
    ("M7_换手不低于2", "is_st == 0 and turn20 >= 2"),
    ("M8_组合", "is_st == 0 and market_return20_all < -0.10 and return_250d > 0.20 and volatility20 > 0.45 and drawdown250 > -0.75"),
    ("M9_组合轻", "is_st == 0 and market_return20_all < -0.10 and return_250d > 0.20"),
]


def main():
    panel = load_panel()
    stocks = load_stocks()
    M = load_matrices()
    rows = []
    for period, start, end in PERIODS:
        for label, flt in VARIANTS:
            cfg = Config(start=start, end=end, universe="all", gate="all",
                         extra_filter=flt, label=f"{period}_{label}")
            sig = build_signals(panel, stocks, cfg)
            res = run(sig, M, stocks, cfg)
            m = metrics(res.equity, res.trades)
            save_result(res, cfg, OUT / period, label)
            rows.append(dict(期间=period, 标签=label, 过滤=flt, 信号日=sig.date.nunique(),
                             候选行=len(sig), 候选股票=sig.code.nunique(), **m))
            print(period, label, {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in m.items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "指标优化汇总.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
