#!/usr/bin/env python3
"""指标过滤组合测试（2017-2026）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from strategy_lib import (Config, build_signals, load_matrices, load_panel, load_stocks,
                          metrics, run, save_result)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "03_结果"

VARIANTS = [
    ("B_含退市_无过滤", ""),
    ("C_排除ST", "is_st == 0"),
    ("D_排除ST_PB正", "is_st == 0 and pb_mrq > 0"),
    ("E_排除ST_股价2元上", "is_st == 0 and close >= 2"),
    ("F_排除ST_换手20不超10", "is_st == 0 and turn20 <= 10"),
    ("G_排除ST_成交额2000万", "is_st == 0 and amount20 >= 2e7"),
    ("H_排除ST_PB正_股价2元上", "is_st == 0 and pb_mrq > 0 and close >= 2"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2017-10-19")
    parser.add_argument("--end", default="2026-10-23")
    parser.add_argument("--tag", default="2017_2026")
    args = parser.parse_args()
    panel = load_panel()
    stocks = load_stocks()
    M = load_matrices()
    rows = []
    for label, flt in VARIANTS:
        cfg = Config(start=args.start, end=args.end, universe="all", gate="all",
                     extra_filter=flt, label=label)
        sig = build_signals(panel, stocks, cfg)
        res = run(sig, M, stocks, cfg)
        m = metrics(res.equity, res.trades)
        save_result(res, cfg, OUT / f"指标组合_{args.tag}", label)
        rows.append(dict(标签=label, 过滤=flt or "无", 信号日=sig.date.nunique(),
                         候选行=len(sig), 候选股票=sig.code.nunique(), **m))
        print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"指标组合_{args.tag}.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
