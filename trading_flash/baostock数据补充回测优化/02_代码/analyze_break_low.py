#!/usr/bin/env python3
"""T+3 开盘跌破信号日最低价：后续收益、远期收益、最大浮亏与收回情况。

主口径：C 池前二组合，实际价比较（T+3 开盘 < T 日最低价）。
复权口径用于核对除权除息造成的假破位。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lib import Config, build_signals, load_panel, load_stocks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "07_破信号低点"
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = [
    ("2017-2026", "2017-10-19", "2026-10-23"),
    ("2005-2016", "2005-01-01", "2016-12-31"),
    ("2005-2026", "2005-01-01", "2026-10-23"),
]
POOLS = [
    ("C_含退市_排除风险警示", "all", "all", "is_st == 0"),
    ("A_仅当前上市", "listed", "listed", ""),
    ("B_含退市_无ST过滤", "all", "all", ""),
]


def trade_path(period: str, label: str) -> Path:
    if label.startswith("C"):
        return ROOT / "06_只买第一名/结果" / period / f"{label}_买前二_逐笔.csv"
    tag = period.replace("-", "_")
    if label.startswith("B"):
        return ROOT / "03_结果" / tag / "B_含退市股_逐笔.csv"
    return ROOT / "03_结果" / tag / f"{label}_逐笔.csv"


def main():
    panel = load_panel(columns=[
        "code", "date", "open", "high", "low", "close", "preclose",
        "hfq_open", "hfq_high", "hfq_low", "hfq_close", "amount20",
        "listing_session", "return_250d", "drawdown250", "rsi14",
        "close_location", "is_st", "pb_mrq", "turn20",
        "market_return20_all", "market_return20_listed", "market_return",
    ])
    panel["date"] = pd.to_datetime(panel.date)
    stocks = load_stocks()
    bars = pd.read_parquet(
        ROOT / "01_数据/bars.parquet",
        columns=["code", "date", "hfq_open", "hfq_high", "hfq_low", "hfq_close"],
    )
    bars["date"] = pd.to_datetime(bars.date)
    by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in bars.groupby("code")}

    details = []
    stats = []
    for period, start, end in PERIODS:
        for label, universe, gate, extra in POOLS:
            if label.startswith(("A", "B")) and period == "2005-2016":
                continue
            path = trade_path(period, label)
            if not path.exists():
                continue
            cfg = Config(start=start, end=end, universe=universe, gate=gate,
                         extra_filter=extra, max_rank=2, label=f"{label}_top2")
            sig = build_signals(panel, stocks, cfg)
            tr = pd.read_csv(path, parse_dates=["signal_date", "entry_date", "exit_date"])
            tr = tr.merge(
                sig[["date", "code", "rank", "low", "hfq_low", "drawdown250", "rsi14"]],
                left_on=["signal_date", "code"], right_on=["date", "code"], how="left",
            )
            for r in tr.itertuples():
                g = by_code.get(r.code)
                if g is None:
                    continue
                pos = g.index[g.date == r.entry_date]
                if len(pos) == 0:
                    continue
                i = int(pos[0])
                seg = g.iloc[i:i + 16]
                if len(seg) < 2:
                    continue
                fwd = {}
                for k in [1, 3, 5, 10, 15]:
                    fwd[f"r{k}"] = (seg.hfq_close.iloc[k] / r.entry_hfq - 1
                                    if len(seg) > k else np.nan)
                mae = seg.hfq_low.min() / r.entry_hfq - 1
                mfe = seg.hfq_high.max() / r.entry_hfq - 1
                # 收回：买入后前 5 个交易日（含买入日）是否有收盘/最高回到信号低点之上
                rec5_close = bool((seg.hfq_close.iloc[:6] > r.hfq_low).any())
                rec5_high = bool((seg.hfq_high.iloc[:6] >= r.hfq_low).any())
                rec3_high = bool((seg.hfq_high.iloc[:4] >= r.hfq_low).any())
                close5_above = (seg.hfq_close.iloc[5] > r.hfq_low) if len(seg) > 5 else np.nan
                details.append(dict(
                    期间=period, 股票池=label, 代码=r.code, 排名=int(r.rank),
                    信号日=r.signal_date, 买入日=r.entry_date, 卖出日=r.exit_date,
                    信号最低=r.low, 买入开盘=r.entry_px,
                    信号最低_复权=r.hfq_low, 买入开盘_复权=r.entry_hfq,
                    破位=bool(r.entry_hfq < r.hfq_low - 1e-9),
                    破位_实际=bool(r.entry_px < r.low - 1e-9),
                    破位深度=r.entry_hfq / r.hfq_low - 1,
                    破位深度_实际=r.entry_px / r.low - 1,
                    实际收益=r.ret, 净盈亏=r.proceeds - r.invested,
                    五日收盘收回=rec5_close, 五日最高收回=rec5_high,
                    三日最高收回=rec3_high, 第五日收盘在上方=close5_above,
                    最大浮亏=mae, 最大浮盈=mfe,
                    **fwd,
                    退出=r.exit_reason,
                ))

    df = pd.DataFrame(details)
    df.to_csv(OUT / "破信号低点_交易明细.csv", index=False, encoding="utf-8-sig")

    c = df[df["股票池"].str.startswith("C")].copy()
    for period in ["2017-2026", "2005-2016", "2005-2026"]:
        p = c[c["期间"] == period]
        if p.empty:
            continue
        groups = [
            ("未破", p[~p["破位"]]),
            ("破位", p[p["破位"]]),
            ("破位+5日收盘收回", p[p["破位"] & p["五日收盘收回"]]),
            ("破位+5日未收回", p[p["破位"] & ~p["五日收盘收回"]]),
        ]
        for name, g in groups:
            stats.append(dict(
                期间=period, 组=name, 笔数=len(g),
                实际收益均值=g["实际收益"].mean(), 实际收益中位=g["实际收益"].median(),
                胜率=(g["实际收益"] > 0).mean(),
                五日=g["r5"].mean(), 十日=g["r10"].mean(), 十五日=g["r15"].mean(),
                最大浮亏均值=g["最大浮亏"].mean(), 最大浮盈均值=g["最大浮盈"].mean(),
                浮亏10=(g["最大浮亏"] <= -0.10).mean(),
                浮亏20=(g["最大浮亏"] <= -0.20).mean(),
                止盈20=(g["最大浮盈"] >= 0.20).mean(),
                最差收益=g["实际收益"].min(),
            ))
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "破信号低点_统计.csv", index=False, encoding="utf-8-sig")

    pool_rows = []
    for (period, label), g in df.groupby(["期间", "股票池"]):
        for b, gg in g.groupby("破位"):
            pool_rows.append(dict(
                期间=period, 股票池=label, 组="破位" if b else "未破", 笔数=len(gg),
                实际收益均值=gg["实际收益"].mean(), 实际收益中位=gg["实际收益"].median(),
                胜率=(gg["实际收益"] > 0).mean(), 十五日=gg["r15"].mean(),
                最大浮亏均值=gg["最大浮亏"].mean(), 最大浮盈均值=gg["最大浮盈"].mean(),
                最差收益=gg["实际收益"].min(),
            ))
    pd.DataFrame(pool_rows).to_csv(OUT / "破信号低点_分池统计.csv", index=False, encoding="utf-8-sig")

    print(st.to_string(index=False))
    print("\n实际价与复权价破位判定不一致：")
    print(df[df["破位"] != df["破位_实际"]][
        ["期间", "股票池", "代码", "信号日", "买入日", "信号最低", "买入开盘",
         "信号最低_复权", "买入开盘_复权", "实际收益"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
