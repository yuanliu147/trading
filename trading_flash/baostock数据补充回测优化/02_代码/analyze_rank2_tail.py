#!/usr/bin/env python3
"""榜二（rank=2）买入后的尾部：收盘收益、持有期最大浮亏、单日最大跌幅。

只统计真正成交的 C 池（含退市、信号日排除风险警示）前二组合中的 rank=2。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from strategy_lib import Config, build_signals, load_panel, load_stocks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "06_只买第一名"
RES = OUT / "结果"
PERIODS = [
    ("2017-2026", "2017-10-19", "2026-10-23"),
    ("2005-2016", "2005-01-01", "2016-12-31"),
    ("2005-2026", "2005-01-01", "2026-10-23"),
]


def main():
    panel = load_panel()
    stocks = load_stocks()
    names = stocks.set_index("code")["code_name"].to_dict()
    bars = pd.read_parquet(
        ROOT / "01_数据/bars.parquet",
        columns=["code", "date", "hfq_open", "hfq_high", "hfq_low", "hfq_close"],
    )
    bars["date"] = pd.to_datetime(bars.date)
    by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in bars.groupby("code")}

    stat_rows = []
    case_rows = []
    for period, start, end in PERIODS:
        cfg = Config(start=start, end=end, universe="all", gate="all",
                     extra_filter="is_st == 0", max_rank=2, label="C_top2")
        sig = build_signals(panel, stocks, cfg)
        tr = pd.read_csv(
            RES / period / "C_含退市_排除风险警示_买前二_逐笔.csv",
            parse_dates=["signal_date", "entry_date", "exit_date"],
        )
        tr = tr.merge(
            sig[["date", "code", "rank", "drawdown250", "rsi14", "close_location"]],
            left_on=["signal_date", "code"], right_on=["date", "code"], how="left",
        )
        tr["pnl"] = tr.proceeds - tr.invested
        rows = []
        for r in tr.itertuples():
            g = by_code.get(r.code)
            if g is None or not np.isfinite(r.entry_hfq) or r.entry_hfq <= 0:
                continue
            seg = g[(g.date >= r.entry_date) & (g.date <= r.exit_date)].copy()
            if seg.empty:
                continue
            mae = seg.hfq_low.min() / r.entry_hfq - 1
            mfe = seg.hfq_high.max() / r.entry_hfq - 1
            # 单日跌幅：买入日按 收盘/买入开盘；之后按 收盘/前收
            entry_row = seg.iloc[0]
            day1 = entry_row.hfq_close / entry_row.hfq_open - 1 if entry_row.hfq_open > 0 else np.nan
            later = seg.hfq_close.pct_change().iloc[1:]
            worst_day = np.nanmin([day1] + later.dropna().tolist()) if len(later) else day1
            rows.append(dict(
                code=r.code, code_name=names.get(r.code, ""), rank=int(r.rank),
                signal_date=r.signal_date, entry_date=r.entry_date, exit_date=r.exit_date,
                ret=r.ret, pnl=r.pnl, mae=mae, mfe=mfe, worst_day=worst_day,
                exit_reason=r.exit_reason, delisted=r.delisted,
                drawdown250=r.drawdown250, rsi14=r.rsi14,
            ))
        df = pd.DataFrame(rows)
        df["period"] = period

        for rk in [1, 2]:
            g = df[df["rank"] == rk]
            stat_rows.append(dict(
                期间=period, 排名=rk, 笔数=len(g),
                收盘亏损10=int((g.ret <= -0.10).sum()),
                收盘亏损15=int((g.ret <= -0.15).sum()),
                收盘亏损20=int((g.ret <= -0.20).sum()),
                收盘亏损30=int((g.ret <= -0.30).sum()),
                浮亏20=int((g.mae <= -0.20).sum()),
                浮亏30=int((g.mae <= -0.30).sum()),
                单日跌8=int((g.worst_day <= -0.08).sum()),
                单日跌10=int((g.worst_day <= -0.10).sum()),
                收盘最差=g.ret.min(), 浮亏最深=g.mae.min(),
                单日最差=g.worst_day.min(), 单笔最大亏损=g.pnl.min(),
                平均单笔=g.ret.mean(), 胜率=(g.ret > 0).mean(),
            ))
        case_rows.append(df[df["rank"] == 2].copy())

    stat = pd.DataFrame(stat_rows)
    stat.to_csv(OUT / "榜二尾部统计.csv", index=False, encoding="utf-8-sig")
    cases = pd.concat(case_rows, ignore_index=True)
    # 取每个期间收盘最差、浮亏最深、亏损金额最大三组并集
    keep = set()
    for period in cases.period.unique():
        g = cases[cases.period == period]
        keep |= set(g.nsmallest(6, "ret").index)
        keep |= set(g.nsmallest(6, "mae").index)
        keep |= set(g.nsmallest(3, "pnl").index)
    detail = cases.loc[sorted(keep)].sort_values(["period", "ret"])
    detail.to_csv(OUT / "榜二大跌明细.csv", index=False, encoding="utf-8-sig")

    # 报告用精简个案：2005-2016 收盘最差 6 只 + 000948（盘中 -34.1% 但收盘 -12.0%）
    # + 2017-2026 收盘最差 3 笔 + 600633（盘中 -15.2% 但收盘 +6.7%）
    old = cases[cases["period"] == "2005-2016"].nsmallest(6, "ret").drop_duplicates("code")
    nanya = cases[(cases["period"] == "2005-2016") & (cases["code"] == "sz.000948")]
    old = pd.concat([old, nanya]).drop_duplicates(subset=["period", "code", "entry_date"])
    recent = cases[cases["period"] == "2017-2026"].nsmallest(3, "ret")
    whipsaw = cases[(cases["period"] == "2017-2026") & (cases["code"] == "sh.600633")]
    curated = (pd.concat([old, recent, whipsaw])
               .drop_duplicates(subset=["period", "code", "entry_date"])
               .sort_values(["period", "ret"]))
    curated.to_csv(OUT / "榜二大跌个案.csv", index=False, encoding="utf-8-sig")

    print(stat.to_string(index=False))
    print("\n个案：")
    cols = ["period", "code", "code_name", "signal_date", "entry_date", "exit_date",
            "ret", "mae", "worst_day", "pnl", "exit_reason", "delisted"]
    print(curated[cols].to_string(index=False))


if __name__ == "__main__":
    main()
