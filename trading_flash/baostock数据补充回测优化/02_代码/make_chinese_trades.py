#!/usr/bin/env python3
"""把关键逐笔转成中文交易记录（CSV + 供 Excel 使用）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "03_结果"
E = ROOT / "05_证据"
OUT = R / "交易记录"
OUT.mkdir(exist_ok=True)

REASON = {
    "max_holding": "到期",
    "take_profit_intraday": "盘中止盈",
    "take_profit_intraday_open": "开盘止盈",
    "take_profit_open": "开盘止盈",
    "pending_max_holding_no_bar": "停牌顺延到期",
    "pending_max_holding_limit_down": "跌停顺延到期",
    "stop": "止损",
    "stop_open": "开盘止损",
    "eod": "期末估值",
    "delist_last_close": "退市最后价",
}


def to_display(code: str) -> str:
    if code.startswith(("sh.", "sz.")):
        return code[3:] + "." + code[:2].upper()
    return code


def convert(path: Path, name: str) -> pd.DataFrame:
    tr = pd.read_csv(path, parse_dates=["signal_date", "entry_date", "exit_date"])
    st = pd.read_csv(E / "主板股票清单.csv")[["code", "code_name"]].rename(columns={"code_name": "名称"})
    tr = tr.merge(st, on="code", how="left")
    df = pd.DataFrame({
        "代码": tr.code.map(to_display),
        "名称": tr["名称"].fillna(""),
        "信号日": tr.signal_date.dt.strftime("%Y-%m-%d"),
        "买入日": tr.entry_date.dt.strftime("%Y-%m-%d"),
        "卖出日": tr.exit_date.dt.strftime("%Y-%m-%d"),
        "买入价": tr.entry_px,
        "卖出价": tr.exit_px,
        "股数": tr.shares,
        "投入金额": tr.invested,
        "卖出金额": tr.proceeds,
        "净盈亏": tr.proceeds - tr.invested,
        "净收益率": tr.ret,
        "后复权收益率": tr.hfq_ret,
        "实际价涨跌幅": tr.actual_px_ret,
        "复权因子": tr.split_factor,
        "退出原因": tr.exit_reason.map(REASON).fillna(tr.exit_reason),
        "持有交易日": tr.holding_sessions,
        "信号日回撤": tr.drawdown250_signal,
        "信号日风险警示": tr.is_st_signal.map({1: "是", 0: "否"}),
        "后来退市": tr.delisted.map({1: "是", 0: "否"}),
    })
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(name, len(df))
    return df


def main():
    convert(R / "指标组合_2017_2026" / "C_排除ST_逐笔.csv", "交易记录_基准C_2017-2026")
    convert(R / "指标优化/2005-2026" / "C_排除ST_逐笔.csv", "交易记录_基准C_2005-2026")
    convert(R / "2017_2026" / "B_含退市股_逐笔.csv", "交易记录_含退市B_2017-2026")
    convert(R / "风险叠加/2005-2026" / "C_止损25_仓位35_逐笔.csv", "交易记录_推荐配置_2005-2026")
    convert(R / "风险叠加/2017-2026" / "C_止损25_仓位35_逐笔.csv", "交易记录_推荐配置_2017-2026")


if __name__ == "__main__":
    main()
