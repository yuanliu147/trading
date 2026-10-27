#!/usr/bin/env python3
"""信号级事件研究：新指标与 T+3 开盘买入、15 日持有的前瞻收益。

说明：
  - 样本为所有满足冻结条件的信号（不只是每日前 2），用于看指标的横截面区分度；
  - 前瞻收益按该股“实际交易日”滚动：T+3 开盘至 T+17 收盘（含 T+3 共 15 个交易日）；
  - 这是描述性统计，信号日在同一天高度聚集，不能当作独立样本做显著性检验；
  - 退市/长期停牌导致 15 日窗口不完整的信号单独统计，不进入均值。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from strategy_lib import PANEL, load_panel, load_stocks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "03_结果"


def build(panel_path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET threads TO 8; SET memory_limit='12GB';")
    p = str(panel_path).replace("'", "''")
    q = f"""
    WITH fwd AS (
        SELECT code, date, hfq_open, hfq_high, hfq_low, hfq_close,
            lead(hfq_open, 3) OVER w AS entry_hfq,
            lead(hfq_close, 17) OVER w AS exit_hfq,
            max(hfq_high) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 3 FOLLOWING AND 17 FOLLOWING) AS max_high,
            min(hfq_low) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 3 FOLLOWING AND 17 FOLLOWING) AS min_low,
            count(*) OVER (PARTITION BY code ORDER BY date ROWS BETWEEN 3 FOLLOWING AND 17 FOLLOWING) AS n_fwd
        FROM read_parquet('{p}')
        WINDOW w AS (PARTITION BY code ORDER BY date)
    )
    SELECT * FROM fwd
    """
    fwd = con.execute(q).fetchdf()
    fwd["date"] = pd.to_datetime(fwd.date)
    con.close()
    return fwd


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    def cut(x, bins, labels):
        return pd.cut(x, bins=bins, labels=labels, right=False)

    df["ST"] = np.where(df.is_st == 1, "ST", "非ST")
    df["后来退市"] = np.where(df.later_delisted == 1, "是", "否")
    df["PB"] = cut(df.pb_mrq, [-np.inf, 0, 1, 3, 10, np.inf],
                   ["负", "0-1", "1-3", "3-10", ">10"]).astype(object)
    df.loc[df.pb_mrq.isna(), "PB"] = "缺失"
    df["股价"] = cut(df.close, [0, 2, 5, 10, 20, np.inf],
                     ["<2", "2-5", "5-10", "10-20", ">=20"]).astype(object)
    df["20日换手率"] = cut(df.turn20, [0, 1, 2, 4, 7, 10, np.inf],
                          ["<1%", "1-2%", "2-4%", "4-7%", "7-10%", ">10%"]).astype(object)
    df["20日成交额"] = cut(df.amount20, [1e7, 2e7, 5e7, 1e8, 3e8, np.inf],
                          ["1000-2000万", "2000-5000万", "0.5-1亿", "1-3亿", ">3亿"]).astype(object)
    df["20日波动率"] = cut(df.volatility20, [0, 0.30, 0.45, 0.60, 0.80, np.inf],
                          ["<30%", "30-45%", "45-60%", "60-80%", ">80%"]).astype(object)
    df["市场20日"] = cut(df.market_return20_all, [-np.inf, -0.15, -0.10, -0.07, -0.05],
                         ["<-15%", "-15~-10%", "-10~-7%", "-7~-5%"]).astype(object)
    df["250日回撤"] = cut(df.drawdown250, [-np.inf, -0.75, -0.65, -0.55, -0.45],
                         ["<-75%", "-75~-65%", "-65~-55%", "-55~-45%"]).astype(object)
    df["250日涨幅"] = cut(df.return_250d, [0, 0.2, 0.5, 1.0, np.inf],
                         ["0-20%", "20-50%", "50-100%", ">100%"]).astype(object)
    df["换手放大"] = cut(df.turn_ratio1_20, [0, 0.5, 0.8, 1.2, 2, np.inf],
                        ["<0.5", "0.5-0.8", "0.8-1.2", "1.2-2", ">2"]).astype(object)
    return df


def summarize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        for key, g in df.groupby(col, observed=True):
            rows.append(dict(维度=col, 分组=str(key), 样本=len(g),
                             信号日=g.date.nunique(),
                             胜率=(g.fwd15 > 0).mean(),
                             止盈20率=(g.tp20).mean(),
                             均值=g.fwd15.mean(),
                             中位数=g.fwd15.median(),
                             平均MFE=g.mfe15.mean(),
                             平均MAE=g.mae15.mean()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2026-10-23")
    parser.add_argument("--tag", default="2005_2026")
    args = parser.parse_args()
    panel = load_panel()
    stocks = load_stocks()
    delisted = set(stocks.loc[stocks.out_date != "", "code"])
    df = panel[(panel.date >= args.start) & (panel.date <= args.end)]
    df = df[df.listing_session >= 250]
    df = df[df.amount20 >= 10_000_000]
    df = df[df.return_250d > 0]
    df = df[df.drawdown250 < -0.45]
    df = df[df.rsi14 < 35]
    df = df[df.close_location > 0.45]
    df = df[df.market_return20_all < -0.05].copy()
    fwd = build(PANEL)
    df = df.merge(fwd[["code", "date", "entry_hfq", "exit_hfq", "max_high", "min_low", "n_fwd"]],
                  on=["code", "date"], how="left")
    df["later_delisted"] = df.code.isin(delisted).astype(int)
    df["fwd15"] = df.exit_hfq / df.entry_hfq - 1
    df["mfe15"] = df.max_high / df.entry_hfq - 1
    df["mae15"] = df.min_low / df.entry_hfq - 1
    df["tp20"] = (df.mfe15 >= 0.20).astype(int)
    full = df[df.n_fwd == 15].copy()
    trunc = df[df.n_fwd != 15].copy()
    full = add_buckets(full)
    cols = ["ST", "后来退市", "PB", "股价", "20日换手率", "20日成交额",
            "20日波动率", "市场20日", "250日回撤", "250日涨幅", "换手放大"]
    out = summarize(full, cols)
    out.to_csv(OUT / f"事件研究_{args.tag}.csv", index=False, encoding="utf-8-sig")
    trunc_summary = dict(样本=len(trunc), 其中ST=int((trunc.is_st == 1).sum()),
                         其中后来退市=int(trunc.later_delisted.sum()),
                         信号日=int(trunc.date.nunique()))
    (OUT / f"事件研究_截断_{args.tag}.json").write_text(
        __import__("json").dumps(trunc_summary, ensure_ascii=False, indent=2) + "\n")
    print("完整窗口", len(full), "截断", len(trunc), trunc_summary)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
