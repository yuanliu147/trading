# -*- coding: utf-8 -*-
"""当前策略 O03（实际价、整手、费用后账户）按实际成交价分层的盈利差异。

数据源：
  /Users/admin/WorkSpace/trading/trading_hithink/flash 审计修复/results/accounts/O03_c20000/trades.csv
  /Users/admin/WorkSpace/trading/trading_flash/2026-10-24_主板恐慌反转_年化30/01_数据/research_panel.parquet

只做描述统计，不重跑策略、不改候选池。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy import stats

AUDIT = Path("/Users/admin/WorkSpace/trading/trading_hithink/flash 审计修复")
PANEL = Path("/Users/admin/WorkSpace/trading/trading_flash/2026-10-24_主板恐慌反转_年化30/01_数据/research_panel.parquet")
OUT = Path(__file__).resolve().parents[1]
TRADES = AUDIT / "results/accounts/O03_c20000/trades.csv"

BINS = [0, 5, 10, 15, 25, 40, 1e9]
LABELS = ["<5", "5-10", "10-15", "15-25", "25-40", ">=40"]
RNG = np.random.default_rng(20261027)


def boot_ci(x: pd.Series, n: int = 20000) -> tuple[float, float]:
    arr = x.to_numpy(dtype=float)
    idx = RNG.integers(0, len(arr), (n, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bucket_row(d: pd.DataFrame) -> pd.Series:
    r = d["return"]
    lo, hi = boot_ci(r)
    gross_win = d.loc[d.net_pnl > 0, "net_pnl"].sum()
    gross_loss = abs(d.loc[d.net_pnl < 0, "net_pnl"].sum())
    return pd.Series({
        "n": len(d),
        "win": (r > 0).mean(),
        "avg": r.mean(),
        "ci_lo": lo,
        "ci_hi": hi,
        "median": r.median(),
        "cap_weighted": d.net_pnl.sum() / d.buy_total_cost.sum(),
        "profit_factor": gross_win / gross_loss if gross_loss else np.inf,
        "tp_share": (d.exit_reason == "profit_target").mean(),
        "pnl": d.net_pnl.sum(),
        "avg_pnl": d.net_pnl.mean(),
        "market": d.mkt.mean(),
        "excess": (r - d.mkt).mean(),
        "vol20": d.sig_vol.mean(),
    })


def main() -> None:
    t = pd.read_csv(TRADES, parse_dates=["buy_date", "sell_date", "buy_signal_date"])

    # ---- 面板切片：只取这 97 只股票在信号日前 10 天到最后一笔卖出日 ----
    cols = [
        "thscode", "date", "open", "close", "high", "low",
        "hfq_open", "hfq_close", "atr20_pct", "volatility20", "amount20",
        "market_return", "breadth20", "return_250d", "drawdown250",
    ]
    start = (t.buy_signal_date.min() - pd.Timedelta(days=10)).date()
    end = (t.sell_date.max() + pd.Timedelta(days=5)).date()
    filt = (
        ds.field("thscode").isin(sorted(t.thscode.unique()))
        & (ds.field("date") >= start)
        & (ds.field("date") <= end)
    )
    panel = ds.dataset(PANEL, format="parquet").to_table(columns=cols, filter=filt).to_pandas()
    panel["date"] = pd.to_datetime(panel["date"])

    sig = panel.rename(columns={
        "date": "buy_signal_date", "close": "sig_close", "hfq_close": "sig_hfq_close",
        "atr20_pct": "sig_atr", "volatility20": "sig_vol", "amount20": "sig_amount20",
        "return_250d": "sig_ret250", "drawdown250": "sig_dd250", "breadth20": "sig_breadth",
    })[["thscode", "buy_signal_date", "sig_close", "sig_hfq_close", "sig_atr",
        "sig_vol", "sig_amount20", "sig_ret250", "sig_dd250", "sig_breadth"]]

    buy = panel.rename(columns={
        "date": "buy_date", "open": "buy_open", "hfq_open": "buy_hfq_open",
        "close": "buy_close", "hfq_close": "buy_hfq_close",
        "atr20_pct": "buy_atr", "volatility20": "buy_vol", "amount20": "buy_amount20",
    })[["thscode", "buy_date", "buy_open", "buy_hfq_open", "buy_close",
        "buy_hfq_close", "buy_atr", "buy_vol", "buy_amount20"]]

    m = t.merge(sig, on=["thscode", "buy_signal_date"], how="left").merge(
        buy, on=["thscode", "buy_date"], how="left"
    )

    # ---- 持有期路径：MFE / MAE / 同期等权市场收益 ----
    path = []
    for row in m.itertuples():
        s = panel[(panel.thscode == row.thscode) & (panel.date >= row.buy_date) & (panel.date <= row.sell_date)]
        if s.empty:
            path.append({"trade_id": row.trade_id, "mfe": np.nan, "mae": np.nan, "mkt": np.nan})
            continue
        path.append({
            "trade_id": row.trade_id,
            "mfe": s.high.max() / row.buy_price - 1,
            "mae": s.low.min() / row.buy_price - 1,
            "mkt": float(np.prod(1 + s.market_return) - 1),
        })
    m = m.merge(pd.DataFrame(path), on="trade_id", how="left")
    m["lt15"] = m.buy_price < 15
    m["bucket"] = pd.cut(m.buy_price, BINS, labels=LABELS, right=False)
    m["bucket_hfq"] = pd.cut(m.buy_hfq_open, BINS, labels=LABELS, right=False)
    m["bucket_sig"] = pd.cut(m.sig_close, BINS, labels=LABELS, right=False)
    m["year"] = m.buy_date.dt.year
    m["era"] = pd.cut(m.year, [2016, 2019, 2022, 2026], labels=["2017-19", "2020-22", "2023-26"])

    # ---- 1. 六档统计 ----
    stats_table = m.groupby("bucket", observed=True).apply(bucket_row).reset_index()
    stats_out = stats_table.rename(columns={
        "bucket": "价格档", "n": "笔数", "win": "胜率", "avg": "每笔平均收益",
        "ci_lo": "收益95%下限", "ci_hi": "收益95%上限", "median": "每笔中位数",
        "cap_weighted": "资金加权收益", "profit_factor": "盈亏比", "tp_share": "止盈占比",
        "pnl": "净盈亏元", "avg_pnl": "平均每笔净盈亏元", "market": "同期市场平均",
        "excess": "超额收益", "vol20": "平均波动率20",
    })
    stats_out.to_csv(OUT / "股价分层_统计.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # ---- 2. 分位检查 ----
    m["quartile"] = pd.qcut(
        m.buy_price, 4,
        labels=["Q1 ≤6.88", "Q2 6.88-10.53", "Q3 10.53-18.40", "Q4 >18.40"],
    )
    quart_table = m.groupby("quartile", observed=True).apply(bucket_row).reset_index()
    quart_table.rename(columns={
        "quartile": "四分位", "n": "笔数", "win": "胜率", "avg": "每笔平均收益",
        "ci_lo": "收益95%下限", "ci_hi": "收益95%上限", "median": "每笔中位数",
        "cap_weighted": "资金加权收益", "profit_factor": "盈亏比", "tp_share": "止盈占比",
        "pnl": "净盈亏元", "avg_pnl": "平均每笔净盈亏元", "market": "同期市场平均",
        "excess": "超额收益", "vol20": "平均波动率20",
    }).to_csv(OUT / "股价分层_四分位.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # ---- 3. 同一买入日配对 ----
    pairs = []
    for dt, g in m.groupby("buy_date"):
        if len(g) != 2:
            continue
        g = g.sort_values("buy_price")
        cheap, expensive = g.iloc[0], g.iloc[1]
        pairs.append({
            "买入日": dt.strftime("%Y-%m-%d"),
            "便宜价": cheap.buy_price,
            "贵价": expensive.buy_price,
            "便宜收益": cheap["return"],
            "贵收益": expensive["return"],
            "差(便宜-贵)": cheap["return"] - expensive["return"],
            "跨15元档": bool(cheap.lt15 != expensive.lt15),
            "便宜代码": cheap.thscode,
            "贵代码": expensive.thscode,
        })
    q = pd.DataFrame(pairs)
    q.to_csv(OUT / "同日配对_检验.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    qd = q["差(便宜-贵)"]
    q_lo, q_hi = boot_ci(qd)
    paired = {
        "dates": len(q),
        "cheap_wins": int((qd > 0).sum()),
        "mean": float(qd.mean()),
        "median": float(qd.median()),
        "ci_lo": q_lo,
        "ci_hi": q_hi,
        "p_ttest": float(stats.ttest_1samp(qd, 0).pvalue),
        "p_wilcoxon": float(stats.wilcoxon(qd).pvalue),
    }
    cross = q[q["跨15元档"]]
    paired["cross_n"] = len(cross)
    paired["cross_cheap_wins"] = int((cross["差(便宜-贵)"] > 0).sum())
    paired["cross_mean"] = float(cross["差(便宜-贵)"].mean())

    # ---- 4. 换基准：实际价 / 信号日收盘 / 后复权 ----
    baselines = []
    for label, col in [("实际成交价", "buy_price"), ("信号日收盘价", "sig_close"), ("后复权成交价", "buy_hfq_open")]:
        a = m[m[col] < 15]
        b = m[m[col] >= 15]
        test = stats.ttest_ind(a["return"], b["return"], equal_var=False)
        baselines.append({
            "基准": label,
            "低于15笔数": len(a),
            "低于15平均": float(a["return"].mean()),
            "15以上笔数": len(b),
            "15以上平均": float(b["return"].mean()),
            "差值": float(a["return"].mean() - b["return"].mean()),
            "p值": float(test.pvalue),
            "低于15资金加权": float(a.net_pnl.sum() / a.buy_total_cost.sum()),
            "15以上资金加权": float(b.net_pnl.sum() / b.buy_total_cost.sum()),
        })
    base_df = pd.DataFrame(baselines)
    base_df.to_csv(OUT / "换基准对照.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # ---- 4b. 分阶段：方向会不会翻转 ----
    era_rows = []
    for era_name, g in m.groupby("era", observed=True):
        row = {"阶段": str(era_name)}
        for side, d in [("lt15", g[g.lt15]), ("ge15", g[~g.lt15])]:
            row[f"{side}_n"] = len(d)
            row[f"{side}_avg"] = float(d["return"].mean())
            row[f"{side}_capw"] = float(d.net_pnl.sum() / d.buy_total_cost.sum())
        era_rows.append(row)
    era_df = pd.DataFrame(era_rows).rename(columns={
        "lt15_n": "低于15笔数", "lt15_avg": "低于15每笔平均", "lt15_capw": "低于15资金加权",
        "ge15_n": "15以上笔数", "ge15_avg": "15以上每笔平均", "ge15_capw": "15以上资金加权",
    })
    era_df.to_csv(OUT / "分阶段_价格对照.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # ---- 5. 市场解释力 ----
    X = np.column_stack([np.ones(len(m)), m.mkt])
    beta = np.linalg.lstsq(X, m["return"], rcond=None)[0]
    resid = m["return"] - X @ beta
    r2 = 1 - (resid @ resid) / ((m["return"] - m["return"].mean()) @ (m["return"] - m["return"].mean()))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid) / (len(m) - 2)))
    market_fit = {
        "alpha": float(beta[0]), "alpha_t": float(beta[0] / se[0]),
        "beta": float(beta[1]), "beta_t": float(beta[1] / se[1]),
        "r2": float(r2),
    }

    # ---- 6. 2026 与 <5 元档 ----
    y2026 = m[m.year == 2026][["buy_date", "thscode", "buy_price", "return", "net_pnl", "exit_reason", "mkt"]]
    low5 = m[m.buy_price < 5]
    low5_ex2026 = low5[low5.year != 2026]

    # ---- 7. 实际价 vs 后复权价 的拆股例子 ----
    m["hfq_ratio"] = m.buy_hfq_open / m.buy_price
    splits = m.nlargest(5, "hfq_ratio")[["buy_date", "thscode", "buy_price", "buy_hfq_open", "hfq_ratio", "return"]]

    # ---- 8. 逐笔输出 ----
    trade_out = m[[
        "buy_date", "sell_date", "thscode", "buy_price", "buy_hfq_open", "sig_close",
        "return", "net_pnl", "holding_sessions", "exit_reason", "bucket", "bucket_hfq",
        "mkt", "mfe", "mae", "sig_atr", "sig_vol", "sig_amount20", "year", "era",
    ]].rename(columns={
        "buy_date": "买入日", "sell_date": "卖出日", "thscode": "代码", "buy_price": "实际买入价",
        "buy_hfq_open": "后复权买入价", "sig_close": "信号日收盘价", "return": "净收益率",
        "net_pnl": "净盈亏元", "holding_sessions": "持有交易日", "exit_reason": "退出原因",
        "bucket": "价格档", "bucket_hfq": "后复权价格档", "mkt": "同期市场收益",
        "mfe": "期间最大浮盈", "mae": "期间最大浮亏", "sig_atr": "信号日ATR20",
        "sig_vol": "信号日波动率20", "sig_amount20": "信号日成交额20", "year": "年份", "era": "阶段",
    })
    trade_out["退出原因"] = trade_out["退出原因"].map({"profit_target": "止盈", "max_holding": "持有到期"}).fillna(trade_out["退出原因"])
    trade_out.to_csv(OUT / "股价分层_逐笔.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # ---- 9. 汇总 JSON 给 HTML ----
    data = {
        "meta": {
            "n": int(len(m)),
            "start": m.buy_date.min().strftime("%Y-%m-%d"),
            "end": m.sell_date.max().strftime("%Y-%m-%d"),
            "total_pnl": float(m.net_pnl.sum()),
            "total_cost": float(m.buy_total_cost.sum()),
            "win": float((m["return"] > 0).mean()),
            "avg": float(m["return"].mean()),
            "median": float(m["return"].median()),
            "avg_hold": float(m.holding_sessions.mean()),
            "open_n": 2,
            "open_note": "600783 13.27 元、603778 7.51 元，均低于 15 元，2026-10-23 合计浮动约 -1.0%",
        },
        "buckets": stats_table.replace({np.nan: None}).to_dict(orient="records"),
        "quartiles": quart_table.replace({np.nan: None}).to_dict(orient="records"),
        "pairs": q.replace({np.nan: None}).to_dict(orient="records"),
        "paired": paired,
        "baselines": baselines,
        "era_table": era_df.replace({np.nan: None}).to_dict(orient="records"),
        "market_fit": market_fit,
        "y2026": y2026.assign(buy_date=y2026.buy_date.dt.strftime("%Y-%m-%d")).replace({np.nan: None}).to_dict(orient="records"),
        "low5_ex2026": {
            "n": int(len(low5_ex2026)),
            "avg": float(low5_ex2026["return"].mean()),
            "pnl": float(low5_ex2026.net_pnl.sum()),
        },
        "low5_2026": {
            "n": int(len(low5[low5.year == 2026])),
            "avg": float(low5[low5.year == 2026]["return"].mean()),
            "pnl": float(low5[low5.year == 2026].net_pnl.sum()),
        },
        "splits": splits.assign(buy_date=splits.buy_date.dt.strftime("%Y-%m-%d")).round(4).to_dict(orient="records"),
        "trades": m.assign(
            buy_date=m.buy_date.dt.strftime("%Y-%m-%d"),
            sell_date=m.sell_date.dt.strftime("%Y-%m-%d"),
        )[["buy_date", "sell_date", "thscode", "buy_price", "return", "net_pnl", "exit_reason", "bucket", "era"]]
        .replace({np.nan: None}).to_dict(orient="records"),
        "lt15": {
            "n": int(m.lt15.sum()),
            "avg": float(m[m.lt15]["return"].mean()),
            "capw": float(m[m.lt15].net_pnl.sum() / m[m.lt15].buy_total_cost.sum()),
            "win": float((m[m.lt15]["return"] > 0).mean()),
        },
        "ge15": {
            "n": int((~m.lt15).sum()),
            "avg": float(m[~m.lt15]["return"].mean()),
            "capw": float(m[~m.lt15].net_pnl.sum() / m[~m.lt15].buy_total_cost.sum()),
            "win": float((m[~m.lt15]["return"] > 0).mean()),
        },
        "welch_p": float(stats.ttest_ind(
            m.loc[m.lt15, "return"], m.loc[~m.lt15, "return"], equal_var=False
        ).pvalue),
    }
    (OUT / "股价分层_数据.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(stats_table.round(4).to_string(index=False))
    print("\nquartiles\n", quart_table[["quartile", "n", "win", "avg", "median", "cap_weighted"]].round(4).to_string(index=False))
    print("\nbaselines\n", base_df.round(4).to_string(index=False))
    print("\npaired", paired)
    print("market_fit", {k: round(v, 3) for k, v in market_fit.items()})
    print("\nlt15/ge15", data["lt15"], data["ge15"], "p", round(data["welch_p"], 4))
    print("\nlow5 ex2026", data["low5_ex2026"], "low5 2026", data["low5_2026"])
    print("\nsplits\n", splits.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n2026\n", y2026.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
