from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    account = "deep_lowamount_h19_h19_p2_c20000_s10_l10"
    frames = []
    for period in ["development", "validation", "holdout"]:
        path = ROOT / "results" / period / "accounts" / account
        orders = pd.read_csv(path / "orders.csv", parse_dates=["date", "signal_date"])
        trades = pd.read_csv(path / "trades.csv")
        orders = orders[(orders["side"] == "buy") & (orders["status"] == "filled")]
        merged = orders.merge(
            trades[["buy_order_id", "return", "net_pnl", "exit_reason"]],
            left_on="order_id",
            right_on="buy_order_id",
            how="left",
            validate="one_to_one",
        )
        merged["period"] = period
        frames.append(merged)
    inputs = pd.concat(frames, ignore_index=True)
    con = duckdb.connect()
    con.register("filled_inputs", inputs)
    feature_path = str(ROOT / "data" / "work" / "features.parquet").replace("'", "''")
    cluster_path = str(ROOT / "data" / "work" / "cluster_daily.parquet").replace("'", "''")
    detail = con.execute(
        f"""
        SELECT i.period, i.signal_date, i.date AS execution_date, i.thscode,
               i.return AS trade_return, i.net_pnl, i.exit_reason,
               f.close, f.amount20, f.atr20_pct, f.volatility20, f.drawdown250,
               f.return_10d, f.rsi14, f.macd_hist_pct, f.macd_hist_delta,
               f.breadth20, f.market_return20, f.market_volatility20,
               f.sentiment_score, c.peer_return20, c.peer_breadth20
        FROM filled_inputs i
        JOIN read_parquet('{feature_path}') f
          ON f.thscode = i.thscode AND f.date = i.signal_date
        LEFT JOIN read_parquet('{cluster_path}') c
          ON c.thscode = i.thscode AND c.date = i.signal_date
        ORDER BY i.period, i.signal_date, i.thscode
        """
    ).fetchdf()
    detail.to_csv(ROOT / "results" / "holdout" / "entry_feature_attribution.csv", index=False)
    detail["period_year"] = detail["period"] + "_" + pd.to_datetime(detail["signal_date"]).dt.year.astype(str)
    summary = detail.groupby("period_year").agg(
        trades=("thscode", "count"),
        win_rate=("trade_return", lambda x: (x > 0).mean()),
        mean_trade_return=("trade_return", "mean"),
        median_trade_return=("trade_return", "median"),
        median_close=("close", "median"),
        median_amount20=("amount20", "median"),
        median_atr20=("atr20_pct", "median"),
        median_volatility20=("volatility20", "median"),
        median_drawdown250=("drawdown250", "median"),
        median_return10=("return_10d", "median"),
        median_rsi14=("rsi14", "median"),
        mean_macd_hist=("macd_hist_pct", "mean"),
        mean_breadth20=("breadth20", "mean"),
        mean_market_return20=("market_return20", "mean"),
        mean_market_volatility20=("market_volatility20", "mean"),
        mean_sentiment=("sentiment_score", "mean"),
        mean_peer_return20=("peer_return20", "mean"),
    ).reset_index()
    summary.to_csv(ROOT / "results" / "holdout" / "entry_feature_drift.csv", index=False)
    write = ROOT / "results" / "holdout" / "failure_analysis.json"
    write.write_text(json.dumps({
        "purpose": "Post-hoc failure attribution only; not used to change the locked strategy or select another holdout winner",
        "locked_candidate": "deep_lowamount_h19",
        "detail_path": "results/holdout/entry_feature_attribution.csv",
        "summary_path": "results/holdout/entry_feature_drift.csv",
    }, ensure_ascii=False, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
