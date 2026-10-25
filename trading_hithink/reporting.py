from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def performance_from_returns(returns: pd.Series, dates: pd.Series) -> dict:
    returns = returns.fillna(0).to_numpy(dtype=float)
    equity = np.cumprod(1 + returns)
    if not len(equity) or (dates.iloc[-1] - dates.iloc[0]).days <= 0:
        return {"cagr": np.nan, "max_drawdown": np.nan}
    elapsed = (dates.iloc[-1] - dates.iloc[0]).days
    peaks = np.maximum.accumulate(np.concatenate([[1.0], equity]))
    drawdowns = np.concatenate([[1.0], equity]) / peaks - 1
    return {
        "cagr": float(equity[-1] ** (365 / elapsed) - 1),
        "total_return": float(equity[-1] - 1),
        "max_drawdown": float(-drawdowns.min()),
    }


def analyze(account: Path) -> None:
    metrics = json.loads((account / "metrics.json").read_text())
    equity = pd.read_csv(account / "daily_equity.csv", parse_dates=["date"])
    trades = pd.read_csv(account / "trades.csv", parse_dates=["buy_date", "sell_date"])
    orders = pd.read_csv(account / "orders.csv", parse_dates=["date", "signal_date"])
    initial = float(metrics["initial_capital"])
    annual_rows = []
    previous = initial
    for year, group in equity.groupby(equity["date"].dt.year, sort=True):
        group = group.sort_values("date")
        ending = float(group["equity"].iloc[-1])
        path = np.concatenate([[previous], group["equity"].to_numpy(dtype=float)])
        drawdown = path / np.maximum.accumulate(path) - 1
        year_trades = trades[trades["sell_date"].dt.year == year] if not trades.empty else trades
        year_orders = orders[(orders["date"].dt.year == year) & (orders["status"] == "filled")]
        average_equity = float(group["equity"].mean())
        gross_turnover = float(year_orders["gross"].sum()) if not year_orders.empty else 0.0
        annual_rows.append(
            {
                "year": int(year),
                "sessions": len(group),
                "start_equity": previous,
                "end_equity": ending,
                "return": ending / previous - 1,
                "max_drawdown": -float(drawdown.min()),
                "closed_trades": len(year_trades),
                "net_pnl": float(year_trades["net_pnl"].sum()) if len(year_trades) else 0.0,
                "win_rate": float((year_trades["net_pnl"] > 0).mean()) if len(year_trades) else np.nan,
                "average_exposure": float(group["exposure"].mean()),
                "gross_turnover_over_average_equity": gross_turnover / average_equity if average_equity > 0 else np.nan,
            }
        )
        previous = ending
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(account / "annual_returns.csv", index=False)

    market = pd.read_parquet(
        ROOT / "data" / "work" / "market_daily.parquet",
        columns=["date", "market_return", "equal_weight_index", "breadth20", "market_return20", "market_close_ma20", "market_volatility20"],
    )
    market["date"] = pd.to_datetime(market["date"])
    joined = equity[["date", "daily_return", "equity"]].merge(market, on="date", how="left", validate="one_to_one")
    joined["regime"] = np.select(
        [
            (joined["breadth20"] <= 0.35) & (joined["market_return20"] < 0),
            (joined["breadth20"] >= 0.60) & (joined["market_close_ma20"] > 0),
            joined["market_volatility20"] >= 0.25,
        ],
        ["bear_breadth", "bull_breadth", "high_volatility"],
        default="mixed",
    )
    regime = joined.groupby("regime").agg(
        sessions=("date", "count"),
        average_daily_return=("daily_return", "mean"),
        log_return_contribution=("daily_return", lambda x: np.log1p(x.fillna(0)).sum()),
        average_market_return=("market_return", "mean"),
    ).reset_index()
    regime["noncontiguous_compound_return"] = joined.groupby("regime")["daily_return"].apply(
        lambda x: (1 + x.fillna(0)).prod() - 1
    ).to_numpy()
    regime.to_csv(account / "regime_attribution.csv", index=False)

    market_path = joined.dropna(subset=["equal_weight_index"]).copy()
    market_path["equal_weight_return"] = market_path["equal_weight_index"].pct_change().fillna(0)
    benchmark = {
        "current_listed_equal_weight_buy_hold": performance_from_returns(
            market_path["equal_weight_return"], market_path["date"]
        ),
        "survivorship_bias": True,
        "benchmark_costs_included": False,
    }
    gated_return = joined["market_return"].fillna(0) * (joined["market_close_ma20"].shift(1) > 0).fillna(False)
    benchmark["previous_close_ma20_gated_equal_weight"] = performance_from_returns(gated_return, joined["date"])
    valid = joined[["daily_return", "market_return"]].dropna()
    if len(valid) > 2:
        beta = valid["daily_return"].cov(valid["market_return"]) / valid["market_return"].var()
        benchmark["strategy_daily_beta_to_equal_weight"] = float(beta)
        benchmark["strategy_daily_correlation_to_equal_weight"] = float(valid["daily_return"].corr(valid["market_return"]))
    write_json(account / "benchmark_analysis.json", benchmark)

    filled = orders[orders["status"] == "filled"]
    cost_summary = {
        "commissions": float(filled["commission"].sum()),
        "stamp_duty": float(filled["stamp_duty"].sum()),
        "transfer_fees": float(filled["transfer_fee"].sum()),
        "slippage_cost_informational": float(filled["slippage_cost"].sum()),
        "dividend_tax": float(filled["dividend_tax"].sum()),
        "gross_traded_value": float(filled["gross"].sum()),
    }
    cost_summary["explicit_fees_over_initial_capital"] = (
        cost_summary["commissions"] + cost_summary["stamp_duty"] + cost_summary["transfer_fees"] + cost_summary["dividend_tax"]
    ) / initial
    cost_summary["slippage_over_initial_capital"] = cost_summary["slippage_cost_informational"] / initial
    write_json(account / "cost_analysis.json", cost_summary)
    rejection = orders.groupby(["side", "status", "status_reason"], dropna=False).size().rename("count").reset_index()
    rejection.to_csv(account / "order_status_summary.csv", index=False)
    summary = {
        "candidate_id": metrics["candidate_id"],
        "period": metrics["period"],
        "initial_capital": initial,
        "worst_annual_return": float(annual["return"].min()),
        "positive_years": int((annual["return"] > 0).sum()),
        "years": int(len(annual)),
        "annual_gross_turnover_over_average_equity": float(annual["gross_turnover_over_average_equity"].mean()),
        "benchmark": benchmark,
        "costs": cost_summary,
        "current_listed_survivorship_bias": True,
    }
    write_json(account / "analysis.json", summary)
    print(json.dumps({"ok": True, "account": str(account), "worst_year": summary["worst_annual_return"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("account", type=Path)
    args = parser.parse_args()
    analyze(args.account.resolve())


if __name__ == "__main__":
    main()
