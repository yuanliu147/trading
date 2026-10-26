from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_account(account: Path) -> dict:
    metrics = json.loads((account / "metrics.json").read_text())
    equity = pd.read_csv(account / "daily_equity.csv", parse_dates=["date"])
    trades = pd.read_csv(account / "trades.csv", parse_dates=["buy_date", "sell_date", "buy_signal_date", "sell_signal_date"])
    orders = pd.read_csv(account / "orders.csv", parse_dates=["date", "signal_date"])
    filled = orders[orders["status"] == "filled"]
    buys = filled[filled["side"] == "buy"]
    sells = filled[filled["side"] == "sell"]
    checks = {
        "engine_reconciliation": bool(metrics["final_reconciliation_passed"]),
        "cash_flow": float(metrics["max_abs_cash_flow_error"]) < 1e-6,
        "equity_identity": float(metrics["max_abs_equity_identity_error"]) < 1e-6,
        "nonnegative_cash": bool((equity["cash"] >= -1e-8).all()),
        "daily_identity": bool(np.allclose(equity["equity"], equity["cash"] + equity["market_value"] + equity["dividend_receivable"], atol=1e-6)),
        "trade_count": len(trades) == int(metrics["trade_count"]),
        "buy_lots": bool((buys["shares"] % 100 == 0).all()),
        "buy_signal_before_execution": bool((buys["signal_date"] < buys["date"]).all()),
        "sell_signal_before_execution": bool((sells["signal_date"].isna() | (sells["signal_date"] < sells["date"])).all()),
        "t_plus_one": bool((trades["sell_date"] > trades["buy_date"]).all()),
        "no_negative_shares": bool((filled["shares"] > 0).all()),
        "minimum_buy_commission": bool((buys["commission"] >= 5 - 1e-8).all()),
        "minimum_sell_commission": bool((sells["commission"] >= 5 - 1e-8).all()),
        "mainboard_codes": bool(filled["thscode"].str[:3].isin(["600", "601", "603", "605", "000", "001", "002", "003"]).all()),
    }
    if len(trades):
        expected_pnl = (
            trades["sell_net_proceeds"] + trades["dividend_received"] - trades["buy_total_cost"]
        )
        checks["trade_net_pnl"] = bool(np.allclose(expected_pnl, trades["net_pnl"], atol=1e-6))
        checks["trade_return"] = bool(np.allclose(trades["net_pnl"] / trades["buy_total_cost"], trades["return"], atol=1e-8))
        paired = trades.merge(filled[["order_id", "side"]], left_on="buy_order_id", right_on="order_id", how="left")
        checks["buy_order_link"] = bool((paired["side"] == "buy").all())
    elapsed = (equity["date"].iloc[-1] - equity["date"].iloc[0]).days
    cagr = (equity["equity"].iloc[-1] / metrics["initial_capital"]) ** (365 / elapsed) - 1
    checks["cagr_recalculation"] = bool(np.isclose(cagr, metrics["cagr"], atol=1e-10))
    drawdown_path = np.concatenate([[metrics["initial_capital"]], equity["equity"].to_numpy()])
    drawdown = drawdown_path / np.maximum.accumulate(drawdown_path) - 1
    checks["drawdown_recalculation"] = bool(np.isclose(-drawdown.min(), metrics["max_drawdown"], atol=1e-10))
    return {
        "account": str(account.relative_to(ROOT)),
        "passed": all(checks.values()),
        "checks": checks,
        "trades": len(trades),
        "orders": len(orders),
        "cagr": metrics["cagr"],
        "max_drawdown": metrics["max_drawdown"],
        "factor_only_adjustments": metrics.get("factor_only_adjustment_count", 0),
        "ending_open_positions": metrics["ending_open_positions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rerun", action="store_true")
    args = parser.parse_args()
    strategy = json.loads((ROOT / "config" / "final_strategy.json").read_text())
    if not strategy.get("locked_before_holdout"):
        raise RuntimeError("Strategy is not locked")
    rerun_results = []
    for capital in [20000, 50000]:
        name = f"{strategy['candidate_id']}_h{strategy['holding_sessions']}_p{strategy['max_positions']}_c{capital}_s10_l10"
        account = ROOT / "results" / "holdout" / "accounts" / name
        files = [account / "trades.csv", account / "orders.csv", account / "daily_equity.csv"]
        before = {path.name: sha256(path) for path in files}
        if not args.skip_rerun:
            command = [
                sys.executable, str(ROOT / "portfolio.py"),
                "--candidate-id", strategy["candidate_id"],
                "--holding", str(strategy["holding_sessions"]),
                "--positions", str(strategy["max_positions"]),
                "--capital", str(capital), "--period", "holdout",
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"Final rerun failed: {result.stderr}")
            after = {path.name: sha256(path) for path in files}
            rerun_results.append({"capital": capital, "identical": before == after, "before": before, "after": after})
            if before != after:
                raise AssertionError(f"Non-deterministic final output for {capital}")
            reporting = subprocess.run([sys.executable, str(ROOT / "reporting.py"), str(account)], cwd=ROOT, capture_output=True, text=True, check=False)
            if reporting.returncode != 0:
                raise RuntimeError(f"Reporting failed: {reporting.stderr}")
    accounts = []
    for capital in [20000, 50000]:
        name = f"{strategy['candidate_id']}_h{strategy['holding_sessions']}_p{strategy['max_positions']}_c{capital}_s10_l10"
        accounts.append(audit_account(ROOT / "results" / "holdout" / "accounts" / name))
    output = {
        "passed": all(item["passed"] for item in accounts) and all(item["identical"] for item in rerun_results),
        "locked_strategy": {key: strategy[key] for key in ["candidate_id", "holding_sessions", "max_positions"]},
        "rerun": rerun_results,
        "accounts": accounts,
        "data_scope": "Local HiThink DuckDB snapshot only; current-listed main-board history; no delisted reconstruction",
        "survivorship_bias_remains": True,
    }
    path = ROOT / "results" / "holdout" / "verification.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps(output, ensure_ascii=False, default=str))
    if not output["passed"]:
        raise AssertionError("Final verification failed")


if __name__ == "__main__":
    main()
