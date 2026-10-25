from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from portfolio import hypothesis_by_id

ROOT = Path(__file__).resolve().parent


def annual_metrics(path: Path, initial_capital: float) -> tuple[float, int]:
    equity = pd.read_csv(path / "daily_equity.csv", usecols=["date", "equity"], parse_dates=["date"])
    previous = initial_capital
    returns = []
    for _, group in equity.groupby(equity["date"].dt.year):
        ending = float(group["equity"].iloc[-1])
        returns.append(ending / previous - 1)
        previous = ending
    return min(returns), sum(value > 0 for value in returns)


def collect(directory: str, kind: str) -> list[dict]:
    records = []
    base = hypothesis_by_id("engulfing_deep50_lowamount_atr3")
    for path in (ROOT / "results" / "development" / directory).glob("*/metrics.json"):
        metrics = json.loads(path.read_text())
        if kind == "account":
            hypothesis = hypothesis_by_id(metrics["candidate_id"])
            trial_id = f"account_{metrics['candidate_id']}"
            entry = hypothesis["entry"]
            score = hypothesis["score"]
            holding = metrics["holding_sessions"]
            positions = metrics["max_positions"]
            exit_policy = hypothesis["exit"]
        elif kind == "score":
            trial_id = f"score_{metrics['score_label']}"
            entry = base["entry"]
            score = metrics["score_sql"]
            holding = 20
            positions = metrics["max_positions"]
            exit_policy = "FALSE"
        elif kind == "gate":
            trial_id = f"gate_{metrics['gate_label']}"
            entry = f"({base['entry']}) AND ({metrics['gate_sql']})"
            score = "-drawdown250"
            holding = 20
            positions = 2
            exit_policy = "FALSE"
        elif kind == "exit":
            trial_id = f"exit_{metrics['exit_label']}"
            entry = base["entry"]
            score = "-drawdown250"
            holding = 40
            positions = 2
            exit_policy = json.dumps(metrics["engine_overrides"], sort_keys=True)
        elif kind == "universe":
            trial_id = f"universe_{metrics['universe_label']}"
            entry = metrics["entry_sql"]
            score = "-drawdown250"
            holding = metrics["holding_sessions"]
            positions = 2
            exit_policy = "FALSE"
        elif kind == "liquidity":
            trial_id = f"liquidity_{metrics['liquidity_label']}"
            entry = metrics["entry_sql"]
            score = "-drawdown250"
            holding = 19
            positions = 2
            exit_policy = "FALSE"
        else:
            trial_id = "holding_deep250"
            entry = base["entry"]
            score = "-drawdown250"
            holding = metrics["holding_sessions"]
            positions = 2
            exit_policy = "FALSE"
        worst_year, positive_years = annual_metrics(path.parent, metrics["initial_capital"])
        records.append(
            {
                "trial_id": trial_id,
                "trial_kind": kind,
                "entry": entry,
                "score": score,
                "exit_policy": exit_policy,
                "holding_sessions": holding,
                "max_positions": positions,
                "initial_capital": metrics["initial_capital"],
                "cagr": metrics["cagr"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar": metrics["calmar"],
                "trade_count": metrics["trade_count"],
                "top_1_profit_share": metrics["top_1_profit_share"],
                "top_5_profit_share": metrics["top_5_profit_share"],
                "worst_year_return": worst_year,
                "positive_years": positive_years,
                "evidence_path": str(path.parent.relative_to(ROOT)),
            }
        )
    return records


def main() -> None:
    records = []
    for directory, kind in [
        ("accounts", "account"),
        ("score_trials", "score"),
        ("gate_trials", "gate"),
        ("holding_trials", "holding"),
        ("exit_trials", "exit"),
        ("universe_trials", "universe"),
        ("liquidity_trials", "liquidity"),
    ]:
        records.extend(collect(directory, kind))
    raw = pd.DataFrame(records)
    raw.to_csv(ROOT / "results" / "development" / "all_trial_capital_results.csv", index=False)
    keys = ["trial_id", "trial_kind", "entry", "score", "exit_policy", "holding_sessions", "max_positions"]
    rows = []
    for key, group in raw.groupby(keys, dropna=False):
        if set(group["initial_capital"]) != {20000, 50000} or len(group) != 2:
            continue
        rows.append(
            {
                **dict(zip(keys, key)),
                "minimum_cagr": group["cagr"].min(),
                "mean_cagr": group["cagr"].mean(),
                "maximum_drawdown": group["max_drawdown"].max(),
                "minimum_calmar": group["calmar"].min(),
                "minimum_trades": group["trade_count"].min(),
                "maximum_top1_profit_share": group["top_1_profit_share"].max(),
                "maximum_top5_profit_share": group["top_5_profit_share"].max(),
                "worst_year_return": group["worst_year_return"].min(),
                "minimum_positive_years": group["positive_years"].min(),
                "cagr_20000": group.loc[group["initial_capital"] == 20000, "cagr"].iloc[0],
                "cagr_50000": group.loc[group["initial_capital"] == 50000, "cagr"].iloc[0],
                "evidence_paths": " | ".join(group["evidence_path"]),
            }
        )
    combined = pd.DataFrame(rows)
    rules = json.loads((ROOT / "config" / "account_selection.json").read_text())["development"]
    combined["passed_gates"] = (
        (combined["minimum_trades"] >= rules["minimum_closed_trades_each_capital"])
        & (combined["maximum_drawdown"] <= rules["maximum_drawdown_each_capital"])
        & (combined["maximum_top5_profit_share"] <= rules["maximum_top5_profit_share_each_capital"])
        & (combined["minimum_cagr"] >= rules["minimum_cagr_each_capital"])
    )
    combined = combined.sort_values("minimum_cagr", ascending=False)
    combined.to_csv(ROOT / "results" / "development" / "all_account_trials.csv", index=False)
    columns = [
        "trial_id", "holding_sessions", "max_positions", "minimum_cagr", "maximum_drawdown",
        "minimum_trades", "maximum_top5_profit_share", "worst_year_return", "passed_gates",
    ]
    print(combined[columns].head(25).to_string(index=False))
    print(json.dumps({"ok": True, "capital_runs": len(raw), "variants": len(combined), "holdout_used": False}))


if __name__ == "__main__":
    main()
