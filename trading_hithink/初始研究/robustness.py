from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import run_backtest
from portfolio import connection, generate_signals, hypothesis_by_id, load_daily, period_dates, write_json

ROOT = Path(__file__).resolve().parent


def account_name(strategy: dict, capital: int, slippage: float = 0.001, limit_rate: float = 0.10) -> str:
    return (
        f"{strategy['candidate_id']}_h{strategy['holding_sessions']}_p{strategy['max_positions']}"
        f"_c{capital}_s{int(slippage * 10000)}_l{int(limit_rate * 100)}"
    )


def verify_lock(strategy: dict) -> None:
    paths = {
        "hypotheses_sha256": ROOT / "config" / "hypotheses.json",
        "round2_hypotheses_sha256": ROOT / "config" / "round2_hypotheses.json",
        "research_config_sha256": ROOT / "config" / "research.json",
        "account_selection_sha256": ROOT / "config" / "account_selection.json",
        "engine_sha256": ROOT / "engine.py",
    }
    for key, path in paths.items():
        if strategy[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError(f"Locked input changed: {key}")
    feature_hash = json.loads((ROOT / "data" / "work" / "metadata.json").read_text())["files"]["features"]["sha256"]
    if strategy["features_sha256"] != feature_hash:
        raise RuntimeError("Locked features changed")


def block_bootstrap(equity: pd.DataFrame, initial_capital: float, output: Path) -> dict:
    returns = equity["daily_return"].fillna(0).to_numpy(dtype=float)
    dates = pd.to_datetime(equity["date"])
    elapsed_days = (dates.iloc[-1] - dates.iloc[0]).days
    block = 20
    repetitions = 2000
    rng = np.random.default_rng(20261024)
    rows = []
    for repetition in range(repetitions):
        starts = rng.integers(0, len(returns), size=int(np.ceil(len(returns) / block)))
        sampled = np.concatenate([returns[start : start + block] for start in starts])[: len(returns)]
        path = initial_capital * np.cumprod(1 + sampled)
        drawdown = path / np.maximum.accumulate(np.concatenate([[initial_capital], path]))[1:] - 1
        cagr = (path[-1] / initial_capital) ** (365 / elapsed_days) - 1
        rows.append({"repetition": repetition, "cagr": cagr, "max_drawdown": -drawdown.min()})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "block_bootstrap_samples.csv", index=False)
    summary = {
        "method": "Circular moving-block bootstrap of daily account returns; 20-session blocks; not a trade-by-trade market resimulation",
        "repetitions": repetitions,
        "cagr_quantiles": frame["cagr"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
        "max_drawdown_quantiles": frame["max_drawdown"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict(),
        "probability_cagr_above_zero": float((frame["cagr"] > 0).mean()),
        "probability_cagr_above_30pct": float((frame["cagr"] >= 0.30).mean()),
        "probability_max_drawdown_above_40pct": float((frame["max_drawdown"] >= 0.40).mean()),
    }
    write_json(output / "block_bootstrap_summary.json", summary)
    return summary


def main() -> None:
    strategy = json.loads((ROOT / "config" / "final_strategy.json").read_text())
    verify_lock(strategy)
    if not strategy.get("locked_before_holdout"):
        raise RuntimeError("Strategy was not locked before holdout")
    research = json.loads((ROOT / "config" / "research.json").read_text())
    start, end = period_dates("holdout")
    hypothesis = hypothesis_by_id(strategy["candidate_id"])
    con = connection()
    try:
        daily = load_daily(con, start, end)
        candidates, exit_signals = generate_signals(con, hypothesis, start, end)
        random_hypothesis = {**hypothesis, "score": "hash(thscode, CAST(date AS VARCHAR))"}
        random_candidates, _ = generate_signals(con, random_hypothesis, start, end)
    finally:
        con.close()
    events = pd.read_parquet(ROOT / "data" / "snapshot" / "adjustment_events.parquet")
    calendar = list(pd.Index(daily["date"].unique()).sort_values())
    next_session = {calendar[index]: calendar[index + 1] for index in range(len(calendar) - 1)}
    delayed_candidates = candidates.copy()
    delayed_candidates["date"] = delayed_candidates["date"].map(next_session)
    delayed_candidates = delayed_candidates.dropna(subset=["date"])
    scenarios = {
        "slippage20bp": {"slippage": 0.002},
        "slippage30bp": {"slippage": 0.003},
        "conservative_5pct_limits": {"limit_rate": 0.05},
        "half_participation": {"participation": research["execution"]["maximum_known_amount_participation"] / 2},
        "delayed_one_session": {"candidates": delayed_candidates},
        "random_rank_control": {"candidates": random_candidates},
        "double_commission": {"commission_multiplier": 2},
    }
    for offset in [-2, -1, 1, 2]:
        holding = strategy["holding_sessions"] + offset
        if holding > 0:
            scenarios[f"holding_{offset:+d}"] = {"holding_sessions": holding}
    records = []
    for capital in [20000, 50000]:
        baseline_name = account_name(strategy, capital)
        baseline_account = ROOT / "results" / "holdout" / "accounts" / baseline_name
        baseline_trades = pd.read_csv(baseline_account / "trades.csv", parse_dates=["buy_signal_date"])
        top = baseline_trades.nlargest(min(5, len(baseline_trades)), "net_pnl")
        blocked = set(zip(top["buy_signal_date"], top["thscode"]))
        jackknife_candidates = candidates[
            ~candidates.apply(lambda row: (row["date"], row["thscode"]) in blocked, axis=1)
        ].copy()
        scenarios["remove_top5_entry_signals"] = {"candidates": jackknife_candidates}
        for label, overrides in scenarios.items():
            config = {
                "start_date": start,
                "end_date": end,
                "initial_capital": capital,
                "max_positions": strategy["max_positions"],
                "slot_fraction": strategy["slot_fraction"],
                "holding_sessions": overrides.get("holding_sessions", strategy["holding_sessions"]),
                "slippage": overrides.get("slippage", strategy["slippage"]),
                "limit_rate": overrides.get("limit_rate", strategy["limit_rate"]),
                "participation": overrides.get("participation", research["execution"]["maximum_known_amount_participation"]),
                "costs": json.loads(json.dumps(research["costs"])),
            }
            config.update(strategy.get("engine_overrides", {}))
            if "commission_multiplier" in overrides:
                config["costs"]["commission_rate"] *= overrides["commission_multiplier"]
                config["costs"]["minimum_commission"] *= overrides["commission_multiplier"]
            scenario_candidates = overrides.get("candidates", candidates)
            result = run_backtest(daily, events, scenario_candidates, exit_signals, config)
            output = ROOT / "results" / "holdout" / "robustness" / f"{label}_c{capital}"
            output.mkdir(parents=True, exist_ok=True)
            result.trades.to_csv(output / "trades.csv", index=False)
            result.orders.to_csv(output / "orders.csv", index=False)
            result.daily_equity.to_csv(output / "daily_equity.csv", index=False)
            metrics = dict(result.metrics)
            metrics.update({"scenario": label, "capital": capital, "robustness_only_not_for_selection": True})
            write_json(output / "metrics.json", metrics)
            write_json(output / "run_config.json", config)
            records.append({key: metrics.get(key) for key in [
                "scenario", "capital", "cagr", "max_drawdown", "calmar", "trade_count", "win_rate",
                "top_5_profit_share", "final_equity", "factor_only_adjustment_count", "max_abs_cash_flow_error",
            ]})
            print(json.dumps({"scenario": label, "capital": capital, "cagr": metrics["cagr"], "mdd": metrics["max_drawdown"]}))
        bootstrap_output = ROOT / "results" / "holdout" / "robustness" / f"bootstrap_c{capital}"
        bootstrap_output.mkdir(parents=True, exist_ok=True)
        block_bootstrap(pd.read_csv(baseline_account / "daily_equity.csv"), capital, bootstrap_output)
    output = ROOT / "results" / "holdout" / "robustness_summary.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    print(json.dumps({"ok": True, "runs": len(records), "output": str(output), "selection_changed": False}))


if __name__ == "__main__":
    main()
