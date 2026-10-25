from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def account_name(strategy: dict, capital: int) -> str:
    return f"{strategy['candidate_id']}_h{strategy['holding_sessions']}_p{strategy['max_positions']}_c{capital}_s10_l10"


def account_evidence(period: str, strategy: dict, capital: int) -> dict:
    account = ROOT / "results" / period / "accounts" / account_name(strategy, capital)
    metrics = read_json(account / "metrics.json")
    annual = pd.read_csv(account / "annual_returns.csv") if (account / "annual_returns.csv").exists() else None
    result = {
        "account_path": str(account.relative_to(ROOT)),
        "initial_capital": capital,
        "final_equity": metrics["final_equity"],
        "cagr": metrics["cagr"],
        "total_return": metrics["total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar": metrics["calmar"],
        "sharpe": metrics["sharpe"],
        "trade_count": metrics["trade_count"],
        "win_rate": metrics["win_rate"],
        "profit_loss_ratio": metrics["profit_loss_ratio"],
        "holding_time_ratio": metrics["holding_time_ratio"],
        "top_1_profit_share": metrics["top_1_profit_share"],
        "top_5_profit_share": metrics["top_5_profit_share"],
        "ending_open_positions": metrics["ending_open_positions"],
        "factor_only_adjustment_count": metrics.get("factor_only_adjustment_count", 0),
        "cagr_30pct_met": metrics["cagr"] >= 0.30,
        "files": {
            "trades": str((account / "trades.csv").relative_to(ROOT)),
            "orders": str((account / "orders.csv").relative_to(ROOT)),
            "daily_equity": str((account / "daily_equity.csv").relative_to(ROOT)),
        },
    }
    if annual is not None:
        result.update(
            {
                "annual_returns": annual.to_dict("records"),
                "worst_year_return": annual["return"].min(),
                "every_full_year_30pct_met": bool((annual.loc[annual["sessions"] >= 200, "return"] >= 0.30).all()),
                "every_calendar_year_including_partial_30pct_met": bool((annual["return"] >= 0.30).all()),
                "positive_years": int((annual["return"] > 0).sum()),
            }
        )
    for name in ["benchmark_analysis.json", "cost_analysis.json", "regime_attribution.csv"]:
        if (account / name).exists():
            result[name] = str((account / name).relative_to(ROOT))
    return result


def main() -> None:
    strategy = read_json(ROOT / "config" / "final_strategy.json")
    verification = read_json(ROOT / "results" / "holdout" / "verification.json")
    if not verification["passed"]:
        raise RuntimeError("Final verification has not passed")
    development = {str(capital): account_evidence("development", strategy, capital) for capital in [20000, 50000]}
    holdout = {str(capital): account_evidence("holdout", strategy, capital) for capital in [20000, 50000]}
    validation = {str(capital): account_evidence("validation", strategy, capital) for capital in [20000, 50000]}
    snapshot = read_json(ROOT / "data" / "snapshot" / "audit.json")
    trials = pd.read_csv(ROOT / "results" / "development" / "all_account_trials.csv")
    robustness_path = ROOT / "results" / "holdout" / "robustness_summary.csv"
    robustness = None
    if robustness_path.exists():
        frame = pd.read_csv(robustness_path)
        robustness = {
            "path": str(robustness_path.relative_to(ROOT)),
            "runs": len(frame),
            "minimum_scenario_cagr_by_capital": frame.groupby("capital")["cagr"].min().to_dict(),
            "maximum_scenario_drawdown_by_capital": frame.groupby("capital")["max_drawdown"].max().to_dict(),
        }
    summary = {
        "strategy": strategy,
        "data": {
            "source": "local_hithink_duckdb_only",
            "snapshot_path": "data/snapshot/manifest.json",
            "rows": snapshot["summary"]["rows"],
            "symbols": snapshot["summary"]["symbols"],
            "start": snapshot["summary"]["start_date"],
            "end": snapshot["summary"]["end_date"],
            "remote_or_third_party_data_used": False,
            "universe": "current_listed_mainboard",
            "survivorship_bias": True,
            "historical_st_known": False,
            "official_historical_industry_available": False,
        },
        "research_attempts": {
            "account_variants": len(trials),
            "account_variants_passing_development_gates": int(trials["passed_gates"].sum()),
            "validation_finalists": len(read_json(ROOT / "config" / "validation_finalists.json")["candidates"]),
            "holdout_strategies_run_for_selection": 1,
        },
        "development": development,
        "validation": validation,
        "holdout": holdout,
        "targets": {
            "both_capitals_holdout_cagr_30pct_met": all(item["cagr_30pct_met"] for item in holdout.values()),
            "both_capitals_holdout_every_full_year_30pct_met": all(item.get("every_full_year_30pct_met", False) for item in holdout.values()),
            "both_capitals_holdout_every_calendar_year_including_partial_30pct_met": all(item.get("every_calendar_year_including_partial_30pct_met", False) for item in holdout.values()),
            "target_is_not_a_future_return_guarantee": True,
        },
        "robustness": robustness,
        "verification": verification,
        "limitations": [
            "Current-listed history excludes delisted stocks by request; reversal returns are especially exposed to survivorship bias.",
            "Historical ST status and names are unavailable; 5% limit sensitivity is a conservative scenario, not an ST reconstruction.",
            "Daily bars cannot prove opening-auction queue fills or intraday stop paths; fills use next-open with price limits and slippage.",
            "Dynamic comovement clusters are not official industry indices.",
            "Corporate-action cash availability and bonus-share sale timing use disclosed assumptions because exact payment/listing dates are unavailable.",
            "Development tried many variants; validation and one locked holdout reduce but do not eliminate selection risk.",
            "Research evidence only, not investment advice or a guarantee of 30% future returns.",
        ],
    }
    path = ROOT / "results" / "final_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n")
    compact = {
        "candidate_id": strategy["candidate_id"],
        "holding_sessions": strategy["holding_sessions"],
        "max_positions": strategy["max_positions"],
        "validation_cagr": {key: value["cagr"] for key, value in validation.items()},
        "holdout_cagr": {key: value["cagr"] for key, value in holdout.items()},
        "holdout_max_drawdown": {key: value["max_drawdown"] for key, value in holdout.items()},
        "both_capitals_holdout_cagr_30pct_met": summary["targets"]["both_capitals_holdout_cagr_30pct_met"],
        "output": str(path),
    }
    print(json.dumps(compact, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
