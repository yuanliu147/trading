from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from engine import run_backtest

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "data" / "work"
RESULTS = ROOT / "results"

LIVE_ELIGIBILITY = """
    listing_session >= 120
    AND close BETWEEN 2 AND 80
    AND amount20 >= 20000000
    AND previous_hfq_close IS NOT NULL
"""


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def literal(path: Path) -> str:
    return str(path).replace("'", "''")


def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit TO '12GB'")
    return con


def period_dates(period: str) -> tuple[str, str]:
    config = json.loads((ROOT / "config" / "research.json").read_text())["sample"]
    return config[f"{period}_start"], config[f"{period}_end"]


def hypothesis_by_id(identifier: str) -> dict:
    hypotheses = []
    for name in ["hypotheses.json", "round2_hypotheses.json"]:
        config = json.loads((ROOT / "config" / name).read_text())
        hypotheses.extend(config["hypotheses"])
    matches = [item for item in hypotheses if item["id"] == identifier]
    if len(matches) != 1:
        raise KeyError(identifier)
    return matches[0]


def source_join() -> str:
    return (
        f"FROM read_parquet('{literal(WORK / 'features.parquet')}') f "
        f"LEFT JOIN read_parquet('{literal(WORK / 'cluster_daily.parquet')}') c USING (thscode, date) "
        f"LEFT JOIN (SELECT thscode, date, morning_star, three_bar_turnaround, piercing, "
        f"bullish_harami, two_day_recovery, three_white_soldiers, gap_up_confirmation, "
        f"limit_washout_rebound FROM read_parquet('{literal(WORK / 'confirmation_patterns.parquet')}')) p USING (thscode, date)"
    )


def generate_signals(con: duckdb.DuckDBPyConnection, hypothesis: dict, start: str, end: str) -> tuple[pd.DataFrame, set]:
    candidates = con.execute(
        f"""
        SELECT f.date, f.thscode, {hypothesis['score']} AS score, f.amount20
        {source_join()}
        WHERE f.date BETWEEN DATE '{start}' AND DATE '{end}'
          AND {LIVE_ELIGIBILITY} AND ({hypothesis['entry']})
        ORDER BY f.date, score DESC, f.thscode
        """
    ).fetchdf()
    candidates = candidates[np_isfinite(candidates["score"])]
    candidates["date"] = pd.to_datetime(candidates["date"])
    exit_frame = con.execute(
        f"""
        SELECT f.date, f.thscode
        {source_join()}
        WHERE f.date BETWEEN DATE '{start}' AND DATE '{end}' AND ({hypothesis['exit']})
        """
    ).fetchdf()
    exit_frame["date"] = pd.to_datetime(exit_frame["date"])
    exit_signals = set(zip(exit_frame["date"], exit_frame["thscode"]))
    return candidates, exit_signals


def np_isfinite(series: pd.Series):
    import numpy as np

    return np.isfinite(series.to_numpy(dtype=float))


def load_daily(con: duckdb.DuckDBPyConnection, start: str, end: str) -> pd.DataFrame:
    frame = con.execute(
        f"""
        SELECT thscode, date, open, high, low, close, volume, amount,
               reference_close, previous_actual_close, backward_factor, previous_backward_factor,
               up_limit_10, down_limit_10, up_limit_05, down_limit_05,
               factor_only_adjustment, dividend_per_share, per_share_bonus, rights_ratio, rights_price
        FROM read_parquet('{literal(WORK / 'core.parquet')}')
        WHERE date BETWEEN DATE '{start}' AND DATE '{end}'
        ORDER BY date, thscode
        """
    ).fetchdf()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def run_one(args: argparse.Namespace) -> dict:
    if args.period == "holdout":
        locked = ROOT / "config" / "final_strategy.json"
        if not locked.exists():
            raise RuntimeError("Holdout requires config/final_strategy.json")
        strategy = json.loads(locked.read_text())
        expected = [args.candidate_id, args.holding, args.positions]
        actual = [strategy["candidate_id"], strategy["holding_sessions"], strategy["max_positions"]]
        if expected != actual or not strategy.get("locked_before_holdout"):
            raise RuntimeError("Holdout parameters do not match the locked strategy")
        current_hashes = {
            "hypotheses_sha256": hashlib.sha256((ROOT / "config" / "hypotheses.json").read_bytes()).hexdigest(),
            "round2_hypotheses_sha256": hashlib.sha256((ROOT / "config" / "round2_hypotheses.json").read_bytes()).hexdigest(),
            "research_config_sha256": hashlib.sha256((ROOT / "config" / "research.json").read_bytes()).hexdigest(),
            "account_selection_sha256": hashlib.sha256((ROOT / "config" / "account_selection.json").read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256((ROOT / "engine.py").read_bytes()).hexdigest(),
            "features_sha256": json.loads((WORK / "metadata.json").read_text())["files"]["features"]["sha256"],
        }
        for key, value in current_hashes.items():
            if strategy.get(key) != value:
                raise RuntimeError(f"Holdout lock hash mismatch: {key}")
    hypothesis = hypothesis_by_id(args.candidate_id)
    if args.holding not in hypothesis["holding_options"]:
        raise ValueError("Holding period was not pre-registered for this hypothesis")
    start, end = period_dates(args.period)
    con = connection()
    try:
        candidates, exit_signals = generate_signals(con, hypothesis, start, end)
        daily = load_daily(con, start, end)
        events = pd.read_parquet(ROOT / "data" / "snapshot" / "adjustment_events.parquet")
    finally:
        con.close()
    research = json.loads((ROOT / "config" / "research.json").read_text())
    config = {
        "start_date": start,
        "end_date": end,
        "initial_capital": args.capital,
        "max_positions": args.positions,
        "slot_fraction": 1.0 / args.positions,
        "holding_sessions": args.holding,
        "slippage": args.slippage,
        "limit_rate": args.limit_rate,
        "participation": research["execution"]["maximum_known_amount_participation"],
        "costs": research["costs"],
    }
    config.update(hypothesis.get("engine_overrides", {}))
    result = run_backtest(daily=daily, events=events, candidates=candidates, exit_signals=exit_signals, config=config)
    name = f"{args.candidate_id}_h{args.holding}_p{args.positions}_c{args.capital}_s{int(args.slippage * 10000)}_l{int(args.limit_rate * 100)}"
    output = RESULTS / args.period / "accounts" / name
    output.mkdir(parents=True, exist_ok=True)
    result.daily_equity.to_csv(output / "daily_equity.csv", index=False)
    result.trades.to_csv(output / "trades.csv", index=False)
    result.orders.to_csv(output / "orders.csv", index=False)
    result.event_ledger.to_csv(output / "corporate_actions.csv", index=False)
    candidates.to_csv(output / "entry_signals.csv", index=False)
    metrics = dict(result.metrics)
    metrics.update(
        {
            "period": args.period,
            "candidate_id": args.candidate_id,
            "holding_sessions": args.holding,
            "max_positions": args.positions,
            "initial_capital": args.capital,
            "slippage": args.slippage,
            "limit_rate": args.limit_rate,
            "entry_signals": len(candidates),
            "engine_overrides": hypothesis.get("engine_overrides", {}),
            "hypothesis_sha256": hashlib.sha256((ROOT / "config" / "hypotheses.json").read_bytes()).hexdigest(),
            "round2_hypotheses_sha256": hashlib.sha256((ROOT / "config" / "round2_hypotheses.json").read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256((ROOT / "engine.py").read_bytes()).hexdigest(),
            "features_sha256": json.loads((WORK / "metadata.json").read_text())["files"]["features"]["sha256"],
            "survivorship_bias": "current_listed_only",
            "historical_st_known": False,
        }
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_config.json", config)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--holding", type=int, required=True)
    parser.add_argument("--positions", type=int, choices=[2, 3], required=True)
    parser.add_argument("--capital", type=int, choices=[20000, 50000], required=True)
    parser.add_argument("--period", choices=["development", "validation", "holdout"], required=True)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--limit-rate", type=float, choices=[0.05, 0.10], default=0.10)
    args = parser.parse_args()
    metrics = run_one(args)
    print(json.dumps(metrics, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
