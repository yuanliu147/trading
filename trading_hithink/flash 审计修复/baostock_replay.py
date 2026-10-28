from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from baostock_optimization import OUTPUT, literal, write_json
from baostock_signal_research import PANEL, SIGNAL_DATASET
from flash_data import CORE, EVENTS, Candidate, FlashData, data_from_frames, load_config, sha256
from flash_engine import EngineConfig, annual_returns, drawdown_spells, run_backtest
from run_audit import file_manifest

SCENARIOS = [
    "baseline", "no_st", "turn5_low", "turn5_high", "turn1_low", "turn1_high",
    "rank_turn_high", "rank_turn_low", "rank_turn5_high", "rank_turn5_low",
]


def candidate_frame(name: str) -> pd.DataFrame:
    path = OUTPUT / "candidates" / f"{name}.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    if frame.duplicated(["date", "thscode"]).any():
        raise AssertionError(f"Duplicate candidates in {name}")
    return frame


def load_replay_data() -> FlashData:
    config = load_config()
    signals = pd.read_parquet(SIGNAL_DATASET, columns=["thscode"])
    codes = signals.drop_duplicates().sort_values("thscode").reset_index(drop=True)
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit TO '6GB'")
    con.register("selected_codes", codes)
    start = config["data"]["start"]
    end = config["data"]["end"]
    bars = con.execute(f"""
        SELECT p.thscode, p.date,
               p.open, p.high, p.low, p.close,
               p.hfq_open, p.hfq_high, p.hfq_low, p.hfq_close,
               c.volume, c.reference_close, c.previous_actual_close, c.backward_factor,
               c.factor_only_adjustment
        FROM read_parquet('{literal(PANEL)}') p
        JOIN read_parquet('{literal(CORE)}') c USING (thscode, date)
        JOIN selected_codes s ON s.thscode=p.thscode
        WHERE p.date BETWEEN DATE '{start}' AND DATE '{end}'
        ORDER BY p.date, p.thscode
    """).fetchdf()
    sessions = con.execute(f"""
        SELECT DISTINCT date FROM read_parquet('{literal(PANEL)}')
        WHERE date BETWEEN DATE '{start}' AND DATE '{end}' ORDER BY date
    """).fetchdf()
    factor_error = con.execute(f"""
        SELECT max(abs(reference_close-previous_actual_close*previous_backward_factor/backward_factor))
        FROM read_parquet('{literal(CORE)}') c JOIN selected_codes s ON s.thscode=c.thscode
        WHERE previous_backward_factor IS NOT NULL AND backward_factor>0
    """).fetchone()[0]
    con.close()
    if factor_error is not None and factor_error > 1e-8:
        raise AssertionError(f"Reference close mismatch: {factor_error}")
    symbols = codes["thscode"].tolist()
    events = pd.read_parquet(EVENTS, filters=[("thscode", "in", symbols)])
    events = events[pd.to_datetime(events["ex_date"]).between(start, end)].copy()
    source = {
        "panel": str(PANEL),
        "core": str(CORE),
        "events": str(EVENTS),
        "baostock_signal_fields": str(SIGNAL_DATASET),
        "raw_signal_symbols": len(codes),
        "joined_bar_rows": len(bars),
        "reference_price_max_error": float(factor_error or 0),
        "third_party_data_used": True,
        "evaluation_status": "historical_research_not_independent_holdout",
    }
    return data_from_frames(
        bars,
        candidate_frame("baseline"),
        events,
        sessions=pd.DatetimeIndex(sessions["date"]),
        source=source,
    )


def replace_candidates(data: FlashData, frame: pd.DataFrame) -> FlashData:
    date_indices = {date: index for index, date in enumerate(data.dates)}
    symbol_indices = {symbol: index for index, symbol in enumerate(data.symbols)}
    unknown_symbols = set(frame["thscode"]) - set(symbol_indices)
    unknown_dates = set(frame["date"]) - set(date_indices)
    if unknown_symbols or unknown_dates:
        raise AssertionError(f"Unmapped candidate keys: {list(unknown_symbols)[:3]} {list(unknown_dates)[:3]}")
    lookup = {}
    for date, group in frame.groupby("date", sort=True):
        records = []
        for row in group.sort_values(["rank", "thscode"]).itertuples(index=False):
            if row.rank > 10:
                continue
            records.append(Candidate(
                symbol_indices[row.thscode], date_indices[date], int(row.rank),
                float(row.score), float(row.amount20),
            ))
        lookup[date_indices[date]] = records
    source = dict(data.source)
    source["candidate_rows"] = len(frame)
    return FlashData(data.dates, data.symbols, data.values, lookup, data.events, source)


def export_account(result, name: str, scenario: str, capital: int, execution: dict, source: dict) -> dict:
    account = OUTPUT / "accounts" / name
    account.mkdir(parents=True, exist_ok=True)
    result.daily_equity.to_csv(account / "daily_equity.csv", index=False)
    result.trades.to_csv(account / "trades.csv", index=False)
    result.orders.to_csv(account / "orders.csv", index=False)
    result.corporate_actions.to_csv(account / "corporate_actions.csv", index=False)
    result.positions_daily.to_csv(account / "positions_daily.csv", index=False)
    result.open_positions.to_csv(account / "open_positions.csv", index=False)
    annual_returns(result.daily_equity, capital).to_csv(account / "annual.csv", index=False)
    drawdown_spells(result.daily_equity, capital).to_csv(account / "drawdowns.csv", index=False)
    metrics = dict(result.metrics)
    metrics.update({
        "scenario": scenario,
        "account": name,
        "capital": capital,
        "closed_plus_open": metrics["trade_count"] + metrics["ending_open_positions"],
        "evaluation_status": "historical_research_not_independent_holdout",
    })
    write_json(account / "metrics.json", metrics)
    write_json(account / "run_config.json", {
        "strategy_id": "flash_o03_baostock_candidate_research",
        "scenario": scenario,
        "capital": capital,
        "execution": execution,
        "source": source,
        "candidate_sha256": sha256(OUTPUT / "candidates" / f"{scenario}.csv"),
        "evaluation_status": "historical_research_not_independent_holdout",
    })
    write_json(account / "manifest.json", file_manifest(account))
    return metrics


def assert_baseline(metrics: dict) -> None:
    baseline_path = Path(__file__).resolve().parent / "results" / "accounts" / "O03_c100000" / "metrics.json"
    expected = json.loads(baseline_path.read_text())
    keys = [
        "final_equity", "total_return", "cagr", "max_drawdown", "sharpe",
        "trade_count", "ending_open_positions", "win_rate", "buy_order_count",
        "rejected_buy_count", "profit_target_exits", "time_exits",
    ]
    for key in keys:
        left, right = metrics[key], expected[key]
        if isinstance(left, (float, np.floating)):
            if not np.isclose(left, right, rtol=1e-10, atol=1e-7):
                raise AssertionError(f"Baseline {key}: {left} != {right}")
        elif left != right:
            raise AssertionError(f"Baseline {key}: {left} != {right}")


def run_replay(scenarios: list[str], capitals: list[int], suffix: str, overrides: dict) -> pd.DataFrame:
    data = load_replay_data()
    rows = []
    for scenario in scenarios:
        scenario_data = replace_candidates(data, candidate_frame(scenario))
        for capital in capitals:
            execution = dict(load_config()["execution"])
            execution["initial_capital"] = capital
            execution.update(overrides)
            engine_config = EngineConfig.from_dict(execution)
            result = run_backtest(scenario_data, engine_config)
            name = f"{scenario}_c{capital}{suffix}"
            if scenario == "baseline" and capital == 100000 and not overrides:
                assert_baseline(result.metrics)
                print(json.dumps({"baseline_reproduction": "passed"}, ensure_ascii=False), flush=True)
            metrics = export_account(result, name, scenario, capital, execution, scenario_data.source)
            rows.append(metrics)
            print(json.dumps({
                "account": name, "cagr": metrics["cagr"],
                "max_drawdown": metrics["max_drawdown"], "trades": metrics["trade_count"],
                "final_equity": metrics["final_equity"],
            }), flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / f"replay_summary{suffix}.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--capitals", default="100000")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--take-profit-mode", choices=["intraday", "next_open"])
    parser.add_argument("--max-participation", type=float)
    parser.add_argument("--limit-rate", type=float)
    args = parser.parse_args()
    scenarios = args.scenarios.split(",")
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    capitals = [int(value) for value in args.capitals.split(",")]
    overrides = {}
    if args.take_profit_mode is not None:
        overrides["take_profit_mode"] = args.take_profit_mode
    if args.max_participation is not None:
        overrides["max_participation"] = args.max_participation
    if args.limit_rate is not None:
        overrides["limit_rate"] = args.limit_rate
    summary = run_replay(scenarios, capitals, args.suffix, overrides)
    print(summary[["account", "cagr", "max_drawdown", "trade_count", "final_equity"]].to_string(index=False))


if __name__ == "__main__":
    main()
