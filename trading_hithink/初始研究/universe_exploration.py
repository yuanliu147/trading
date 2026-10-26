from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine import run_backtest
from portfolio import connection, generate_signals, hypothesis_by_id, load_daily, period_dates, write_json

ROOT = Path(__file__).resolve().parent

ENTRIES = {
    "engulfing_original": "engulfing_reversal",
    "deep50": "engulfing_deep50",
    "deep50_lowamount": "engulfing_deep50_lowamount",
    "lowamount": "engulfing_lowamount",
}


def main() -> None:
    start, end = period_dates("development")
    research = json.loads((ROOT / "config" / "research.json").read_text())
    con = connection()
    try:
        daily = load_daily(con, start, end)
        candidate_frames = {}
        entries = {}
        for label, identifier in ENTRIES.items():
            base = hypothesis_by_id(identifier)
            entries[label] = base["entry"]
            hypothesis = {**base, "score": "-drawdown250", "exit": "FALSE"}
            candidate_frames[label], _ = generate_signals(con, hypothesis, start, end)
    finally:
        con.close()
    events = pd.read_parquet(ROOT / "data" / "snapshot" / "adjustment_events.parquet")
    records = []
    for label, candidates in candidate_frames.items():
        for holding in [19, 20]:
            for capital in [20000, 50000]:
                config = {
                    "start_date": start,
                    "end_date": end,
                    "initial_capital": capital,
                    "max_positions": 2,
                    "slot_fraction": 0.5,
                    "holding_sessions": holding,
                    "slippage": 0.001,
                    "limit_rate": 0.10,
                    "participation": research["execution"]["maximum_known_amount_participation"],
                    "costs": research["costs"],
                }
                result = run_backtest(daily, events, candidates, set(), config)
                output = ROOT / "results" / "development" / "universe_trials" / f"{label}_h{holding}_c{capital}"
                output.mkdir(parents=True, exist_ok=True)
                result.trades.to_csv(output / "trades.csv", index=False)
                result.daily_equity.to_csv(output / "daily_equity.csv", index=False)
                result.orders.to_csv(output / "orders.csv", index=False)
                metrics = dict(result.metrics)
                metrics.update({"universe_label": label, "entry_sql": entries[label], "entry_signals": len(candidates), "holding_sessions": holding})
                write_json(output / "metrics.json", metrics)
                records.append({key: metrics.get(key) for key in [
                    "universe_label", "holding_sessions", "initial_capital", "entry_signals", "cagr", "max_drawdown",
                    "calmar", "trade_count", "win_rate", "top_1_profit_share", "top_5_profit_share", "holding_time_ratio",
                    "final_equity", "factor_only_adjustment_count",
                ]})
                print(json.dumps({"universe": label, "holding": holding, "capital": capital, "cagr": metrics["cagr"], "mdd": metrics["max_drawdown"]}))
    path = ROOT / "results" / "development" / "universe_trials_summary.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(json.dumps({"ok": True, "runs": len(records), "output": str(path), "holdout_used": False}))


if __name__ == "__main__":
    main()
