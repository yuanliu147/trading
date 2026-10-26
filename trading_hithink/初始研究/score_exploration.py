from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine import run_backtest
from portfolio import connection, generate_signals, hypothesis_by_id, load_daily, period_dates, write_json

ROOT = Path(__file__).resolve().parent

SCORES = {
    "neg_return10": "-return_10d",
    "target_drop10": "-abs(return_10d + 0.10)",
    "target_drop15": "-abs(return_10d + 0.15)",
    "target_rsi25": "-abs(rsi14 - 25)",
    "deep250": "-drawdown250",
    "low_amount": "-amount20",
    "random_daily": "hash(thscode, CAST(date AS VARCHAR))",
}


def main() -> None:
    identifier = "engulfing_deep50_lowamount_atr3"
    base = hypothesis_by_id(identifier)
    start, end = period_dates("development")
    research = json.loads((ROOT / "config" / "research.json").read_text())
    con = connection()
    try:
        daily = load_daily(con, start, end)
        candidate_frames = {}
        for label, score in SCORES.items():
            hypothesis = {**base, "score": score, "exit": "FALSE"}
            candidate_frames[label], _ = generate_signals(con, hypothesis, start, end)
    finally:
        con.close()
    events = pd.read_parquet(ROOT / "data" / "snapshot" / "adjustment_events.parquet")
    records = []
    for label, candidates in candidate_frames.items():
        for positions in [2, 3]:
            for capital in [20000, 50000]:
                config = {
                    "start_date": start,
                    "end_date": end,
                    "initial_capital": capital,
                    "max_positions": positions,
                    "slot_fraction": 1.0 / positions,
                    "holding_sessions": 20,
                    "slippage": 0.001,
                    "limit_rate": 0.10,
                    "participation": research["execution"]["maximum_known_amount_participation"],
                    "costs": research["costs"],
                }
                result = run_backtest(daily, events, candidates, set(), config)
                output = ROOT / "results" / "development" / "score_trials" / f"{label}_p{positions}_c{capital}"
                output.mkdir(parents=True, exist_ok=True)
                result.trades.to_csv(output / "trades.csv", index=False)
                result.daily_equity.to_csv(output / "daily_equity.csv", index=False)
                result.orders.to_csv(output / "orders.csv", index=False)
                metrics = dict(result.metrics)
                metrics.update({"score_label": label, "score_sql": SCORES[label], "max_positions": positions, "initial_capital": capital})
                write_json(output / "metrics.json", metrics)
                records.append({key: metrics.get(key) for key in [
                    "score_label", "max_positions", "initial_capital", "cagr", "max_drawdown", "calmar",
                    "trade_count", "win_rate", "top_1_profit_share", "top_5_profit_share", "holding_time_ratio",
                    "final_equity", "factor_only_adjustment_count", "max_abs_cash_flow_error",
                ]})
                print(json.dumps({"score": label, "positions": positions, "capital": capital, "cagr": metrics["cagr"], "mdd": metrics["max_drawdown"]}))
    path = ROOT / "results" / "development" / "score_trials_summary.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(json.dumps({"ok": True, "runs": len(records), "output": str(path), "holdout_used": False}))


if __name__ == "__main__":
    main()
