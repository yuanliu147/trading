from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

import analyze_signals as a

HERE = Path(__file__).resolve()
OUT = HERE.parent / "results"


def main() -> None:
    summary = json.loads((OUT / "analysis_summary.json").read_text())
    if a.sha256(a.DATABASE) != summary["input_hashes"]["raw_database"]:
        raise AssertionError("Raw DuckDB changed after the current snapshot was frozen")
    outcomes = pd.read_csv(OUT / "signal_outcomes_audited.csv", parse_dates=["date"], low_memory=False)
    signals = outcomes[outcomes["strong_reversal"] & outcomes["date"].eq(pd.Timestamp("2026-10-23"))].copy()
    con = duckdb.connect(str(a.DATABASE), read_only=True)
    con.register("current_signals", signals[["thscode"]].drop_duplicates())
    bars = con.execute(
        """WITH base AS (
               SELECT h.thscode,h.date,h.open,h.high,h.low,h.close,d.amount,
                      d.open actual_open,d.high actual_high,d.low actual_low,d.close actual_close,
                      avg(d.amount) OVER(PARTITION BY h.thscode ORDER BY h.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) amount20
               FROM v_daily_hfq h JOIN v_daily d USING(thscode,date)
               JOIN current_signals s ON h.thscode=s.thscode
           ), returns AS (
               SELECT *,close/lag(close) OVER(PARTITION BY thscode ORDER BY date)-1 ret1 FROM base
           ) SELECT * FROM returns WHERE date IN (DATE '2026-10-23',DATE '2026-10-26') ORDER BY thscode,date"""
    ).fetchdf()
    con.close()
    if bars["date"].max() != pd.Timestamp("2026-10-26"):
        raise AssertionError("Current follow-through requires the 2026-10-26 local close")
    bars["amount_ratio"] = bars["amount"] / bars["amount20"]
    columns = ["open", "high", "low", "close", "actual_open", "actual_high", "actual_low", "actual_close", "ret1", "amount_ratio"]
    table = bars.pivot(index="thscode", columns="date", values=columns)
    table.columns = [f"{field}_{'hfq' if field in {'open', 'high', 'low', 'close', 'ret1'} else 'raw'}_{pd.Timestamp(date):%d}" for field, date in table.columns]
    table = table.reset_index().merge(signals[["thscode", "rank", "drawdown250", "close_location"]], on="thscode", validate="one_to_one").sort_values("rank")
    table["higher_low"] = table["low_hfq_26"] > table["low_hfq_23"]
    table["above_first_open"] = table["close_hfq_26"] > table["open_hfq_23"]
    table["body26"] = (table["close_hfq_26"] - table["open_hfq_26"]) / table["open_hfq_26"]
    table["consolidation"] = table["higher_low"] & table["above_first_open"] & table["body26"].abs().le(.02) & table["amount_ratio_raw_26"].lt(1)
    table["future_return_known"] = False
    table.to_csv(OUT / "current_strong_cohort_followthrough.csv", index=False, encoding="utf-8-sig")
    result = {
        "first_signal_date": "2026-10-23", "latest_known_bar": "2026-10-26", "signals": len(table),
        "higher_low": int(table["higher_low"].sum()), "above_first_open": int(table["above_first_open"].sum()),
        "small_body_low_volume_consolidation": int(table["consolidation"].sum()),
        "future_return_used": False, "raw_database_sha256": summary["input_hashes"]["raw_database"],
        "source_outcomes_sha256": a.sha256(OUT / "signal_outcomes_audited.csv"),
        "code_sha256": a.sha256(HERE), "consolidation_config_sha256": a.sha256(HERE.parent / "followup_config.json"),
    }
    a.save_json("current_followthrough_summary.json", result)
    print(table[["rank", "thscode", "actual_open_raw_23", "actual_close_raw_23", "actual_open_raw_26", "actual_low_raw_26", "actual_close_raw_26", "ret1_hfq_26", "amount_ratio_raw_26", "higher_low", "above_first_open", "consolidation"]].to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(json.dumps(a.json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
