from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import analyze_signals as a

HERE = Path(__file__).resolve()
OUT = HERE.parent / "results"
CONFIG = json.loads((HERE.parent / "followup_config.json").read_text())


def comparison_stats(frame: pd.DataFrame, sample: str, group: str) -> dict:
    both = frame[frame["baseline_closed"] & frame["delay4_closed"]].copy()
    difference = both["delay4_net_return"] - both["baseline_net_return"]
    return {
        "sample": sample, "group": group, "both_closed_n": len(both), "signal_days": both["date"].nunique(),
        "baseline_mean": both["baseline_net_return"].mean(), "delay4_mean": both["delay4_net_return"].mean(),
        "delay4_minus_baseline_mean": difference.mean(), "delay4_minus_baseline_median": difference.median(),
        "delay4_better_share": (difference > CONFIG["comparison_tie_tolerance"]).mean(), "baseline_win_rate": (both["baseline_net_return"] > 0).mean(),
        "delay4_win_rate": (both["delay4_net_return"] > 0).mean(), "baseline_worst": both["baseline_net_return"].min(),
        "delay4_worst": both["delay4_net_return"].min(), "delay4_mean_mae": both["delay4_mae"].mean(),
    }


def main() -> None:
    outcomes = pd.read_csv(OUT / "signal_outcomes_audited.csv", parse_dates=["date", "entry_date", "exit_date"], low_memory=False)
    first = outcomes[outcomes["strong_reversal"]].copy().reset_index(drop=True)
    signals, features, lead, cluster, data, events, dates = a.load_research()
    execution = dict(a.SIGNAL_CONFIG["execution"])
    execution["initial_capital"] = 50000
    execution["max_positions"] = 2
    execution["entry_lag_sessions"] = 4
    config = a.EngineConfig.from_dict(execution)
    symbols = {symbol: i for i, symbol in enumerate(data.symbols)}
    next_keys = first[first["di"] + 1 < len(dates)][["thscode", "di"]].copy()
    next_keys["next_date"] = next_keys["di"].map(lambda i: dates[int(i) + 1])
    con = duckdb.connect()
    con.register("next_keys", next_keys[["thscode", "next_date"]])
    next_features = con.execute(
        f"""SELECT k.thscode,k.next_date,p.amount_ratio1_20,p.close_location,p.macd_hist_pct,p.close_ma5,p.ret1,p.market_return,p.breadth20
            FROM next_keys k JOIN read_parquet('{a.literal(a.PANEL)}') p ON p.thscode=k.thscode AND p.date=k.next_date"""
    ).fetchdf()
    con.close()
    next_features["di"] = next_features["next_date"].map(lambda date: dates.get_loc(date) - 1)
    first = first.merge(next_features.drop(columns=["next_date"]), on=["thscode", "di"], how="left", validate="one_to_one", suffixes=("", "_next"))
    first["next_date"] = first["di"].map(lambda i: dates[int(i) + 1] if int(i) + 1 < len(dates) else pd.NaT)
    signal_keys = set(zip(signals["thscode"], signals["date"]))
    records = []
    for number, row in enumerate(first.itertuples(), 1):
        i = int(row.di)
        symbol_i = symbols[row.thscode]
        candidate = a.Candidate(symbol_i, i, int(row.rank), -float(row.drawdown250), float(row.amount20))
        delayed = a.single_candidate(data, candidate, i, config)
        record = {
            "date": row.date, "thscode": row.thscode, "rank": row.rank, "period": row.period,
            "non_overlapping": row.non_overlapping, "baseline_closed": bool(row.closed),
            "baseline_net_return": row.net_return, "baseline_mae": row.mae,
            "baseline_entry_date": row.entry_date, "baseline_exit_date": row.exit_date,
            "baseline_status": row.status,
            "next_date": dates[i + 1] if i + 1 < len(dates) else pd.NaT,
            "next_still_original_signal": (row.thscode, dates[i + 1]) in signal_keys if i + 1 < len(dates) else np.nan,
        }
        for key, value in delayed.items():
            if key != "signal_index":
                record[f"delay4_{key}"] = value
        if i + 1 < len(dates):
            first_open = data.values["hfq_open"][i, symbol_i]
            first_low = data.values["hfq_low"][i, symbol_i]
            next_open = data.values["hfq_open"][i + 1, symbol_i]
            next_close = data.values["hfq_close"][i + 1, symbol_i]
            next_low = data.values["hfq_low"][i + 1, symbol_i]
            if not all(np.isfinite(value) and value > 0 for value in [first_open, first_low, next_open, next_close, next_low]):
                for key in ["next_return", "next_body", "next_higher_low", "next_above_first_open", "next_breaks_first_low", "next_volume_contracts", "next_consolidation"]:
                    record[key] = np.nan
            else:
                record["next_return"] = next_close / data.values["hfq_close"][i, symbol_i] - 1
                record["next_body"] = (next_close - next_open) / next_open
                record["next_higher_low"] = next_low > first_low
                record["next_above_first_open"] = next_close > first_open
                record["next_breaks_first_low"] = next_low < first_low
                record["next_volume_contracts"] = row.amount_ratio1_20_next < 1 if pd.notna(row.amount_ratio1_20_next) else np.nan
                record["next_consolidation"] = bool(record["next_higher_low"] and record["next_above_first_open"] and abs(record["next_body"]) <= .02 and record["next_volume_contracts"]) if pd.notna(row.amount_ratio1_20_next) else np.nan
        records.append(record)
        if number % 20 == 0:
            print(f"delay pairs {number}/{len(first)}", flush=True)
    pairs = pd.DataFrame(records)
    pairs.to_csv(OUT / "strong_reversal_delay_pairs.csv", index=False, encoding="utf-8-sig")
    samples = [("all", pd.Series(True, index=pairs.index)), ("top2", pairs["rank"] <= 2), ("top10", pairs["rank"] <= 10), ("non_overlapping", pairs["non_overlapping"])]
    groups = [
        ("all", pd.Series(True, index=pairs.index)),
        ("next_consolidation", pairs["next_consolidation"].fillna(False).astype(bool)),
        ("not_next_consolidation", pairs["next_consolidation"].eq(False)),
        ("next_breaks_first_low", pairs["next_breaks_first_low"].eq(True)),
        ("next_still_original_signal", pairs["next_still_original_signal"].eq(True)),
        ("next_not_original_signal", pairs["next_still_original_signal"].eq(False)),
        ("2017_2023", pairs["period"].eq("2017_2023")),
        ("2024_2026", pairs["period"].eq("2024_2026")),
    ]
    table = []
    for sample, sample_mask in samples:
        for group, group_mask in groups:
            table.append(comparison_stats(pairs[sample_mask & group_mask], sample, group))
    table = pd.DataFrame(table)
    table.to_csv(OUT / "strong_reversal_delay_stats.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "first_signals": len(first), "first_closed_baseline": int(pairs["baseline_closed"].sum()),
        "first_closed_delay4": int(pairs["delay4_closed"].sum()),
        "next_consolidation_count": int(pairs["next_consolidation"].eq(True).sum()),
        "next_consolidation_missing": int(pairs["next_consolidation"].isna().sum()),
        "source_outcomes_sha256": a.sha256(OUT / "signal_outcomes_audited.csv"),
        "code_sha256": a.sha256(HERE), "config_sha256": a.sha256(HERE.parent / "followup_config.json"),
        "historical_cutoff": "2026-10-23", "current_outcomes_used": False,
        "scope": CONFIG["status"], "load_source": data.source,
    }
    a.save_json("strong_reversal_delay_summary.json", manifest)
    print(table[(table["sample"].isin(["all", "top2", "non_overlapping"])) & (table["group"].isin(["all", "next_consolidation", "not_next_consolidation", "next_still_original_signal", "next_not_original_signal", "2017_2023", "2024_2026"]))].round(4).to_string(index=False))
    print(json.dumps(a.json_safe(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
