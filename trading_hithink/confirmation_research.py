from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from diagnostics import connection, prepare_eligible

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "data" / "work"


def literal(path: Path) -> str:
    return str(path).replace("'", "''")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def build_patterns(con) -> None:
    output = WORK / "confirmation_patterns.parquet"
    if output.exists():
        raise FileExistsError(output)
    sql = f"""
        WITH x AS (
            SELECT thscode, date, hfq_open, hfq_high, hfq_low, hfq_close, ret1,
                   previous_hfq_open, previous_hfq_high, previous_hfq_low, previous_hfq_close,
                   lag(hfq_open, 2) OVER g AS open2,
                   lag(hfq_high, 2) OVER g AS high2,
                   lag(hfq_low, 2) OVER g AS low2,
                   lag(hfq_close, 2) OVER g AS close2,
                   lag(hfq_open, 3) OVER g AS open3,
                   lag(hfq_close, 3) OVER g AS close3,
                   lag(ret1, 1) OVER g AS previous_ret1
            FROM read_parquet('{literal(WORK / 'core.parquet')}')
            WINDOW g AS (PARTITION BY thscode ORDER BY date)
        )
        SELECT *,
            close2 < open2 AND abs(close2 - open2) / nullif(open2, 0) > 0.025
              AND abs(previous_hfq_close - previous_hfq_open) / nullif(previous_hfq_open, 0) < 0.012
              AND hfq_close > hfq_open AND (hfq_close - hfq_open) / nullif(hfq_open, 0) > 0.015
              AND hfq_close > (open2 + close2) / 2 AS morning_star,
            close2 < open2 AND previous_hfq_close < previous_hfq_open
              AND hfq_close > hfq_open AND hfq_close > previous_hfq_high AND ret1 > 0.02 AS three_bar_turnaround,
            previous_hfq_close < previous_hfq_open
              AND (previous_hfq_open - previous_hfq_close) / nullif(previous_hfq_open, 0) > 0.03
              AND hfq_open < previous_hfq_low AND hfq_close > hfq_open
              AND hfq_close > (previous_hfq_open + previous_hfq_close) / 2
              AND hfq_close < previous_hfq_open AS piercing,
            previous_hfq_close < previous_hfq_open
              AND (previous_hfq_open - previous_hfq_close) / nullif(previous_hfq_open, 0) > 0.03
              AND hfq_open > previous_hfq_close AND hfq_close < previous_hfq_open
              AND hfq_close > hfq_open
              AND abs(hfq_close - hfq_open) < 0.5 * abs(previous_hfq_close - previous_hfq_open) AS bullish_harami,
            previous_hfq_close < open2 AND hfq_close > open2
              AND hfq_close > hfq_open AND ret1 > 0.02 AS two_day_recovery,
            close2 > open2 AND previous_hfq_close > previous_hfq_open AND hfq_close > hfq_open
              AND close2 > close3 AND previous_hfq_close > close2 AND hfq_close > previous_hfq_close
              AND (close2 - open2) / nullif(open2, 0) > 0.01
              AND (previous_hfq_close - previous_hfq_open) / nullif(previous_hfq_open, 0) > 0.01
              AND (hfq_close - hfq_open) / nullif(hfq_open, 0) > 0.01
              AND (close2 - low2) / nullif(high2 - low2, 0) > 0.5
              AND (previous_hfq_close - previous_hfq_low) / nullif(previous_hfq_high - previous_hfq_low, 0) > 0.5
              AND (hfq_close - hfq_low) / nullif(hfq_high - hfq_low, 0) > 0.5 AS three_white_soldiers,
            hfq_open > previous_hfq_close * 1.01 AND hfq_close > hfq_open
              AND hfq_close > previous_hfq_high AND ret1 > 0.02 AS gap_up_confirmation,
            previous_ret1 < -0.07 AND hfq_close > hfq_open AND ret1 > 0.02
              AND (hfq_close - hfq_low) / nullif(hfq_high - hfq_low, 0) > 0.7 AS limit_washout_rebound
        FROM x
    """
    (WORK / "build_confirmation_patterns.sql").write_text(sql)
    con.execute(f"COPY ({sql}) TO '{literal(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def entry_for(config: dict, pattern: dict) -> str:
    return config["base_entry"].replace("@PATTERN@", pattern["pattern_key"]).replace("@RSI@", str(pattern["rsi_max"]))


def diagnostics(con, config: dict) -> None:
    prepare_eligible(con, "development")
    con.execute(
        f"""
        CREATE TEMP TABLE confirmed_eligible AS
        SELECT e.*, p.morning_star, p.three_bar_turnaround, p.piercing,
               p.bullish_harami, p.two_day_recovery, p.three_white_soldiers,
               p.gap_up_confirmation, p.limit_washout_rebound
        FROM eligible e
        JOIN read_parquet('{literal(WORK / 'confirmation_patterns.parquet')}') p USING (thscode, date)
        """
    )
    summaries = []
    yearly_frames = []
    for pattern in config["patterns"]:
        entry = entry_for(config, pattern)
        con.execute(
            f"""
            CREATE TEMP TABLE confirmation_selected AS
            SELECT e.*, b.baseline5, b.baseline10, b.baseline20, b.baseline60
            FROM confirmed_eligible e JOIN daily_baseline b USING (date)
            WHERE fwd20 IS NOT NULL AND ({entry})
            """
        )
        summary = con.execute(
            """
            SELECT count(*) AS observations, count(DISTINCT thscode) AS symbols,
                   count(DISTINCT date) AS active_days,
                   avg(fwd5) AS mean_fwd5, avg(fwd10) AS mean_fwd10,
                   avg(fwd20) AS mean_fwd20, avg(fwd60) AS mean_fwd60,
                   median(fwd20) AS median_fwd20, avg((fwd20 > 0)::INTEGER) AS win_rate20,
                   avg(fwd20 - baseline20) AS excess20,
                   avg(fwd20 - baseline20) - 0.006 AS net60bp_excess20
            FROM confirmation_selected
            """
        ).fetchdf().iloc[0].to_dict()
        yearly = con.execute(
            """
            SELECT year(date) AS year, count(*) AS observations, avg(fwd20) AS mean_fwd20,
                   avg(fwd20 - baseline20) AS excess20
            FROM confirmation_selected GROUP BY year(date) ORDER BY year
            """
        ).fetchdf()
        summary.update({"id": pattern["id"], "pattern_key": pattern["pattern_key"], "entry": entry})
        summary["positive_excess_years"] = int((yearly["excess20"] > 0).sum())
        summary["worst_year_excess20"] = yearly["excess20"].min()
        summaries.append(summary)
        yearly.insert(0, "id", pattern["id"])
        yearly_frames.append(yearly)
        con.execute("DROP TABLE confirmation_selected")
    summary_frame = pd.DataFrame(summaries).sort_values("net60bp_excess20", ascending=False)
    summary_frame.to_csv(ROOT / "results" / "development" / "confirmation_summary.csv", index=False)
    pd.concat(yearly_frames, ignore_index=True).to_csv(ROOT / "results" / "development" / "confirmation_yearly.csv", index=False)
    eligible = summary_frame[
        (summary_frame["observations"] >= config["selection"]["minimum_development_signals"])
        & (summary_frame["positive_excess_years"] >= config["selection"]["minimum_positive_excess_years"])
    ]
    candidates = []
    for row in eligible.head(config["selection"]["maximum_development_account_candidates"]).to_dict("records"):
        candidates.append({
            "id": row["id"],
            "entry": row["entry"],
            "exit": config["exit"],
            "score": config["score"],
            "holding_options": config["holding_options"],
            "position_options": config["position_options"],
            "capital_options": config["capital_options"],
            "diagnostic_evidence": row,
            "research_round": 5,
        })
    write_json(ROOT / "config" / "confirmation_candidates.json", {
        "stage": "development_confirmation_candidates",
        "holdout_used": False,
        "source_config_sha256": hashlib.sha256((ROOT / "config" / "confirmation_hypotheses.json").read_bytes()).hexdigest(),
        "candidates": candidates,
    })
    print(summary_frame[["id", "observations", "active_days", "mean_fwd5", "mean_fwd20", "net60bp_excess20", "positive_excess_years"]].to_string(index=False))


def main() -> None:
    con = connection()
    try:
        build_patterns(con)
        config = json.loads((ROOT / "config" / "confirmation_hypotheses.json").read_text())
        diagnostics(con, config)
    finally:
        con.close()


if __name__ == "__main__":
    main()
