from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "data" / "work"
RESULTS = ROOT / "results"

FACTOR_FAMILIES = {
    "return_5d": "trend_reversal",
    "return_10d": "trend_reversal",
    "return_20d": "trend_reversal",
    "return_60d": "trend_reversal",
    "return_120d": "trend_reversal",
    "close_ma20": "trend_macd",
    "close_ma60": "trend_macd",
    "close_ma120": "trend_macd",
    "ma20_ma60": "trend_macd",
    "ma20_slope5": "trend_macd",
    "ma60_slope10": "trend_macd",
    "macd_dif_pct": "trend_macd",
    "macd_dea_pct": "trend_macd",
    "macd_hist_pct": "trend_macd",
    "macd_hist_delta": "trend_macd",
    "rsi14": "reversal_volatility",
    "kdj_j": "reversal_volatility",
    "breakout20": "structure",
    "breakout60": "structure",
    "breakout120": "structure",
    "drawdown60": "structure",
    "drawdown250": "structure",
    "price_position60": "structure",
    "distance_swing_high": "structure",
    "distance_swing_low": "structure",
    "days_since_swing_high": "structure",
    "days_since_swing_low": "structure",
    "week_return4": "multiscale",
    "week_return12": "multiscale",
    "week_close_ma4": "multiscale",
    "month_return3": "multiscale",
    "month_close_ma6": "multiscale",
    "atr20_pct": "reversal_volatility",
    "volatility20": "reversal_volatility",
    "volatility60": "reversal_volatility",
    "volatility_contraction": "reversal_volatility",
    "range_pct": "candle_volume",
    "range20_pct": "candle_volume",
    "candle_body": "candle_volume",
    "upper_shadow": "candle_volume",
    "lower_shadow": "candle_volume",
    "close_location": "candle_volume",
    "gap_open": "candle_volume",
    "amount_ratio1_20": "candle_volume",
    "amount_ratio5_20": "candle_volume",
    "amount_ratio20_60": "candle_volume",
    "amount_up_down20": "candle_volume",
    "breadth20": "market_regime",
    "breadth60": "market_regime",
    "market_return5": "market_regime",
    "market_return20": "market_regime",
    "market_volatility20": "market_regime",
    "sentiment_score": "limit_emotion",
    "sentiment_change5": "limit_emotion",
    "limit_up10_share": "limit_emotion",
    "limit_down10_share": "limit_emotion",
    "broken_limit_up10_share": "limit_emotion",
    "peer_return5": "comovement_cluster",
    "peer_return20": "comovement_cluster",
    "peer_close_ma20": "comovement_cluster",
    "peer_breadth20": "comovement_cluster",
    "peer_breadth60": "comovement_cluster",
    "cluster_relative20": "comovement_cluster",
    "cluster_strength": "comovement_cluster",
}

BASE_ELIGIBILITY = """
    listing_session >= 120
    AND close BETWEEN 2 AND 80
    AND amount20 >= 20000000
    AND previous_hfq_close IS NOT NULL
    AND next_hfq_open IS NOT NULL
    AND next_actual_open IS NOT NULL
    AND next_up_limit_10 IS NOT NULL
    AND next_actual_open < next_up_limit_10 - 0.0001
"""


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit TO '16GB'")
    temp = WORK / "duckdb_tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory TO '{str(temp).replace(chr(39), chr(39) * 2)}'")
    return con


def period_bounds(period: str) -> tuple[str, str]:
    boundaries = json.loads((WORK / "sample_boundaries.json").read_text())[period]
    return boundaries["first_session"][:10], boundaries["last_uncontaminated_signal_for_60d"][:10]


def prepare_eligible(con: duckdb.DuckDBPyConnection, period: str) -> int:
    start, end = period_bounds(period)
    if period == "holdout":
        raise ValueError("Holdout diagnostics are not available during selection")
    features = str(WORK / "features.parquet").replace("'", "''")
    clusters = str(WORK / "cluster_daily.parquet").replace("'", "''")
    con.execute(
        f"""
        CREATE TEMP TABLE eligible AS
        SELECT f.*,
               c.cluster_id, c.cluster_strength, c.cluster_size,
               c.peer_return1, c.peer_return5, c.peer_return20, c.peer_return60,
               c.peer_close_ma20, c.peer_close_ma60, c.peer_breadth20, c.peer_breadth60,
               c.peer_limit_up_share, c.cluster_relative20,
               open_after_5_sessions / nullif(next_hfq_open, 0) - 1 AS fwd5,
               open_after_10_sessions / nullif(next_hfq_open, 0) - 1 AS fwd10,
               open_after_20_sessions / nullif(next_hfq_open, 0) - 1 AS fwd20,
               open_after_60_sessions / nullif(next_hfq_open, 0) - 1 AS fwd60,
               next_hfq_open / nullif(hfq_close, 0) - 1 AS executable_gap
        FROM read_parquet('{features}') f
        LEFT JOIN read_parquet('{clusters}') c USING (thscode, date)
        WHERE date BETWEEN DATE '{start}' AND DATE '{end}' AND {BASE_ELIGIBILITY}
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE daily_baseline AS
        SELECT date, avg(fwd5) AS baseline5, avg(fwd10) AS baseline10,
               avg(fwd20) AS baseline20, avg(fwd60) AS baseline60
        FROM eligible GROUP BY date
        """
    )
    return con.execute("SELECT count(*) FROM eligible").fetchone()[0]


def case_bin(factor: str, thresholds: list[float]) -> str:
    parts = [f"WHEN {factor} <= {value:.17g} THEN {index}" for index, value in enumerate(thresholds)]
    return "CASE " + " ".join(parts) + " ELSE 9 END"


def factor_diagnostics(con: duckdb.DuckDBPyConnection, period: str, output: Path) -> None:
    threshold_path = RESULTS / "development" / "factor_thresholds.json"
    if period == "development":
        all_thresholds = {}
    else:
        all_thresholds = json.loads(threshold_path.read_text())
    decile_frames = []
    yearly_frames = []
    summaries = []
    for factor, family in FACTOR_FAMILIES.items():
        if period == "development":
            values = con.execute(
                f"SELECT quantile_cont({factor}, 0.1), quantile_cont({factor}, 0.2), quantile_cont({factor}, 0.3), "
                f"quantile_cont({factor}, 0.4), quantile_cont({factor}, 0.5), quantile_cont({factor}, 0.6), "
                f"quantile_cont({factor}, 0.7), quantile_cont({factor}, 0.8), quantile_cont({factor}, 0.9) "
                f"FROM eligible WHERE {factor} IS NOT NULL"
            ).fetchone()
            thresholds = [float(value) for value in values]
            all_thresholds[factor] = thresholds
        else:
            thresholds = all_thresholds[factor]
        bin_expression = case_bin(factor, thresholds)
        con.execute(
            f"""
            CREATE TEMP TABLE factor_binned AS
            SELECT e.date, e.fwd5, e.fwd10, e.fwd20, e.fwd60,
                   b.baseline5, b.baseline10, b.baseline20, b.baseline60,
                   {bin_expression} AS decile
            FROM eligible e JOIN daily_baseline b USING (date)
            WHERE {factor} IS NOT NULL AND fwd20 IS NOT NULL
            """
        )
        deciles = con.execute(
            """
            SELECT decile, count(*) AS observations,
                   avg(fwd5) AS mean_fwd5, avg(fwd10) AS mean_fwd10,
                   avg(fwd20) AS mean_fwd20, avg(fwd60) AS mean_fwd60,
                   median(fwd20) AS median_fwd20,
                   avg((fwd20 > 0)::INTEGER) AS win_rate20,
                   avg(fwd20 - baseline20) AS excess20,
                   avg(fwd20 - baseline20) - 0.006 AS net60bp_excess20
            FROM factor_binned GROUP BY decile ORDER BY decile
            """
        ).fetchdf()
        deciles.insert(0, "factor", factor)
        deciles.insert(0, "period", period)
        decile_frames.append(deciles)
        yearly = con.execute(
            """
            SELECT year(date) AS year, decile, count(*) AS observations,
                   avg(fwd20) AS mean_fwd20, avg(fwd20 - baseline20) AS excess20
            FROM factor_binned GROUP BY year(date), decile ORDER BY year, decile
            """
        ).fetchdf()
        yearly.insert(0, "factor", factor)
        yearly.insert(0, "period", period)
        yearly_frames.append(yearly)
        if period == "development":
            eligible_deciles = deciles[deciles["observations"] >= 80]
            selected_bin = int(eligible_deciles.loc[eligible_deciles["excess20"].idxmax(), "decile"])
        else:
            previous = pd.read_csv(RESULTS / "development" / "factor_summary.csv")
            selected_bin = int(previous.loc[previous["factor"] == factor, "selected_bin"].iloc[0])
        selected = deciles[deciles["decile"] == selected_bin].iloc[0]
        selected_yearly = yearly[yearly["decile"] == selected_bin]
        low = deciles[deciles["decile"] == 0]
        high = deciles[deciles["decile"] == 9]
        high_low = float(high["excess20"].iloc[0] - low["excess20"].iloc[0]) if len(high) and len(low) else np.nan
        monotonicity = deciles["decile"].corr(deciles["excess20"], method="spearman")
        summaries.append(
            {
                "period": period,
                "family": family,
                "factor": factor,
                "selected_bin": selected_bin,
                "selected_bin_observations": int(selected["observations"]),
                "selected_bin_fwd20": selected["mean_fwd20"],
                "selected_bin_excess20": selected["excess20"],
                "selected_bin_net60bp_excess20": selected["net60bp_excess20"],
                "high_low_excess20": high_low,
                "monotonicity": monotonicity,
                "selected_positive_excess_years": int((selected_yearly["excess20"] > 0).sum()),
                "selected_years": int(selected_yearly["year"].nunique()),
                "worst_year_excess20": selected_yearly["excess20"].min(),
            }
        )
        con.execute("DROP TABLE factor_binned")
    decile_result = pd.concat(decile_frames, ignore_index=True)
    yearly_result = pd.concat(yearly_frames, ignore_index=True)
    summary_result = pd.DataFrame(summaries)
    if period == "development":
        summary_result = summary_result.sort_values("selected_bin_net60bp_excess20", ascending=False)
        write_json(threshold_path, all_thresholds)
    decile_result.to_csv(output / "factor_deciles.csv", index=False)
    yearly_result.to_csv(output / "factor_yearly.csv", index=False)
    summary_result.to_csv(output / "factor_summary.csv", index=False)


def hypothesis_diagnostics(con: duckdb.DuckDBPyConnection, period: str, output: Path) -> None:
    config = json.loads((ROOT / "config" / "hypotheses.json").read_text())
    total_sessions = con.execute("SELECT count(DISTINCT date) FROM eligible").fetchone()[0]
    summaries = []
    yearly_frames = []
    regime_frames = []
    for hypothesis in config["hypotheses"]:
        identifier = hypothesis["id"]
        con.execute(
            f"""
            CREATE TEMP TABLE hypothesis_selected AS
            SELECT e.date, e.thscode, e.fwd5, e.fwd10, e.fwd20, e.fwd60, e.executable_gap,
                   e.breadth20, e.market_return20, e.market_close_ma20, e.market_volatility20,
                   b.baseline5, b.baseline10, b.baseline20, b.baseline60
            FROM eligible e JOIN daily_baseline b USING (date)
            WHERE fwd20 IS NOT NULL AND ({hypothesis['entry']})
            """
        )
        summary = con.execute(
            """
            SELECT count(*) AS observations, count(DISTINCT thscode) AS symbols,
                   count(DISTINCT date) AS active_days,
                   avg(fwd5) AS mean_fwd5, avg(fwd10) AS mean_fwd10,
                   avg(fwd20) AS mean_fwd20, avg(fwd60) AS mean_fwd60,
                   median(fwd20) AS median_fwd20,
                   avg((fwd20 > 0)::INTEGER) AS win_rate20,
                   avg(fwd20 - baseline20) AS excess20,
                   avg(fwd20 - baseline20) - 0.006 AS net60bp_excess20,
                   avg(executable_gap) AS mean_executable_gap,
                   stddev_samp(fwd20) AS std_fwd20
            FROM hypothesis_selected
            """
        ).fetchdf().iloc[0].to_dict()
        summary.update(
            {
                "period": period,
                "id": identifier,
                "family": hypothesis["family"],
                "active_day_share": summary["active_days"] / total_sessions if total_sessions else 0,
                "signals_per_session": summary["observations"] / total_sessions if total_sessions else 0,
            }
        )
        summaries.append(summary)
        yearly = con.execute(
            """
            SELECT year(date) AS year, count(*) AS observations,
                   avg(fwd20) AS mean_fwd20, avg(fwd20 - baseline20) AS excess20,
                   avg((fwd20 > 0)::INTEGER) AS win_rate20
            FROM hypothesis_selected GROUP BY year(date) ORDER BY year
            """
        ).fetchdf()
        yearly.insert(0, "id", identifier)
        yearly.insert(0, "period", period)
        yearly_frames.append(yearly)
        regimes = con.execute(
            """
            SELECT CASE
                     WHEN breadth20 <= 0.35 AND market_return20 < 0 THEN 'bear_breadth'
                     WHEN breadth20 >= 0.60 AND market_close_ma20 > 0 THEN 'bull_breadth'
                     WHEN market_volatility20 >= 0.25 THEN 'high_volatility'
                     ELSE 'mixed'
                   END AS regime,
                   count(*) AS observations, avg(fwd20) AS mean_fwd20,
                   avg(fwd20 - baseline20) AS excess20, avg((fwd20 > 0)::INTEGER) AS win_rate20
            FROM hypothesis_selected GROUP BY 1 ORDER BY 1
            """
        ).fetchdf()
        regimes.insert(0, "id", identifier)
        regimes.insert(0, "period", period)
        regime_frames.append(regimes)
        con.execute("DROP TABLE hypothesis_selected")
    summary_frame = pd.DataFrame(summaries)
    if period == "development":
        summary_frame = summary_frame.sort_values("net60bp_excess20", ascending=False)
    else:
        development_order = pd.read_csv(RESULTS / "development" / "hypothesis_summary.csv")["id"].tolist()
        summary_frame["_order"] = summary_frame["id"].map({value: index for index, value in enumerate(development_order)})
        summary_frame = summary_frame.sort_values("_order").drop(columns="_order")
    summary_frame.to_csv(output / "hypothesis_summary.csv", index=False)
    pd.concat(yearly_frames, ignore_index=True).to_csv(output / "hypothesis_yearly.csv", index=False)
    pd.concat(regime_frames, ignore_index=True).to_csv(output / "hypothesis_regimes.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=["development", "validation"], required=True)
    parser.add_argument("--kind", choices=["all", "factors", "hypotheses"], default="all")
    args = parser.parse_args()
    output = RESULTS / args.period
    output.mkdir(parents=True, exist_ok=True)
    con = connection()
    try:
        observations = prepare_eligible(con, args.period)
        if args.kind in ["all", "factors"]:
            factor_diagnostics(con, args.period, output)
        if args.kind in ["all", "hypotheses"]:
            hypothesis_diagnostics(con, args.period, output)
        write_json(
            output / f"{args.kind}_run_metadata.json",
            {
                "period": args.period,
                "kind": args.kind,
                "eligible_rows": observations,
                "bounds": period_bounds(args.period),
                "base_eligibility": BASE_ELIGIBILITY,
                "entry_execution_diagnostic": "Signal at close, next open entry; next-open limit-up observations are not executable",
                "cost_proxy": "60bp subtracted for diagnostics only; not a substitute for account backtest",
                "features_sha256": file_sha256(WORK / "metadata.json"),
                "hypotheses_sha256": file_sha256(ROOT / "config" / "hypotheses.json"),
                "holdout_used": False,
            },
        )
        print(json.dumps({"ok": True, "period": args.period, "eligible_rows": observations, "output": str(output)}))
    finally:
        con.close()


if __name__ == "__main__":
    main()
