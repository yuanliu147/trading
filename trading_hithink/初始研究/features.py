from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAPSHOT = DATA / "snapshot"
WORK = DATA / "work"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def literal(path: Path) -> str:
    return str(path).replace("'", "''")


def connection() -> duckdb.DuckDBPyConnection:
    WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit TO '16GB'")
    con.execute(f"SET temp_directory TO '{literal(WORK / 'duckdb_tmp')}'")
    return con


def core_sql() -> str:
    sql = (ROOT / "sql" / "build_core.sql").read_text()
    return sql.replace("@DAILY@", literal(SNAPSHOT / "mainboard_daily.parquet")).replace(
        "@EVENTS@", literal(SNAPSHOT / "adjustment_events.parquet")
    )


def build_core(con: duckdb.DuckDBPyConnection, explain: bool) -> None:
    duplicates = con.execute(
        f"""
        SELECT count(*) FROM (
            SELECT thscode, ex_date FROM read_parquet('{literal(SNAPSHOT / 'adjustment_events.parquet')}')
            GROUP BY thscode, ex_date HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    if duplicates:
        raise AssertionError(f"Duplicate corporate action keys: {duplicates}")
    sql = core_sql()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "build_core_executed.sql").write_text(sql)
    if explain:
        plan = con.execute("EXPLAIN " + sql).fetchdf()
        print(plan.to_string(index=False))
        return
    output = WORK / "core.parquet"
    if output.exists():
        raise FileExistsError(output)
    con.execute(f"COPY ({sql}) TO '{literal(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")


def build_oscillators(con: duckdb.DuckDBPyConnection) -> None:
    output = WORK / "oscillators.parquet"
    if output.exists():
        raise FileExistsError(output)
    source = WORK / "core.parquet"
    frame = con.execute(
        f"SELECT thscode, date, hfq_high, hfq_low, hfq_close FROM read_parquet('{literal(source)}')"
    ).fetchdf()
    frame = frame.sort_values(["thscode", "date"], kind="stable").reset_index(drop=True)
    grouped = frame.groupby("thscode", sort=False)
    close = frame["hfq_close"]
    ema12 = grouped["hfq_close"].transform(lambda x: x.ewm(span=12, adjust=False, min_periods=26).mean())
    ema26 = grouped["hfq_close"].transform(lambda x: x.ewm(span=26, adjust=False, min_periods=26).mean())
    dif = ema12 - ema26
    dea = dif.groupby(frame["thscode"], sort=False).transform(
        lambda x: x.ewm(span=9, adjust=False, min_periods=9).mean()
    )
    delta = close.groupby(frame["thscode"], sort=False).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.groupby(frame["thscode"], sort=False).transform(
        lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    average_loss = loss.groupby(frame["thscode"], sort=False).transform(
        lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    low9 = grouped["hfq_low"].transform(lambda x: x.rolling(9, min_periods=9).min())
    high9 = grouped["hfq_high"].transform(lambda x: x.rolling(9, min_periods=9).max())
    stoch = (close - low9) / (high9 - low9).replace(0, np.nan)
    k = stoch.groupby(frame["thscode"], sort=False).transform(
        lambda x: x.ewm(alpha=1 / 3, adjust=False).mean()
    )
    d = k.groupby(frame["thscode"], sort=False).transform(
        lambda x: x.ewm(alpha=1 / 3, adjust=False).mean()
    )
    previous_dif = dif.groupby(frame["thscode"], sort=False).shift(1)
    previous_dea = dea.groupby(frame["thscode"], sort=False).shift(1)
    result = frame[["thscode", "date"]].copy()
    result["macd_dif_pct"] = dif / close
    result["macd_dea_pct"] = dea / close
    result["macd_hist_pct"] = 2 * (dif - dea) / close
    result["macd_hist_delta"] = result["macd_hist_pct"].groupby(frame["thscode"], sort=False).diff()
    result["macd_golden_cross"] = ((dif > dea) & (previous_dif <= previous_dea)).astype("int8")
    result["macd_dead_cross"] = ((dif < dea) & (previous_dif >= previous_dea)).astype("int8")
    result["rsi14"] = 100 - 100 / (1 + average_gain / average_loss.replace(0, np.nan))
    result["stoch9"] = stoch
    result["kdj_k"] = k
    result["kdj_d"] = d
    result["kdj_j"] = 3 * k - 2 * d
    con.register("oscillator_frame", result)
    con.execute(f"COPY oscillator_frame TO '{literal(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.unregister("oscillator_frame")


def build_market(con: duckdb.DuckDBPyConnection) -> None:
    source = WORK / "core.parquet"
    market = con.execute(
        f"""
        SELECT date,
               count(*) AS stocks,
               count(DISTINCT thscode) AS distinct_stocks,
               avg(ret1) AS market_return,
               median(ret1) AS median_return,
               stddev_samp(ret1) AS return_dispersion,
               quantile_cont(ret1, 0.1) AS return_p10,
               quantile_cont(ret1, 0.9) AS return_p90,
               sum(amount) AS market_amount,
               avg(CASE WHEN listing_session >= 20 THEN (close_ma20 > 0)::INTEGER END) AS breadth20,
               avg(CASE WHEN listing_session >= 60 THEN (close_ma60 > 0)::INTEGER END) AS breadth60,
               avg(CASE WHEN listing_session >= 120 THEN (close_ma120 > 0)::INTEGER END) AS breadth120,
               avg(CASE WHEN ret1 IS NOT NULL THEN (ret1 > 0)::INTEGER END) AS advance_share,
               avg(CASE WHEN reference_close IS NOT NULL THEN limit_up_10::INTEGER END) AS limit_up10_share,
               avg(CASE WHEN reference_close IS NOT NULL THEN limit_down_10::INTEGER END) AS limit_down10_share,
               avg(CASE WHEN reference_close IS NOT NULL THEN broken_limit_up_10::INTEGER END) AS broken_limit_up10_share,
               avg(CASE WHEN reference_close IS NOT NULL THEN limit_up_05::INTEGER END) AS limit_up05_share,
               avg(CASE WHEN reference_close IS NOT NULL THEN limit_down_05::INTEGER END) AS limit_down05_share,
               avg(CASE WHEN listing_session >= 250 THEN (breakout250 >= 0)::INTEGER END) AS new_high250_share,
               avg(CASE WHEN listing_session >= 250 THEN (drawdown250 <= -0.5)::INTEGER END) AS half_below_high250_share
        FROM read_parquet('{literal(source)}')
        GROUP BY date
        ORDER BY date
        """
    ).fetchdf()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").reset_index(drop=True)
    returns = market["market_return"].fillna(0)
    market["equal_weight_index"] = (1 + returns).cumprod()
    index = market["equal_weight_index"]
    market["market_return5"] = index.pct_change(5, fill_method=None)
    market["market_return20"] = index.pct_change(20, fill_method=None)
    market["market_return60"] = index.pct_change(60, fill_method=None)
    market["market_close_ma20"] = index / index.rolling(20).mean() - 1
    market["market_ma20_slope5"] = index.rolling(20).mean().pct_change(5, fill_method=None)
    market["market_volatility20"] = market["market_return"].rolling(20).std() * np.sqrt(252)
    market["breadth20_change5"] = market["breadth20"].diff(5)
    market["amount_ratio20_60"] = (
        market["market_amount"].rolling(20).mean() / market["market_amount"].rolling(60).mean()
    )

    def rolling_z(column: str) -> pd.Series:
        series = market[column]
        mean = series.rolling(250, min_periods=120).mean()
        std = series.rolling(250, min_periods=120).std()
        return (series - mean) / std.replace(0, np.nan)

    market["sentiment_score"] = (
        0.35 * rolling_z("limit_up10_share")
        + 0.30 * rolling_z("market_amount")
        + 0.20 * rolling_z("new_high250_share")
        + 0.15 * rolling_z("breadth20")
    )
    market["sentiment_change5"] = market["sentiment_score"].diff(5)
    for output in [WORK / "market_daily.parquet", WORK / "market_daily.csv"]:
        if output.exists():
            raise FileExistsError(output)
    market.to_parquet(WORK / "market_daily.parquet", index=False)
    market.to_csv(WORK / "market_daily.csv", index=False)


def build_enriched(con: duckdb.DuckDBPyConnection) -> None:
    core = literal(WORK / "core.parquet")
    oscillators = literal(WORK / "oscillators.parquet")
    market = literal(WORK / "market_daily.parquet")
    output = WORK / "features.parquet"
    if output.exists():
        raise FileExistsError(output)
    sql = f"""
        WITH weekly_raw AS (
            SELECT thscode, date_trunc('week', date) AS week, max(date) AS week_end,
                   arg_max(hfq_close, date) AS week_close, max(hfq_high) AS week_high,
                   min(hfq_low) AS week_low, sum(amount) AS week_amount
            FROM read_parquet('{core}')
            GROUP BY thscode, date_trunc('week', date)
        ), weekly AS (
            SELECT *,
                   week_close / nullif(avg(week_close) OVER (PARTITION BY thscode ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 0) - 1 AS week_close_ma4,
                   week_close / nullif(lag(week_close, 4) OVER (PARTITION BY thscode ORDER BY week), 0) - 1 AS week_return4,
                   week_close / nullif(lag(week_close, 12) OVER (PARTITION BY thscode ORDER BY week), 0) - 1 AS week_return12,
                   week_close / nullif(max(week_high) OVER (PARTITION BY thscode ORDER BY week ROWS BETWEEN 12 PRECEDING AND CURRENT ROW), 0) - 1 AS week_drawdown12
            FROM weekly_raw
        ), monthly_raw AS (
            SELECT thscode, date_trunc('month', date) AS month, max(date) AS month_end,
                   arg_max(hfq_close, date) AS month_close, max(hfq_high) AS month_high,
                   min(hfq_low) AS month_low, sum(amount) AS month_amount
            FROM read_parquet('{core}')
            GROUP BY thscode, date_trunc('month', date)
        ), monthly AS (
            SELECT *,
                   month_close / nullif(avg(month_close) OVER (PARTITION BY thscode ORDER BY month ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) - 1 AS month_close_ma6,
                   month_close / nullif(lag(month_close, 3) OVER (PARTITION BY thscode ORDER BY month), 0) - 1 AS month_return3,
                   month_close / nullif(lag(month_close, 12) OVER (PARTITION BY thscode ORDER BY month), 0) - 1 AS month_return12
            FROM monthly_raw
        ), high_swings AS (
            SELECT thscode, date AS high_confirmation_date, confirmed_swing_high
            FROM read_parquet('{core}') WHERE swing_high_confirmed
        ), low_swings AS (
            SELECT thscode, date AS low_confirmation_date, confirmed_swing_low
            FROM read_parquet('{core}') WHERE swing_low_confirmed
        )
        SELECT c.*, o.* EXCLUDE (thscode, date), m.* EXCLUDE (date),
               w.week_close_ma4, w.week_return4, w.week_return12, w.week_drawdown12,
               mo.month_close_ma6, mo.month_return3, mo.month_return12,
               hs.high_confirmation_date, hs.confirmed_swing_high,
               ls.low_confirmation_date, ls.confirmed_swing_low,
               c.hfq_close / nullif(hs.confirmed_swing_high, 0) - 1 AS distance_swing_high,
               c.hfq_close / nullif(ls.confirmed_swing_low, 0) - 1 AS distance_swing_low,
               date_diff('day', hs.high_confirmation_date, c.date) AS days_since_swing_high,
               date_diff('day', ls.low_confirmation_date, c.date) AS days_since_swing_low
        FROM read_parquet('{core}') c
        JOIN read_parquet('{oscillators}') o USING (thscode, date)
        JOIN read_parquet('{market}') m USING (date)
        ASOF LEFT JOIN weekly w ON c.thscode = w.thscode AND w.week_end < date_trunc('week', c.date)
        ASOF LEFT JOIN monthly mo ON c.thscode = mo.thscode AND mo.month_end < date_trunc('month', c.date)
        ASOF LEFT JOIN high_swings hs ON c.thscode = hs.thscode AND hs.high_confirmation_date <= c.date
        ASOF LEFT JOIN low_swings ls ON c.thscode = ls.thscode AND ls.low_confirmation_date <= c.date
    """
    (WORK / "build_features_executed.sql").write_text(sql)
    con.execute(f"COPY ({sql}) TO '{literal(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")


def write_metadata(con: duckdb.DuckDBPyConnection) -> None:
    market = pd.read_parquet(WORK / "market_daily.parquet", columns=["date"])
    dates = pd.DatetimeIndex(market["date"])
    config = json.loads((ROOT / "config" / "research.json").read_text())
    boundaries = {}
    for name in ["development", "validation", "holdout"]:
        start = pd.Timestamp(config["sample"][f"{name}_start"])
        end = pd.Timestamp(config["sample"][f"{name}_end"])
        period_dates = dates[(dates >= start) & (dates <= end)]
        first_index = dates.get_loc(period_dates[0])
        last_index = dates.get_loc(period_dates[-1])
        label_index = max(first_index - 1, last_index - config["sample"]["embargo_sessions"] - 1)
        boundaries[name] = {
            "first_session": period_dates[0],
            "last_session": period_dates[-1],
            "last_uncontaminated_signal_for_60d": dates[label_index],
        }
    write_json(WORK / "sample_boundaries.json", boundaries)
    audit = con.execute(
        f"""
        WITH x AS (
            SELECT *, lag(backward_factor) OVER (PARTITION BY thscode ORDER BY date) AS previous_factor
            FROM read_parquet('{literal(WORK / 'core.parquet')}')
        )
        SELECT count(*) FILTER (WHERE dividend_per_share <> 0 OR per_share_bonus <> 0 OR rights_ratio <> 0) AS corporate_action_rows,
               count(*) FILTER (WHERE factor_only_adjustment) AS factor_only_adjustments,
               median(abs(previous_actual_close / nullif(reference_close, 0) - backward_factor / nullif(previous_factor, 0)))
                   FILTER (WHERE previous_factor IS NOT NULL) AS median_reference_factor_error,
               max(abs(previous_actual_close / nullif(reference_close, 0) - backward_factor / nullif(previous_factor, 0)))
                   FILTER (WHERE previous_factor IS NOT NULL) AS max_reference_factor_error,
               count(*) FILTER (WHERE previous_factor IS NOT NULL AND abs(backward_factor / previous_factor - 1) > 0.000001
                                AND dividend_per_share = 0 AND per_share_bonus = 0 AND rights_ratio = 0) AS unexplained_factor_jumps
        FROM x
        """
    ).fetchdf().iloc[0].to_dict()
    write_json(WORK / "reference_price_audit.json", audit)
    files = {}
    for name in ["core", "oscillators", "market_daily", "features"]:
        path = WORK / f"{name}.parquet"
        row_count = con.execute(f"SELECT count(*) FROM read_parquet('{literal(path)}')").fetchone()[0]
        files[name] = {"path": str(path.relative_to(ROOT)), "rows": row_count, "bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        WORK / "metadata.json",
        {
            "source_snapshot": "data/snapshot/manifest.json",
            "build_core_sql_sha256": sha256(ROOT / "sql" / "build_core.sql"),
            "features_py_sha256": sha256(Path(__file__)),
            "remote_data_used": False,
            "files": files,
            "reference_price_audit": audit,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--derived-only", action="store_true")
    args = parser.parse_args()
    con = connection()
    try:
        if not args.derived_only:
            build_core(con, args.explain)
            if args.explain:
                return
            build_oscillators(con)
        elif args.explain:
            raise ValueError("--explain and --derived-only cannot be combined")
        build_market(con)
        build_enriched(con)
        write_metadata(con)
        print(json.dumps({"ok": True, "features": str(WORK / 'features.parquet')}))
    finally:
        con.close()


if __name__ == "__main__":
    main()
