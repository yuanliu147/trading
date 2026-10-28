from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parent
REPAIR = HERE.parents[1]
sys.path.insert(0, str(REPAIR))
from flash_data import INITIAL, PANEL, literal, sha256  # noqa: E402

MOTHER = PANEL.parents[1]
RESULTS = OUT / "results"
CONFIG = json.loads((OUT / "analysis_config.json").read_text())
DATABASE = Path(CONFIG["data"]["database"])
TODAY = pd.Timestamp(CONFIG["data"]["cutoff"])
FIRST = pd.Timestamp(CONFIG["data"]["first_signal_date"])
SECOND = pd.Timestamp(CONFIG["data"]["second_signal_date"])
PREVIOUS = REPAIR / "603778_信号对比"
STRATEGY = json.loads((REPAIR / "config/strategy.json").read_text())
FIXED = json.loads((PREVIOUS / "analysis_config.json").read_text())
SIGNAL_MODULE = MOTHER / "02_代码/今日信号_数据库.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def save_json(name: str, value: Any) -> None:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (pd.Timestamp, Path)):
            return str(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.bool_):
            return bool(item)
        return item

    (RESULTS / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_quality() -> tuple[dict[str, Any], pd.DataFrame]:
    con = duckdb.connect(str(DATABASE), read_only=True)
    latest = pd.Timestamp(con.execute("SELECT max(date) FROM v_daily").fetchone()[0])
    if latest != TODAY:
        raise AssertionError(f"Expected {TODAY.date()}, found {latest.date()}")
    batch_columns = [row[1] for row in con.execute("PRAGMA table_info('_import_batches')").fetchall()]
    batch = con.execute("SELECT * FROM _import_batches ORDER BY completed_at DESC,started_at DESC LIMIT 1").fetchdf()
    if batch.iloc[0]["status"] != "complete":
        raise AssertionError("Latest daily import is not complete")
    daily = con.execute(
        """SELECT d.date,count(*) AS row_count,count(DISTINCT d.thscode) AS symbols,
                  count(*) FILTER(WHERE h.date IS NULL) missing_hfq,
                  count(*) FILTER(WHERE d.open IS NULL OR d.high IS NULL OR d.low IS NULL OR d.close IS NULL
                                      OR d.open<=0 OR d.high<=0 OR d.low<=0 OR d.close<=0) invalid_ohlc,
                  count(*) FILTER(WHERE d.volume=0 OR d.amount=0) zero_activity,
                  count(*) FILTER(WHERE d.high<d.low OR d.high<d.open OR d.high<d.close
                                      OR d.low>d.open OR d.low>d.close) inconsistent_ohlc,
                  count(*) FILTER(WHERE h.open IS NULL OR h.high IS NULL OR h.low IS NULL OR h.close IS NULL) null_hfq,
                  count(*) FILTER(WHERE abs(h.close/d.close-h.open/d.open)>1e-8
                                      OR abs(h.close/d.close-h.high/d.high)>1e-8
                                      OR abs(h.close/d.close-h.low/d.low)>1e-8) inconsistent_hfq
           FROM v_daily d LEFT JOIN v_daily_hfq h USING(thscode,date)
           WHERE d.date>=DATE '2026-10-23' GROUP BY d.date ORDER BY d.date"""
    ).fetchdf()
    latest_quality = daily[daily["date"] == TODAY].iloc[0]
    for field in ["missing_hfq", "invalid_ohlc", "zero_activity", "inconsistent_ohlc", "null_hfq", "inconsistent_hfq"]:
        if int(latest_quality[field]) != 0:
            raise AssertionError(f"Latest quality failure: {field}")
    duplicates = con.execute(
        "SELECT count(*) FROM (SELECT thscode,date,count(*) n FROM v_daily WHERE date=? GROUP BY thscode,date HAVING n>1)", [TODAY]
    ).fetchone()[0]
    if duplicates:
        raise AssertionError("Duplicate current bars")
    meta = dict(con.execute("SELECT key,value FROM _meta").fetchall())
    missing_today = con.execute(
        "SELECT count(*) FROM (SELECT thscode,max(date) last_date FROM v_daily GROUP BY thscode) WHERE last_date<?", [TODAY]
    ).fetchone()[0]
    result = {
        "database": DATABASE,
        "database_sha256": sha256(DATABASE),
        "cutoff": TODAY,
        "latest_import": batch.iloc[0].to_dict(),
        "latest_import_columns": batch_columns,
        "metadata": meta,
        "duplicates": duplicates,
        "current_universe_symbols_without_bar_today": missing_today,
        "missing_today_note": "A symbol without a current bar is not automatically a corrupt row; suspension or listing status is not reconstructed here.",
        "remote_or_third_party_data_used": False,
    }
    con.close()
    return result, daily


def stock_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    result = []
    for symbol, group in raw.groupby("thscode", sort=False):
        g = group.sort_values("date").copy().reset_index(drop=True)
        close = g["hfq_close"]
        previous = close.shift(1)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        ema12 = close.ewm(span=12, adjust=False, min_periods=26).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
        g["ret1"] = close.pct_change(fill_method=None)
        g["return_20d"] = close.pct_change(20, fill_method=None)
        g["rsi14"] = 100 - 100 / (1 + safe_div(gain, loss))
        g["macd_hist_pct"] = 2 * (dif - dea) / close
        g["macd_hist_delta"] = g["macd_hist_pct"].diff()
        g["close_location"] = safe_div(g["close"] - g["low"], g["high"] - g["low"])
        g["drawdown250"] = close / g["hfq_high"].rolling(250, min_periods=250).max() - 1
        g["return_250d"] = close / close.shift(250) - 1
        g["amount20"] = g["amount"].rolling(20, min_periods=20).mean()
        g["amount_ratio1_20"] = g["amount"] / g["amount20"]
        g["amount_ratio5_20"] = g["amount"].rolling(5, min_periods=5).mean() / g["amount20"]
        for window in [5, 10, 20, 60, 250]:
            g[f"ma{window}_actual"] = close.rolling(window, min_periods=window).mean() / g["backward_factor"]
            g[f"close_ma{window}"] = close / close.rolling(window, min_periods=window).mean() - 1
        g["ma20_slope5"] = g["close_ma20"] * np.nan
        ma20 = close.rolling(20, min_periods=20).mean()
        g["ma20_slope5"] = ma20 / ma20.shift(5) - 1
        true_range = pd.concat([g["hfq_high"] - g["hfq_low"], (g["hfq_high"] - previous).abs(), (g["hfq_low"] - previous).abs()], axis=1).max(axis=1)
        g["atr20_pct"] = (true_range / previous).rolling(20, min_periods=20).mean()
        g["range_pct"] = (g["hfq_high"] - g["hfq_low"]) / previous
        g["candle_body"] = (close - g["hfq_open"]) / previous
        g["upper_shadow"] = (g["hfq_high"] - g[["hfq_open", "hfq_close"]].max(axis=1)) / previous
        g["lower_shadow"] = (g[["hfq_open", "hfq_close"]].min(axis=1) - g["hfq_low"]) / previous
        g["BOR"] = (g["hfq_low"] < g["hfq_low"].shift(1)) & (close > previous) & (close > g["hfq_open"])
        g["strong_reversal"] = (g["drawdown250"] <= FIXED["fixed_conditions"]["deep_drawdown250_lte"]) & g["BOR"] & (g["close_location"] > FIXED["fixed_conditions"]["high_close_location_gt"])
        completed = g[g["date"] <= FIRST]
        weekly = completed.set_index("date")["hfq_close"].resample("W-FRI").last().dropna()
        last_row = g.index[-1]
        g.loc[last_row, "completed_week_return4"] = weekly.iloc[-1] / weekly.iloc[-5] - 1 if len(weekly) >= 5 else np.nan
        g.loc[last_row, "completed_week_return12"] = weekly.iloc[-1] / weekly.iloc[-13] - 1 if len(weekly) >= 13 else np.nan
        g.loc[last_row, "completed_week_close_drawdown12"] = weekly.iloc[-1] / weekly.iloc[-12:].max() - 1 if len(weekly) >= 12 else np.nan
        g.loc[last_row, "week_to_date_return"] = close.iloc[-1] / completed["hfq_close"].iloc[-1] - 1
        previous_month = g[g["date"] < TODAY.replace(day=1)]
        g.loc[last_row, "month_to_date_return"] = close.iloc[-1] / previous_month["hfq_close"].iloc[-1] - 1 if not previous_month.empty else np.nan
        result.append(g)
    return pd.concat(result, ignore_index=True)


def load_stock_bars(symbols: list[str]) -> pd.DataFrame:
    con = duckdb.connect(str(DATABASE), read_only=True)
    con.register("requested_symbols", pd.DataFrame({"thscode": symbols}))
    raw = con.execute(
        """SELECT d.thscode,d.date,d.open,d.high,d.low,d.close,d.volume,d.amount,
                  h.open hfq_open,h.high hfq_high,h.low hfq_low,h.close hfq_close,f.backward_factor
           FROM v_daily d JOIN v_daily_hfq h USING(thscode,date)
           JOIN calc_adjust_factor_daily f USING(thscode,date)
           JOIN requested_symbols s ON d.thscode=s.thscode ORDER BY d.thscode,d.date"""
    ).fetchdf()
    con.close()
    raw["date"] = pd.to_datetime(raw["date"])
    return raw


def previous_cohorts() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(PREVIOUS / "results/current_strong_cohort_followthrough.csv")
    outcomes = pd.read_csv(PREVIOUS / "results/signal_outcomes_audited.csv", parse_dates=["date"], low_memory=False)
    first = outcomes[outcomes["strong_reversal"] & outcomes["date"].eq(FIRST)].copy()
    if len(first) != 10 or set(first["thscode"]) != set(old["thscode"]):
        raise AssertionError("First-signal cohort differs from the frozen research")
    old_summary = json.loads((PREVIOUS / "results/analysis_summary.json").read_text())
    for name, path in [("panel", PANEL), ("core", INITIAL / "data/work/core.parquet"), ("events", INITIAL / "data/snapshot/adjustment_events.parquet")]:
        if sha256(path) != old_summary["input_hashes"][name]:
            raise AssertionError(f"Frozen input changed: {name}")
    second = pd.read_csv(MOTHER / "03_结果/当前信号_2026-10-26.csv", parse_dates=["date"])
    return first.sort_values("rank"), second.sort_values("rank")


def check_frozen_bars(raw: pd.DataFrame) -> int:
    frozen = pd.read_csv(PREVIOUS / "results/current_strong_cohort_followthrough.csv")
    comparisons = 0
    for row in frozen.itertuples():
        for day in [23, 26]:
            date = pd.Timestamp(f"2026-10-{day}")
            actual = raw[raw["thscode"].eq(row.thscode) & raw["date"].eq(date)]
            if len(actual) != 1:
                raise AssertionError(f"Missing frozen bar: {row.thscode} {date.date()}")
            actual = actual.iloc[0]
            for field in ["open", "high", "low", "close"]:
                for prefix, column in [("actual", f"actual_{field}_raw_{day}"), ("hfq", f"{field}_hfq_{day}")]:
                    expected = getattr(row, column)
                    source_field = f"hfq_{field}" if prefix == "hfq" else field
                    if not np.isclose(actual[source_field], expected, rtol=0, atol=1e-10):
                        raise AssertionError(f"Revised frozen bar: {row.thscode} {date.date()} {column}")
                    comparisons += 1
    return comparisons


def market_snapshots(wide: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market = market.sort_index().copy()
    index = (1 + market["market_return"].fillna(0)).cumprod()
    for window in [5, 20, 60]:
        market[f"market_return{window}"] = index.pct_change(window, fill_method=None)
    market["close_ma20"] = index / index.rolling(20).mean() - 1
    market["ma20_slope5"] = market["close_ma20"] * np.nan
    market["ma20_slope5"] = index.rolling(20).mean() / index.rolling(20).mean().shift(5) - 1
    for date in [FIRST, SECOND, TODAY]:
        subset = wide[wide["date"] == date]
        down_limit = np.round(subset["ref_close"] * .9, 2)
        market.loc[date, "advance_share"] = (subset["ret1"] > 0).mean()
        market.loc[date, "limit_down10_share"] = (subset["actual_close"] <= down_limit + 1e-4).mean()
        market.loc[date, "limit_up10_share"] = subset["limit_up_10"].mean()
    output = market.loc[[FIRST, SECOND, TODAY]].reset_index(names="date")
    output["market_proxy"] = "mainboard_prefix_equal_weight_not_official_index"
    previous = pd.read_csv(PREVIOUS / "results/current_market.csv", parse_dates=["date"]).set_index("date")
    for date in [FIRST, SECOND]:
        for field in ["market_return", "market_return20", "breadth20", "close_ma20", "ma20_slope5", "advance_share", "limit_down10_share", "limit_up10_share"]:
            if not np.isclose(output.set_index("date").loc[date, field], previous.loc[date, field], rtol=0, atol=1e-10):
                raise AssertionError(f"Frozen market mismatch: {date.date()} {field}")
    return output


def original_signals(wide: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    signal = STRATEGY["signal"]
    eligible = wide[wide["date"].eq(TODAY) & wide["listing_session"].ge(250) & wide["amount20"].ge(signal["minimum_amount20"])].copy()
    eligible["market_return20"] = market.loc[TODAY, "market_return20"]
    eligible["breadth20"] = market.loc[TODAY, "breadth20"]
    eligible["cond_dd"] = eligible["drawdown250"] < signal["drawdown250_lt"]
    eligible["cond_r250"] = eligible["return_250d"] > signal["return_250d_gt"]
    eligible["cond_rsi"] = eligible["rsi14"] < signal["rsi14_lt"]
    eligible["cond_cl"] = eligible["close_location"] > signal["close_location_gt"]
    gate = bool(market.loc[TODAY, "market_return20"] < signal["market_return20_lt"])
    eligible["cond_all"] = eligible[["cond_dd", "cond_r250", "cond_rsi", "cond_cl"]].all(axis=1) & gate
    signals = eligible[eligible["cond_all"]].sort_values(["drawdown250", "thscode"]).copy()
    signals["rank"] = np.arange(1, len(signals) + 1)
    return eligible, signals, gate


def peer_snapshots() -> pd.DataFrame:
    symbols = CONFIG["data"]["comparison_symbols"]
    membership_path = INITIAL / "data/work/cluster_membership.parquet"
    targets = CONFIG["data"]["comparison_symbols"]
    placeholders = ",".join(f"'{symbol}'" for symbol in targets)
    con = duckdb.connect(str(DATABASE), read_only=True)
    query = f"""
        WITH targets AS (
            SELECT thscode peer_target,cluster_id FROM read_parquet('{literal(membership_path)}')
            WHERE effective_month=DATE '2026-10-01' AND thscode IN ({placeholders}) AND cluster_id>=0
        ), peers AS (
            SELECT t.peer_target,m.* FROM read_parquet('{literal(membership_path)}') m
            JOIN targets t ON m.cluster_id=t.cluster_id
            WHERE m.effective_month=DATE '2026-10-01' AND m.cluster_id>=0
        ), base AS (
            SELECT p.peer_target,p.thscode,h.date,h.close,
                   h.close/lag(h.close) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date)-1 ret1,
                   h.close/lag(h.close,5) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date)-1 ret5,
                   h.close/lag(h.close,20) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date)-1 ret20,
                   h.close/lag(h.close,60) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date)-1 ret60,
                   h.close/avg(h.close) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)-1 ma20d,
                   h.close/avg(h.close) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)-1 ma60d
            FROM peers p JOIN v_daily_hfq h ON p.thscode=h.thscode WHERE h.date<=?
        ) SELECT peer_target AS "target",date,count(*) AS members,avg(ret1) peer_return1,avg(ret5) peer_return5,
                 avg(ret20) peer_return20,avg(ret60) peer_return60,avg((ma20d>0)::INTEGER) peer_breadth20,
                 avg((ma60d>0)::INTEGER) peer_breadth60
          FROM base WHERE date=? AND thscode<>peer_target GROUP BY peer_target,date
    """
    peers = con.execute(query, [TODAY, TODAY]).fetchdf()
    con.close()
    membership = pd.read_parquet(membership_path, filters=[("effective_month", "=", pd.Timestamp("2026-10-01")), ("thscode", "in", symbols)])
    membership["target"] = membership["thscode"]
    output = pd.DataFrame({"target": symbols}).merge(membership[["target", "cluster_id", "cluster_strength", "cluster_size", "cluster_train_end"]], on="target", how="left")
    output["date"] = TODAY
    output = output.merge(peers, on=["target", "date"], how="left")
    output["official_industry_index"] = False
    output["note"] = np.where(output["cluster_id"].fillna(-1) < 0, "no_lagged_PCA_peer_group", "lagged_PCA_peer_group_not_official_industry")
    return output


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    initial_hash = sha256(DATABASE)
    if initial_hash != CONFIG["data"]["raw_database_sha256_at_discovery"]:
        raise AssertionError("Database changed after discovery; create a new snapshot instead of mixing cutoffs")
    quality, daily_quality = source_quality()
    daily_quality.to_csv(RESULTS / "source_daily_quality.csv", index=False)
    first, second = previous_cohorts()
    source = load_module(SIGNAL_MODULE, "local_original_signal_2026_10_27")
    wide, market = source.features(source.load(CONFIG["data"]["original_feature_start"]))
    market_output = market_snapshots(wide, market)
    eligible, signals, gate = original_signals(wide, market)
    market_output.to_csv(RESULTS / "market_snapshots.csv", index=False, encoding="utf-8-sig")
    symbols = sorted(set(first["thscode"]) | set(second.loc[second["rank"] <= 10, "thscode"]) | set(signals["thscode"]) | set(CONFIG["data"]["comparison_symbols"]))
    raw = load_stock_bars(symbols)
    frozen_comparisons = check_frozen_bars(raw)
    indicators = stock_indicators(raw)
    current = indicators[indicators["date"].eq(TODAY)].copy()
    if len(current[current["thscode"].eq("603778.SH")]) != 1:
        raise AssertionError("603778 current bar missing")
    rank_today = signals.set_index("thscode")["rank"]
    current["original_rank_today"] = current["thscode"].map(rank_today)
    current["in_original_signal_today"] = current["original_rank_today"].notna()
    current["in_execution_top10_today"] = current["original_rank_today"].le(10)
    current["first_23_strong_rank"] = current["thscode"].map(first.set_index("thscode")["rank"])
    current["second_26_original_rank"] = current["thscode"].map(second.set_index("thscode")["rank"])
    current["market_return20"] = market_output.set_index("date").loc[TODAY, "market_return20"]
    current["breadth20"] = market_output.set_index("date").loc[TODAY, "breadth20"]
    current["completed_week_through"] = FIRST
    current["week_to_date_complete"] = False
    current["month_to_date_complete"] = False
    current["higher_timeframe_note"] = "Week-to-date and month-to-date are unfinished periods; weekly references use completed data through 2026-10-23."
    current.to_csv(RESULTS / "current_stock_features.csv", index=False, encoding="utf-8-sig")

    signal_columns = ["date", "rank", "thscode", "actual_close", "ref_close", "limit_up_05", "limit_up_10", "return_250d", "drawdown250", "rsi14", "close_location", "amount20", "market_return20", "breadth20"]
    current_signals = signals[signal_columns].merge(
        current[["thscode", "open", "high", "low", "close", "amount", "amount_ratio1_20", "ret1", "return_20d", "BOR", "strong_reversal", "macd_hist_pct", "macd_hist_delta", "ma5_actual", "ma10_actual", "ma20_actual", "close_ma20", "upper_shadow"]],
        on="thscode", how="left", validate="one_to_one"
    )
    current_signals["in_execution_top10"] = current_signals["rank"] <= 10
    current_signals["future_trade_outcome_known"] = False
    current_signals.to_csv(RESULTS / "current_original_signals.csv", index=False, encoding="utf-8-sig")

    latest = current.set_index("thscode")
    follow_rows = []
    for row in first.itertuples():
        base23 = raw[raw["thscode"].eq(row.thscode) & raw["date"].eq(FIRST)].iloc[0]
        base26 = raw[raw["thscode"].eq(row.thscode) & raw["date"].eq(SECOND)].iloc[0]
        record = {"thscode": row.thscode, "first_signal_date": FIRST, "first_rank": row.rank,
                  "first_open": base23["open"], "first_high": base23["high"], "first_low": base23["low"], "first_close": base23["close"],
                  "first_hfq_open": base23["hfq_open"], "first_hfq_high": base23["hfq_high"], "first_hfq_low": base23["hfq_low"], "first_hfq_close": base23["hfq_close"],
                  "second_low": base26["low"], "second_high": base26["high"], "second_close": base26["close"]}
        if row.thscode in latest.index:
            bar = latest.loc[row.thscode]
            observed = True
            for field in ["date", "open", "high", "low", "close", "hfq_open", "hfq_high", "hfq_low", "hfq_close", "volume", "amount", "ret1", "rsi14", "macd_hist_pct", "macd_hist_delta", "close_location", "amount_ratio1_20", "strong_reversal", "candle_body"]:
                record[field] = bar[field]
            record["return_since_first_close"] = bar["hfq_close"] / base23["hfq_close"] - 1
            record["higher_low_than_first"] = bar["hfq_low"] > base23["hfq_low"]
            record["below_first_low"] = bar["hfq_low"] < base23["hfq_low"]
            record["above_first_open"] = bar["hfq_close"] > base23["hfq_open"]
            record["close_above_first_high"] = bar["hfq_close"] > base23["hfq_high"]
            record["high_above_first_high"] = bar["hfq_high"] > base23["hfq_high"]
            record["close_above_second_high"] = bar["hfq_close"] > base26["hfq_high"]
            record["close_below_second_low"] = bar["hfq_close"] < base26["hfq_low"]
            record["macd_hist_positive"] = bar["macd_hist_pct"] > 0
            record["consolidation_27"] = bool(record["higher_low_than_first"] and record["above_first_open"] and abs((bar["hfq_close"] - bar["hfq_open"]) / bar["hfq_open"]) <= .02 and bar["amount_ratio1_20"] < 1)
            record["original_rank_today"] = rank_today.get(row.thscode, np.nan)
        else:
            observed = False
            record.update({"return_since_first_close": np.nan, "original_rank_today": np.nan})
        record["bar27_observed"] = observed
        record["completed_trade_outcome_known"] = False
        follow_rows.append(record)
    follow = pd.DataFrame(follow_rows).sort_values("first_rank")
    follow.to_csv(RESULTS / "cohort_23_followthrough.csv", index=False, encoding="utf-8-sig")

    second_follow = second.merge(current, on="thscode", how="left", suffixes=("_signal", "_current"), validate="one_to_one")
    second_follow["return_since_26_close"] = second_follow["hfq_close"] / second_follow["thscode"].map(raw[raw["date"].eq(SECOND)].set_index("thscode")["hfq_close"]) - 1
    second_follow["original_rank_today"] = second_follow["thscode"].map(rank_today)
    second_follow["future_trade_outcome_known"] = False
    second_follow.to_csv(RESULTS / "cohort_26_followthrough.csv", index=False, encoding="utf-8-sig")
    peers = peer_snapshots()
    peers.to_csv(RESULTS / "current_peer_proxy.csv", index=False, encoding="utf-8-sig")
    if sha256(DATABASE) != initial_hash:
        raise AssertionError("Database changed while the analysis was running")
    quality["database_hash_unchanged_during_analysis"] = True
    quality["frozen_ohlc_comparisons_passed"] = frozen_comparisons
    quality["mainboard_eligible_symbols_today"] = len(eligible)
    quality["tracked_stock_symbols"] = len(symbols)
    quality["tracked_stock_bars_today"] = len(current)
    quality["scope_limitations"] = [
        "Daily data only; no intraday queue, tick or order-book evidence.",
        "Current listed universe; delisted stocks and historical ST flags are not reconstructed.",
        "No official industry index or historical industry constituents used for 603778.",
        "Current two-day observations are not the strategy's T+3 entry or completed 15-session trade outcomes.",
    ]
    save_json("source_quality.json", quality)
    target = latest.loc["603778.SH"]
    observed_follow = follow[follow["bar27_observed"]]
    top2_23 = observed_follow[observed_follow["first_rank"] <= 2]
    second_top2 = second_follow[second_follow["rank"] <= 2]
    summary = {
        "cutoff": TODAY,
        "raw_database_sha256": initial_hash,
        "code_sha256": sha256(HERE),
        "config_sha256": sha256(OUT / "analysis_config.json"),
        "original_signal_module_sha256": sha256(SIGNAL_MODULE),
        "original_signal_module": SIGNAL_MODULE,
        "market_gate_open": gate,
        "original_signal_count": len(signals),
        "original_top10_count": int((signals["rank"] <= 10).sum()),
        "strong_reversal_count_within_original_signals": int(current_signals["strong_reversal"].sum()),
        "target": {"thscode": target.name, **{field: target[field] for field in ["date", "open", "high", "low", "close", "volume", "amount", "ret1", "rsi14", "macd_hist_pct", "macd_hist_delta", "close_location", "amount_ratio1_20", "drawdown250", "return_250d", "return_20d", "ma5_actual", "ma10_actual", "ma20_actual", "ma60_actual", "close_ma20", "completed_week_return4", "completed_week_return12", "week_to_date_return", "month_to_date_return", "BOR", "strong_reversal", "in_original_signal_today", "original_rank_today"]}},
        "target_T2_structure": {field: follow.loc[follow["thscode"].eq("603778.SH"), field].iloc[0] for field in ["return_since_first_close", "high_above_first_high", "close_above_first_high", "below_first_low", "above_first_open", "close_below_second_low", "consolidation_27", "macd_hist_positive"]},
        "first_23_strong_cohort": {
            "signals": len(follow), "observed": len(observed_follow),
            "positive_since_23_close": int(observed_follow["return_since_first_close"].gt(0).sum()),
            "mean_since_23_close": observed_follow["return_since_first_close"].mean(),
            "median_since_23_close": observed_follow["return_since_first_close"].median(),
            "close_above_23_high": int(observed_follow["close_above_first_high"].sum()),
            "intraday_below_23_low": int(observed_follow["below_first_low"].sum()),
            "close_above_23_open": int(observed_follow["above_first_open"].sum()),
            "small_body_low_volume_consolidation_27": int(observed_follow["consolidation_27"].sum()),
            "still_original_signal_today": int(observed_follow["original_rank_today"].notna().sum()),
            "top2_mean_since_23_close": top2_23["return_since_first_close"].mean(),
        },
        "second_26_top2_T1": {
            "rows": len(second_top2),
            "observed": int(second_top2["return_since_26_close"].notna().sum()),
            "mean_since_26_close": second_top2["return_since_26_close"].mean(),
            "note": "Only T+1, shares 603778 with the first cohort; not independent and not a completed trade result.",
        },
        "execution": CONFIG["execution_status"],
        "interpretation_boundary": "New observable evidence is not proof of future returns or an independent out-of-sample strategy validation; no individualized buy/sell instruction or position sizing is generated.",
    }
    save_json("analysis_summary.json", summary)
    columns = ["thscode", "original_rank_today", "open", "high", "low", "close", "ret1", "close_location", "rsi14", "macd_hist_pct", "macd_hist_delta", "amount_ratio1_20", "ma20_actual", "close_ma20", "BOR", "strong_reversal"]
    print(current[current["thscode"].isin(CONFIG["data"]["comparison_symbols"])][columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
    print("\nTODAY ORIGINAL TOP10\n", current_signals[current_signals["rank"] <= 10][["rank", "thscode", "close", "ret1", "drawdown250", "rsi14", "close_location", "strong_reversal"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
    print("\n23 STRONG COHORT\n", follow[["first_rank", "thscode", "close", "ret1", "return_since_first_close", "close_above_first_high", "below_first_low", "above_first_open", "consolidation_27", "original_rank_today"]].to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
    print("\nMARKET\n", market_output.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
    print("\nPEERS\n", peers.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)
    print("\n", json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
