from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parent
sys.path.insert(0, str(HERE.parents[1]))
from flash_data import CORE, EVENTS, INITIAL, PANEL, Candidate, data_from_frames, literal, load_config, sha256  # noqa: E402
from flash_engine import EngineConfig, run_backtest  # noqa: E402

RESULTS = OUT / "results"
DATABASE = Path("/Users/admin/Library/Application Support/hithink-finance/data/market.duckdb")
MOTHER = PANEL.parents[1]
CONFIG = json.loads((OUT / "analysis_config.json").read_text())
SIGNAL_CONFIG = load_config()
BAR_COLUMNS = [
    "open", "high", "low", "close", "hfq_open", "hfq_high", "hfq_low", "hfq_close",
    "volume", "reference_close", "previous_actual_close", "backward_factor",
]
EVENT_COLUMNS = ["dividend_per_share", "per_share_bonus", "rights_ratio", "rights_price"]
FEATURE_COLUMNS = [
    "thscode", "date", "close", "hfq_close", "amount", "amount20", "drawdown250", "return_250d",
    "return_60d", "return_20d", "rsi14", "close_location", "macd_hist_pct", "macd_hist_delta",
    "amount_ratio1_20", "amount_ratio5_20", "close_ma5", "close_ma10", "close_ma20", "close_ma60",
    "ma20_slope5", "atr20_pct", "range_pct", "candle_body", "upper_shadow", "lower_shadow",
    "distance_low20", "distance_low60", "price_position60", "week_return4", "week_return12",
    "week_drawdown12", "month_return3", "month_return12", "market_return", "market_return5",
    "market_return20", "market_return60", "breadth20", "advance_share", "sentiment_score",
    "sentiment_change5", "limit_down10_share",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(name: str, value: Any) -> None:
    (RESULTS / name).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def stock_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    result = []
    for symbol, group in frame.groupby("thscode", sort=False):
        g = group.sort_values("date").copy().reset_index(drop=True)
        close = g["hfq_close"]
        previous_close = close.shift(1)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        ema12 = close.ewm(span=12, adjust=False, min_periods=26).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
        g["ret1"] = close.pct_change(fill_method=None)
        g["rsi14"] = 100 - 100 / (1 + safe_div(gain, loss))
        g["macd_hist_pct"] = 2 * (dif - dea) / close
        g["macd_hist_delta"] = g["macd_hist_pct"].diff()
        g["close_location"] = safe_div(g["close"] - g["low"], g["high"] - g["low"])
        g["drawdown250"] = close / g["hfq_high"].rolling(250, min_periods=250).max() - 1
        g["return_250d"] = close / close.shift(250) - 1
        g["return_20d"] = close / close.shift(20) - 1
        g["amount20"] = g["amount"].rolling(20, min_periods=20).mean()
        g["amount_ratio1_20"] = g["amount"] / g["amount20"]
        g["amount_ratio5_20"] = g["amount"].rolling(5, min_periods=5).mean() / g["amount20"]
        for window in [5, 10, 20, 60, 250]:
            g[f"ma{window}"] = close.rolling(window, min_periods=window).mean()
            g[f"close_ma{window}"] = close / g[f"ma{window}"] - 1
        g["ma20_slope5"] = g["ma20"] / g["ma20"].shift(5) - 1
        true_range = pd.concat([g["hfq_high"] - g["hfq_low"], (g["hfq_high"] - previous_close).abs(), (g["hfq_low"] - previous_close).abs()], axis=1).max(axis=1)
        g["atr20_pct"] = (true_range / previous_close).rolling(20, min_periods=20).mean()
        g["range_pct"] = (g["hfq_high"] - g["hfq_low"]) / previous_close
        g["candle_body"] = (g["hfq_close"] - g["hfq_open"]) / previous_close
        g["upper_shadow"] = (g["hfq_high"] - g[["hfq_open", "hfq_close"]].max(axis=1)) / previous_close
        g["lower_shadow"] = (g[["hfq_open", "hfq_close"]].min(axis=1) - g["hfq_low"]) / previous_close
        g["BOR"] = (g["hfq_low"] < g["hfq_low"].shift(1)) & (close > previous_close) & (close > g["hfq_open"])
        g["close_change_from_2026_10_23"] = close / close[g["date"].eq(pd.Timestamp("2026-10-23"))].iloc[0] - 1
        result.append(g)
    return pd.concat(result, ignore_index=True)


def current_snapshots() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = duckdb.connect(str(DATABASE), read_only=True).execute(
        """SELECT d.thscode,d.date,d.open,d.high,d.low,d.close,d.volume,d.amount,
                  h.open hfq_open,h.high hfq_high,h.low hfq_low,h.close hfq_close
           FROM v_daily d JOIN v_daily_hfq h USING(thscode,date)
           WHERE d.thscode IN ('603778.SH','605376.SH','002240.SZ') ORDER BY d.thscode,d.date"""
    ).fetchdf()
    raw["date"] = pd.to_datetime(raw["date"])
    indicators = stock_indicators(raw)
    latest_date = indicators["date"].max()
    if latest_date != pd.Timestamp("2026-10-26"):
        raise AssertionError(f"Unexpected raw-data cutoff: {latest_date}")
    frozen = duckdb.connect().execute(
        f"SELECT {','.join(FEATURE_COLUMNS)} FROM read_parquet(?) WHERE date=DATE '2026-10-23' AND thscode IN ('603778.SH','605376.SH','002240.SZ')", [str(PANEL)]
    ).fetchdf()
    frozen["date"] = pd.to_datetime(frozen["date"])
    current = indicators[indicators["thscode"].isin(CONFIG["data"]["comparison_symbols"])].copy()
    historical_raw = indicators[indicators["date"].eq(pd.Timestamp("2026-10-23"))].sort_values("thscode").reset_index(drop=True)
    frozen = frozen.sort_values("thscode").reset_index(drop=True)
    for column in ["close", "hfq_close", "rsi14", "macd_hist_pct", "macd_hist_delta", "close_location", "drawdown250", "return_250d", "return_20d", "amount_ratio1_20", "amount_ratio5_20", "close_ma5", "close_ma10", "close_ma20", "close_ma60", "ma20_slope5", "atr20_pct", "range_pct", "candle_body", "upper_shadow", "lower_shadow"]:
        if not np.allclose(historical_raw[column], frozen[column], rtol=0, atol=1e-10, equal_nan=True):
            raise AssertionError(f"Raw/frozen mismatch on 2026-10-23: {column}")
    context_columns = [
        "thscode", "date", "distance_low20", "distance_low60", "price_position60",
        "week_return4", "week_return12", "week_drawdown12", "month_return3", "month_return12",
        "market_return", "market_return5", "market_return20", "market_return60", "breadth20",
        "advance_share", "sentiment_score", "sentiment_change5", "limit_down10_share",
    ]
    historical = historical_raw.merge(frozen[context_columns], on=["thscode", "date"], how="left", validate="one_to_one")
    current["data_origin"] = "local_raw_duckdb"
    historical["data_origin"] = "frozen_local_duckdb_panel"
    snapshots = pd.concat([historical, current[current["date"] == latest_date]], ignore_index=True, sort=False)
    files = {23: MOTHER / "03_结果" / "当前信号_2026-10-23.csv", 26: MOTHER / "03_结果" / "当前信号_2026-10-26.csv"}
    ranks = []
    for day, path in files.items():
        cohort = pd.read_csv(path)
        cohort["date"] = pd.to_datetime(cohort["date"])
        cohort["cohort"] = f"2026-10-{day}"
        ranks.append(cohort)
    rank_frame = pd.concat(ranks, ignore_index=True)
    snapshots = snapshots.merge(rank_frame[["date", "thscode", "rank"]], on=["date", "thscode"], how="left")
    snapshots["signal_count"] = snapshots["date"].map(rank_frame.groupby("date").size())
    snapshots["in_original_signal"] = snapshots["rank"].notna()
    snapshots["strong_reversal"] = (snapshots["drawdown250"] <= -.65) & snapshots["BOR"] & (snapshots["close_location"] > .8)
    snapshots["planned_entry"] = snapshots["date"].map({pd.Timestamp("2026-10-23"): pd.Timestamp("2026-10-28"), pd.Timestamp("2026-10-26"): pd.Timestamp("2026-10-29")})
    snapshots["bar_range_pct"] = (snapshots["high"] - snapshots["low"]) / snapshots.groupby("thscode")["close"].shift(1)
    for column in ["week_return4", "week_return12", "week_drawdown12", "month_return3", "month_return12"]:
        completed_period = historical.set_index("thscode")[column]
        snapshots.loc[snapshots["date"] == latest_date, column] = snapshots.loc[snapshots["date"] == latest_date, "thscode"].map(completed_period)
    snapshots["higher_timeframe_basis"] = "completed week through 2026-10-23; completed month through 2026-09-30"
    snapshots["entry_status"] = "not_known_at_cutoff"
    snapshots.to_csv(RESULTS / "current_stock_snapshots.csv", index=False, encoding="utf-8-sig")
    cohorts = rank_frame[rank_frame["rank"] <= 2].merge(snapshots.drop(columns=["rank"]), on=["date", "thscode"], how="left", validate="one_to_one")
    cohorts.to_csv(RESULTS / "current_two_cohorts.csv", index=False, encoding="utf-8-sig")

    module_path = MOTHER / "02_代码" / "今日信号_数据库.py"
    spec = importlib.util.spec_from_file_location("local_signal_snapshot", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wide, market = module.features(module.load())
    market = market.sort_index()
    market["ma20"] = (1 + market["market_return"].fillna(0)).cumprod().rolling(20).mean()
    market["close_ma20"] = (1 + market["market_return"].fillna(0)).cumprod() / market["ma20"] - 1
    market["ma20_slope5"] = market["ma20"] / market["ma20"].shift(5) - 1
    for date in [pd.Timestamp("2026-10-23"), latest_date]:
        subset = wide[wide["date"] == date]
        down_limit = np.round(subset["ref_close"] * .9, 2)
        values = [
            ("advance_share", (subset["ret1"] > 0).mean()),
            ("limit_down10_share", (subset["actual_close"] <= down_limit + 1e-4).mean()),
            ("limit_up10_share", subset["limit_up_10"].mean()),
        ]
        for field, value in values:
            market.loc[date, field] = value
    for day in [23, 26]:
        expected = pd.read_csv(files[day]).iloc[0]
        date = pd.Timestamp(f"2026-10-{day}")
        for field in ["market_return20", "breadth20"]:
            if not math.isclose(float(market.loc[date, field]), float(expected[field]), abs_tol=1e-10):
                raise AssertionError(f"Current market {field} mismatch on {date.date()}")
    market_output = market.loc[[pd.Timestamp("2026-10-23"), latest_date]].reset_index(names="date")
    market_output.to_csv(RESULTS / "current_market.csv", index=False, encoding="utf-8-sig")
    del wide, market, frozen
    gc.collect()
    return snapshots, cohorts, market_output


def peer_snapshots() -> pd.DataFrame:
    con = duckdb.connect(str(DATABASE), read_only=True)
    con.execute("SET threads TO 1")
    query = f"""
        WITH targets AS (
            SELECT thscode AS peer_target,cluster_id FROM read_parquet('{literal(INITIAL / 'data/work/cluster_membership.parquet')}')
            WHERE effective_month=DATE '2026-10-01' AND thscode IN ('603778.SH','605376.SH','002240.SZ') AND cluster_id>=0
        ), peers AS (
            SELECT t.peer_target,m.* FROM read_parquet('{literal(INITIAL / 'data/work/cluster_membership.parquet')}') m
            JOIN targets t ON m.cluster_id=t.cluster_id
            WHERE m.effective_month=DATE '2026-10-01' AND m.cluster_id>=0
        ), base AS (
            SELECT p.peer_target,p.thscode,h.date,h.close,
                   h.close/lag(h.close,5) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date)-1 ret5,
                   h.close/lag(h.close,20) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date)-1 ret20,
                   h.close/lag(h.close,60) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date)-1 ret60,
                   h.close/avg(h.close) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)-1 ma20d,
                   h.close/avg(h.close) OVER(PARTITION BY p.peer_target,p.thscode ORDER BY h.date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)-1 ma60d
            FROM peers p JOIN v_daily_hfq h ON p.thscode=h.thscode WHERE h.date<=DATE '2026-10-26'
        ), aggregate AS (
            SELECT peer_target,date,count(*) members,avg(ret5) peer_return5,avg(ret20) peer_return20,avg(ret60) peer_return60,
                   avg((ma20d>0)::INTEGER) peer_breadth20,avg((ma60d>0)::INTEGER) peer_breadth60
            FROM base WHERE date IN (DATE '2026-10-23',DATE '2026-10-26') AND thscode<>peer_target GROUP BY peer_target,date
        ) SELECT * FROM aggregate ORDER BY peer_target,date
    """
    peers = con.execute(query).fetchdf().rename(columns={"peer_target": "target"})
    con.close()
    membership = pd.read_parquet(INITIAL / "data/work/cluster_membership.parquet", filters=[("effective_month", "=", pd.Timestamp("2026-10-01")), ("thscode", "in", CONFIG["data"]["comparison_symbols"])])
    membership["target"] = membership["thscode"]
    result = pd.MultiIndex.from_product([CONFIG["data"]["comparison_symbols"], pd.to_datetime(["2026-10-23", "2026-10-26"])], names=["target", "date"]).to_frame(index=False)
    result = result.merge(membership[["target", "cluster_id", "cluster_strength", "cluster_size", "cluster_train_end"]], on="target", how="left").merge(peers, on=["target", "date"], how="left")
    result["official_industry_index"] = False
    result["note"] = np.where(result["cluster_id"].fillna(-1) < 0, "no_lagged_PCA_peer_group", "lagged_PCA_peer_group_not_official_industry")
    result.to_csv(RESULTS / "current_peer_proxy.csv", index=False, encoding="utf-8-sig")
    return result


def load_research() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, pd.DatetimeIndex]:
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit TO '4GB'")
    signal = SIGNAL_CONFIG["signal"]
    signals = con.execute(
        f"""WITH eligible AS (
                SELECT thscode,date,drawdown250,amount20,return_250d,rsi14,close_location,market_return20 FROM read_parquet('{literal(PANEL)}')
                WHERE amount20>={signal['minimum_amount20']}
            ), selected AS (
                SELECT *,row_number() OVER(PARTITION BY date ORDER BY drawdown250,thscode) rank
                FROM eligible WHERE drawdown250<{signal['drawdown250_lt']} AND return_250d>{signal['return_250d_gt']}
                    AND rsi14<{signal['rsi14_lt']} AND close_location>{signal['close_location_gt']}
                    AND market_return20<{signal['market_return20_lt']}
            ) SELECT thscode,date,amount20,rank FROM selected ORDER BY date,rank,thscode"""
    ).fetchdf()
    signals["date"] = pd.to_datetime(signals["date"])
    if len(signals) != SIGNAL_CONFIG["data"]["expected_signal_count"]:
        raise AssertionError("Signal count differs from the frozen strategy")
    con.register("signal_keys", signals[["thscode", "date"]])
    feature_select = [column for column in FEATURE_COLUMNS if column != "amount20"]
    features = con.execute(f"SELECT {','.join(feature_select)} FROM read_parquet('{literal(PANEL)}') p JOIN signal_keys s USING(thscode,date)").fetchdf()
    dates = pd.DatetimeIndex(con.execute(f"SELECT DISTINCT date FROM read_parquet('{literal(PANEL)}') ORDER BY date").fetchdf()["date"])
    lead = signals[["thscode", "date"]].copy()
    date_index = {date: i for i, date in enumerate(dates)}
    lead["lead_date"] = lead["date"].map(lambda x: dates[min(date_index[x] + 2, len(dates) - 1)])
    lead["has_two_sessions"] = lead["date"].map(date_index) + 2 < len(dates)
    con.register("lead_keys", lead)
    lead_features = con.execute(
        f"""SELECT l.thscode,l.date,l.lead_date,l.has_two_sessions,p.hfq_close lead_hfq_close,
                   p.close_ma5 lead_close_ma5,p.macd_hist_pct lead_macd_hist_pct,p.market_return lead_market_return
            FROM lead_keys l JOIN read_parquet('{literal(PANEL)}') p ON p.thscode=l.thscode AND p.date=l.lead_date"""
    ).fetchdf()
    bars = con.execute(
        f"""SELECT p.thscode,p.date,p.open,p.high,p.low,p.close,p.hfq_open,p.hfq_high,p.hfq_low,p.hfq_close,
                   c.volume,c.reference_close,c.previous_actual_close,c.backward_factor,c.factor_only_adjustment
            FROM read_parquet('{literal(PANEL)}') p JOIN read_parquet('{literal(CORE)}') c USING(thscode,date)
            JOIN (SELECT DISTINCT thscode FROM signal_keys) s ON p.thscode=s.thscode ORDER BY p.date,p.thscode"""
    ).fetchdf()
    cluster = con.execute(
        f"""SELECT c.thscode,c.date,c.cluster_id,c.peer_return5,c.peer_breadth20,c.peer_breadth60,c.cluster_relative20
            FROM read_parquet('{literal(INITIAL / 'data/work/cluster_daily.parquet')}') c JOIN signal_keys s USING(thscode,date)"""
    ).fetchdf()
    con.close()
    for frame in [features, lead_features, bars, cluster]:
        frame["date"] = pd.to_datetime(frame["date"])
    symbols = sorted(signals["thscode"].unique())
    events = pd.read_parquet(EVENTS, filters=[("thscode", "in", symbols)])
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    events_before_filter = len(events)
    events = events[events["ex_date"].between(dates[0], dates[-1])].copy()
    source = {"corporate_action_rows_outside_research_window": int(events_before_filter - len(events))}
    data = data_from_frames(bars, signals, events, sessions=dates, source=source)
    return signals, features, lead_features, cluster, data, events, dates


def single_candidate(data: Any, candidate: Candidate, signal_index: int, config: EngineConfig) -> dict[str, Any]:
    last_index = min(signal_index + 60, len(data.dates) - 1)
    sessions = data.dates[signal_index:last_index + 1]
    symbol = data.symbols[candidate.symbol]
    frame: dict[str, Any] = {"date": sessions, "thscode": symbol}
    for name in BAR_COLUMNS + ["factor_only_adjustment"]:
        frame[name] = data.values[name][signal_index:last_index + 1, candidate.symbol]
    local_candidate = pd.DataFrame([{"thscode": symbol, "date": sessions[0], "rank": 1, "score": candidate.score, "amount20": candidate.amount20}])
    event_rows = []
    for (date_index, symbol_index), event in data.events.items():
        if symbol_index == candidate.symbol and signal_index <= date_index <= last_index:
            event_rows.append({"thscode": symbol, "ex_date": data.dates[date_index], **{name: getattr(event, name) for name in EVENT_COLUMNS}})
    local = data_from_frames(pd.DataFrame(frame), local_candidate, pd.DataFrame(event_rows), sessions=sessions)
    result = run_backtest(local, config)
    entry_i = signal_index + config.entry_lag_sessions
    planned_exit_i = entry_i + config.holding_sessions - 1
    output: dict[str, Any] = {"signal_index": signal_index, "status": "no_entry_observed", "closed": False, "scheduled_window_observed": planned_exit_i < len(data.dates)}
    if not result.orders.empty:
        order = result.orders.iloc[0]
        output.update(
            status=order["status"], rejection=order["status_reason"],
            entry_date=order["date"] if order["status"] == "filled" else pd.NaT,
            entry_price=order["price"], raw_entry_open=order["raw_open"], buy_shares=order["shares"],
            buy_gross=order["gross"], buy_commission=order["commission"], buy_transfer_fee=order["transfer_fee"],
            buy_total_cost=order["gross"] + order["commission"] + order["transfer_fee"], participation=order["participation"],
        )
    if not result.trades.empty:
        trade = result.trades.iloc[0]
        exit_i = data.dates.get_loc(trade["sell_date"])
        output.update(
            closed=True, exit_date=trade["sell_date"], exit_price=trade["sell_price"], exit_reason=trade["exit_reason"],
            shares=trade["shares"], bonus_shares=trade["bonus_shares"], sell_gross=trade["sell_gross"],
            sell_commission=trade["sell_commission"], sell_stamp_duty=trade["sell_stamp_duty"], sell_transfer_fee=trade["sell_transfer_fee"],
            sell_net_proceeds=trade["sell_net_proceeds"], dividend_received_gross=trade["dividend_received_gross"],
            dividend_tax=trade["dividend_tax"], dividend_received=trade["dividend_received"],
            net_return=trade["return"], net_pnl=trade["net_pnl"], holding_sessions=trade["holding_sessions"],
            entry_hfq_open=data.values["hfq_open"][entry_i, candidate.symbol],
            corporate_action_count=result.metrics["corporate_action_count"], deferred_sell_count=result.metrics["deferred_sell_count"],
            account_final_equity=result.daily_equity.iloc[-1]["equity"], account_reconciled=result.metrics["final_reconciliation_passed"],
        )
        path = slice(entry_i, exit_i + 1)
        output["mfe"] = np.nanmax(data.values["hfq_high"][path, candidate.symbol]) / output["entry_hfq_open"] - 1
        output["mae"] = np.nanmin(data.values["hfq_low"][path, candidate.symbol]) / output["entry_hfq_open"] - 1
    elif not result.open_positions.empty:
        position = result.open_positions.iloc[0]
        output.update(status="open_at_cutoff", marked_shares=position["shares"], marked_price=position["last_close"], marked_value=position["market_value"], marked_cash=result.daily_equity.iloc[-1]["cash"], marked_equity=result.daily_equity.iloc[-1]["equity"])
    output["signal_close_hfq"] = data.values["hfq_close"][signal_index, candidate.symbol]
    output["signal_high_hfq"] = data.values["hfq_high"][signal_index, candidate.symbol]
    output["signal_low_hfq"] = data.values["hfq_low"][signal_index, candidate.symbol]
    output["entry_gap_from_signal"] = data.values["hfq_open"][entry_i, candidate.symbol] / output["signal_close_hfq"] - 1 if entry_i < len(data.dates) else np.nan
    if signal_index + 2 < len(data.dates):
        lead_low = data.values["hfq_low"][signal_index + 1:signal_index + 3, candidate.symbol]
        lead_close = data.values["hfq_close"][signal_index + 2, candidate.symbol]
        output["lead2_stock_return"] = lead_close / output["signal_close_hfq"] - 1
        output["lead2_above_signal_high"] = lead_close > output["signal_high_hfq"] if np.isfinite(lead_close) else np.nan
        output["lead2_below_signal_low"] = np.nanmin(lead_low) < output["signal_low_hfq"] if np.isfinite(lead_low).any() else np.nan
    return output


def mark_sequences(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    result = frame.copy()
    index = {date: i for i, date in enumerate(dates)}
    result["di"] = result["date"].map(index)
    result = result.sort_values(["thscode", "di", "rank"]).reset_index(drop=True)
    gaps = result.groupby("thscode")["di"].diff()
    result["episode_first"] = gaps.isna() | (gaps > CONFIG["samples"]["episode_gap_sessions"])
    result["episode_id"] = result.groupby("thscode")["episode_first"].cumsum()
    non_overlapping = []
    previous: dict[str, int] = {}
    for row in result.itertuples():
        keep = row.thscode not in previous or row.di - previous[row.thscode] >= CONFIG["samples"]["independent_signal_gap_sessions"]
        non_overlapping.append(keep)
        if keep:
            previous[row.thscode] = row.di
    result["non_overlapping"] = non_overlapping
    return result


def wilson(successes: int, count: int) -> tuple[float, float]:
    if not count:
        return np.nan, np.nan
    p = successes / count
    denominator = 1 + 1.96 ** 2 / count
    center = (p + 1.96 ** 2 / (2 * count)) / denominator
    spread = 1.96 * math.sqrt(p * (1 - p) / count + 1.96 ** 2 / (4 * count ** 2)) / denominator
    return center - spread, center + spread


def stats(frame: pd.DataFrame, sample: str, condition: str) -> dict[str, Any]:
    f = frame[frame["closed"] & frame["net_return"].notna()]
    count = len(f)
    wins = int((f["net_return"] > 0).sum())
    low, high = wilson(wins, count)
    return {
        "sample": sample, "condition": condition, "closed_n": count, "signal_days": f["date"].nunique(), "symbols": f["thscode"].nunique(),
        "win_rate": wins / count if count else np.nan, "naive_wilson_low": low, "naive_wilson_high": high,
        "mean": f["net_return"].mean(), "median": f["net_return"].median(), "p10": f["net_return"].quantile(.1),
        "minimum": f["net_return"].min(), "target_share": f["exit_reason"].eq("profit_target").mean() if count else np.nan,
        "mean_mae": f["mae"].mean(), "mean_entry_gap": f["entry_gap_from_signal"].mean(),
    }


def date_bootstrap(frame: pd.DataFrame, strong: pd.Series, other: pd.Series) -> dict[str, Any]:
    closed = frame[frame["closed"] & frame["net_return"].notna()].copy()
    strong = strong.loc[closed.index].to_numpy(dtype=bool)
    other = other.loc[closed.index].to_numpy(dtype=bool)
    dates = pd.DatetimeIndex(sorted(closed["date"].unique()))
    codes = closed["date"].map({date: i for i, date in enumerate(dates)}).to_numpy()
    count = len(dates)
    sums = np.zeros((count, 4))
    for mask, offset in [(strong, 0), (other, 2)]:
        sums[:, offset] = np.bincount(codes[mask], weights=closed.loc[mask, "net_return"], minlength=count)
        sums[:, offset + 1] = np.bincount(codes[mask], minlength=count)
    rng = np.random.default_rng(CONFIG["validation"]["bootstrap_seed"])
    differences = []
    for _ in range(CONFIG["validation"]["date_cluster_bootstrap_repetitions"]):
        chosen = rng.integers(0, count, count)
        totals = sums[chosen].sum(axis=0)
        if totals[1] and totals[3]:
            differences.append(totals[0] / totals[1] - totals[2] / totals[3])
    return {"bootstrap_n": len(differences), "difference_p025": np.quantile(differences, .025) if differences else np.nan, "difference_median": np.median(differences) if differences else np.nan, "difference_p975": np.quantile(differences, .975) if differences else np.nan, "scope": "resample_signal_days_descriptive_not_strategy_selection_inference"}


def research_analysis() -> dict[str, Any]:
    signals, features, lead, cluster, data, events, dates = load_research()
    full = signals.merge(features, on=["thscode", "date"], how="left", validate="one_to_one").merge(lead, on=["thscode", "date"], how="left", validate="one_to_one").merge(cluster, on=["thscode", "date"], how="left", validate="one_to_one")
    bor = pd.DataFrame({"thscode": data.symbols})
    bor_flags = {}
    raw_flags = {}
    for symbol_index, symbol in enumerate(data.symbols):
        close = data.values["hfq_close"][:, symbol_index]
        low = data.values["hfq_low"][:, symbol_index]
        open_ = data.values["hfq_open"][:, symbol_index]
        bor_flags[symbol] = (low < np.roll(low, 1)) & (close > np.roll(close, 1)) & (close > open_)
        raw_close = data.values["close"][:, symbol_index]
        raw_low = data.values["low"][:, symbol_index]
        raw_open = data.values["open"][:, symbol_index]
        raw_flags[symbol] = (raw_low < np.roll(raw_low, 1)) & (raw_close > np.roll(raw_close, 1)) & (raw_close > raw_open)
    full["BOR"] = [bool(bor_flags[row.thscode][dates.get_loc(row.date)]) for row in full.itertuples()]
    full["raw_BOR"] = [bool(raw_flags[row.thscode][dates.get_loc(row.date)]) for row in full.itertuples()]
    full = mark_sequences(full, dates)
    symbol_index = {symbol: i for i, symbol in enumerate(data.symbols)}
    execution = dict(SIGNAL_CONFIG["execution"])
    execution["initial_capital"] = CONFIG["execution"]["standalone_capital"]
    execution["max_positions"] = CONFIG["execution"]["max_positions"]
    engine_config = EngineConfig.from_dict(execution)
    outcomes = []
    for number, row in enumerate(full.itertuples(), 1):
        signal_index = dates.get_loc(row.date)
        candidate = Candidate(symbol_index[row.thscode], signal_index, row.rank, -row.drawdown250, row.amount20)
        outcomes.append(single_candidate(data, candidate, signal_index, engine_config))
        if number % 500 == 0:
            print(f"audited standalone accounts {number}/{len(full)}", flush=True)
    outcomes = pd.DataFrame(outcomes)
    full = pd.concat([full.reset_index(drop=True), outcomes.drop(columns=["signal_index"])], axis=1)
    market = pd.read_parquet(INITIAL / "data/work/market_daily.parquet")
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").reset_index(drop=True)
    market["ma250"] = market["equal_weight_index"].rolling(250).mean()
    market["mkt_ret250"] = market["equal_weight_index"].pct_change(250, fill_method=None)
    market["mkt_dd250"] = market["equal_weight_index"] / market["equal_weight_index"].rolling(250).max() - 1
    market_index = {date: i for i, date in enumerate(market["date"])}
    lead2_market = []
    for row in full.itertuples():
        if row.has_two_sessions:
            i = market_index[row.date]
            lead2_market.append(market.loc[i + 2, "equal_weight_index"] / market.loc[i, "equal_weight_index"] - 1)
        else:
            lead2_market.append(np.nan)
    full["lead2_market_return"] = lead2_market
    market_phase = market.set_index("date")
    full["market_phase"] = [("A" if market_phase.loc[row.date, "equal_weight_index"] > market_phase.loc[row.date, "ma250"] and market_phase.loc[row.date, "mkt_ret250"] > 0 else "B" if market_phase.loc[row.date, "mkt_ret250"] > 0 else "C" if market_phase.loc[row.date, "equal_weight_index"] > market_phase.loc[row.date, "ma250"] else "D") for row in full.itertuples()]
    full["deep"] = full["drawdown250"] <= CONFIG["fixed_conditions"]["deep_drawdown250_lte"]
    full["high_close"] = full["close_location"] > CONFIG["fixed_conditions"]["high_close_location_gt"]
    full["strong_reversal"] = full["deep"] & full["BOR"] & full["high_close"]
    full["period"] = np.where(full["date"].dt.year < 2024, "2017_2023", "2024_2026")
    full["rank_group"] = np.select([full["rank"] <= 2, full["rank"] <= 10], ["top2", "rank3_10"], default="above10")
    full["unadjusted_report17"] = full["deep"] & full["raw_BOR"] & full["high_close"] & full["rank"].le(2) & full["scheduled_window_observed"]
    full.to_csv(RESULTS / "signal_outcomes_audited.csv", index=False, encoding="utf-8-sig")

    conditions = [
        ("all", pd.Series(True, index=full.index)),
        ("deep", full["deep"]),
        ("deep_BOR_high_close", full["strong_reversal"]),
        ("deep_without_complete_reversal", full["deep"] & ~full["strong_reversal"]),
        ("deep_BOR", full["deep"] & full["BOR"]),
        ("deep_high_close", full["deep"] & full["high_close"]),
        ("deep_BOR_not_high_close", full["deep"] & full["BOR"] & ~full["high_close"]),
        ("deep_not_BOR_not_high_close", full["deep"] & ~full["BOR"] & ~full["high_close"]),
        ("raw_report17", full["unadjusted_report17"]),
    ]
    samples = {
        "top2": full["rank"] <= 2,
        "top2_episode_first": (full["rank"] <= 2) & full["episode_first"],
        "top2_2017_2023": (full["rank"] <= 2) & (full["period"] == "2017_2023"),
        "top2_2024_2026": (full["rank"] <= 2) & (full["period"] == "2024_2026"),
        "top10": full["rank"] <= 10,
        "top10_non_overlapping": (full["rank"] <= 10) & full["non_overlapping"],
        "top10_2017_2023": (full["rank"] <= 10) & (full["period"] == "2017_2023"),
        "top10_2024_2026": (full["rank"] <= 10) & (full["period"] == "2024_2026"),
        "rank3_10": full["rank"].between(3, 10),
        "above10": full["rank"] > 10,
        "all_signals": pd.Series(True, index=full.index),
        "all_non_overlapping": full["non_overlapping"],
    }
    summary_rows = []
    for sample, sample_mask in samples.items():
        for condition, condition_mask in conditions:
            summary_rows.append(stats(full[sample_mask & condition_mask], sample, condition))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "fixed_condition_stats.csv", index=False, encoding="utf-8-sig")
    bootstraps = {}
    for sample, mask in [("top2", full["rank"] <= 2), ("top10_non_overlapping", (full["rank"] <= 10) & full["non_overlapping"]), ("all_non_overlapping", full["non_overlapping"])]:
        subset = full[mask].copy()
        bootstraps[sample] = date_bootstrap(subset, subset["strong_reversal"], subset["deep"] & ~subset["strong_reversal"])
    save_json("strong_reversal_date_bootstrap.json", bootstraps)

    feature_definitions = {
        "BOR": full["BOR"],
        "close_location_gt_0.8": full["high_close"],
        "MACD_hist_gt_0": full["macd_hist_pct"] > 0,
        "MACD_hist_improving": full["macd_hist_delta"] > 0,
        "amount_ratio1_20_gte_1": full["amount_ratio1_20"] >= 1,
        "amount_ratio5_20_gte_1": full["amount_ratio5_20"] >= 1,
        "RSI_lt_30": full["rsi14"] < 30,
        "prior_250d_gain_gt_100pct": full["return_250d"] > 1,
        "close_ma20_gte_minus10pct": full["close_ma20"] >= -.1,
        "above_MA5": full["close_ma5"] > 0,
        "MA20_slope5_positive": full["ma20_slope5"] > 0,
        "month_return3_gt_minus30pct": full["month_return3"] > -.3,
        "sentiment_change5_positive": full["sentiment_change5"] > 0,
        "lagged_peer_return5_gt_market5": full["peer_return5"] > full["market_return5"],
        "lagged_peer_breadth20_gt_market": full["peer_breadth20"] > full["breadth20"],
        "T2_stock_return_positive": full["lead2_stock_return"] > 0,
        "T2_close_above_signal_high": full["lead2_above_signal_high"].fillna(False),
        "T1_T2_low_below_signal_low": full["lead2_below_signal_low"].fillna(False),
        "T2_above_MA5": full["lead_close_ma5"] > 0,
        "T2_MACD_hist_positive": full["lead_macd_hist_pct"] > 0,
        "T1_T2_market_return_positive": full["lead2_market_return"] > 0,
    }
    feature_sources = {
        "BOR": None, "close_location_gt_0.8": "close_location",
        "MACD_hist_gt_0": "macd_hist_pct", "MACD_hist_improving": "macd_hist_delta",
        "amount_ratio1_20_gte_1": "amount_ratio1_20", "amount_ratio5_20_gte_1": "amount_ratio5_20",
        "RSI_lt_30": "rsi14", "prior_250d_gain_gt_100pct": "return_250d",
        "close_ma20_gte_minus10pct": "close_ma20", "above_MA5": "close_ma5",
        "MA20_slope5_positive": "ma20_slope5", "month_return3_gt_minus30pct": "month_return3",
        "sentiment_change5_positive": "sentiment_change5", "lagged_peer_return5_gt_market5": "peer_return5",
        "lagged_peer_breadth20_gt_market": "peer_breadth20", "T2_stock_return_positive": "lead2_stock_return",
        "T2_close_above_signal_high": "lead_hfq_close", "T1_T2_low_below_signal_low": "lead2_below_signal_low",
        "T2_above_MA5": "lead_close_ma5", "T2_MACD_hist_positive": "lead_macd_hist_pct",
        "T1_T2_market_return_positive": "lead2_market_return",
    }
    feature_rows = []
    for sample, sample_mask in [("top2_deep", (full["rank"] <= 2) & full["deep"]), ("top10_deep", (full["rank"] <= 10) & full["deep"]), ("all_deep", full["deep"]), ("all_deep_non_overlapping", full["deep"] & full["non_overlapping"])]:
        for feature, mask in feature_definitions.items():
            source = feature_sources[feature]
            valid = pd.Series(True, index=full.index) if source is None else full[source].notna()
            if feature == "T2_close_above_signal_high":
                valid &= full["has_two_sessions"]
            for period in ["all", "2017_2023", "2024_2026"]:
                period_mask = pd.Series(True, index=full.index) if period == "all" else full["period"].eq(period)
                selected = sample_mask & period_mask & valid
                feature_rows.append(stats(full[selected & mask], f"{sample}_{period}", f"{feature}=true"))
                feature_rows.append(stats(full[selected & ~mask], f"{sample}_{period}", f"{feature}=false"))
    pd.DataFrame(feature_rows).to_csv(RESULTS / "fixed_feature_stats.csv", index=False, encoding="utf-8-sig")

    paired = []
    for symbol, group in full.groupby("thscode"):
        group = group.sort_values("di")
        previous = None
        for row in group.itertuples():
            if previous is not None and 1 <= row.di - previous.di <= 3:
                paired.append({"thscode": symbol, "first_date": previous.date, "second_date": row.date, "gap": row.di - previous.di, "first_rank": previous.rank, "second_rank": row.rank, "first_strong": previous.strong_reversal, "second_strong": row.strong_reversal, "first_closed": previous.closed, "second_closed": row.closed, "first_return": previous.net_return, "second_return": row.net_return, "second_minus_first": row.net_return - previous.net_return})
            previous = row
    paired = pd.DataFrame(paired)
    paired.to_csv(RESULTS / "adjacent_signal_pairs.csv", index=False, encoding="utf-8-sig")
    pair_stats = []
    for label, group in [("all", paired), ("first_strong_second_not", paired[paired["first_strong"] & ~paired["second_strong"]]), ("both_top10", paired[(paired["first_rank"] <= 10) & (paired["second_rank"] <= 10)])]:
        closed = group[group["first_closed"] & group["second_closed"]]
        pair_stats.append({"pair_sample": label, "pairs": len(closed), "first_mean": closed["first_return"].mean(), "second_mean": closed["second_return"].mean(), "difference_mean": closed["second_minus_first"].mean(), "difference_median": closed["second_minus_first"].median(), "second_better_share": (closed["second_minus_first"] > 0).mean(), "first_win_rate": (closed["first_return"] > 0).mean(), "second_win_rate": (closed["second_return"] > 0).mean()})
    pd.DataFrame(pair_stats).to_csv(RESULTS / "adjacent_signal_pair_stats.csv", index=False, encoding="utf-8-sig")
    market_stats = []
    for rank_name, mask in [("top10", full["rank"] <= 10), ("all", pd.Series(True, index=full.index))]:
        for phase in ["A", "B", "C", "D"]:
            market_stats.append(stats(full[mask & full["market_phase"].eq(phase)], rank_name, f"market_phase_{phase}"))
    pd.DataFrame(market_stats).to_csv(RESULTS / "market_phase_stats.csv", index=False, encoding="utf-8-sig")
    report17 = full[full["unadjusted_report17"]].copy()
    report17.to_csv(RESULTS / "original_17_case_reconciliation.csv", index=False, encoding="utf-8-sig")
    status = full.groupby(["rank_group", "status"], dropna=False).size().rename("n").reset_index()
    status.to_csv(RESULTS / "execution_status.csv", index=False, encoding="utf-8-sig")
    return {
        "signals": len(full), "closed": int(full["closed"].sum()), "top10": int((full["rank"] <= 10).sum()),
        "top2": int((full["rank"] <= 2).sum()), "hfq_vs_raw_BOR_disagreements": int((full["BOR"] != full["raw_BOR"]).sum()),
        "original_17_cases": len(report17), "original_17_closed": int(report17["closed"].sum()),
        "original_17_audited_mean": report17["net_return"].mean(), "original_17_audited_win_rate": (report17["net_return"] > 0).mean(),
        "strict_BOR_high_top2": int(((full["rank"] <= 2) & full["strong_reversal"]).sum()),
        "standalone_execution": "50000 initial equity, two-position slot sizing, one isolated candidate per account",
        "note": "standalone outcomes remove slot competition; they are not account returns or an independent holdout",
        "load_source": data.source,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    input_manifest = json.loads((HERE.parents[1] / "audit/input_manifest.json").read_text())
    for name, path in [("panel", PANEL), ("core", CORE), ("events", EVENTS)]:
        if sha256(path) != input_manifest["input_sha256"][name]:
            raise AssertionError(f"Frozen input changed: {name}")
    snapshots, cohorts, market = current_snapshots()
    peers = peer_snapshots()
    current_columns = ["date", "thscode", "open", "high", "low", "close", "close_location", "rsi14", "macd_hist_pct", "amount_ratio1_20", "close_ma20", "drawdown250", "return_250d", "BOR", "rank", "strong_reversal"]
    current_table = snapshots[snapshots["date"].isin(pd.to_datetime(["2026-10-23", "2026-10-26"]))][current_columns]
    print(current_table.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    print(market.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    print(peers.to_string(index=False, float_format=lambda value: f"{value:.5f}"), flush=True)
    summary = research_analysis()
    summary["input_hashes"] = {"panel": sha256(PANEL), "core": sha256(CORE), "events": sha256(EVENTS), "raw_database": sha256(DATABASE)}
    summary["code_hash"] = sha256(HERE)
    summary["config_hash"] = sha256(OUT / "analysis_config.json")
    summary["current_cutoff"] = "2026-10-26"
    summary["historical_return_cutoff"] = "2026-10-23"
    con = duckdb.connect(str(DATABASE), read_only=True)
    raw_count, raw_symbols, raw_zero_activity = con.execute("SELECT count(*),count(DISTINCT thscode),count(*) FILTER(WHERE volume=0 OR amount=0) FROM v_daily WHERE date=DATE '2026-10-26'").fetchone()
    con.close()
    summary["raw_database_rows_on_cutoff"] = raw_count
    summary["raw_database_symbols_on_cutoff"] = raw_symbols
    summary["raw_database_zero_activity_rows_on_cutoff"] = raw_zero_activity
    summary["remote_data_used"] = False
    save_json("analysis_summary.json", summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
