"""Point-in-time baseline signals."""

import pandas as pd

from .config import ResearchConfig


KEY_COLUMNS = ["date", "code"]


def _keys(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(KEY_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Missing alignment columns: {sorted(missing)}")
    keys = frame[KEY_COLUMNS].copy()
    keys["date"] = keys["date"].astype(str)
    return keys


def assert_aligned_inputs(panel: pd.DataFrame, signals: pd.DataFrame) -> None:
    """Fail before execution unless panel and signals have identical daily keys."""
    panel_keys = _keys(panel)
    signal_keys = _keys(signals)
    if panel_keys.empty or signal_keys.empty:
        raise ValueError("Backtest panel and signals must not be empty")
    if panel_keys.duplicated().any() or signal_keys.duplicated().any():
        raise ValueError("Backtest panel and signals require unique (date, code) keys")
    if len(panel_keys) != len(signal_keys):
        raise ValueError(
            f"Backtest row-count mismatch: panel={len(panel_keys)} signals={len(signal_keys)}"
        )
    joined = panel_keys.merge(
        signal_keys,
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    mismatched = joined["_merge"].ne("both")
    if mismatched.any():
        counts = joined.loc[mismatched, "_merge"].value_counts().to_dict()
        raise ValueError(f"Backtest (date, code) key mismatch: {counts}")
    panel_dates = set(panel_keys["date"])
    signal_dates = set(signal_keys["date"])
    if panel_dates != signal_dates:
        raise ValueError("Backtest panel and signal date windows differ")


def select_backtest_window(
    panel: pd.DataFrame, signals: pd.DataFrame, config: ResearchConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select by each frame's own dates, then join signals one-to-one by key."""
    panel = panel.copy()
    signals = signals.copy()
    panel["date"] = panel["date"].astype(str)
    signals["date"] = signals["date"].astype(str)
    panel_window = panel.loc[
        panel["date"].between(config.start_date, config.end_date)
    ].reset_index(drop=True)
    signal_window = signals.loc[
        signals["date"].between(config.start_date, config.end_date)
    ].reset_index(drop=True)
    aligned_signals = panel_window[KEY_COLUMNS].merge(
        signal_window,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    assert_aligned_inputs(panel_window, aligned_signals)
    return panel_window, aligned_signals


def compute_signals(panel: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    result = panel.copy().sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    grouped = result.groupby("code", sort=False)

    result["ma200"] = grouped["signal_close"].transform(
        lambda values: values.rolling(config.trend_window).mean()
    )
    result["channel_high_120"] = grouped["signal_high"].transform(
        lambda values: values.shift(1).rolling(config.entry_lookback).max()
    )
    result["channel_low_120"] = grouped["signal_low"].transform(
        lambda values: values.shift(1).rolling(config.exit_lookback).min()
    )
    previous_close = grouped["signal_close"].shift(config.entry_lookback)
    result["rank_return_120"] = result["signal_close"] / previous_close - 1

    previous_ma = result.groupby("code", sort=False)["ma200"].shift(config.trend_slope_window)
    in_main_window = result["date"].between(config.start_date, config.end_date)
    non_st = result["is_st"].eq(0).fillna(False)
    tradable = result["trade_status"].eq(1).fillna(False)
    valid = result["data_valid"].eq(True).fillna(False)
    no_execution_conflict = ~result["execution_conflict"].fillna(True).astype(bool)
    result["entry_signal"] = (
        result["signal_close"].gt(result["ma200"])
        & result["ma200"].gt(previous_ma)
        & result["signal_close"].gt(result["channel_high_120"])
        & result["rank_return_120"].notna()
        & non_st
        & tradable
        & valid
        & no_execution_conflict
        & in_main_window
    ).astype(bool)
    result["exit_signal"] = result["signal_close"].lt(result["channel_low_120"]).astype(bool)
    result["st_exit"] = result["is_st"].eq(1).fillna(False).astype(bool)
    return result
