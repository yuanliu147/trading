"""Point-in-time baseline signals."""

import pandas as pd

from .config import ResearchConfig


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
