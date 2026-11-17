"""Point-in-time signals for the recovered legacy candidate B."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .legacy_data import A_ENTRY_REQUIRED_FIELDS, candidate_entry_valid


KEYS = ["date", "code"]
ELIGIBILITY_START = "2017-10-13"
GAP_MAX = 0.05


@dataclass(frozen=True)
class LegacyBResult:
    signals: pd.DataFrame
    audit: dict


def load_industry_csv(path: str | Path) -> pd.DataFrame:
    industry = pd.read_csv(path)
    parts = industry["code"].str.split(".")
    industry["code"] = parts.str[1] + "." + parts.str[0].str.upper()
    industry["start_date"] = pd.to_datetime(industry["start_date"])
    industry["industry"] = industry["industry"].astype(str)
    return industry.sort_values(["code", "start_date"]).reset_index(drop=True)


def industry_matrix(dates, codes, industry: pd.DataFrame) -> pd.DataFrame:
    """Last effective industry record per date, never a current-snapshot backfill."""
    ids = {name: i + 1 for i, name in enumerate(sorted(industry["industry"].dropna().unique()))}
    out = np.zeros((len(dates), len(codes)), dtype="int16")
    positions = {code: j for j, code in enumerate(codes)}
    date_values = pd.DatetimeIndex(dates).values
    for code, group in industry.groupby("code", sort=False):
        column = positions.get(code)
        if column is None:
            continue
        starts = pd.to_datetime(group["start_date"]).values.astype("datetime64[ns]")
        values = group["industry"].map(ids).fillna(0).astype("int16").values
        index = np.searchsorted(starts, date_values, side="right") - 1
        out[:, column] = np.where(index >= 0, values[np.maximum(index, 0)], 0)
    return pd.DataFrame(out, index=pd.DatetimeIndex(dates), columns=codes)


def _matrix(panel: pd.DataFrame, column: str) -> pd.DataFrame:
    return panel.pivot(index="date", columns="code", values=column).sort_index().astype(float)


def _map_sector(industry_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    mapped = np.full(industry_ids.shape, np.nan, dtype=float)
    for industry_id in range(1, values.shape[1]):
        mapped[:, industry_id] = values[:, industry_id]
    return np.take_along_axis(mapped, industry_ids.astype(int), axis=1)


def compute_legacy_b_signals(
    full_a_panel: pd.DataFrame,
    main_panel: pd.DataFrame,
    industry: pd.DataFrame,
    config,
    eligibility_start: str = ELIGIBILITY_START,
) -> LegacyBResult:
    """Compute B's frozen point-in-time entries and data-gate audits."""
    candidate = next(c for c in config.candidates if c.name == "early_trend_breadth")
    full = full_a_panel.copy()
    main = main_panel.copy()
    full["date"] = pd.to_datetime(full["date"])
    main["date"] = pd.to_datetime(main["date"])
    full = full.sort_values(["date", "code"])
    main = main.sort_values(["date", "code"])

    close = _matrix(full, "signal_close")
    open_ = _matrix(full, "signal_open")
    high = _matrix(full, "signal_high")
    low = _matrix(full, "signal_low")
    amount = _matrix(full, "amount")
    raw_close = _matrix(full, "close")
    dates = close.index
    codes = close.columns

    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    up_struct = ((close > ma60) & (ma20 > ma60) & (ma60 > ma60.shift(20))).fillna(False)
    cumulative = up_struct.cumsum()
    reset = cumulative.where(~up_struct).ffill().fillna(0)
    trend_age = cumulative - reset

    previous = close.shift(1).values
    tr = np.maximum.reduce([
        high.values - low.values,
        np.abs(high.values - previous),
        np.abs(low.values - previous),
    ])
    tr = pd.DataFrame(tr, index=dates, columns=codes)
    atr_pct = tr.rolling(14, min_periods=10).mean() / close
    dd60 = close / close.rolling(60, min_periods=30).max() - 1
    amt20 = amount.rolling(20, min_periods=15).mean()
    r1 = close.pct_change(fill_method=None)
    r20 = close / close.shift(20) - 1

    limit = pd.DataFrame(0.098, index=dates, columns=codes)
    growth = [c for c in codes if c.startswith(("300", "301"))]
    if growth:
        limit.loc[dates >= "2020-08-24", growth] = 0.196
    star = [c for c in codes if c.startswith("688")]
    if star:
        limit.loc[:, star] = 0.196
    bj = [c for c in codes if c.startswith(("8", "4"))]
    if bj:
        limit.loc[:, bj] = 0.296
    lu_count20 = (r1 >= limit).astype(float).rolling(20, min_periods=10).sum()
    market_breadth20 = (close > ma20).astype(float).where(close.notna()).mean(axis=1)

    industry_frame = industry_matrix(dates, codes, industry)
    industry_array = industry_frame.values.astype(int)
    above20 = (close > ma20).astype(float).where(close.notna()).values
    ret20 = r20.values
    sector_breadth = np.full((len(dates), int(industry_array.max()) + 1), np.nan)
    sector_ret = np.full_like(sector_breadth, np.nan)
    for industry_id in range(1, sector_breadth.shape[1]):
        mask = industry_array == industry_id
        if not mask.any():
            continue
        sector_breadth[:, industry_id] = np.nanmean(np.where(mask, above20, np.nan), axis=1)
        sector_ret[:, industry_id] = np.nanmean(np.where(mask, ret20, np.nan), axis=1)
    sector_breadth20 = _map_sector(industry_array, sector_breadth)
    sector_ret20 = _map_sector(industry_array, sector_ret)

    valid = (
        up_struct.values
        & (amt20.values >= candidate.amount_ma20_min)
        & (raw_close.values >= candidate.actual_close_min)
        & (trend_age.values >= candidate.trend_age_min)
        & (trend_age.values <= candidate.trend_age_max)
        & (atr_pct.values <= candidate.atr_pct_max)
        & (dd60.values > candidate.dd60_threshold)
        & (lu_count20.values == 0)
        & (market_breadth20.values[:, None] > candidate.market_breadth_threshold)
        & (sector_breadth20 >= candidate.sector_breadth_min)
        & (sector_ret20 > candidate.sector_ret20_threshold)
        & (industry_array > 0)
    )
    date_ok = (dates >= pd.Timestamp(eligibility_start)) & (dates <= pd.Timestamp(config.end_date))
    entry_matrix = pd.DataFrame(valid & date_ok[:, None], index=dates, columns=codes).fillna(False)
    entry_rows = np.argwhere(entry_matrix.values)
    entries = pd.DataFrame({
        "date": dates[entry_rows[:, 0]],
        "code": codes[entry_rows[:, 1]],
        "entry_signal": True,
    })
    for name, matrix in (
        ("amt20", amt20), ("atr_pct", atr_pct), ("sector_breadth20", sector_breadth20),
        ("sector_ret20", sector_ret20), ("market_breadth20", market_breadth20),
        ("trend_age", trend_age), ("dd60", dd60), ("lu_count20", lu_count20),
    ):
        if isinstance(matrix, pd.Series):
            entries[name] = matrix.iloc[entry_rows[:, 0]].values
        else:
            values = matrix.values if hasattr(matrix, "values") else np.asarray(matrix)
            entries[name] = values[entry_rows[:, 0], entry_rows[:, 1]]

    signals = main[["date", "code"]].copy()
    signals["date"] = signals["date"].dt.strftime("%Y-%m-%d")
    entries["date"] = entries["date"].dt.strftime("%Y-%m-%d")
    signals = signals.merge(entries, on=KEYS, how="left", validate="one_to_one")
    signals["entry_signal"] = signals["entry_signal"].eq(True).fillna(False)

    required = list(dict.fromkeys([
        "open", "high", "low", "close", "prev_close", "signal_open", "signal_high",
        "signal_low", "signal_close", "volume", "amount", "trade_status", "is_st",
        "ipo_date", "execution_conflict",
    ]))
    main_fields = main[["date", "code", *required]].copy()
    main_fields["date"] = main_fields["date"].dt.strftime("%Y-%m-%d")
    proposed = signals.loc[signals["entry_signal"], ["date", "code"]].merge(
        main_fields, on=KEYS, how="left", validate="one_to_one"
    )
    valid_mask = candidate_entry_valid(proposed) if len(proposed) else pd.Series(dtype=bool)
    st_mask = proposed["is_st"].eq(1).fillna(True) if len(proposed) else pd.Series(dtype=bool)
    accepted = (
        proposed.loc[valid_mask.values & ~st_mask.values, ["date", "code"]]
        if len(proposed) else proposed
    )
    signals = signals.merge(
        accepted.assign(entry_signal=True), on=KEYS, how="left", suffixes=("", "_valid"),
        validate="one_to_one",
    )
    signals["entry_signal"] = signals["entry_signal_valid"].eq(True).fillna(False)
    signals = signals.drop(columns="entry_signal_valid")
    signals["legacy_rank_score"] = -signals["amt20"]
    signals["rank_return_120"] = signals["legacy_rank_score"]
    signals["stop_ref"] = np.nan
    signals["stop_pct"] = 0.20

    active = full[["date", "code"]].drop_duplicates()
    active_positions = (
        dates.get_indexer(pd.to_datetime(active["date"])),
        codes.get_indexer(active["code"]),
    )
    active_industry = industry_array[active_positions[0], active_positions[1]] > 0
    active = active.assign(industry_present=active_industry)
    daily_industry = active.groupby("date")["industry_present"].mean()
    window = (daily_industry.index >= pd.Timestamp(config.start_date)) & (
        daily_industry.index <= pd.Timestamp(config.end_date)
    )
    daily_industry = daily_industry.loc[window]
    final = signals.loc[signals["entry_signal"]]
    final_keys = set(zip(final["date"], final["code"]))
    tradeable = proposed.loc[
        proposed["trade_status"].eq(1)
        & proposed["volume"].gt(0)
        & proposed["amount"].gt(0)
        & proposed["execution_conflict"].eq(False).fillna(False)
    ]
    tradeable_valid = candidate_entry_valid(tradeable) if len(tradeable) else pd.Series(dtype=bool)
    audit = {
        "eligibility_start": eligibility_start,
        "proposed_entries": int(len(proposed)),
        "tradeable_proposed": int(len(tradeable)),
        "invalid_tradeable_proposed": int((~tradeable_valid).sum()) if len(tradeable) else 0,
        "st_rejected_proposed": int(st_mask.sum()) if len(proposed) else 0,
        "entry_count": int(len(final)),
        "entry_codes": int(final["code"].nunique()),
        "first_entry": str(final["date"].min()) if len(final) else None,
        "last_entry": str(final["date"].max()) if len(final) else None,
        "industry_daily_min": float(np.nanmin(daily_industry)) if len(daily_industry) else 0.0,
        "industry_days_below_min": int((daily_industry < config.industry_coverage_min).sum()),
        "industry_entry_coverage": float(
            sum(1 for date, code in final_keys if
                industry_frame.loc[pd.Timestamp(date), code] > 0) / len(final_keys)
        ) if final_keys else 0.0,
        "required_entry_coverage": float(tradeable_valid.mean()) if len(tradeable) else 0.0,
    }
    return LegacyBResult(signals=signals, audit=audit)
