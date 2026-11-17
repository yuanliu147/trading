"""Feature families for the selection layer. Single source of truth."""
from __future__ import annotations

FAMILIES: dict[str, list[str]] = {
    "A趋势": [
        "ret5", "ret10", "ret20", "ret60", "ret120", "ret250",
        "ex_ew20", "ex_ew60", "ex_ew120",
        "dist_ma20", "dist_ma60", "dist_ma120", "dist_ma250",
        "ma20_slope20", "ma60_slope20", "ma120_slope20",
        "trend_r2_60", "trend_r2_120", "trend_slope_60", "trend_slope_120",
        "up_ratio20", "up_ratio60", "ma_alignment",
        "ma20_ma60_spread", "ma60_ma120_spread",
        "new_high_60", "new_high_250", "dd250", "days_since_hh250",
        "hh20_ratio", "prior_runup20", "prior_runup60",
        "price_efficiency60", "box_range60",
    ],
    "B波动": [
        "vol20", "vol60", "vol250", "downvol20", "atr20n", "atr60n",
        "beta60", "beta250", "r2mkt60", "resid_vol60",
        "skew60", "kurt60", "maxret20", "minret20",
        "vol_of_vol60", "max_dd_250", "ulcer250",
        "limit_up20", "limit_up60", "limit_down20", "limit_down60",
        "gap_abs20", "st_days250",
    ],
    "C流动性": [
        "log_amt20", "log_amt60", "amt_trend", "amihud20", "log_mcap",
        "turn20", "turn60", "turn_ratio", "turn_pct250", "amt_pct250", "vol_pct250",
    ],
    "D量价回踩": [
        "vol_ratio5_20", "vol_ratio20_60", "updown_vol20", "updown_vol60",
        "obv_slope20", "pullback_vol_ratio",
        "pullback_depth", "pullback_depth_atr", "retrace_ratio", "days_since_hh20",
        "close_loc", "body", "upper_shadow", "lower_shadow",
        "signal_day_ret", "signal_day_gap", "signal_day_turn_ratio",
        "ma20_touch_count10", "vwap20_dev", "vol_price_corr20",
    ],
    "E估值": [
        "log_pe", "log_pb", "pe_neg", "pe_med_ratio250", "pb_med_ratio250",
    ],
    "F行业市场": [
        "ind_ret20", "ind_ret60", "ind_ret120", "ind_ex_ew20",
        "ind_mom_rank", "ind_breadth20", "ind_vol20",
        "stock_ind_ex20", "stock_ind_ex60",
        "mkt_gate", "hs300_gate", "market_breadth20", "market_breadth60",
        "mkt_ret20", "mkt_vol20", "mkt_dd250",
    ],
    "G结构": [
        "board_main", "board_gem", "board_star", "age_years",
    ],
    "H趋势持续性": [
        "vr5", "vr20", "hurst5", "hurst20", "autocorr60",
    ],
}

ALL_FEATURES = [f for fs in FAMILIES.values() for f in fs]
FAMILY_OF = {f: fam for fam, fs in FAMILIES.items() for f in fs}

# theory score B, fixed before looking at outcomes
THEORY_SIGN = {
    "trend_r2_120": +1,
    "ma_alignment": +1,
    "vol60": -1,
    "pullback_depth_atr": -1,
    "pullback_vol_ratio": -1,
    "ind_ex_ew20": +1,
    "log_amt20": +1,
    "dist_ma60": -1,
}


def add_derived(df):
    """Add features that are simple combinations of stored columns."""
    import numpy as np
    df = df.copy()
    df["ex_ew20"] = df["ret20"] - df["mkt_ret20"]
    df["ex_ew60"] = df["ret60"] - df["mkt_ret60"]
    df["ex_ew120"] = df["ret120"] - df["mkt_ret120"]
    df["turn_ratio"] = df["turn20"] / df["turn60"].replace(0, float("nan"))
    df["hurst5"] = np.log(df["vr5"].clip(lower=1e-8)) / np.log(5)
    df["hurst20"] = np.log(df["vr20"].clip(lower=1e-8)) / np.log(20)
    return df
