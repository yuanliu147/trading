"""Point-in-time signals for frozen legacy candidate A only."""

import numpy as np
import pandas as pd

from .legacy_data import A_ENTRY_REQUIRED_FIELDS, SIGNAL_FIELDS, candidate_entry_valid


KEYS = ["date", "code"]
MARKET_CODE = "000001.SH"
INDICATOR_COLUMNS = [
    "source_row_index",
    "ma5",
    "ma20",
    "ma20_previous",
    "amount_ma20",
    "volume_ma20",
    "high20_prev",
    "low10",
    "wdif",
    "wdea",
    "whist",
    "m1",
    "m2",
    "m3",
    "hit_count",
    "history_count",
    "amt60",
    "vol120",
    "range250",
    "act",
    "limfrac",
    "gapfrac",
]
FEATURE_COLUMNS = ["amt60", "vol120", "range250", "act", "limfrac", "gapfrac"]


def _candidate_config(config):
    return next(
        candidate for candidate in config.candidates if candidate.name == "three_of_two_low_vol"
    )


def _method_hits(m1, m2, m3):
    """Count only explicit true values; nullable and missing booleans fail closed."""
    flags = pd.concat([m1, m2, m3], axis=1).apply(lambda column: column.fillna(False))
    flags = flags.astype(bool)
    hits = flags.sum(axis=1).astype(int)
    return hits, hits.ge(2)


def order_legacy_a_candidates(candidates):
    """Use A's authoritative score, descending, with code as the final tie-break."""
    return candidates.sort_values(
        ["date", "legacy_rank_score", "code"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _history_indicators(block, candidate):
    close = block["signal_close"].astype(float).reset_index(drop=True)
    open_ = block["signal_open"].astype(float).reset_index(drop=True)
    high = block["signal_high"].astype(float).reset_index(drop=True)
    low = block["signal_low"].astype(float).reset_index(drop=True)
    volume = block["volume"].astype(float).reset_index(drop=True)
    amount = block["amount"].astype(float).reset_index(drop=True)

    ma5 = close.rolling(candidate.stock_ma_fast).mean()
    ma20 = close.rolling(candidate.stock_ma_slow).mean()
    volume_ma20 = volume.rolling(candidate.m3_volume_window).mean()
    amount_ma20 = amount.rolling(candidate.stock_ma_slow).mean()
    wdif = close.ewm(span=candidate.m2_ema_fast, adjust=False).mean() - close.ewm(
        span=candidate.m2_ema_slow, adjust=False
    ).mean()
    wdea = wdif.ewm(span=candidate.m2_signal_ema, adjust=False).mean()
    whist = 2 * (wdif - wdea)
    m1 = (ma5.gt(ma20) & ma5.shift(1).le(ma20.shift(1))).fillna(False)
    m2 = (whist.gt(0) & wdea.gt(wdea.shift(1))).fillna(False)
    m3 = (
        close.gt(high.rolling(candidate.m3_breakout_window).max().shift(1))
        & volume.ge(candidate.m3_volume_ratio_min * volume_ma20)
    ).fillna(False)
    hits, _ = _method_hits(m1, m2, m3)

    returns = close.pct_change()
    prefix = str(block["code"].iloc[0]).split(".")[0][:3]
    limit_threshold = (
        candidate.stock_growth_limit_threshold
        if prefix in ("300", "301", "688")
        else candidate.stock_main_limit_threshold
    )
    amount_long = amount.rolling(candidate.stock_activity_long_window).mean()
    activity = amount.rolling(candidate.stock_activity_short_window).mean() / amount_long - 1
    activity = activity.where(amount_long.gt(0))
    range_high = high.rolling(candidate.stock_range_window).max()
    range_low = low.rolling(candidate.stock_range_window).min()

    return pd.DataFrame(
        {
            "source_row_index": np.arange(len(block), dtype=np.int64),
            "ma5": ma5,
            "ma20": ma20,
            "ma20_previous": ma20.shift(1),
            "amount_ma20": amount_ma20,
            "volume_ma20": volume_ma20,
            "high20_prev": high.rolling(candidate.m3_breakout_window).max().shift(1),
            "low10": low.rolling(candidate.low10_window).min(),
            "wdif": wdif,
            "wdea": wdea,
            "whist": whist,
            "m1": m1.astype(float),
            "m2": m2.astype(float),
            "m3": m3.astype(float),
            "hit_count": hits.astype(float),
            "history_count": np.arange(1, len(block) + 1, dtype=np.int64),
            "amt60": amount.rolling(60).mean(),
            "vol120": returns.rolling(candidate.stock_vol_window, min_periods=1).std(ddof=0)
            * np.sqrt(252),
            "range250": range_high / range_low - 1,
            "act": activity,
            "limfrac": returns.abs()
            .ge(limit_threshold)
            .astype(float)
            .rolling(candidate.stock_limit_window)
            .mean(),
            "gapfrac": open_.div(close.shift(1))
            .sub(1)
            .abs()
            .gt(candidate.stock_gap_threshold)
            .astype(float)
            .rolling(candidate.stock_gap_window)
            .mean(),
        }
    )


def _add_stock_indicators(context, candidate):
    result = context.sort_values(KEYS[::-1], kind="stable").reset_index(drop=True)
    outputs = {
        column: np.full(len(result), np.nan, dtype=float) for column in INDICATOR_COLUMNS
    }
    for positions in result.groupby("code", sort=False).indices.values():
        indicators = _history_indicators(result.iloc[positions], candidate)
        for column in INDICATOR_COLUMNS:
            outputs[column][positions] = indicators[column].to_numpy()
    for column in INDICATOR_COLUMNS:
        result[column] = outputs[column]
    result["source_row_index"] = result["source_row_index"].astype(np.int64)
    result["history_count"] = result["history_count"].astype(np.int64)
    return result


def _cross_section(context, candidate):
    eligible = context.loc[
        context["rank_eligible"].eq(True).fillna(False)
        & context["history_count"].ge(candidate.stock_history_min)
        & context["amt60"].ge(candidate.stock_amount_ma60_min),
        [*KEYS, *FEATURE_COLUMNS],
    ].copy()
    if eligible.empty:
        eligible["s_vol"] = pd.Series(dtype=float)
        eligible["s_heat"] = pd.Series(dtype=float)
        eligible["stock_filter_ok"] = pd.Series(dtype=bool)
        return eligible[[*KEYS, "s_vol", "s_heat", "stock_filter_ok"]]

    grouped = eligible.groupby("date", sort=False)
    eligible["vol120_rank"] = grouped["vol120"].rank(pct=True)
    eligible["range250_rank"] = grouped["range250"].rank(pct=True)
    eligible["act_rank"] = grouped["act"].rank(pct=True)
    eligible["limfrac_rank"] = grouped["limfrac"].rank(pct=True)
    eligible["gapfrac_rank"] = grouped["gapfrac"].rank(pct=True)
    eligible["s_vol"] = eligible[["vol120_rank", "range250_rank"]].mean(axis=1)
    eligible["s_heat"] = eligible[["act_rank", "limfrac_rank", "gapfrac_rank"]].mean(
        axis=1
    )
    eligible["stock_filter_ok"] = (
        eligible["s_vol"].lt(candidate.s_vol_max)
        & eligible["s_heat"].ge(candidate.s_heat_min)
    ).fillna(False)
    return eligible[[*KEYS, "s_vol", "s_heat", "stock_filter_ok"]]


def _market_panel(index_panel, candidate):
    market = index_panel.loc[
        index_panel["code"].eq(MARKET_CODE), ["date", "code", "close"]
    ].copy()
    market["date"] = market["date"].astype(str)
    market = market.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    if market.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate market-index key")
    market = market.rename(columns={"code": "market_code", "close": "market_close"})
    market["market_ma20"] = market["market_close"].rolling(
        candidate.market_ma_window
    ).mean()
    market["market_ma20_previous"] = market["market_ma20"].shift(1)
    market["market_ok"] = (
        market["market_close"].gt(market["market_ma20"])
        & market["market_ma20"].gt(market["market_ma20_previous"])
    ).fillna(False)
    return market


def _apply_cooldown(result, cooldown):
    accepted = np.zeros(len(result), dtype=bool)
    rejected = np.zeros(len(result), dtype=bool)
    previous = np.full(len(result), np.nan)
    raw = result["raw_signal"].to_numpy(dtype=bool)
    source_rows = result["source_row_index"].to_numpy(dtype=float, na_value=np.nan)
    for positions in result.groupby("code", sort=False).indices.values():
        last_accepted = None
        for position in positions[raw[positions]]:
            if not np.isfinite(source_rows[position]):
                rejected[position] = True
                continue
            source_row = int(source_rows[position])
            if last_accepted is not None:
                previous[position] = last_accepted
            if last_accepted is None or source_row - last_accepted >= cooldown:
                accepted[position] = True
                last_accepted = source_row
            else:
                rejected[position] = True
    result["accepted_signal"] = accepted
    result["cooldown_rejected"] = rejected
    result["previous_accepted_row_index"] = previous


def compute_legacy_a_signals(
    context_panel, main_panel, index_panel, config
) -> pd.DataFrame:
    """Compute A's fixed close signals without exits, sizing, or account returns."""
    candidate = _candidate_config(config)
    required_context = {
        *KEYS,
        "signal_open",
        "signal_high",
        "signal_low",
        "signal_close",
        "volume",
        "amount",
        "is_main_board",
        "rank_eligible",
    }
    missing = required_context - set(context_panel)
    if missing:
        raise ValueError(f"Missing candidate-A context columns: {sorted(missing)}")
    if not set(KEYS).issubset(main_panel) or not set(KEYS).issubset(index_panel):
        raise ValueError("Candidate-A inputs require date/code keys")

    context = context_panel.copy()
    context["date"] = context["date"].astype(str)
    if context.duplicated(KEYS).any():
        raise ValueError("Duplicate candidate-A context key")
    context = _add_stock_indicators(context, candidate)
    cross_section = _cross_section(context, candidate)

    main = main_panel.copy()
    main["date"] = main["date"].astype(str)
    if main.duplicated(KEYS).any():
        raise ValueError("Duplicate candidate-A main-panel key")
    main = main.drop(columns=["is_main_board", *SIGNAL_FIELDS], errors="ignore")
    source = context[[*KEYS, "is_main_board", *SIGNAL_FIELDS, *INDICATOR_COLUMNS]]
    result = main.merge(
        source,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    result["context_key_present"] = result["_merge"].eq("both")
    result = result.drop(columns="_merge").merge(
        cross_section,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    result = result.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    for column in ("m1", "m2", "m3"):
        result[column] = result[column].eq(1).fillna(False)
    result["hit_count"] = result["hit_count"].fillna(0).astype(int)
    result["source_row_index"] = result["source_row_index"].astype("Int64")

    market = _market_panel(index_panel, candidate)
    result["market_code"] = MARKET_CODE
    result = result.merge(
        market,
        on=["date", "market_code"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).drop(columns="market_code")
    result["market_ok"] = result["market_ok"].eq(True).fillna(False)
    result["stock_filter_ok"] = result["stock_filter_ok"].eq(True).fillna(False)

    _, enough_methods = _method_hits(result["m1"], result["m2"], result["m3"])
    stock_trend = result["signal_close"].gt(result["ma20"]) & result["ma20"].gt(
        result["ma20_previous"]
    )
    liquidity = result["amount_ma20"].ge(candidate.amount_ma20_min)
    result["raw_signal"] = (
        stock_trend.fillna(False)
        & liquidity.fillna(False)
        & enough_methods
        & result["market_ok"]
        & result["stock_filter_ok"]
    ).astype(bool)
    volume_ratio = result["volume"].div(result["volume_ma20"])
    result["legacy_rank_score"] = (
        result["hit_count"] * candidate.score_hit_weight
        + volume_ratio.fillna(0) * candidate.score_volume_weight
    )
    result["rank_return_120"] = result["legacy_rank_score"]

    _apply_cooldown(result, candidate.cooldown)
    in_window = result["date"].between(config.start_date, config.end_date)
    main_board = result["is_main_board"].eq(True).fillna(False)
    result["proposed_entry"] = result["accepted_signal"] & in_window & main_board
    valid_entry = candidate_entry_valid(result)
    result["entry_signal"] = (
        result["proposed_entry"] & valid_entry & result["is_st"].eq(0).fillna(False)
    ).astype(bool)
    result["stop_ref"] = result["low10"]
    result["stop_pct"] = candidate.entry_stop_pct
    return result.sort_values(["code", "date"], kind="stable").reset_index(drop=True)


def audit_legacy_a_entries(signals, context_panel, config):
    """Audit only actually accepted/proposed entries, one explicit key per row."""
    proposed = signals.loc[signals["proposed_entry"].eq(True), KEYS].copy()
    proposed["date"] = proposed["date"].astype(str)
    if proposed.duplicated(KEYS).any():
        raise ValueError("Duplicate proposed-entry key")
    right = context_panel.reindex(columns=[*KEYS, *A_ENTRY_REQUIRED_FIELDS]).copy()
    right["date"] = right["date"].astype(str)
    if right.duplicated(KEYS).any():
        raise ValueError("Duplicate entry-audit context key")
    audit = proposed.merge(
        right,
        on=KEYS,
        how="left",
        validate="one_to_one",
        sort=True,
        indicator=True,
    )
    complete = audit[list(A_ENTRY_REQUIRED_FIELDS)].notna().all(axis=1) & audit[
        "_merge"
    ].eq("both")
    audit["required_fields_complete"] = complete
    valid = candidate_entry_valid(audit) & audit["_merge"].eq("both")
    audit["candidate_entry_valid"] = valid.astype(bool)
    if len(audit):
        audit["missing_fields"] = audit.apply(
            lambda row: "|".join(
                field for field in A_ENTRY_REQUIRED_FIELDS if pd.isna(row[field])
            ),
            axis=1,
        )
    else:
        audit["missing_fields"] = pd.Series(dtype="string")
    denominator = len(audit)
    numerator = int(valid.sum())
    coverage = numerator / denominator if denominator else 0.0
    threshold = config.required_entry_coverage
    passes = bool(
        denominator > 0
        and coverage >= threshold
        and audit["_merge"].eq("both").all()
    )
    summary = {
        "denominator": denominator,
        "valid_numerator": numerator,
        "coverage": coverage,
        "threshold": threshold,
        "passes": passes,
        "candidate_a_status": "EVALUABLE" if passes else "NOT_EVALUABLE",
    }
    return audit.drop(columns="_merge"), summary
