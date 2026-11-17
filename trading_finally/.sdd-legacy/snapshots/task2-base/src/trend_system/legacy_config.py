"""Strict loader for the frozen legacy-candidate research configuration."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class LegacySource:
    path: str
    sha256: str


@dataclass(frozen=True)
class ThreeOfTwoLowVolConfig:
    name: str
    evaluation_status: str
    sources: tuple[LegacySource, ...]
    cooldown: int
    max_slots: int
    max_weight: float
    max_new_per_day: int
    market_ma_window: int
    stock_ma_fast: int
    stock_ma_slow: int
    amount_ma20_min: int
    m2_ema_fast: int
    m2_ema_slow: int
    m2_signal_ema: int
    m3_breakout_window: int
    m3_volume_window: int
    m3_volume_ratio_min: float
    low10_window: int
    low10_includes_signal_day: bool
    entry_stop_pct: float
    skip_if_stop_at_or_above_entry: bool
    max_hold: int
    entry_day_counts_toward_hold: bool
    suspension_counts_toward_hold: bool
    exit_priority: tuple[str, ...]
    stock_history_tail: int
    stock_history_min: int
    stock_amount_ma60_min: int
    stock_vol_window: int
    stock_range_window: int
    stock_activity_short_window: int
    stock_activity_long_window: int
    stock_limit_window: int
    stock_gap_window: int
    stock_gap_threshold: float
    stock_main_limit_threshold: float
    stock_growth_limit_threshold: float
    s_vol_max: float
    s_heat_min: float
    score_hit_weight: int
    score_volume_weight: float


@dataclass(frozen=True)
class EarlyTrendBreadthConfig:
    name: str
    evaluation_status: str
    sources: tuple[LegacySource, ...]
    not_evaluable_fields: tuple[str, ...]
    eligibility_start: str
    max_slots: int
    max_weight: float
    risk_per_trade: float
    heat_reject_threshold: float
    max_hold: int
    entry_day_counts_toward_hold: bool
    suspension_counts_toward_hold: bool
    exit_priority: tuple[str, ...]
    market_breadth_threshold: float
    trend_age_min: int
    trend_age_max: int
    actual_close_min: int
    atr_pct_max: float
    dd60_threshold: float
    sector_breadth_min: float
    sector_ret20_threshold: float
    amount_ma20_min: int
    ranking: str
    lu_filter: bool
    gap_filter: bool
    gap_max: float
    catastrophe_stop_pct: float
    risk_atr_multiple: int
    risk_size_min: float
    risk_size_max: float
    min_position_weight: float


@dataclass(frozen=True)
class LegacyResearchConfig:
    hithink_db_path: str
    baostock_db_path: str
    start_date: str
    end_date: str
    initial_cash: int
    commission_rate: float
    min_commission: int
    slippage_bps: int
    random_seed: int
    max_main_drawdown: float
    unrejected_max_drawdown: float
    random_paths: int
    random_positive_min: int
    stress_cost_multiplier: int
    stress_slippage_bps: int
    rolling_three_year_positive_threshold: float
    point_in_time_coverage_min: float
    industry_coverage_min: float
    required_entry_coverage: float
    candidates: tuple[ThreeOfTwoLowVolConfig | EarlyTrendBreadthConfig, ...]


_CONFIG_FIELDS = frozenset(field.name for field in fields(LegacyResearchConfig))
_CANDIDATE_CLASSES = {
    "three_of_two_low_vol": ThreeOfTwoLowVolConfig,
    "early_trend_breadth": EarlyTrendBreadthConfig,
}
_CANDIDATE_ORDER = tuple(_CANDIDATE_CLASSES)
_DATABASE_PATH_FIELDS = ("hithink_db_path", "baostock_db_path")


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _field_names(config_class) -> frozenset[str]:
    return frozenset(field.name for field in fields(config_class))


def _check_fields(data, config_class, kind: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{kind.capitalize()} must be a JSON object")
    expected = _field_names(config_class)
    unknown = set(data) - expected
    if unknown:
        raise ValueError(f"Unknown {kind} fields: {sorted(unknown)}")
    missing = expected - set(data)
    if missing:
        raise ValueError(f"Missing {kind} fields: {sorted(missing)}")


def _scalar_value(name: str, value, expected_type):
    if expected_type is str:
        if type(value) is not str:
            raise ValueError(f"Invalid field type: {name}")
        return value
    if expected_type is bool:
        if type(value) is not bool:
            raise ValueError(f"Invalid field type: {name}")
        return value
    if expected_type is int:
        if type(value) is not int:
            raise ValueError(f"Invalid field type: {name}")
        return value
    if expected_type is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid field type: {name}")
        return float(value)
    raise TypeError(f"Unsupported config field type: {name}")


def _string_tuple(name: str, value) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ValueError(f"Invalid field type: {name}")
    return tuple(value)


def _load_source(data) -> LegacySource:
    _check_fields(data, LegacySource, "source")
    source = LegacySource(
        path=_scalar_value("path", data["path"], str),
        sha256=_scalar_value("sha256", data["sha256"], str),
    )
    if not source.path or not source.sha256:
        raise ValueError("Source path and SHA256 must be non-empty")
    return source


def _load_candidate(data, expected_name: str):
    candidate_class = _CANDIDATE_CLASSES[expected_name]
    _check_fields(data, candidate_class, "candidate")
    values = {}
    for field in fields(candidate_class):
        value = data[field.name]
        if field.name == "sources":
            if not isinstance(value, list):
                raise ValueError("Invalid field type: sources")
            values[field.name] = tuple(_load_source(source) for source in value)
        elif field.type == tuple[str, ...]:
            values[field.name] = _string_tuple(field.name, value)
        else:
            values[field.name] = _scalar_value(field.name, value, field.type)
    return candidate_class(**values)


def _candidate_names(data) -> dict[str, dict]:
    if not isinstance(data, list) or len(data) != 2:
        raise ValueError("Legacy config must contain exactly two candidates")
    by_name = {}
    for candidate in data:
        if not isinstance(candidate, dict) or type(candidate.get("name")) is not str:
            raise ValueError("Each candidate must be an object with a string name")
        name = candidate["name"]
        if name not in _CANDIDATE_CLASSES or name in by_name:
            raise ValueError(f"Invalid or duplicate candidate: {name}")
        by_name[name] = candidate
    if set(by_name) != set(_CANDIDATE_ORDER):
        raise ValueError("Legacy config candidate set does not match the frozen pair")
    return by_name


def load_legacy_config(path: str | Path) -> LegacyResearchConfig:
    try:
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(
                file,
                parse_constant=_reject_non_json_constant,
                object_pairs_hook=_reject_duplicate_fields,
            )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load legacy config: {path}") from error

    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")

    unknown = set(data) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    missing = _CONFIG_FIELDS - set(data)
    if missing:
        raise ValueError(f"Missing config fields: {sorted(missing)}")

    values = {}
    for field in fields(LegacyResearchConfig):
        if field.name == "candidates":
            continue
        values[field.name] = _scalar_value(field.name, data[field.name], field.type)

    for field in _DATABASE_PATH_FIELDS:
        if not values[field]:
            raise ValueError(f"Missing database path: {field}")

    candidates_by_name = _candidate_names(data["candidates"])
    values["candidates"] = tuple(
        _load_candidate(candidates_by_name[name], name) for name in _CANDIDATE_ORDER
    )
    return LegacyResearchConfig(**values)
