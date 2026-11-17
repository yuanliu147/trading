"""Strict loader for the recovered candidate-B research configuration."""

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path

from .legacy_config import EarlyTrendBreadthConfig, LegacySource


@dataclass(frozen=True)
class LegacyBRecoveryConfig:
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
    industry_coverage_min: float
    required_entry_coverage: float
    status: str
    industry_source: LegacySource
    candidates: tuple[EarlyTrendBreadthConfig, ...]


OFFICIAL_CONFIG_SHA256 = "d77b88c7303b1567406e0f396b242a7b991d8998186ee704ba0f45f766552c1e"
EXPECTED_SOURCES = {
    "/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/02_代码/30b_组合回测_含退市.py":
        "8406120aedd2ad21cbbca35e87fb784262434730b3399697b7afbe822b4686e8",
    "/Users/admin/WorkSpace/trading/2026-10-20_涨幅趋势_全维度定制/02_代码/10b_构建全A面板_含退市.py":
        "d45892c97a2995d1c3fc7349c2dd9162be6b1d11825a330673136c88300f3d37",
}
EXPECTED_INDUSTRY = {
    "/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据/申万行业分类_时点.csv":
        "3ebc1a6c22c1b17f137470bb0c040c31b055b5d805fe29f4b55ce53d8ee62872",
}
_TOP_FIELDS = frozenset(field.name for field in fields(LegacyBRecoveryConfig))
_CANDIDATE_FIELDS = frozenset(field.name for field in fields(EarlyTrendBreadthConfig))
_SOURCE_FIELDS = frozenset(("path", "sha256"))


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


def _require_exact(mapping: dict, expected: frozenset, where: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be an object")
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing or unknown:
        raise ValueError(f"{where} fields mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite")
    return float(value)


def _integer(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer")
    return value


def _source(value, where: str) -> LegacySource:
    _require_exact(value, _SOURCE_FIELDS, where)
    if not isinstance(value["path"], str) or not isinstance(value["sha256"], str):
        raise ValueError(f"{where} path/sha256 must be strings")
    return LegacySource(path=value["path"], sha256=value["sha256"])


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_b_config(path: str | Path) -> LegacyBRecoveryConfig:
    if _sha256(path) != OFFICIAL_CONFIG_SHA256:
        raise ValueError("B recovery config hash does not match the frozen official config")
    raw = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    _require_exact(raw, _TOP_FIELDS, "root")
    if raw["status"] not in ("PENDING_DATA_GATE", "EVALUABLE"):
        raise ValueError("B recovery status must be PENDING_DATA_GATE or EVALUABLE")
    if not isinstance(raw["candidates"], list) or len(raw["candidates"]) != 1:
        raise ValueError("B recovery requires exactly one candidate")
    candidate_raw = raw["candidates"][0]
    _require_exact(candidate_raw, _CANDIDATE_FIELDS, "candidate")
    if candidate_raw["name"] != "early_trend_breadth":
        raise ValueError("B recovery candidate must be early_trend_breadth")
    if candidate_raw["evaluation_status"] not in ("PENDING_DATA_GATE", "EVALUABLE"):
        raise ValueError("B candidate evaluation_status invalid")
    if candidate_raw["not_evaluable_fields"]:
        raise ValueError("B recovery requires no not_evaluable_fields")
    sources = tuple(
        _source(source, f"candidate.sources[{index}]")
        for index, source in enumerate(candidate_raw["sources"])
    )
    if {source.path: source.sha256 for source in sources} != EXPECTED_SOURCES:
        raise ValueError("B recovery candidate source set/hashes do not match the frozen lineage")
    industry = _source(raw["industry_source"], "industry_source")
    if {industry.path: industry.sha256} != EXPECTED_INDUSTRY:
        raise ValueError("B recovery industry source does not match the frozen lineage")
    not_evaluable = candidate_raw["not_evaluable_fields"]
    if not isinstance(not_evaluable, list) or not all(isinstance(x, str) for x in not_evaluable):
        raise ValueError("not_evaluable_fields must be a string list")
    for key in ("entry_day_counts_toward_hold", "suspension_counts_toward_hold",
                "lu_filter", "gap_filter"):
        if not isinstance(candidate_raw[key], bool):
            raise ValueError(f"candidate.{key} must be boolean")
    if not isinstance(candidate_raw["ranking"], str) or not isinstance(candidate_raw["exit_priority"], list):
        raise ValueError("candidate ranking/exit_priority type invalid")
    candidate = EarlyTrendBreadthConfig(
        name=candidate_raw["name"],
        evaluation_status=candidate_raw["evaluation_status"],
        sources=sources,
        not_evaluable_fields=tuple(not_evaluable),
        eligibility_start=candidate_raw["eligibility_start"],
        max_slots=_integer(candidate_raw["max_slots"], "max_slots"),
        max_weight=_number(candidate_raw["max_weight"], "max_weight"),
        risk_per_trade=_number(candidate_raw["risk_per_trade"], "risk_per_trade"),
        heat_reject_threshold=_number(candidate_raw["heat_reject_threshold"], "heat_reject_threshold"),
        max_hold=_integer(candidate_raw["max_hold"], "max_hold"),
        entry_day_counts_toward_hold=candidate_raw["entry_day_counts_toward_hold"],
        suspension_counts_toward_hold=candidate_raw["suspension_counts_toward_hold"],
        exit_priority=tuple(candidate_raw["exit_priority"]),
        market_breadth_threshold=_number(candidate_raw["market_breadth_threshold"], "market_breadth_threshold"),
        trend_age_min=_integer(candidate_raw["trend_age_min"], "trend_age_min"),
        trend_age_max=_integer(candidate_raw["trend_age_max"], "trend_age_max"),
        actual_close_min=_number(candidate_raw["actual_close_min"], "actual_close_min"),
        atr_pct_max=_number(candidate_raw["atr_pct_max"], "atr_pct_max"),
        dd60_threshold=_number(candidate_raw["dd60_threshold"], "dd60_threshold"),
        sector_breadth_min=_number(candidate_raw["sector_breadth_min"], "sector_breadth_min"),
        sector_ret20_threshold=_number(candidate_raw["sector_ret20_threshold"], "sector_ret20_threshold"),
        amount_ma20_min=_integer(candidate_raw["amount_ma20_min"], "amount_ma20_min"),
        ranking=candidate_raw["ranking"],
        lu_filter=candidate_raw["lu_filter"],
        gap_filter=candidate_raw["gap_filter"],
        gap_max=_number(candidate_raw["gap_max"], "gap_max"),
        catastrophe_stop_pct=_number(candidate_raw["catastrophe_stop_pct"], "catastrophe_stop_pct"),
        risk_atr_multiple=_integer(candidate_raw["risk_atr_multiple"], "risk_atr_multiple"),
        risk_size_min=_number(candidate_raw["risk_size_min"], "risk_size_min"),
        risk_size_max=_number(candidate_raw["risk_size_max"], "risk_size_max"),
        min_position_weight=_number(candidate_raw["min_position_weight"], "min_position_weight"),
    )
    return LegacyBRecoveryConfig(
        start_date=raw["start_date"],
        end_date=raw["end_date"],
        initial_cash=_integer(raw["initial_cash"], "initial_cash"),
        commission_rate=_number(raw["commission_rate"], "commission_rate"),
        min_commission=_integer(raw["min_commission"], "min_commission"),
        slippage_bps=_integer(raw["slippage_bps"], "slippage_bps"),
        random_seed=_integer(raw["random_seed"], "random_seed"),
        max_main_drawdown=_number(raw["max_main_drawdown"], "max_main_drawdown"),
        unrejected_max_drawdown=_number(raw["unrejected_max_drawdown"], "unrejected_max_drawdown"),
        random_paths=_integer(raw["random_paths"], "random_paths"),
        random_positive_min=_integer(raw["random_positive_min"], "random_positive_min"),
        industry_coverage_min=_number(raw["industry_coverage_min"], "industry_coverage_min"),
        required_entry_coverage=_number(raw["required_entry_coverage"], "required_entry_coverage"),
        status=raw["status"],
        industry_source=industry,
        candidates=(candidate,),
    )
