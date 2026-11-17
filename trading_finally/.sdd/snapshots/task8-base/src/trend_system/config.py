import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ResearchConfig:
    hithink_db_path: str
    baostock_db_path: str
    start_date: str = "2016-10-13"
    end_date: str = "2026-10-29"
    initial_cash: int = 20000
    slots: int = 3
    max_weight: float = 0.20
    max_exposure: float = 0.60
    entry_lookback: int = 120
    trend_window: int = 200
    trend_slope_window: int = 20
    exit_lookback: int = 20
    commission_rate: float = 0.00025
    min_commission: float = 5
    slippage_bps: int = 10
    random_seed: int = 20261101


_CONFIG_FIELDS = frozenset(field.name for field in fields(ResearchConfig))
_DATABASE_PATH_FIELDS = ("hithink_db_path", "baostock_db_path")


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def load_config(path: str | Path) -> ResearchConfig:
    try:
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file, parse_constant=_reject_non_json_constant)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load config: {path}") from error

    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")

    unknown = set(data) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")

    for field in _DATABASE_PATH_FIELDS:
        if not isinstance(data.get(field), str) or not data[field]:
            raise ValueError(f"Missing database path: {field}")

    return ResearchConfig(**data)
