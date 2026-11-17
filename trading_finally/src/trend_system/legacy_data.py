"""Frozen-panel point-in-time context and feasibility gates for legacy candidates."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .data import PRICE_THRESHOLD
from .legacy_config import LegacyResearchConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = PROJECT_ROOT / "sql" / "legacy_context.sql"
RAW_PRICE_FIELDS = ("open", "high", "low", "close")
SIGNAL_FIELDS = tuple(f"signal_{field}" for field in RAW_PRICE_FIELDS)
CONTEXT_REQUIRED_FIELDS = (
    *RAW_PRICE_FIELDS,
    "prev_close",
    *SIGNAL_FIELDS,
    "volume",
    "amount",
    "trade_status",
    "is_st",
    "ipo_date",
)
A_ENTRY_REQUIRED_FIELDS = (*CONTEXT_REQUIRED_FIELDS, "execution_conflict")
FROZEN_INPUT_MANIFEST = {
    "full_a_original": {
        "path": PROJECT_ROOT / "data" / "legacy_full_a_panel.parquet",
        "sha256": "8183acce5607b04946d80c05477fc26fd60791748a590200e5096d2d0e525a85",
    },
    "main": {
        "path": PROJECT_ROOT / "data" / "main_panel.parquet",
        "sha256": "6a51ed0e7958aa1549dcf156e4387badcfced0236fd812d2bc1fd894aa3acc1a",
    },
    "hithink_signal": {
        "path": PROJECT_ROOT / "data" / "hithink_signal_panel.parquet",
        "sha256": "6c7c8c070eca05e273fa67463787d085bbc8c8fbe38ed8ffc00d1b7245ccc53d",
    },
    "industry": {
        "path": PROJECT_ROOT / "data" / "legacy_industry_panel.parquet",
        "sha256": "6f49b7207340b863380c7c6c389224024ab639def8c64e3ce3b3ee9530a2f53d",
    },
    "pre_amendment_audit": {
        "path": PROJECT_ROOT / "results" / "legacy_context_audit_pre_amendment.json",
        "sha256": "c30375094d8584131e529a50833d69ab5fabe479042e68bc05d2bc5f1a4ce899",
    },
}


@dataclass(frozen=True)
class LegacyContextResult:
    full_a_panel_path: Path
    source_full_a_panel_path: Path
    industry_panel_path: Path
    audit_path: Path
    full_a_rows: int
    industry_rows: int
    candidate_a_status: str
    candidate_b_status: str


def _complete_mask(frame: pd.DataFrame, required_fields) -> pd.Series:
    if any(field not in frame for field in required_fields):
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[list(required_fields)].notna().all(axis=1)


def point_in_time_industry(records: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """Select only industry records whose explicit effective interval contains the date."""
    required = {"code", "industry", "valid_from", "valid_to"}
    missing = required - set(records)
    if missing:
        raise ValueError(f"Missing industry columns: {sorted(missing)}")
    selected = records.loc[
        records["valid_from"].notna()
        & (records["valid_from"].astype(str) <= as_of_date)
        & (
            records["valid_to"].isna()
            | (records["valid_to"].astype(str) >= as_of_date)
        )
    ].copy()
    if selected["code"].duplicated().any():
        raise ValueError("Overlapping point-in-time industry records")
    return selected.reset_index(drop=True)


def daily_coverage(
    active_universe: pd.DataFrame,
    panel: pd.DataFrame,
    required_fields,
) -> pd.DataFrame:
    """Use the active date/code grid as the fixed denominator for every day."""
    if active_universe.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate active-universe key")
    if panel.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate panel key")
    active = active_universe[["date", "code"]].copy()
    complete = panel.loc[_complete_mask(panel, required_fields), ["date", "code"]]
    complete = complete.merge(active, on=["date", "code"], how="inner")
    denominator = active.groupby("date", as_index=False).size().rename(
        columns={"size": "denominator"}
    )
    numerator = complete.groupby("date", as_index=False).size().rename(
        columns={"size": "numerator"}
    )
    result = denominator.merge(numerator, on="date", how="left", validate="one_to_one")
    result["numerator"] = result["numerator"].fillna(0).astype(int)
    result["coverage"] = result["numerator"] / result["denominator"]
    return result[["date", "denominator", "numerator", "coverage"]]


def coverage_gate_passes(daily: pd.DataFrame, minimum: float) -> bool:
    """Every active day must meet the threshold; pooled coverage is not a substitute."""
    return bool(
        len(daily)
        and daily["denominator"].gt(0).all()
        and daily["coverage"].notna().all()
        and daily["coverage"].ge(minimum).all()
    )


def required_field_coverage(panel: pd.DataFrame, required_fields) -> float:
    if not len(panel):
        return 0.0
    return float(_complete_mask(panel, required_fields).mean())


def assert_expected_trading_dates(actual_dates, expected_dates) -> None:
    """Fail closed when the panel drops or adds an entire expected trading day."""
    actual = set(actual_dates)
    expected = set(expected_dates)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ValueError(f"Missing expected trading dates: {missing}")
    if unexpected:
        raise ValueError(f"Unexpected trading dates: {unexpected}")


def refresh_context_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Exempt NULL liquidity only for normal suspensions, never for ranking eligibility."""
    result = frame.copy()
    valid_raw = result[[*RAW_PRICE_FIELDS, "prev_close"]].gt(0).all(axis=1)
    valid_signal = result[list(SIGNAL_FIELDS)].gt(0).all(axis=1)
    result["data_valid"] = valid_raw & valid_signal
    normal_trading = result["trade_status"].eq(1)
    normal_suspension = result["trade_status"].eq(0)
    base_complete = _complete_mask(
        result,
        (
            *RAW_PRICE_FIELDS,
            "prev_close",
            *SIGNAL_FIELDS,
            "trade_status",
            "is_st",
            "ipo_date",
        ),
    )
    trading_liquidity_valid = (
        normal_trading
        & result[["volume", "amount"]].notna().all(axis=1)
        & result["volume"].ge(0)
        & result["amount"].ge(0)
    )
    suspension_liquidity_valid = (
        normal_suspension
        & (result["volume"].isna() | result["volume"].ge(0))
        & (result["amount"].isna() | result["amount"].ge(0))
    )
    result["context_complete"] = (
        result["data_valid"]
        & base_complete
        & (trading_liquidity_valid | suspension_liquidity_valid)
    )
    result["rank_eligible"] = result["context_complete"] & normal_trading
    return result


def candidate_entry_valid(frame: pd.DataFrame) -> pd.Series:
    """Strict reusable validity predicate for Task 3's actually triggered entries."""
    complete = _complete_mask(frame, A_ENTRY_REQUIRED_FIELDS)
    if not complete.any():
        return pd.Series(False, index=frame.index, dtype=bool)
    valid = (
        complete
        & frame[[*RAW_PRICE_FIELDS, "prev_close", *SIGNAL_FIELDS]].gt(0).all(axis=1)
        & frame["trade_status"].eq(1)
        & frame["volume"].gt(0)
        & frame["amount"].gt(0)
        & frame["execution_conflict"].eq(False).fillna(False)
    )
    return valid.fillna(False).astype(bool)


def complete_hfq_signals(
    panel: pd.DataFrame, hithink: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill only NULL hfq fields at exact keys from positive frozen hfq values."""
    if panel.duplicated(["date", "code"]).any() or hithink.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate hfq completion key")
    left = panel.copy()
    left["date"] = left["date"].astype(str)
    right = hithink[["date", "code", *SIGNAL_FIELDS]].copy()
    right["date"] = right["date"].astype(str)
    right = right.loc[right[list(SIGNAL_FIELDS)].gt(0).all(axis=1)]
    helper_names = {field: f"_hithink_{field}" for field in SIGNAL_FIELDS}
    right = right.rename(columns=helper_names)
    merged = left.merge(
        right,
        on=["date", "code"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    normal_trading = merged["trade_status"].eq(1)
    fill_masks = {}
    for field in SIGNAL_FIELDS:
        helper = helper_names[field]
        fill_masks[field] = (
            normal_trading & merged[field].isna() & merged[helper].notna()
        )
        merged.loc[fill_masks[field], field] = merged.loc[fill_masks[field], helper]
    filled = pd.DataFrame(fill_masks).any(axis=1)
    audit = merged.loc[filled, ["date", "code"]].copy()
    if len(audit):
        filled_fields = pd.DataFrame(fill_masks).loc[filled]
        audit["completed_fields"] = filled_fields.apply(
            lambda row: "|".join(field for field in SIGNAL_FIELDS if row[field]),
            axis=1,
        )
    else:
        audit["completed_fields"] = pd.Series(dtype="string")
    audit["source"] = "frozen_hithink_signal_panel"
    return merged.drop(columns=list(helper_names.values())), audit.reset_index(drop=True)


def evaluate_candidate_statuses(
    config: LegacyResearchConfig, gates: dict[str, bool]
) -> dict[str, str]:
    """Apply Task 2 gates; A's actual-triggered-entry gate belongs to Task 3."""
    candidates = {candidate.name: candidate for candidate in config.candidates}
    candidate_a = candidates["three_of_two_low_vol"]
    candidate_b = candidates["early_trend_breadth"]
    a_pass = (
        candidate_a.evaluation_status == "PENDING_DATA_GATE"
        and gates["full_a_point_in_time"]
    )
    b_pass = (
        candidate_b.evaluation_status == "PENDING_DATA_GATE"
        and not candidate_b.not_evaluable_fields
        and gates["full_a_point_in_time"]
        and gates["industry_point_in_time"]
        and gates["b_entry_fields"]
    )
    return {
        "three_of_two_low_vol": "PENDING_DATA_GATE" if a_pass else "NOT_EVALUABLE",
        "early_trend_breadth": "EVALUABLE" if b_pass else "NOT_EVALUABLE",
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_inputs() -> dict:
    result = {}
    for name, expected in FROZEN_INPUT_MANIFEST.items():
        path = expected["path"]
        actual = _sha256(path)
        if actual != expected["sha256"]:
            raise ValueError(f"Frozen input manifest mismatch: {name}")
        result[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return result


def _daily_records(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": str(row.date),
            "denominator": int(row.denominator),
            "numerator": int(row.numerator),
            "coverage": float(row.coverage),
        }
        for row in frame.itertuples(index=False)
    ]


def _coverage_summary(frame: pd.DataFrame, threshold: float) -> dict:
    return {
        "passes": coverage_gate_passes(frame, threshold),
        "minimum": float(frame["coverage"].min()),
        "days_below_threshold": int(frame["coverage"].lt(threshold).sum()),
        "daily": _daily_records(frame),
    }


def _completion_reason_counts(original: pd.DataFrame, completed: pd.DataFrame, config) -> dict:
    original_main = original["is_main_board"] & original["date"].between(
        config.start_date, config.end_date
    )
    completed_main = completed["is_main_board"] & completed["date"].between(
        config.start_date, config.end_date
    )
    before_signal_missing = ~original[list(SIGNAL_FIELDS)].gt(0).all(axis=1)
    after_signal_missing = ~completed[list(SIGNAL_FIELDS)].gt(0).all(axis=1)
    normal_before = original["trade_status"].eq(1)
    normal_after = completed["trade_status"].eq(1)
    suspended_null = (
        completed_main
        & completed["trade_status"].eq(0)
        & (completed["volume"].isna() | completed["amount"].isna())
    )
    return {
        "normal_trading_main_rows_with_incomplete_signal_before": int(
            (original_main & normal_before & before_signal_missing).sum()
        ),
        "normal_trading_main_rows_with_incomplete_signal_after": int(
            (completed_main & normal_after & after_signal_missing).sum()
        ),
        "normal_suspension_main_rows_with_null_liquidity_unchanged": int(
            suspended_null.sum()
        ),
        "remaining_context_incomplete_rows_all_a": int(
            (~completed.loc[
                completed["date"].between(config.start_date, config.end_date),
                "context_complete",
            ]).sum()
        ),
        "remaining_reason_counts_not_additive": {
            "raw_or_prev_close_not_positive": int(
                (~completed[[*RAW_PRICE_FIELDS, "prev_close"]].gt(0).all(axis=1)).sum()
            ),
            "signal_ohlc_not_positive_after_completion": int(after_signal_missing.sum()),
            "normal_trading_null_liquidity": int(
                (
                    normal_after
                    & (completed["volume"].isna() | completed["amount"].isna())
                ).sum()
            ),
            "negative_liquidity": int(
                (completed["volume"].lt(0) | completed["amount"].lt(0)).sum()
            ),
            "unsupported_or_missing_trade_status": int(
                (~completed["trade_status"].isin([0, 1])).sum()
            ),
        },
    }


def build_legacy_context(config: LegacyResearchConfig) -> LegacyContextResult:
    """Amend context exclusively from frozen, hash-verified Parquet inputs."""
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    completed_path = data_dir / "legacy_full_a_panel_completed.parquet"
    audit_path = results_dir / "legacy_context_audit.json"
    temp_path = completed_path.with_suffix(".parquet.tmp")
    inputs_before = _verify_frozen_inputs()
    original_audit = json.loads(
        FROZEN_INPUT_MANIFEST["pre_amendment_audit"]["path"].read_text(encoding="utf-8")
    )
    original = pd.read_parquet(FROZEN_INPUT_MANIFEST["full_a_original"]["path"])
    original["date"] = original["date"].astype(str)
    if original.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate original full-A key")
    hithink = pd.read_parquet(
        FROZEN_INPUT_MANIFEST["hithink_signal"]["path"],
        columns=["date", "code", *SIGNAL_FIELDS],
    )
    completed, completion_audit = complete_hfq_signals(original, hithink)
    completed = refresh_context_flags(completed)

    main_dates = pd.read_parquet(
        FROZEN_INPUT_MANIFEST["main"]["path"], columns=["date"]
    )["date"].astype(str)
    expected_dates = set(
        main_dates.loc[main_dates.between(config.start_date, config.end_date)]
    )
    performance = completed["date"].between(config.start_date, config.end_date)
    actual_dates = set(completed.loc[performance, "date"])
    assert_expected_trading_dates(actual_dates, expected_dates)

    active = completed.loc[performance, ["date", "code"]]
    complete_rows = completed.loc[performance & completed["context_complete"]]
    after_daily = daily_coverage(active, complete_rows, ("close",)).sort_values("date")
    before_daily = pd.DataFrame(original_audit["coverage"]["full_a"]["daily"])
    assert_expected_trading_dates(set(before_daily["date"]), expected_dates)
    denominator_check = before_daily[["date", "denominator"]].merge(
        after_daily[["date", "denominator"]],
        on="date",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    if not denominator_check["denominator_before"].equals(
        denominator_check["denominator_after"]
    ):
        raise ValueError("Amendment changed the active-universe denominator")

    before_summary = _coverage_summary(before_daily, config.point_in_time_coverage_min)
    after_summary = _coverage_summary(after_daily, config.point_in_time_coverage_min)
    industry_daily = after_daily[["date", "denominator"]].copy()
    industry_daily["numerator"] = 0
    industry_daily["coverage"] = 0.0
    gates = {
        "full_a_point_in_time": after_summary["passes"],
        "a_entry_fields": False,
        "industry_point_in_time": False,
        "b_entry_fields": False,
    }
    statuses = evaluate_candidate_statuses(config, gates)

    completed.to_parquet(temp_path, index=False)
    inputs_after = _verify_frozen_inputs()
    if inputs_after != inputs_before:
        raise ValueError("Frozen input changed during amendment")
    temp_path.replace(completed_path)

    main_performance = performance & completed["is_main_board"]
    strict_valid_diagnostic = candidate_entry_valid(completed.loc[main_performance])
    industry = pd.read_parquet(FROZEN_INPUT_MANIFEST["industry"]["path"])
    audit = {
        "config": asdict(config),
        "amendment": {
            "round": 1,
            "approved_on": "2026-11-04",
            "source_level_rebuild_reproducible": False,
            "frozen_input_reproducible": True,
            "source_databases_connected": False,
            "current_source_hashes_computed": False,
            "original_failure_evidence_preserved": True,
            "pre_amendment_audit": str(FROZEN_INPUT_MANIFEST["pre_amendment_audit"]["path"]),
        },
        "source_level_rebuild_reproducible": False,
        "frozen_input_reproducible": True,
        "construction": {
            "input_mode": "frozen hash-verified Parquet only; no source database connection",
            "performance_start_date": config.start_date,
            "end_date": config.end_date,
            "expected_trading_dates": "distinct dates from frozen main_panel.parquet, never self-derived from full-A rows",
            "active_denominator": "unchanged original ipo_date <= date < out_date grid",
            "suspension_rule": "trade_status=0 may have NULL volume/amount; all other context remains valid; rank_eligible=false",
            "normal_trading_rule": "trade_status=1 requires complete nonnegative volume/amount and valid raw/hfq context",
            "entry_gate": "deferred to Task 3 actual triggered and proposed entries before any account returns",
            "entry_valid_predicate": "complete fields, positive raw/hfq prices, trade_status=1, positive volume/amount, execution_conflict=false",
            "price_threshold": PRICE_THRESHOLD,
            "signals_implemented": False,
            "returns_run": False,
        },
        "input_panels": inputs_before,
        "sql": {
            "path": str(SQL_PATH),
            "sha256": _sha256(SQL_PATH),
            "executed_in_fix_round": False,
        },
        "panels": {
            "full_a_original": {
                "path": str(FROZEN_INPUT_MANIFEST["full_a_original"]["path"]),
                "rows": len(original),
                "sha256": inputs_before["full_a_original"]["sha256"],
                "preserved": True,
            },
            "full_a_completed": {
                "path": str(completed_path),
                "rows": len(completed),
                "codes": int(completed["code"].nunique()),
                "first_date": str(completed["date"].min()),
                "last_date": str(completed["date"].max()),
                "primary_key": ["date", "code"],
                "performance_dates": len(actual_dates),
                "sha256": _sha256(completed_path),
            },
            "industry": {
                "path": str(FROZEN_INPUT_MANIFEST["industry"]["path"]),
                "rows": len(industry),
                "sha256": inputs_before["industry"]["sha256"],
            },
        },
        "hfq_completion": {
            "rule": "exact date/code; same-direction hfq; normal trading; fill original NULL only; no overwrite or time-series fill",
            "completed_rows": len(completion_audit),
            "completed_field_counts": {
                field: int(
                    completion_audit["completed_fields"].str.contains(field, regex=False).sum()
                )
                for field in SIGNAL_FIELDS
            },
            "keys": completion_audit.to_dict("records"),
            "before_after_remaining": _completion_reason_counts(original, completed, config),
        },
        "coverage": {
            "full_a": {
                "threshold": config.point_in_time_coverage_min,
                "before": before_summary,
                "after": after_summary,
                "passes": after_summary["passes"],
                "daily": after_summary["daily"],
            },
            "candidate_a_entry": {
                "deferred_to_task3": True,
                "threshold": config.required_entry_coverage,
                "denominator": None,
                "coverage": None,
                "passes": None,
                "predicate": "trend_system.legacy_data.candidate_entry_valid",
                "all_active_stock_day_diagnostic_not_a_gate": {
                    "denominator": int(main_performance.sum()),
                    "strict_valid_numerator_after": int(strict_valid_diagnostic.sum()),
                    "strict_valid_coverage_after": float(strict_valid_diagnostic.mean()),
                },
            },
            "industry": {
                "source": original_audit["coverage"]["industry"]["source"],
                "threshold": config.industry_coverage_min,
                "passes": False,
                "entry_coverage": 0.0,
                "daily": _daily_records(industry_daily),
            },
        },
        "candidate_statuses": statuses,
        "candidate_b_rule_freeze": {
            "evaluation_status": config.candidates[1].evaluation_status,
            "not_evaluable_fields": list(config.candidates[1].not_evaluable_fields),
            "restored": False,
        },
        "signals_implemented": False,
        "returns_run": False,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return LegacyContextResult(
        full_a_panel_path=completed_path,
        source_full_a_panel_path=FROZEN_INPUT_MANIFEST["full_a_original"]["path"],
        industry_panel_path=FROZEN_INPUT_MANIFEST["industry"]["path"],
        audit_path=audit_path,
        full_a_rows=len(completed),
        industry_rows=len(industry),
        candidate_a_status=statuses["three_of_two_low_vol"],
        candidate_b_status=statuses["early_trend_breadth"],
    )
