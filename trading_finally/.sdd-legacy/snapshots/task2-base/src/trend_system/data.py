import hashlib
import json
import platform
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import ResearchConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQL_ROOT = PROJECT_ROOT / "sql"
PANEL_START_DATE = "2010-01-01"
STRESS_END_DATE = "2015-12-31"
PRICE_THRESHOLD = 0.011
VOLUME_REL_THRESHOLD = 1e-5
AMOUNT_REL_THRESHOLD = 1e-6
RAW_PRICE_FIELDS = ("open", "high", "low", "close")
SIGNAL_FIELDS = ("signal_open", "signal_high", "signal_low", "signal_close")
PANEL_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "prev_close",
    *SIGNAL_FIELDS,
    "volume",
    "amount",
    "trade_status",
    "is_st",
    "ipo_date",
    "out_date",
    "data_valid",
    "execution_conflict",
    "adjustment_source_conflict",
    "source_conflict",
]
_MAIN_BOARD_SQL = (
    "substr({code}, 1, 3) IN "
    "('600', '601', '603', '605', '000', '001', '002', '003')"
)


@dataclass(frozen=True)
class PanelBuildResult:
    main_panel_path: Path
    stress_panel_path: Path
    corporate_actions_path: Path
    hithink_signal_panel_path: Path
    index_panel_path: Path
    audit_path: Path
    conflicts_path: Path
    exclusions_path: Path
    main_rows: int
    stress_rows: int
    conflict_rows: int
    exclusion_rows: int


def normalize_code(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Unable to convert code: {value!r}")
    match = re.fullmatch(r"(sh|sz)\.(\d{6})", value)
    if match:
        return f"{match.group(2)}.{match.group(1).upper()}"
    if re.fullmatch(r"\d{6}\.(SH|SZ)", value):
        return value
    raise ValueError(f"Unable to convert code: {value!r}")


def _normalize_codes(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("Unable to convert null code")
    strings = values.astype(str)
    if strings.str.fullmatch(r"\d{6}\.(SH|SZ)").all():
        return strings
    return strings.map(normalize_code)


def _reject_duplicate_keys(frame: pd.DataFrame) -> None:
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("duplicate (date, code) key")


def _reject_negative_prices(frame: pd.DataFrame, columns) -> None:
    present = [column for column in columns if column in frame]
    if present and frame[present].lt(0).any().any():
        raise ValueError("negative price is not allowed")


def _assemble_panel(raw: pd.DataFrame, hfq: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    hfq = hfq.copy()
    basic = basic.copy()
    for frame in (raw, hfq, basic):
        frame["code"] = _normalize_codes(frame["code"])
    _reject_duplicate_keys(raw)
    _reject_duplicate_keys(hfq)
    if basic["code"].duplicated().any():
        raise ValueError("duplicate stock_basic code")
    _reject_negative_prices(raw, (*RAW_PRICE_FIELDS, "prev_close", "preclose"))
    _reject_negative_prices(hfq, RAW_PRICE_FIELDS)

    adjusted = hfq[["date", "code", *RAW_PRICE_FIELDS]].rename(
        columns={field: f"signal_{field}" for field in RAW_PRICE_FIELDS}
    )
    joined = raw.merge(adjusted, on=["date", "code"], how="left", validate="one_to_one")
    joined = joined.merge(basic[["code", "ipo_date", "out_date"]], on="code", validate="many_to_one")
    if "prev_close" not in joined:
        joined = joined.rename(columns={"preclose": "prev_close"})
    return _finalize_panel(joined)


def _finalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(PANEL_COLUMNS) - set(frame) - {
        "data_valid",
        "execution_conflict",
        "adjustment_source_conflict",
        "source_conflict",
    }
    if missing:
        raise ValueError(f"Missing panel columns: {sorted(missing)}")
    panel = frame.copy()
    panel["code"] = _normalize_codes(panel["code"])
    panel["date"] = panel["date"].astype(str)
    _reject_duplicate_keys(panel)
    _reject_negative_prices(panel, (*RAW_PRICE_FIELDS, "prev_close", *SIGNAL_FIELDS))
    valid_raw = panel[[*RAW_PRICE_FIELDS, "prev_close"]].gt(0).all(axis=1)
    valid_signal = panel[list(SIGNAL_FIELDS)].gt(0).all(axis=1)
    panel["data_valid"] = (valid_raw & valid_signal).astype(bool)
    panel["execution_conflict"] = False
    panel["adjustment_source_conflict"] = False
    panel["source_conflict"] = False
    panel["ipo_date"] = panel["ipo_date"].replace("", pd.NA)
    panel["out_date"] = panel["out_date"].replace("", pd.NA)
    return panel[PANEL_COLUMNS].reset_index(drop=True)


def _relative_difference(left: pd.Series, right: pd.Series) -> pd.Series:
    left_values = left.astype(float)
    right_values = right.astype(float)
    denominator = np.maximum(left_values.abs(), right_values.abs())
    difference = (left_values - right_values).abs() / denominator.replace(0, np.nan)
    return difference.fillna(0).where(
        left_values.notna() & right_values.notna(), np.nan
    )


def _compare_overlap(
    panel: pd.DataFrame,
    hithink: pd.DataFrame,
    base_factors: dict[str, tuple[float, float]] | None = None,
):
    """Compare one chronological chunk and retain baostock values on conflicts."""
    if base_factors is None:
        base_factors = {}
    compared = panel.copy()
    hithink = hithink.copy()
    hithink["code"] = _normalize_codes(hithink["code"])
    hithink["date"] = hithink["date"].astype(str)
    _reject_duplicate_keys(hithink)
    _reject_negative_prices(
        hithink,
        [f"hithink_{field}" for field in (*RAW_PRICE_FIELDS, *SIGNAL_FIELDS)],
    )

    left = panel.copy()
    left["_panel_index"] = left.index
    merged = left.merge(
        hithink,
        on=["date", "code"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    both = merged["_merge"] == "both"
    hithink_only = merged["_merge"] == "right_only"
    flags = pd.DataFrame(index=merged.index)
    absolute_differences = {}
    relative_differences = {}

    for field in RAW_PRICE_FIELDS:
        other = f"hithink_{field}"
        difference = (merged[field] - merged[other]).abs()
        absolute_differences[field] = difference
        missing = merged[field].isna() ^ merged[other].isna()
        flags[field] = both & (missing | (difference > PRICE_THRESHOLD + 1e-12))

    for field, threshold in (
        ("volume", VOLUME_REL_THRESHOLD),
        ("amount", AMOUNT_REL_THRESHOLD),
    ):
        other = f"hithink_{field}"
        difference = _relative_difference(merged[field], merged[other])
        relative_differences[field] = difference
        missing = merged[field].isna() ^ merged[other].isna()
        flags[field] = both & (missing | (difference > threshold))

    baostock_factor = (merged["signal_close"] / merged["close"]).where(
        (merged["close"] > 0) & (merged["signal_close"] > 0)
    )
    hithink_factor = (merged["hithink_signal_close"] / merged["hithink_close"]).where(
        (merged["hithink_close"] > 0) & (merged["hithink_signal_close"] > 0)
    )
    candidates = merged.loc[
        both & baostock_factor.notna() & hithink_factor.notna(), ["date", "code"]
    ].copy()
    candidates["baostock_factor"] = baostock_factor.loc[candidates.index]
    candidates["hithink_factor"] = hithink_factor.loc[candidates.index]
    candidates = candidates.sort_values(["code", "date"]).drop_duplicates("code")
    for row in candidates.itertuples(index=False):
        if row.code not in base_factors:
            base_factors[row.code] = (row.baostock_factor, row.hithink_factor)

    base_baostock = merged["code"].map(
        {code: factors[0] for code, factors in base_factors.items()}
    )
    base_hithink = merged["code"].map(
        {code: factors[1] for code, factors in base_factors.items()}
    )
    normalized_differences = {}
    normalized_values = {}
    for field in SIGNAL_FIELDS:
        other = f"hithink_{field}"
        normalized_baostock = merged[field] / base_baostock
        normalized_hithink = merged[other] / base_hithink
        normalized_values[field] = (normalized_baostock, normalized_hithink)
        difference = (normalized_baostock - normalized_hithink).abs()
        normalized_differences[field] = difference
        missing = merged[field].isna() | merged[other].isna()
        flags[field] = both & (missing | (difference > PRICE_THRESHOLD + 1e-12))
    flags["adjustment_normalization_unavailable"] = (
        both & (base_baostock.isna() | base_hithink.isna())
    )
    flags["baostock_raw_missing"] = hithink_only

    conflict_mask = flags.any(axis=1)
    execution_mask = flags[list(RAW_PRICE_FIELDS)].any(axis=1)
    adjustment_mask = flags[
        [*SIGNAL_FIELDS, "adjustment_normalization_unavailable"]
    ].any(axis=1)
    volume_amount_mask = flags[["volume", "amount"]].any(axis=1)
    conflict_fields = pd.Series("", index=merged.index, dtype=object)
    for field in flags:
        selected = flags[field] & conflict_mask
        conflict_fields.loc[selected] += np.where(
            conflict_fields.loc[selected] == "", field, "|" + field
        )

    details = pd.DataFrame(
        {
            "date": merged["date"],
            "code": merged["code"],
            "presence": merged["_merge"].astype(str),
            "conflict_fields": conflict_fields,
            "execution_conflict": execution_mask,
            "adjustment_source_conflict": adjustment_mask,
            "baostock_adjust_factor": baostock_factor,
            "hithink_adjust_factor": hithink_factor,
            "volume_rel_diff": relative_differences["volume"],
            "amount_rel_diff": relative_differences["amount"],
        }
    )
    for field in RAW_PRICE_FIELDS:
        details[f"baostock_{field}"] = merged[field]
        details[f"hithink_{field}"] = merged[f"hithink_{field}"]
        details[f"{field}_abs_diff"] = absolute_differences[field]
    for field in SIGNAL_FIELDS:
        details[f"baostock_{field}"] = merged[field]
        details[f"hithink_{field}"] = merged[f"hithink_{field}"]
        details[f"{field}_normalized_abs_diff"] = normalized_differences[field]
    for field in ("volume", "amount"):
        details[f"baostock_{field}"] = merged[field]
        details[f"hithink_{field}"] = merged[f"hithink_{field}"]

    conflicts = details.loc[conflict_mask].reset_index(drop=True)
    for column, mask in (
        ("source_conflict", conflict_mask),
        ("execution_conflict", execution_mask),
        ("adjustment_source_conflict", adjustment_mask),
    ):
        panel_rows = merged.loc[both & mask, "_panel_index"]
        compared.loc[panel_rows, column] = True
    normalized_close = normalized_values["signal_close"]
    diagnostics = pd.DataFrame(
        {
            "normalized_close_relative_diff": _relative_difference(*normalized_close),
            "absolute_factor_scale_relative_diff": _relative_difference(
                baostock_factor, hithink_factor
            ),
            "adjustment_source_conflict": adjustment_mask,
        }
    )
    diagnostics = diagnostics.loc[both & diagnostics.notna().all(axis=1)].reset_index(
        drop=True
    )
    stats = {
        "baostock_rows": int(len(panel)),
        "hithink_rows": int(len(hithink)),
        "matched_rows": int(both.sum()),
        "baostock_only_rows": int((merged["_merge"] == "left_only").sum()),
        "hithink_only_rows": int(hithink_only.sum()),
        "conflict_rows": int(len(conflicts)),
        "execution_conflict_rows": int(
            details.loc[conflict_mask, "execution_conflict"].sum()
        ),
        "adjustment_source_conflict_rows": int(
            details.loc[conflict_mask, "adjustment_source_conflict"].sum()
        ),
        "volume_amount_conflict_rows": int((conflict_mask & volume_amount_mask).sum()),
        "field_conflict_counts": {
            field: int(flags[field].sum()) for field in flags if flags[field].any()
        },
        "adjustment_diagnostics": diagnostics,
    }
    return compared, conflicts, stats


def _hithink_signal_panel(hithink: pd.DataFrame) -> pd.DataFrame:
    panel = hithink.copy()
    panel["code"] = _normalize_codes(panel["code"])
    panel["date"] = panel["date"].astype(str)
    _reject_duplicate_keys(panel)
    panel = panel.rename(
        columns={
            f"hithink_{field}": field
            for field in (*RAW_PRICE_FIELDS, *SIGNAL_FIELDS, "volume", "amount")
        }
    )
    columns = [
        "date",
        "code",
        *RAW_PRICE_FIELDS,
        *SIGNAL_FIELDS,
        "volume",
        "amount",
    ]
    panel = panel[columns]
    panel["adjustment_factor"] = (panel["signal_close"] / panel["close"]).where(
        (panel["close"] > 0) & (panel["signal_close"] > 0)
    )
    panel["data_valid"] = (
        panel[list(RAW_PRICE_FIELDS)].gt(0).all(axis=1)
        & panel[list(SIGNAL_FIELDS)].gt(0).all(axis=1)
    )
    return panel.reset_index(drop=True)


def _summarize_adjustment_diagnostics(diagnostics: pd.DataFrame) -> dict:
    relative = diagnostics["normalized_close_relative_diff"]
    scale = diagnostics["absolute_factor_scale_relative_diff"]
    quantile_points = {
        "min": 0,
        "p25": 0.25,
        "p50": 0.5,
        "p75": 0.75,
        "p90": 0.9,
        "p95": 0.95,
        "p99": 0.99,
        "p99_9": 0.999,
        "max": 1,
    }
    different_scale = scale > 1e-6
    result = {
        "metric": "abs(normalized_baostock_close-normalized_hithink_close)/max(abs(normalized_baostock_close),abs(normalized_hithink_close))",
        "rows": int(len(diagnostics)),
        "quantiles": {
            name: float(relative.quantile(point))
            for name, point in quantile_points.items()
        },
        "relative_difference_bands": {
            "eq_0": int((relative == 0).sum()),
            "gt_0_le_0.001": int(((relative > 0) & (relative <= 0.001)).sum()),
            "gt_0.001_le_0.01": int(((relative > 0.001) & (relative <= 0.01)).sum()),
            "gt_0.01": int((relative > 0.01).sum()),
        },
        "absolute_factor_scale": {
            "metric": "relative difference of same-day hfq_close/raw_close before normalization",
            "difference_threshold": 1e-6,
            "different_scale_rows": int(different_scale.sum()),
            "different_scale_with_normalized_close_le_0.001": int(
                (different_scale & (relative <= 0.001)).sum()
            ),
        },
        "precision_band_upper": 0.001,
        "material_difference_threshold": 0.01,
    }
    conflict_relative = relative.loc[diagnostics["adjustment_source_conflict"].astype(bool)]
    result["adjustment_conflict_subset"] = {
        "rows": int(len(conflict_relative)),
        "quantiles": {
            name: float(conflict_relative.quantile(point))
            for name, point in quantile_points.items()
        },
        "relative_difference_bands": {
            "eq_0": int((conflict_relative == 0).sum()),
            "gt_0_le_0.001": int(
                ((conflict_relative > 0) & (conflict_relative <= 0.001)).sum()
            ),
            "gt_0.001_le_0.01": int(
                ((conflict_relative > 0.001) & (conflict_relative <= 0.01)).sum()
            ),
            "gt_0.01": int((conflict_relative > 0.01).sum()),
        },
    }
    return result


def _conflict_exclusions(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    execution = panel["execution_conflict"]
    adjustment = panel["adjustment_source_conflict"] & ~execution
    audit_only = panel["source_conflict"] & ~execution & ~panel["adjustment_source_conflict"]
    for mask, reason, detail in (
        (execution, "execution_conflict", "retained; ineligible for baseline entry and holding"),
        (adjustment, "adjustment_source_conflict", "retained; independent sensitivity required"),
        (audit_only, "source_conflict", "retained; audit only; no independent blocking decision"),
    ):
        excluded = panel.loc[mask, ["date", "code"]].copy()
        excluded["reason"] = reason
        excluded["rows"] = 1
        excluded["detail"] = detail
        frames.append(excluded)
    return pd.concat(frames, ignore_index=True)


def effective_corporate_actions(events: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    effective = events.loc[events["date"].astype(str) <= as_of_date].copy()
    return effective.sort_values(["date", "code"]).reset_index(drop=True)


def load_panel(path: str | Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    missing = set(PANEL_COLUMNS) - set(panel)
    if missing:
        raise ValueError(f"Missing panel columns: {sorted(missing)}")
    _reject_duplicate_keys(panel)
    return panel


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sql_file(name: str) -> str:
    return (SQL_ROOT / name).read_text(encoding="utf-8")


def _window(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "codes": int(frame["code"].nunique()),
        "first_date": str(frame["date"].min()) if len(frame) else None,
        "last_date": str(frame["date"].max()) if len(frame) else None,
    }


def build_panel(config: ResearchConfig) -> PanelBuildResult:
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    baostock_sql = _read_sql_file("baostock_panel.sql")
    hithink_sql = _read_sql_file("hithink_overlap.sql")
    baostock_path = Path(config.baostock_db_path)
    hithink_path = Path(config.hithink_db_path)
    database_hashes = {
        "baostock": _sha256(baostock_path),
        "hithink": _sha256(hithink_path),
    }

    sqlite = sqlite3.connect(f"{baostock_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        raw_panel = pd.read_sql_query(baostock_sql, sqlite, params=[config.end_date])
        panel = _finalize_panel(raw_panel)
        index_panel = pd.read_sql_query(
            """
            SELECT substr(code, 4, 6) || '.' || upper(substr(code, 1, 2)) AS code,
                   date, open, high, low, close, preclose AS prev_close, volume, amount
            FROM index_kline_d WHERE date <= ?
            """,
            sqlite,
            params=[config.end_date],
        )
        baostock_source_rows = pd.read_sql_query(
            """
            SELECT adjustflag, COUNT(*) AS rows, MIN(date) AS first_date,
                   MAX(date) AS last_date, COUNT(DISTINCT code) AS codes
            FROM stock_kline_d GROUP BY adjustflag
            """,
            sqlite,
        ).to_dict("records")
        incomplete = pd.read_sql_query(
            """
            SELECT code, SUM(adjustflag = 1) AS hfq_rows,
                   SUM(adjustflag = 3) AS raw_rows
            FROM stock_kline_d GROUP BY code
            HAVING SUM(adjustflag = 1) <> SUM(adjustflag = 3)
            """,
            sqlite,
        )
        main_filter = _MAIN_BOARD_SQL.format(code="substr(code, 4, 6)")
        universe = pd.read_sql_query(
            f"""
            SELECT type, status, COUNT(*) AS codes
            FROM stock_basic
            WHERE ({main_filter}) AND (code LIKE 'sh.%' OR code LIKE 'sz.%')
            GROUP BY type, status
            """,
            sqlite,
        )
        excluded_universe = pd.read_sql_query(
            f"""
            SELECT substr(b.code, 4, 6) || '.' || upper(substr(b.code, 1, 2)) AS code,
                   b.type, b.status
            FROM stock_basic AS b
            WHERE NOT (b.type = 1 AND ({_MAIN_BOARD_SQL.format(code='substr(b.code, 4, 6)')})
                       AND (b.code LIKE 'sh.%' OR b.code LIKE 'sz.%'))
            """,
            sqlite,
        )
    finally:
        sqlite.close()

    overlap_totals = Counter()
    field_totals = Counter()
    conflict_frames = []
    hithink_panel_frames = []
    diagnostic_frames = []
    base_factors = {}
    start_year = pd.Timestamp(config.start_date).year
    end_year = pd.Timestamp(config.end_date).year
    duck = duckdb.connect(str(hithink_path), read_only=True)
    try:
        for year in range(start_year, end_year + 1):
            chunk_start = max(config.start_date, f"{year}-01-01")
            chunk_end = min(config.end_date, f"{year}-12-31")
            hithink_chunk = duck.execute(
                hithink_sql, [chunk_start, chunk_end]
            ).fetchdf()
            panel_index = panel.index[
                (panel["date"] >= chunk_start) & (panel["date"] <= chunk_end)
            ]
            matched_keys = panel.loc[panel_index, ["date", "code"]]
            hithink_panel_frames.append(
                _hithink_signal_panel(hithink_chunk).merge(
                    matched_keys, on=["date", "code"], how="inner", validate="one_to_one"
                )
            )
            compared, conflicts, stats = _compare_overlap(
                panel.loc[panel_index], hithink_chunk, base_factors
            )
            for column in (
                "source_conflict",
                "execution_conflict",
                "adjustment_source_conflict",
            ):
                panel.loc[panel_index, column] = compared[column]
            if not conflicts.empty:
                conflict_frames.append(conflicts)
            diagnostic_frames.append(stats["adjustment_diagnostics"])
            for key, value in stats.items():
                if key not in ("field_conflict_counts", "adjustment_diagnostics"):
                    overlap_totals[key] += value
            field_totals.update(stats["field_conflict_counts"])

        event_filter = _MAIN_BOARD_SQL.format(code="thscode")
        events = duck.execute(
            f"""
            SELECT thscode AS code, CAST(ex_date AS VARCHAR) AS date,
                   dividend_per_share, per_share_bonus, rights_ratio, rights_price,
                   CAST(ex_date AS VARCHAR) AS source_date, batch_id
            FROM raw_adjustment_events
            WHERE ({event_filter}) AND (thscode LIKE '%.SH' OR thscode LIKE '%.SZ')
            """,
        ).fetchdf()
        hithink_source_rows = {}
        for name in ("v_daily", "v_daily_hfq"):
            hithink_source_rows[name] = duck.execute(
                f"SELECT COUNT(*) AS rows, MIN(date) AS first_date, "
                f"MAX(date) AS last_date, COUNT(DISTINCT thscode) AS codes FROM {name}"
            ).fetchdf().iloc[0].to_dict()
    finally:
        duck.close()

    hithink_panel = pd.concat(hithink_panel_frames, ignore_index=True)
    _reject_duplicate_keys(hithink_panel)
    adjustment_diagnostics = _summarize_adjustment_diagnostics(
        pd.concat(diagnostic_frames, ignore_index=True)
    )
    conflicts = (
        pd.concat(conflict_frames, ignore_index=True)
        if conflict_frames
        else pd.DataFrame(columns=["date", "code", "presence", "conflict_fields"])
    )
    corporate_actions = effective_corporate_actions(events, config.end_date)
    _reject_duplicate_keys(corporate_actions)
    stress = panel.loc[
        (panel["date"] >= PANEL_START_DATE) & (panel["date"] <= STRESS_END_DATE)
    ].reset_index(drop=True)

    exclusion_frames = []
    universe_exclusions = excluded_universe.copy()
    universe_exclusions["date"] = pd.NA
    universe_exclusions["reason"] = np.where(
        universe_exclusions["type"] == 1, "non_main_board", "non_stock_security"
    )
    universe_exclusions["detail"] = "code-level universe exclusion"
    universe_exclusions["rows"] = pd.NA
    exclusion_frames.append(
        universe_exclusions[["date", "code", "reason", "rows", "detail"]]
    )
    future_events = events.loc[events["date"] > config.end_date].copy()
    if not future_events.empty:
        future_events["reason"] = "future_corporate_action"
        future_events["rows"] = 1
        future_events["detail"] = "ex_date is after panel end"
        exclusion_frames.append(
            future_events[["date", "code", "reason", "rows", "detail"]]
        )
    hithink_only = conflicts.loc[conflicts["presence"] == "right_only"].copy()
    if not hithink_only.empty:
        hithink_only["reason"] = "hithink_only_not_inserted"
        hithink_only["rows"] = 1
        hithink_only["detail"] = "no same-key baostock raw row"
        exclusion_frames.append(
            hithink_only[["date", "code", "reason", "rows", "detail"]]
        )
    excluded = panel.loc[~panel["data_valid"], ["date", "code"]].copy()
    excluded["reason"] = "data_valid_false"
    excluded["rows"] = 1
    excluded["detail"] = "retained; ineligible for signals"
    exclusion_frames.append(excluded)
    exclusion_frames.append(_conflict_exclusions(panel))
    exclusions = pd.concat(exclusion_frames, ignore_index=True)

    paths = {
        "main": data_dir / "main_panel.parquet",
        "stress": data_dir / "stress_panel.parquet",
        "corporate_actions": data_dir / "corporate_actions.parquet",
        "hithink_signal": data_dir / "hithink_signal_panel.parquet",
        "index": data_dir / "index_panel.parquet",
        "conflicts": results_dir / "data_conflicts.parquet",
        "exclusions": results_dir / "data_exclusions.csv",
        "audit": results_dir / "data_audit.json",
    }
    panel.to_parquet(paths["main"], index=False)
    stress.to_parquet(paths["stress"], index=False)
    corporate_actions.to_parquet(paths["corporate_actions"], index=False)
    hithink_panel.to_parquet(paths["hithink_signal"], index=False)
    index_panel.to_parquet(paths["index"], index=False)
    conflicts.to_parquet(paths["conflicts"], index=False)
    exclusions.to_csv(paths["exclusions"], index=False)

    main_board_codes = int(universe.loc[universe["type"] == 1, "codes"].sum())
    delisted_codes = int(
        universe.loc[(universe["type"] == 1) & (universe["status"] == 0), "codes"].sum()
    )
    missing_hfq = panel[list(SIGNAL_FIELDS)].isna().any(axis=1)
    performance = panel["date"].between(config.start_date, config.end_date)
    audit = {
        "config": asdict(config),
        "panel_construction": {
            "start_date": PANEL_START_DATE,
            "performance_start_date": config.start_date,
            "end_date": config.end_date,
            "stress_window": [PANEL_START_DATE, STRESS_END_DATE],
            "price_threshold": PRICE_THRESHOLD,
            "volume_relative_threshold": VOLUME_REL_THRESHOLD,
            "amount_relative_threshold": AMOUNT_REL_THRESHOLD,
            "adjusted_comparison": "per-code first common valid hfq_close/raw_close normalization; 0.011 yuan after normalization",
            "source_conflict": "audit-only union of raw OHLC, volume, amount and adjusted-series conflicts; not independently blocking",
            "execution_conflict": "raw OHLC conflicts only; blocks baseline entry and holding",
            "adjustment_source_conflict": "normalized adjusted-series conflicts only; requires independent sensitivity, not an execution-validity verdict",
            "read_only": True,
        },
        "databases": {
            "baostock": {
                "path": str(baostock_path),
                "sha256": database_hashes["baostock"],
                "source_rows": baostock_source_rows,
            },
            "hithink": {
                "path": str(hithink_path),
                "sha256": database_hashes["hithink"],
                "source_rows": hithink_source_rows,
            },
        },
        "sql": {
            name: {
                "path": str(SQL_ROOT / name),
                "sha256": _sha256(SQL_ROOT / name),
                "text": _read_sql_file(name),
            }
            for name in ("baostock_panel.sql", "hithink_overlap.sql")
        },
        "panels": {
            "main": {**_window(panel), "performance_rows": int(performance.sum())},
            "stress": {**_window(stress), "source_mode": "baostock_single_source"},
            "index": _window(index_panel),
            "hithink_signal": _window(hithink_panel),
            "corporate_actions": _window(corporate_actions),
        },
        "overlap": {
            "start_date": config.start_date,
            "end_date": config.end_date,
            **dict(overlap_totals),
            "conflict_codes": int(conflicts["code"].nunique()),
            "panel_source_conflict_rows": int(panel["source_conflict"].sum()),
            "panel_execution_conflict_rows": int(panel["execution_conflict"].sum()),
            "panel_adjustment_source_conflict_rows": int(
                panel["adjustment_source_conflict"].sum()
            ),
            "matched_conflict_rows": int(
                overlap_totals["conflict_rows"] - overlap_totals["hithink_only_rows"]
            ),
            "field_conflict_counts": dict(field_totals),
        },
        "normalized_adjustment_diagnostics": adjustment_diagnostics,
        "adjustment_completeness": {
            "all_source_incomplete_codes": int(len(incomplete)),
            "main_board_incomplete_codes": int(
                incomplete["code"].map(normalize_code).str[:3].isin(
                    {"600", "601", "603", "605", "000", "001", "002", "003"}
                ).sum()
            ),
            "panel_missing_hfq_rows": int(missing_hfq.sum()),
            "panel_missing_hfq_codes": int(panel.loc[missing_hfq, "code"].nunique()),
            "panel_invalid_rows": int((~panel["data_valid"]).sum()),
            "panel_invalid_codes": int(panel.loc[~panel["data_valid"], "code"].nunique()),
            "treatment": "raw rows retained with data_valid=False; no future-row backfill and no whole-stock deletion",
        },
        "universe": {
            "main_board_codes": main_board_codes,
            "delisted_main_board_codes": delisted_codes,
            "listed_main_board_codes": main_board_codes - delisted_codes,
            "panel_codes": int(panel["code"].nunique()),
            "excluded_universe_codes": int(len(excluded_universe)),
        },
        "exclusions": {
            "rows": int(len(exclusions)),
            "reason_counts": {
                str(key): int(value)
                for key, value in exclusions["reason"].value_counts().items()
            },
        },
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "duckdb": duckdb.__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
    }
    paths["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    if panel["date"].max() != config.end_date:
        raise ValueError("Main panel does not reach configured end date")
    _reject_duplicate_keys(panel)
    return PanelBuildResult(
        main_panel_path=paths["main"],
        stress_panel_path=paths["stress"],
        corporate_actions_path=paths["corporate_actions"],
        hithink_signal_panel_path=paths["hithink_signal"],
        index_panel_path=paths["index"],
        audit_path=paths["audit"],
        conflicts_path=paths["conflicts"],
        exclusions_path=paths["exclusions"],
        main_rows=len(panel),
        stress_rows=len(stress),
        conflict_rows=len(conflicts),
        exclusion_rows=len(exclusions),
    )
