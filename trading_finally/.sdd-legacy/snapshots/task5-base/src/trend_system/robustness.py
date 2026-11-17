"""Frozen robustness matrix and mechanical research gates.

All strategy variants use the upstream signal, account, and metric functions.  The
only exception to signal generation is the frozen randomization of entry priority
or entry membership; both transformations are explicit and auditable.
"""

import gc
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow

from . import execution
from .broker import BacktestResult, execution_integrity, run_backtest
from .config import ResearchConfig
from .data import load_panel
from .metrics import annual_and_rolling, performance_summary
from .report import render_final_report
from .risk import build_risk_report
from .signals import compute_signals, select_backtest_window


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_FIELDS = ("signal_open", "signal_high", "signal_low", "signal_close")
RAW_PRICE_FIELDS = ("open", "high", "low", "close", "prev_close")
NEIGHBORHOOD_VALUES = (
    ("entry_lookback", 90),
    ("entry_lookback", 150),
    ("trend_window", 180),
    ("trend_window", 220),
    ("trend_slope_window", 10),
    ("trend_slope_window", 30),
    ("exit_lookback", 15),
    ("exit_lookback", 25),
)
RANDOM_PATHS = 20
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_BLOCK_SIZE = 20
FROZEN_GATE_IDS = (
    "data_execution_integrity",
    "main_window_positive",
    "rolling_3y_majority_positive",
    "double_cost_positive_and_executable",
    "neighborhood_at_least_6_positive",
    "random_order_at_least_16_positive",
    "remaining_years_equity_pnl_positive",
    "fully_after_close_reproducible",
    "hithink_common_positive_mdd_and_direction",
    "drawdown_boundary",
)


@dataclass(frozen=True)
class RobustnessResult:
    summary: dict
    variants: pd.DataFrame
    bootstrap: pd.DataFrame
    descriptive: pd.DataFrame
    verdict: str
    output_dir: Path
    stress_output_dir: Path
    report_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_config(config: ResearchConfig) -> bytes:
    return json.dumps(
        asdict(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _config_hash(config: ResearchConfig) -> str:
    return hashlib.sha256(_canonical_config(config)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def neighborhood_configs(config: ResearchConfig) -> list[tuple[str, ResearchConfig]]:
    """Return exactly the eight preregistered one-factor configurations."""
    return [
        (f"{field}_{value}", replace(config, **{field: value}))
        for field, value in NEIGHBORHOOD_VALUES
    ]


def evaluate_gates(summary: dict) -> str:
    """Apply the frozen mechanical veto without selecting a historical variant."""
    try:
        raw_drawdown = (
            summary["max_drawdown"]
            if "max_drawdown" in summary
            else summary["baseline"]["max_drawdown"]
        )
        drawdown = float(raw_drawdown)
    except (KeyError, TypeError, ValueError):
        return "REJECTED"
    gates = summary.get("gates", summary.get("mechanical_gates", []))
    if not isinstance(gates, list) or len(gates) != len(FROZEN_GATE_IDS):
        return "REJECTED"
    if any(not isinstance(gate, dict) for gate in gates):
        return "REJECTED"
    identifiers = [gate.get("id") for gate in gates]
    if any(not isinstance(identifier, str) for identifier in identifiers):
        return "REJECTED"
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(FROZEN_GATE_IDS):
        return "REJECTED"
    if not np.isfinite(drawdown):
        return "REJECTED"
    passed_values = [gate.get("passed") for gate in gates]
    if any(type(value) is not bool for value in passed_values) or not all(passed_values):
        return "REJECTED"
    if drawdown > 0.20:
        return "REJECTED"
    if drawdown > 0.10:
        return "BOUNDARY"
    return "UNREJECTED"


def _with_previous_ma(signals: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    result = signals.copy()
    ordered = result.sort_values(["code", "date"], kind="stable")
    previous = ordered.groupby("code", sort=False)["ma200"].shift(
        config.trend_slope_window
    )
    result["previous_ma"] = previous.reindex(result.index)
    return result


def _rng(config: ResearchConfig, experiment: int, path_id: int | None = None):
    entropy = [config.random_seed, experiment]
    if path_id is not None:
        entropy.append(path_id)
    return np.random.default_rng(np.random.SeedSequence(entropy))


def random_order_signals(
    signals: pd.DataFrame, config: ResearchConfig, path_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomize entry priority while preserving every entry and exit signal."""
    if not 0 <= path_id < RANDOM_PATHS:
        raise ValueError("path_id must identify one of the 20 frozen paths")
    result = signals.copy()
    entry = result["entry_signal"].astype(bool)
    scores = _rng(config, experiment=1, path_id=path_id).random(int(entry.sum()))
    result.loc[entry, "rank_return_120"] = scores
    log = result.loc[entry, ["date", "code"]].copy().reset_index(drop=True)
    log["random_rank"] = result.loc[entry, "rank_return_120"].to_numpy()
    log.insert(0, "path_id", path_id)
    return result, log


def _history_complete(signals: pd.DataFrame) -> pd.Series:
    required = ["ma200", "channel_high_120", "channel_low_120", "rank_return_120"]
    missing = set(required) - set(signals)
    if missing:
        raise ValueError(f"Missing signal history columns: {sorted(missing)}")
    complete = signals[required].notna().all(axis=1)
    if "previous_ma" in signals:
        complete &= signals["previous_ma"].notna()
    return complete


def random_entry_signals(
    signals: pd.DataFrame, config: ResearchConfig, path_id: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace each day's entries from the eligible non-entry universe.

    Selection is without replacement within a date.  Real rank_return_120 values
    are deliberately untouched so the upstream broker still ranks replacements
    by their actual momentum.  An undersized pool is a hard failure.
    """
    if not 0 <= path_id < RANDOM_PATHS:
        raise ValueError("path_id must identify one of the 20 frozen paths")
    required_columns = {
        "date",
        "code",
        "is_st",
        "trade_status",
        "data_valid",
        "execution_conflict",
        "entry_signal",
        "rank_return_120",
    }
    missing = required_columns - set(signals)
    if missing:
        raise ValueError(f"Missing random-entry columns: {sorted(missing)}")

    eligible = (
        signals["is_st"].eq(0).fillna(False)
        & signals["trade_status"].eq(1).fillna(False)
        & signals["data_valid"].eq(True).fillna(False)
        & ~signals["execution_conflict"].fillna(True).astype(bool)
        & ~signals["entry_signal"].astype(bool)
        & _history_complete(signals)
    )
    entry_counts = signals.loc[signals["entry_signal"].astype(bool)].groupby("date").size()
    pools = {
        day: group.index.to_numpy()
        for day, group in signals.loc[eligible].groupby("date", sort=False)
    }
    for day, required in entry_counts.items():
        available = len(pools.get(day, []))
        if available < required:
            raise RuntimeError(
                "random entry pool insufficient: "
                f"date={day} required={int(required)} available={available}"
            )

    result = signals.copy()
    rng = _rng(config, experiment=2, path_id=path_id)
    log_rows = []
    for day, required_series in entry_counts.items():
        required = int(required_series)
        original_indices = result.index[
            result["date"].eq(day) & result["entry_signal"].astype(bool)
        ]
        pool = pools[day]
        selected_indices = rng.choice(pool, size=required, replace=False)
        result.loc[original_indices, "entry_signal"] = False
        result.loc[selected_indices, "entry_signal"] = True
        log_rows.append(
            {
                "path_id": path_id,
                "date": str(day),
                "required": required,
                "available": int(len(pool)),
                "original_codes": "|".join(sorted(result.loc[original_indices, "code"].astype(str))),
                "replacement_codes": "|".join(sorted(result.loc[selected_indices, "code"].astype(str))),
            }
        )
    log = pd.DataFrame(
        log_rows,
        columns=[
            "path_id",
            "date",
            "required",
            "available",
            "original_codes",
            "replacement_codes",
        ],
    )
    return result, log


def annual_equity_contributions(
    daily_account: pd.DataFrame, initial_cash: float
) -> pd.DataFrame:
    """Calculate calendar-year account equity PnL, not closed-trade PnL."""
    account = daily_account.copy()
    account["_date"] = pd.to_datetime(account["date"])
    account = account.sort_values("_date", kind="stable")
    rows = []
    previous_equity = float(initial_cash)
    previous_date = account["_date"].iloc[0]
    for year, group in account.groupby(account["_date"].dt.year, sort=True):
        end_equity = float(group["equity"].iloc[-1])
        end_date = group["_date"].iloc[-1]
        pnl = end_equity - previous_equity
        rows.append(
            {
                "year": str(year),
                "start_date": previous_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "start_equity": previous_equity,
                "end_equity": end_equity,
                "equity_pnl": pnl,
                "return_pct": pnl / previous_equity if previous_equity else np.nan,
                "is_top_contributor": False,
            }
        )
        previous_equity = end_equity
        previous_date = end_date
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("daily_account must not be empty")
    top_index = result["equity_pnl"].idxmax()
    result.loc[top_index, "is_top_contributor"] = True
    return result


def block_bootstrap(
    daily_account: pd.DataFrame,
    seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
) -> pd.DataFrame:
    """Moving-block resampling of account returns, never of trade averages."""
    account = daily_account.sort_values("date", kind="stable")
    returns = account["equity"].astype(float).pct_change().dropna().to_numpy()
    if len(returns) < block_size:
        raise ValueError("At least one complete return block is required")
    if iterations <= 0 or block_size <= 0:
        raise ValueError("iterations and block_size must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    blocks_needed = int(np.ceil(len(returns) / block_size))
    rows = []
    for replicate in range(iterations):
        starts = rng.integers(0, len(returns) - block_size + 1, size=blocks_needed)
        sample = np.concatenate(
            [returns[start : start + block_size] for start in starts]
        )[: len(returns)]
        nav = np.concatenate(([1.0], np.cumprod(1.0 + sample)))
        drawdown = nav / np.maximum.accumulate(nav) - 1.0
        rows.append(
            {
                "replicate": replicate,
                "total_return": float(nav[-1] - 1.0),
                "max_drawdown": float(-drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def _save_backtest(
    output_dir: Path,
    backtest: BacktestResult,
    summary: dict,
    annual: pd.DataFrame,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "daily_account.parquet": backtest.daily_account,
        "daily_positions.parquet": backtest.daily_positions,
        "trades.parquet": backtest.trades,
        "orders.parquet": backtest.orders,
        "round_trips.parquet": backtest.round_trips,
        "action_log.parquet": backtest.action_log,
    }
    for name, frame in tables.items():
        frame.to_parquet(output_dir / name, index=False)
    annual.to_csv(output_dir / "annual_and_rolling.csv", index=False)
    _write_json(output_dir / "performance_summary.json", summary)
    _write_json(output_dir / "variant.json", metadata)


def _run_and_save(
    *,
    family: str,
    name: str,
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    config: ResearchConfig,
    output_dir: Path,
    changed_fields: dict | None = None,
    double_cost: bool = False,
) -> tuple[dict, dict, BacktestResult]:
    started = time.perf_counter()
    run_config = config
    if double_cost:
        run_config = replace(
            config,
            commission_rate=config.commission_rate * 2,
            min_commission=config.min_commission * 2,
            slippage_bps=20,
        )
        original_costs = execution.transaction_costs

        def doubled_costs(date, amount, side, multiplier=1.0):
            return original_costs(date, amount, side, multiplier=2.0 * multiplier)

        with patch.object(execution, "transaction_costs", doubled_costs):
            backtest = run_backtest(panel, signals, corporate_actions, run_config)
    else:
        backtest = run_backtest(panel, signals, corporate_actions, run_config)
    summary = performance_summary(backtest.daily_account, backtest.round_trips)
    annual = annual_and_rolling(backtest.daily_account)
    metadata = {
        "family": family,
        "name": name,
        "status": "completed",
        "config": asdict(run_config),
        "config_sha256": _config_hash(run_config),
        "changed_fields": changed_fields or {},
        "double_all_fee_components": double_cost,
    }
    _save_backtest(output_dir, backtest, summary, annual, metadata)
    record = {
        "family": family,
        "name": name,
        "status": "completed",
        "start_date": summary["start_date"],
        "end_date": summary["end_date"],
        "daily_account_rows": int(len(backtest.daily_account)),
        "trade_rows": int(len(backtest.trades)),
        "round_trip_rows": int(len(backtest.round_trips)),
        "final_equity": summary["final_equity"],
        "total_return": summary["total_return"],
        "max_drawdown": summary["max_drawdown"],
        "rolling_3y_positive_share": summary["rolling_3y_positive_share"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "changed_fields": changed_fields or {},
        "output_dir": _relative(output_dir),
    }
    print(
        f"[robustness] {family}/{name}: completed; "
        f"return={summary['total_return']:.6f}; mdd={summary['max_drawdown']:.6f}",
        flush=True,
    )
    return record, summary, backtest


def _common_source_inputs(
    main_window: pd.DataFrame, hithink: pd.DataFrame, config: ResearchConfig
):
    hithink_columns = ["date", "code", *SIGNAL_FIELDS]
    missing = set(hithink_columns) - set(hithink)
    if missing:
        raise ValueError(f"Missing hithink signal columns: {sorted(missing)}")
    adjusted = hithink[hithink_columns].copy()
    adjusted["date"] = adjusted["date"].astype(str)
    if adjusted.duplicated(["date", "code"]).any():
        raise ValueError("duplicate hithink (date, code) key")
    adjusted = adjusted.rename(columns={field: f"hithink_{field}" for field in SIGNAL_FIELDS})
    merged = main_window.merge(
        adjusted, on=["date", "code"], how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("No common source rows")
    baostock_panel = merged.drop(columns=[f"hithink_{field}" for field in SIGNAL_FIELDS])
    hithink_panel = baostock_panel.copy()
    raw_valid = hithink_panel[list(RAW_PRICE_FIELDS)].gt(0).all(axis=1)
    for field in SIGNAL_FIELDS:
        hithink_panel[field] = merged[f"hithink_{field}"].to_numpy()
    hithink_valid = hithink_panel[list(SIGNAL_FIELDS)].gt(0).all(axis=1)
    hithink_panel["data_valid"] = raw_valid & hithink_valid

    baostock_signals = _with_previous_ma(compute_signals(baostock_panel, config), config)
    hithink_signals = _with_previous_ma(compute_signals(hithink_panel, config), config)
    complete = (
        _history_complete(baostock_signals)
        & _history_complete(hithink_signals)
        & baostock_signals["data_valid"].astype(bool)
        & hithink_signals["data_valid"].astype(bool)
    )
    if not complete.any():
        raise ValueError("No common date has complete indicator warmup")
    common_start = str(baostock_signals.loc[complete, "date"].min())
    common_config = replace(config, start_date=common_start)
    baostock_panel, baostock_signals = select_backtest_window(
        baostock_panel, baostock_signals, common_config
    )
    hithink_panel, hithink_signals = select_backtest_window(
        hithink_panel, hithink_signals, common_config
    )
    if not baostock_panel[["date", "code"]].equals(hithink_panel[["date", "code"]]):
        raise RuntimeError("Common-source panels have divergent keys or calendars")
    if not baostock_signals[["date", "code"]].equals(hithink_signals[["date", "code"]]):
        raise RuntimeError("Common-source signals have divergent keys or calendars")
    return (
        baostock_panel,
        hithink_panel,
        baostock_signals,
        hithink_signals,
        common_config,
        common_start,
    )


def _signal_differences(
    baostock: pd.DataFrame, hithink: pd.DataFrame, output_dir: Path
) -> dict:
    keys = ["date", "code"]
    if not baostock[keys].equals(hithink[keys]):
        raise RuntimeError("Signal panels are not aligned")
    columns = ["entry_signal", "exit_signal", "st_exit"]
    different = pd.Series(False, index=baostock.index)
    result = baostock[keys].copy()
    for column in columns:
        left = baostock[column].astype(bool)
        right = hithink[column].astype(bool)
        changed = left.ne(right)
        result[f"baostock_{column}"] = left
        result[f"hithink_{column}"] = right
        different |= changed
    differences = result.loc[different].reset_index(drop=True)
    differences.to_parquet(output_dir / "signal_differences.parquet", index=False)
    return {
        "difference_rows": int(len(differences)),
        "entry_signal_differences": int(
            baostock["entry_signal"].astype(bool).ne(hithink["entry_signal"].astype(bool)).sum()
        ),
        "exit_signal_differences": int(
            baostock["exit_signal"].astype(bool).ne(hithink["exit_signal"].astype(bool)).sum()
        ),
        "st_exit_differences": int(
            baostock["st_exit"].astype(bool).ne(hithink["st_exit"].astype(bool)).sum()
        ),
    }


def _descriptive_analysis(
    daily_account: pd.DataFrame, index_panel: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, dict]:
    account = daily_account[["date", "equity"]].copy()
    account["date"] = pd.to_datetime(account["date"])
    account = account.sort_values("date", kind="stable").reset_index(drop=True)
    account["account_return"] = account["equity"].astype(float).pct_change().fillna(0.0)

    def indexed(code: str) -> pd.DataFrame:
        selected = index_panel.loc[index_panel["code"] == code, ["date", "close"]].copy()
        if selected.empty:
            raise ValueError(f"Descriptive index is unavailable: {code}")
        selected["date"] = pd.to_datetime(selected["date"])
        return selected.sort_values("date").drop_duplicates("date", keep="last").set_index("date")

    market = indexed("000300.SH")
    market["ma200"] = market["close"].rolling(200).mean()
    market["previous_ma200"] = market["ma200"].shift(20)
    market_aligned = market.reindex(account["date"]).ffill()
    trending = market_aligned["close"].gt(market_aligned["ma200"]) & market_aligned[
        "ma200"
    ].gt(market_aligned["previous_ma200"])
    falling = market_aligned["close"].lt(market_aligned["ma200"]) & market_aligned[
        "ma200"
    ].lt(market_aligned["previous_ma200"])
    known = market_aligned[["close", "ma200", "previous_ma200"]].notna().all(axis=1)
    account["market_regime"] = np.select(
        [known & trending, known & falling], ["trend", "down"], default="sideways"
    )
    account.loc[(~known).to_numpy(), "market_regime"] = "unclassified"

    large = indexed("000300.SH")["close"]
    small = indexed("000852.SH")["close"]
    large_return = large.pct_change(60).reindex(account["date"]).ffill()
    small_return = small.pct_change(60).reindex(account["date"]).ffill()
    account["large_cap_60d_return"] = large_return.to_numpy()
    account["small_cap_60d_return"] = small_return.to_numpy()
    spread = small_return - large_return
    account["size_return_spread_60d"] = spread.to_numpy()
    account["size_style"] = np.select(
        [spread.gt(0), spread.lt(0)], ["small_led", "large_led"], default="balanced"
    )
    account.loc[spread.isna().to_numpy(), "size_style"] = "unclassified"

    group_rows = []
    for group_type in ("market_regime", "size_style"):
        for group_name, group in account.groupby(group_type, sort=True):
            returns = group["account_return"].astype(float)
            group_rows.append(
                {
                    "group_type": group_type,
                    "group": group_name,
                    "days": int(len(group)),
                    "compound_return": float((1.0 + returns).prod() - 1.0),
                    "mean_daily_return": float(returns.mean()),
                    "positive_day_share": float(returns.gt(0).mean()),
                }
            )
    groups = pd.DataFrame(group_rows)
    export = account.copy()
    export["date"] = export["date"].dt.strftime("%Y-%m-%d")
    export.to_csv(output_dir / "descriptive_daily.csv", index=False)
    groups.to_csv(output_dir / "descriptive_groups.csv", index=False)
    formulas = {
        "post_hoc_only": "这些规则与参数未预冻结，只作事后描述，不参与门槛，不形成交易闸门。",
        "market_regime": "沪深300：close>MA200且MA200>MA200.shift(20)为trend；close<MA200且MA200<MA200.shift(20)为down；其余为sideways；历史不足为unclassified。",
        "size_style": "中证1000(000852.SH)60日收益减沪深300(000300.SH)60日收益；差值>0为small_led，<0为large_led，=0为balanced，历史不足为unclassified。",
        "group_return": "组内账户日收益r按product(1+r)-1复合；同时披露日收益均值、正收益日占比与天数。",
    }
    return groups, {"formulas": formulas, "groups": group_rows}


def _gate(identifier: str, passed: bool, actual, requirement: str) -> dict:
    if isinstance(actual, (np.floating, np.integer)):
        actual = actual.item()
    return {
        "id": identifier,
        "passed": bool(passed),
        "actual": actual,
        "requirement": requirement,
    }


def _freeze_main(output_dir: Path, config: ResearchConfig) -> dict:
    files = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {"main_freeze.json", "summary.json"}:
            continue
        files[str(path.relative_to(output_dir))] = _sha256(path)
    freeze = {
        "created_at": _utc_now(),
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "scope": "all main-window robustness outputs frozen before historical stress",
        "files": files,
        "dependencies": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    _write_json(output_dir / "main_freeze.json", freeze)
    return freeze


def run_robustness(config: ResearchConfig) -> RobustnessResult:
    """Execute every frozen experiment, then run the single-source stress last."""
    started_at = _utc_now()
    started = time.perf_counter()
    root = PROJECT_ROOT
    data_dir = root / "data"
    output_dir = root / "results" / "robustness"
    stress_output_dir = root / "results" / "stress_2010_2015"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = {
        name: data_dir / name
        for name in (
            "main_panel.parquet",
            "hithink_signal_panel.parquet",
            "index_panel.parquet",
            "corporate_actions.parquet",
            "stress_panel.parquet",
        )
    }
    input_manifest = {
        "inputs": {
            path.name: {"path": _relative(path), "sha256": _sha256(path)}
            for path in input_paths.values()
        }
    }
    _write_json(output_dir / "input_manifest.json", input_manifest)

    main_panel = load_panel(data_dir / "main_panel.parquet")
    corporate_actions = pd.read_parquet(data_dir / "corporate_actions.parquet")
    baseline_signals_full = _with_previous_ma(compute_signals(main_panel, config), config)
    main_window, baseline_signals = select_backtest_window(
        main_panel, baseline_signals_full, config
    )
    del baseline_signals_full

    baseline_record, baseline_summary, baseline_backtest = _run_and_save(
        family="baseline",
        name="baseline",
        panel=main_window,
        signals=baseline_signals,
        corporate_actions=corporate_actions,
        config=config,
        output_dir=output_dir / "baseline",
    )
    variant_records = []

    for name, variant_config in neighborhood_configs(config):
        signals_full = compute_signals(main_panel, variant_config)
        _, signals = select_backtest_window(main_panel, signals_full, variant_config)
        del signals_full
        baseline_values = asdict(config)
        variant_values = asdict(variant_config)
        changed = {
            key: variant_values[key]
            for key in baseline_values
            if variant_values[key] != baseline_values[key]
        }
        if len(changed) != 1:
            raise RuntimeError("Neighborhood variant is not one-factor")
        record, _, _ = _run_and_save(
            family="neighborhood",
            name=name,
            panel=main_window,
            signals=signals,
            corporate_actions=corporate_actions,
            config=variant_config,
            output_dir=output_dir / "neighborhood" / name,
            changed_fields=changed,
        )
        variant_records.append(record)
        del signals
        gc.collect()

    hithink = pd.read_parquet(data_dir / "hithink_signal_panel.parquet")
    hithink_source_start = str(hithink["date"].min())
    (
        baostock_common_panel,
        hithink_common_panel,
        baostock_common_signals,
        hithink_common_signals,
        common_config,
        common_start,
    ) = _common_source_inputs(main_window, hithink, config)
    del hithink
    common_dir = output_dir / "dual_source"
    common_dir.mkdir(parents=True, exist_ok=True)
    baostock_record, baostock_common_summary, _ = _run_and_save(
        family="dual_source",
        name="baostock_common",
        panel=baostock_common_panel,
        signals=baostock_common_signals,
        corporate_actions=corporate_actions,
        config=common_config,
        output_dir=common_dir / "baostock_common",
        changed_fields={"start_date": common_start, "window": "common_only"},
    )
    variant_records.append(baostock_record)
    hithink_record, hithink_common_summary, _ = _run_and_save(
        family="dual_source",
        name="hithink_common",
        panel=hithink_common_panel,
        signals=hithink_common_signals,
        corporate_actions=corporate_actions,
        config=common_config,
        output_dir=common_dir / "hithink_common",
        changed_fields={"start_date": common_start, "signal_source": "hithink_hfq"},
    )
    variant_records.append(hithink_record)
    signal_difference_summary = _signal_differences(
        baostock_common_signals, hithink_common_signals, common_dir
    )
    common_window = {
        "warmup_source_start": hithink_source_start,
        "start_date": common_start,
        "end_date": config.end_date,
        "selection_rule": "首个两源指标ma200、previous_ma、双通道和rank_return_120均完整且data_valid的共同日期",
        "both_sources_start_in_cash_without_positions": True,
        "common_window_is_not_full_main_window": True,
        "baostock_total_return": baostock_common_summary["total_return"],
        "baostock_max_drawdown": baostock_common_summary["max_drawdown"],
        "hithink_total_return": hithink_common_summary["total_return"],
        "hithink_max_drawdown": hithink_common_summary["max_drawdown"],
        **signal_difference_summary,
    }
    del (
        baostock_common_panel,
        hithink_common_panel,
        baostock_common_signals,
        hithink_common_signals,
        main_panel,
    )
    gc.collect()

    double_record, double_summary, double_backtest = _run_and_save(
        family="double_cost",
        name="double_fees_20bp_slippage",
        panel=main_window,
        signals=baseline_signals,
        corporate_actions=corporate_actions,
        config=config,
        output_dir=output_dir / "double_cost" / "double_fees_20bp_slippage",
        changed_fields={"all_fee_multiplier": 2, "slippage_bps": 20},
        double_cost=True,
    )
    variant_records.append(double_record)

    for path_id in range(RANDOM_PATHS):
        signals, order_log = random_order_signals(baseline_signals, config, path_id)
        path_dir = output_dir / "random_order" / f"path_{path_id:02d}"
        path_dir.mkdir(parents=True, exist_ok=True)
        order_log.to_csv(path_dir / "random_order_log.csv", index=False)
        record, _, _ = _run_and_save(
            family="random_order",
            name=f"path_{path_id:02d}",
            panel=main_window,
            signals=signals,
            corporate_actions=corporate_actions,
            config=config,
            output_dir=path_dir,
            changed_fields={"path_id": path_id, "seed": config.random_seed},
        )
        variant_records.append(record)
        del signals
        gc.collect()

    for path_id in range(RANDOM_PATHS):
        signals, replacement_log = random_entry_signals(baseline_signals, config, path_id)
        path_dir = output_dir / "random_entry" / f"path_{path_id:02d}"
        path_dir.mkdir(parents=True, exist_ok=True)
        replacement_log.to_csv(path_dir / "replacement_log.csv", index=False)
        record, _, _ = _run_and_save(
            family="random_entry",
            name=f"path_{path_id:02d}",
            panel=main_window,
            signals=signals,
            corporate_actions=corporate_actions,
            config=config,
            output_dir=path_dir,
            changed_fields={"path_id": path_id, "seed": config.random_seed},
        )
        variant_records.append(record)
        del signals
        gc.collect()

    portfolio_configs = (
        ("2_slots_20pct", replace(config, slots=2, max_weight=0.20, max_exposure=0.40)),
        ("4_slots_15pct", replace(config, slots=4, max_weight=0.15, max_exposure=0.60)),
    )
    for name, variant_config in portfolio_configs:
        changed = {
            key: asdict(variant_config)[key]
            for key in asdict(config)
            if asdict(variant_config)[key] != asdict(config)[key]
        }
        record, _, _ = _run_and_save(
            family="portfolio",
            name=name,
            panel=main_window,
            signals=baseline_signals,
            corporate_actions=corporate_actions,
            config=variant_config,
            output_dir=output_dir / "portfolio" / name,
            changed_fields=changed,
        )
        variant_records.append(record)

    annual_equity = annual_equity_contributions(
        baseline_backtest.daily_account, config.initial_cash
    )
    annual_equity.to_csv(output_dir / "annual_equity.csv", index=False)
    top = annual_equity.loc[annual_equity["is_top_contributor"]].iloc[0]
    remaining_equity_pnl = float(
        annual_equity.loc[~annual_equity["is_top_contributor"], "equity_pnl"].sum()
    )
    profit_concentration = {
        "method": "账户年度权益盈亏；首年相对初始现金，后续年份相对上一年末权益",
        "top_year": str(top["year"]),
        "top_year_equity_pnl": float(top["equity_pnl"]),
        "remaining_equity_pnl": remaining_equity_pnl,
        "uses_realized_annual_contribution": False,
    }

    bootstrap = block_bootstrap(
        baseline_backtest.daily_account,
        seed=config.random_seed,
        iterations=BOOTSTRAP_ITERATIONS,
        block_size=BOOTSTRAP_BLOCK_SIZE,
    )
    bootstrap.to_csv(output_dir / "bootstrap_20d_1000.csv", index=False)
    quantiles = bootstrap["total_return"].quantile([0.025, 0.5, 0.975])
    bootstrap_summary = {
        "iterations": BOOTSTRAP_ITERATIONS,
        "block_size": BOOTSTRAP_BLOCK_SIZE,
        "seed": config.random_seed,
        "method": "对基线账户日收益作有放回移动日期块抽样；不执行1000次账户回测",
        "exploratory_only": True,
        "total_return_quantiles": {
            "p02_5": float(quantiles.loc[0.025]),
            "p50": float(quantiles.loc[0.5]),
            "p97_5": float(quantiles.loc[0.975]),
        },
        "positive_return_share": float(bootstrap["total_return"].gt(0).mean()),
    }

    index_panel = pd.read_parquet(data_dir / "index_panel.parquet")
    descriptive_groups, descriptive = _descriptive_analysis(
        baseline_backtest.daily_account, index_panel, output_dir
    )
    variants = pd.DataFrame(variant_records)
    variants.to_csv(output_dir / "variant_matrix.csv", index=False)

    neighborhood = variants.loc[variants["family"] == "neighborhood"]
    random_order = variants.loc[variants["family"] == "random_order"]
    neighborhood_positive = int(neighborhood["total_return"].gt(0).sum())
    random_order_positive = int(random_order["total_return"].gt(0).sum())
    hithink_return = hithink_common_summary["total_return"]
    baostock_return = baostock_common_summary["total_return"]
    opposite_direction = (hithink_return > 0 > baostock_return) or (
        baostock_return > 0 > hithink_return
    )
    baseline_integrity = execution_integrity(
        main_window, baseline_signals, baseline_backtest, config
    )
    double_integrity = execution_integrity(
        main_window,
        baseline_signals,
        double_backtest,
        replace(
            config,
            commission_rate=config.commission_rate * 2,
            min_commission=config.min_commission * 2,
            slippage_bps=20,
        ),
        fee_multiplier=2.0,
    )
    risk_summary = build_risk_report(
        main_window,
        baseline_signals,
        baseline_backtest,
        config,
        output_dir / "risk",
    )
    drawdown = float(baseline_summary["max_drawdown"])
    gates = [
        _gate(
            "data_execution_integrity",
            baseline_integrity["all_passed"],
            baseline_integrity,
            "数据与执行关键证据逐项通过：键/窗口、现金、整手、既定开盘价/滑点/夹取/费用/金额、权益恒等式和次一账户日/合法顺延",
        ),
        _gate(
            "main_window_positive",
            baseline_summary["total_return"] > 0,
            baseline_summary["total_return"],
            "> 0",
        ),
        _gate(
            "rolling_3y_majority_positive",
            len(baseline_summary["rolling_3y_returns"]) > 0
            and baseline_summary["rolling_3y_positive_share"] > 0.5,
            {
                "positive_share": baseline_summary["rolling_3y_positive_share"],
                "windows": len(baseline_summary["rolling_3y_returns"]),
            },
            "> 0.5",
        ),
        _gate(
            "double_cost_positive_and_executable",
            double_summary["total_return"] > 0 and double_integrity["all_passed"],
            {"total_return": double_summary["total_return"], "execution": double_integrity},
            "> 0，且所有成交保持既定开盘规则与收盘后时序",
        ),
        _gate(
            "neighborhood_at_least_6_positive",
            neighborhood_positive >= 6,
            {"positive": neighborhood_positive, "total": len(neighborhood)},
            ">= 6 / 8",
        ),
        _gate(
            "random_order_at_least_16_positive",
            random_order_positive >= 16,
            {"positive": random_order_positive, "total": len(random_order)},
            ">= 16 / 20",
        ),
        _gate(
            "remaining_years_equity_pnl_positive",
            remaining_equity_pnl > 0,
            profit_concentration,
            "> 0；使用账户年度权益盈亏",
        ),
        _gate(
            "fully_after_close_reproducible",
            baseline_integrity["key_set_and_window"]
            and baseline_integrity["next_open_after_signal"]
            and baseline_integrity["legal_fill_prices"],
            baseline_integrity,
            "键/窗口、既定开盘成交与收盘后次一账户日/合法顺延证据分别通过，无隐藏盘中条件",
        ),
        _gate(
            "hithink_common_positive_mdd_and_direction",
            hithink_return > 0
            and hithink_common_summary["max_drawdown"] <= 0.20
            and not opposite_direction,
            {
                "hithink_total_return": hithink_return,
                "hithink_max_drawdown": hithink_common_summary["max_drawdown"],
                "baostock_common_total_return": baostock_return,
                "opposite_direction": opposite_direction,
            },
            "同花顺共同窗口净收益>0且MDD<=0.20，两源方向不得相反",
        ),
        _gate(
            "drawdown_boundary",
            drawdown <= 0.20,
            drawdown,
            "MDD<=0.10且其他门槛通过为UNREJECTED；0.10<MDD<=0.20为BOUNDARY；MDD>0.20为REJECTED",
        ),
    ]
    summary = {
        "started_at": started_at,
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "baseline_record": baseline_record,
        "baseline": baseline_summary,
        "max_drawdown": drawdown,
        "variants": variant_records,
        "annual_equity": annual_equity.to_dict("records"),
        "profit_concentration": profit_concentration,
        "common_window": common_window,
        "bootstrap": bootstrap_summary,
        "descriptive": descriptive,
        "execution_integrity": {
            "baseline": baseline_integrity,
            "double_cost": double_integrity,
        },
        "alignment": {
            "method": "所有窗口按signals自身date筛选，并以(date,code)一对一连接；执行前硬断言",
            "baseline_panel_rows": int(len(main_window)),
            "baseline_signal_rows": int(len(baseline_signals)),
            "baseline_key_difference_count": 0,
            "dual_source_key_difference_count": 0,
        },
        "risk_report": risk_summary,
        "input_manifest": input_manifest,
        "gates": gates,
        "outputs": {
            "robustness": _relative(output_dir),
            "stress": _relative(stress_output_dir),
        },
    }
    summary["verdict"] = evaluate_gates(summary)
    summary["run_status"] = {
        "status": "main_completed_stress_pending",
        "completed_real_variants": len(variant_records),
        "stress_executed": False,
    }
    _write_json(output_dir / "main_summary.json", summary)
    freeze = _freeze_main(output_dir, config)

    # This is deliberately the last strategy run.  It cannot alter any main result.
    stress_config = replace(config, start_date="2010-01-01", end_date="2015-12-31")
    stress_panel = load_panel(data_dir / "stress_panel.parquet")
    stress_signals = compute_signals(stress_panel, stress_config)
    stress_panel, stress_signals = select_backtest_window(
        stress_panel, stress_signals, stress_config
    )
    _, stress_performance, _ = _run_and_save(
        family="historical_stress",
        name="stress_2010_2015",
        panel=stress_panel,
        signals=stress_signals,
        corporate_actions=corporate_actions,
        config=stress_config,
        output_dir=stress_output_dir,
        changed_fields={"start_date": "2010-01-01", "end_date": "2015-12-31"},
    )
    stress_summary = {
        **stress_performance,
        "status": "completed",
        "source_mode": "baostock_single_source",
        "limitation": "2010—2015仅单源历史压力，不承担双源确认，不用于回头调参或替换基线",
        "output_dir": _relative(stress_output_dir),
    }
    summary["stress"] = stress_summary
    summary["main_freeze"] = {
        "path": _relative(output_dir / "main_freeze.json"),
        "config_sha256": freeze["config_sha256"],
        "frozen_file_count": len(freeze["files"]),
        "created_at": freeze["created_at"],
    }
    summary["run_status"] = {
        "status": "completed",
        "completed_real_variants": len(variant_records),
        "bootstrap_return_resamples": len(bootstrap),
        "stress_executed": True,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output_dir / "summary.json", summary)
    report_path = render_final_report(summary, root / "90_研究结论.md")
    return RobustnessResult(
        summary=summary,
        variants=variants,
        bootstrap=bootstrap,
        descriptive=descriptive_groups,
        verdict=summary["verdict"],
        output_dir=output_dir,
        stress_output_dir=stress_output_dir,
        report_path=report_path,
    )
