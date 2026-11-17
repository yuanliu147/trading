"""Run the frozen legacy candidate A through the audited account ledger."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

from .broker import BacktestResult, execution_integrity, run_backtest
from .data import load_panel
from .legacy_policy import LegacyPolicy
from .metrics import annual_and_rolling, performance_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_A = "three_of_two_low_vol"
SIGNAL_COLUMNS = (
    "date",
    "code",
    "entry_signal",
    "legacy_rank_score",
    "rank_return_120",
    "stop_ref",
    "ma5",
    "ma20",
    "whist",
    "signal_close",
)
TABLE_FILES = {
    "daily_account": "daily_account.parquet",
    "daily_positions": "daily_positions.parquet",
    "trades": "trades.parquet",
    "orders": "orders.parquet",
    "round_trips": "round_trips.parquet",
    "action_log": "action_log.parquet",
}


@dataclass(frozen=True)
class LegacyRunResult:
    candidate_id: str
    status: str
    output_dir: Path
    backtest: BacktestResult
    summary: dict
    annual_rolling: pd.DataFrame
    manifest: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(frame: pd.DataFrame) -> None:
    keys = frame[["date", "code"]]
    if keys.duplicated().any():
        raise ValueError("Duplicate (date, code) keys")


def prepare_legacy_frames(
    panel: pd.DataFrame, signals: pd.DataFrame, config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the frozen window, derive the panel factor and join by exact keys."""
    panel = panel.copy()
    signals = signals.copy()
    panel["date"] = panel["date"].astype(str)
    signals["date"] = signals["date"].astype(str)
    if "adjustment_factor" not in panel:
        positive = (panel["close"] > 0) & (panel["signal_close"] > 0)
        panel["adjustment_factor"] = (
            panel["signal_close"] / panel["close"]
        ).where(positive)

    panel = panel.loc[
        panel["date"].between(config.start_date, config.end_date)
    ].reset_index(drop=True)
    signals = signals.loc[
        signals["date"].between(config.start_date, config.end_date)
    ].reset_index(drop=True)
    _reject_duplicate_keys(panel)
    _reject_duplicate_keys(signals)

    panel_keys = set(zip(panel["date"], panel["code"]))
    signal_keys = set(zip(signals["date"], signals["code"]))
    if panel_keys != signal_keys or len(panel) != len(signals):
        raise ValueError(
            "Panel and signal key sets differ: "
            f"panel={len(panel)} signals={len(signals)} "
            f"panel_only={len(panel_keys - signal_keys)} "
            f"signal_only={len(signal_keys - panel_keys)}"
        )
    if set(panel["date"]) != set(signals["date"]):
        raise ValueError("Panel and signal date sets differ")

    entries = signals["entry_signal"].astype(bool)
    entry_keys = signals.loc[entries, ["date", "code"]]
    entry_factors = entry_keys.merge(
        panel[["date", "code", "adjustment_factor"]],
        on=["date", "code"],
        how="left",
        validate="one_to_one",
    )["adjustment_factor"]
    if entry_factors.isna().any() or (entry_factors <= 0).any():
        raise ValueError("Entry signals require finite positive adjustment factors")
    return panel, signals


def run_legacy_frames(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    config,
) -> BacktestResult:
    prepared_panel, prepared_signals = prepare_legacy_frames(panel, signals, config)
    return run_backtest(
        prepared_panel,
        prepared_signals,
        corporate_actions,
        config,
        policy=LegacyPolicy(config),
    )


def validate_legacy_status(summary: dict, candidate_id: str) -> None:
    if candidate_id != CANDIDATE_A:
        raise ValueError(f"Candidate {candidate_id!r} is not evaluable in this stage")
    if summary.get("candidate_a_status") != "EVALUABLE":
        raise ValueError("Candidate A data gate is not EVALUABLE")
    if summary.get("account_backtest_allowed") is not True:
        raise ValueError("Account backtest is not allowed by the data gate")


def _load_verified_summary(root: Path) -> dict:
    summary_path = root / "results" / "legacy_a_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_legacy_status(summary, CANDIDATE_A)
    registered = summary["outputs"]["signals"]
    actual_hash = _sha256(Path(registered["path"]))
    if actual_hash != registered["sha256"]:
        raise ValueError("Frozen candidate-A signal hash does not match its summary")
    coverage = summary["outputs"]["entry_coverage"]
    if _sha256(Path(coverage["path"])) != coverage["sha256"]:
        raise ValueError("Frozen candidate-A coverage hash does not match its summary")
    if summary.get("account_returns_run") is not False:
        raise ValueError("Candidate-A summary must predate any account return run")
    return summary


def _accounting_checks(
    panel: pd.DataFrame, signals: pd.DataFrame, result: BacktestResult, config
) -> dict:
    account = result.daily_account
    trades = result.trades
    buys = trades.loc[trades["side"] == "buy"]
    return {
        "cash_nonnegative": bool((account["cash"] >= -1e-8).all()),
        "whole_lot_buys": bool((buys["quantity"] % 100 == 0).all()),
        "equity_identity": bool(
            (
                (account["equity"] - account["cash"] - account["receivables"]
                 - account["market_value"]).abs()
                <= 1e-6
            ).all()
        ),
        "trades_after_signal_date": bool(
            (trades["date"] > trades["signal_date"]).all()
        ),
        "no_signal_day_trade": bool(
            not (trades["date"] == trades["signal_date"]).any()
        ),
        "execution_integrity": execution_integrity(
            panel, signals, result, config
        ),
    }


def _manifest(
    config,
    frozen_summary: dict,
    performance: dict,
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    result: BacktestResult,
    checks: dict,
    root: Path,
) -> dict:
    config_path = root / "config" / "legacy_candidates.json"
    signal_path = Path(frozen_summary["outputs"]["signals"]["path"])
    coverage_path = Path(frozen_summary["outputs"]["entry_coverage"]["path"])
    panel_path = root / "data" / "main_panel.parquet"
    actions_path = root / "data" / "corporate_actions.parquet"
    code_files = [
        "src/trend_system/legacy_runner.py",
        "src/trend_system/legacy_policy.py",
        "src/trend_system/legacy_a.py",
        "src/trend_system/broker.py",
        "src/trend_system/execution.py",
    ]
    return {
        "candidate_id": CANDIDATE_A,
        "status": "COMPLETED",
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "inputs": {
            "main_panel": {"path": str(panel_path), "sha256": _sha256(panel_path)},
            "signals": {
                "path": str(signal_path),
                "sha256": _sha256(signal_path),
                "rows": int(len(signals)),
            },
            "entry_coverage": {
                "path": str(coverage_path),
                "sha256": _sha256(coverage_path),
            },
            "corporate_actions": {
                "path": str(actions_path),
                "sha256": _sha256(actions_path),
            },
            "candidate_a_summary": {
                "path": str(root / "results" / "legacy_a_summary.json"),
                "sha256": _sha256(root / "results" / "legacy_a_summary.json"),
            },
        },
        "code": {
            name: {"sha256": _sha256(root / name)} for name in code_files
        },
        "window": {
            "start_date": str(panel["date"].min()),
            "end_date": str(panel["date"].max()),
        },
        "rows": {
            "panel": int(len(panel)),
            "signals": int(len(signals)),
            "daily_account": int(len(result.daily_account)),
            "trades": int(len(result.trades)),
            "orders": int(len(result.orders)),
            "round_trips": int(len(result.round_trips)),
        },
        "keys": {
            "panel_unique": int(len(set(zip(panel["date"], panel["code"])))),
            "signal_unique": int(len(set(zip(signals["date"], signals["code"])))),
        },
        "policy": {
            "class": "LegacyPolicy",
            "ordering": "legacy_rank_score DESC, code ASC",
            "max_slots": 4,
            "max_weight": 0.20,
            "max_new_per_day": 2,
            "stop_pct": 0.08,
            "max_hold": 40,
            "skip_if_stop_at_or_above_entry": True,
        },
        "checks": checks,
        "summary": performance,
        "returns_run": True,
    }


def run_legacy_candidate(candidate_id: str, config, root: Path | None = None) -> LegacyRunResult:
    """Run the single frozen candidate-A main version on hash-verified inputs."""
    root = PROJECT_ROOT if root is None else Path(root)
    frozen_summary = _load_verified_summary(root)
    validate_legacy_status(frozen_summary, candidate_id)

    panel = load_panel(root / "data" / "main_panel.parquet")
    signals = pd.read_parquet(
        frozen_summary["outputs"]["signals"]["path"], columns=list(SIGNAL_COLUMNS)
    )
    actions = pd.read_parquet(root / "data" / "corporate_actions.parquet")
    prepared_panel, prepared_signals = prepare_legacy_frames(panel, signals, config)
    backtest = run_backtest(
        prepared_panel,
        prepared_signals,
        actions,
        config,
        policy=LegacyPolicy(config),
    )
    summary = performance_summary(backtest.daily_account, backtest.round_trips)
    annual_rolling = annual_and_rolling(backtest.daily_account)
    checks = _accounting_checks(prepared_panel, prepared_signals, backtest, config)

    output_directory = root / "results" / "legacy_candidates" / candidate_id
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, filename in TABLE_FILES.items():
        getattr(backtest, name).to_parquet(output_directory / filename, index=False)
    annual_rolling.to_csv(output_directory / "annual_and_rolling.csv", index=False)
    (output_directory / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        config, frozen_summary, summary, prepared_panel, prepared_signals, backtest, checks, root
    )
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return LegacyRunResult(
        candidate_id=candidate_id,
        status="COMPLETED",
        output_dir=output_directory,
        backtest=backtest,
        summary=summary,
        annual_rolling=annual_rolling,
        manifest=manifest,
    )
