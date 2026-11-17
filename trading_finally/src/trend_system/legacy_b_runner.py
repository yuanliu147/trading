"""Run the recovered legacy candidate B through the audited account ledger."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .broker import BacktestResult, execution_integrity, run_backtest
from .data import load_panel
from .legacy_b_policy import LegacyBPolicy
from .legacy_runner import PROJECT_ROOT, _sha256, prepare_legacy_frames
from .metrics import annual_and_rolling, performance_summary


CANDIDATE_B = "early_trend_breadth"


@dataclass(frozen=True)
class LegacyBRunResult:
    status: str
    output_dir: Path
    backtest: BacktestResult
    summary: dict
    annual_rolling: pd.DataFrame
    manifest: dict


def _checks(panel, signals, backtest, config) -> dict:
    account = backtest.daily_account
    trades = backtest.trades
    buys = trades.loc[trades["side"] == "buy"]
    return {
        "cash_nonnegative": bool((account["cash"] >= -1e-8).all()),
        "whole_lot_buys": bool((buys["quantity"] % 100 == 0).all()),
        "equity_identity": bool(
            (
                (account["equity"] - account["cash"] - account["receivables"]
                 - account["market_value"]).abs() <= 1e-6
            ).all()
        ),
        "trades_after_signal_date": bool((trades["date"] > trades["signal_date"]).all()),
        "execution_integrity": execution_integrity(panel, signals, backtest, config),
    }


def run_legacy_b(config, config_path=None, root: Path | None = None) -> LegacyBRunResult:
    root = PROJECT_ROOT if root is None else Path(root)
    summary_path = root / "results" / "legacy_b_summary.json"
    audit_path = root / "results" / "legacy_b_recovery_audit.json"
    if not audit_path.exists():
        raise ValueError("Candidate B recovery audit is missing")
    recovery = json.loads(audit_path.read_text(encoding="utf-8"))
    if recovery.get("restored") is not True or recovery.get("status") != "EVALUABLE":
        raise ValueError("Candidate B recovery audit is not restored/EVALUABLE")
    official_config = root / "config" / "legacy_b_recovery.json"
    if recovery["config"]["sha256"] != _sha256(official_config):
        raise ValueError("Candidate B recovery audit does not match the official config")
    if config_path is None or _sha256(Path(config_path)) != recovery["config"]["sha256"]:
        raise ValueError("Candidate B actual --config does not match the recovery audit")
    frozen = json.loads(summary_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "EVALUABLE":
        raise ValueError("Candidate B data gate is not EVALUABLE")
    if frozen.get("account_returns_run") is not False:
        raise ValueError("Candidate B summary must predate any account return run")
    signal_info = frozen["outputs"]["signals"]
    if _sha256(Path(signal_info["path"])) != signal_info["sha256"]:
        raise ValueError("Frozen candidate-B signal hash does not match its summary")

    panel = load_panel(root / "data" / "main_panel.parquet")
    signals = pd.read_parquet(
        signal_info["path"],
        columns=["date", "code", "entry_signal", "atr_pct", "amt20",
                 "legacy_rank_score", "rank_return_120", "stop_pct"],
    )
    actions = pd.read_parquet(root / "data" / "corporate_actions.parquet")
    prepared_panel, prepared_signals = prepare_legacy_frames(panel, signals, config)
    backtest = run_backtest(
        prepared_panel, prepared_signals, actions, config, policy=LegacyBPolicy(config)
    )
    summary = performance_summary(backtest.daily_account, backtest.round_trips)
    annual = annual_and_rolling(backtest.daily_account)
    checks = _checks(prepared_panel, prepared_signals, backtest, config)

    output_dir = root / "results" / "legacy_candidates" / CANDIDATE_B
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("daily_account", "daily_positions", "trades", "orders",
                 "round_trips", "action_log"):
        getattr(backtest, name).to_parquet(output_dir / f"{name}.parquet", index=False)
    annual.to_csv(output_dir / "annual_and_rolling.csv", index=False)
    (output_dir / "performance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "candidate": CANDIDATE_B,
        "status": "COMPLETED",
        "window": {
            "start_date": str(prepared_panel["date"].min()),
            "end_date": str(prepared_panel["date"].max()),
        },
        "rows": {
            "panel": int(len(prepared_panel)),
            "signals": int(len(prepared_signals)),
            "daily_account": int(len(backtest.daily_account)),
            "trades": int(len(backtest.trades)),
            "orders": int(len(backtest.orders)),
            "round_trips": int(len(backtest.round_trips)),
        },
        "config": {"path": str(root / "config" / "legacy_b_recovery.json"),
                   "sha256": _sha256(root / "config" / "legacy_b_recovery.json")},
        "recovery_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
        "code": {
            name: {"sha256": _sha256(root / name)} for name in (
                "src/trend_system/legacy_b_runner.py",
                "src/trend_system/legacy_b_policy.py",
                "src/trend_system/legacy_b.py",
                "src/trend_system/legacy_b_config.py",
                "src/trend_system/broker.py",
                "src/trend_system/execution.py",
            )
        },
        "inputs": {
            "signals": signal_info,
            "candidate_b_summary": {
                "path": str(summary_path), "sha256": _sha256(summary_path)
            },
            "main_panel": {
                "path": str(root / "data" / "main_panel.parquet"),
                "sha256": _sha256(root / "data" / "main_panel.parquet"),
            },
            "corporate_actions": {
                "path": str(root / "data" / "corporate_actions.parquet"),
                "sha256": _sha256(root / "data" / "corporate_actions.parquet"),
            },
        },
        "policy": {
            "class": "LegacyBPolicy",
            "ordering": "amt20 ASC, code ASC",
            "max_slots": 6,
            "max_weight": 0.20,
            "risk_per_trade": 0.005,
            "heat_reject_threshold": 0.0301,
            "max_hold": 20,
            "catastrophe_stop_pct": 0.20,
            "gap_filter": "skip entry if open > prev_close * 1.05",
        },
        "checks": checks,
        "summary": summary,
        "returns_run": True,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return LegacyBRunResult(
        status="COMPLETED", output_dir=output_dir, backtest=backtest,
        summary=summary, annual_rolling=annual, manifest=manifest,
    )
