from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from flash_data import CORE, EVENTS, INITIAL, PANEL, ROOT, load_config, load_flash_data, literal, sha256
from flash_engine import EngineConfig, annual_returns, drawdown_spells, run_backtest

AUDIT = ROOT / "flash 审计修复"
RESULTS = AUDIT / "results"
EVIDENCE = AUDIT / "audit"
LEGACY = AUDIT / "legacy"
MOTHER = ROOT.parent / "trading_flash" / "2026-10-24_主板恐慌反转_年化30"
SECOND = ROOT.parent / "trading_flash" / "不同时间入场点回测"
ACCOUNT_FILES = [
    "daily_equity.csv", "trades.csv", "orders.csv", "corporate_actions.csv",
    "positions_daily.csv", "open_positions.csv", "annual.csv", "drawdowns.csv",
    "metrics.json", "run_config.json",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def file_manifest(path: Path) -> dict[str, Any]:
    return {
        name: {"sha256": sha256(path / name), "bytes": (path / name).stat().st_size}
        for name in ACCOUNT_FILES
    }


def capture_baseline() -> None:
    sources = {
        "panic_lib.py": MOTHER / "02_代码" / "panic_lib.py",
        "actual_engine_fixed.py": MOTHER / "02_代码" / "actual_engine_fixed.py",
        "verify_delivery.py": MOTHER / "02_代码" / "verify_delivery.py",
        "verify_actual_fixed.py": MOTHER / "02_代码" / "verify_actual_fixed.py",
        "timing_lib.py": SECOND / "02_代码" / "timing_lib.py",
        "timing_verify_delivery.py": SECOND / "02_代码" / "verify_delivery.py",
        "mother_README.md": MOTHER / "README.md",
        "mother_reproduction.md": MOTHER / "03_数据与复现.md",
        "timing_README.md": SECOND / "README.md",
        "original_actual_trades.csv": MOTHER / "03_结果" / "实际价_修正版_逐笔.csv",
        "original_actual_equity.csv": MOTHER / "03_结果" / "实际价_修正版_每日权益.csv",
        "original_actual_metrics.json": MOTHER / "03_结果" / "实际价_修正版_指标.json",
    }
    LEGACY.mkdir(parents=True, exist_ok=True)
    manifest_path = LEGACY / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = {}
    for name, source in sources.items():
        target = LEGACY / name
        if target.exists():
            if name in previous and sha256(target) != previous[name]["sha256"]:
                raise RuntimeError(f"Legacy evidence was modified: {name}")
            records[name] = previous.get(name, {"source": str(source), "sha256": sha256(target), "bytes": target.stat().st_size})
            continue
        if name in previous:
            raise RuntimeError(f"Legacy evidence is missing: {name}")
        shutil.copy2(source, target)
        records[name] = {"source": str(source), "sha256": sha256(source), "bytes": source.stat().st_size}
    write_json(manifest_path, records)


def baseline_evidence() -> None:
    equity = pd.read_csv(MOTHER / "03_结果" / "实际价_修正版_每日权益.csv", parse_dates=["date"])
    trades = pd.read_csv(MOTHER / "03_结果" / "实际价_修正版_逐笔.csv", parse_dates=["entry_date", "exit_date"])
    con = duckdb.connect()
    field_types = con.execute(
        f"""SELECT min(typeof(up_limit_05)) AS up_limit_05_type,
                   min(typeof(limit_up_05)) AS limit_up_05_type,
                   count(*) FILTER (up_limit_05 = 1) AS legacy_price_equal_one,
                   count(*) FILTER (limit_up_05) AS boolean_sample_count
            FROM read_parquet('{literal(PANEL)}')"""
    ).fetchdf().iloc[0].to_dict()
    signal_day = con.execute(
        f"""SELECT count(*) FILTER (
                        drawdown250 < -.45 AND return_250d > 0 AND rsi14 < 35
                        AND close_location > .45 AND market_return20 < -.05
                    ) AS legacy_unfiltered,
                   count(*) FILTER (
                        drawdown250 < -.45 AND return_250d > 0 AND rsi14 < 35
                        AND close_location > .45 AND market_return20 < -.05
                        AND NOT limit_up_05
                    ) AS explicit_5pct_sample_filter
            FROM read_parquet('{literal(PANEL)}')
            WHERE date = '2026-10-23' AND amount20 >= 10000000"""
    ).fetchdf().iloc[0].to_dict()
    limit_case = con.execute(
        f"""SELECT thscode, date, open, high, low, close, reference_close,
                   down_limit_10, down_limit_05, volume
            FROM read_parquet('{literal(CORE)}')
            WHERE thscode = '000595.SZ' AND date BETWEEN '2025-01-27' AND '2025-02-05'
            ORDER BY date"""
    ).fetchdf()
    con.close()
    evidence = {
        "negative_cash_days": int((equity["cash"] < -0.005).sum()),
        "minimum_cash": float(equity["cash"].min()),
        "negative_cash_dates": [str(value.date()) for value in equity.loc[equity["cash"] < -0.005, "date"]],
        "original_capital": 100000,
        "maximum_entry_price": float(trades["entry_px"].max()),
        "maximum_entry": trades.loc[trades["entry_px"].idxmax(), ["thscode", "entry_date", "entry_px"]].to_dict(),
        "up_limit_05_field": field_types,
        "signal_day_2026_10_23": signal_day,
        "limit_down_000595": limit_case.to_dict("records"),
        "out_of_sample_note": "The 2024-2026 segment was used in grid evaluation and selection; no untouched segment remains.",
        "wencai_note": "The manual Wencai cross-check is excluded from this local-only audit.",
        "survivorship_note": "Delisted stocks are not reconstructed, as requested; this is disclosed rather than silently repaired.",
    }
    write_json(EVIDENCE / "baseline_evidence.json", evidence)


def code_hashes() -> dict[str, str]:
    return {path.name: sha256(path) for path in sorted(AUDIT.glob("*.py"))}


def input_manifest(config: dict[str, Any]) -> dict[str, Any]:
    core_metadata = json.loads((INITIAL / "data" / "work" / "metadata.json").read_text())
    snapshot_manifest = json.loads((INITIAL / "data" / "snapshot" / "manifest.json").read_text())
    panel_expected = (MOTHER / "05_证据" / "关键文件sha256.txt").read_text().splitlines()[0].split()[0]
    hashes = {"panel": sha256(PANEL), "core": sha256(CORE), "events": sha256(EVENTS)}
    if hashes["panel"] != panel_expected:
        raise AssertionError("Frozen research panel hash mismatch")
    if hashes["core"] != core_metadata["files"]["core"]["sha256"]:
        raise AssertionError("Core hash mismatch")
    if hashes["events"] != snapshot_manifest["exports"]["adjustment_events"]["file_sha256"]:
        raise AssertionError("Corporate action snapshot hash mismatch")
    feature_link = MOTHER / "01_数据" / "features.parquet"
    expected_link = INITIAL / "data" / "work" / "features.parquet"
    if not feature_link.is_symlink() or feature_link.resolve() != expected_link.resolve():
        raise AssertionError("Migrated feature symlink is incorrect")
    return {
        "data_policy": "local_hithink_duckdb_derived_frozen_files_only",
        "remote_data_used": False,
        "third_party_data_used": False,
        "input_sha256": hashes,
        "strategy_sha256": sha256(AUDIT / "config" / "strategy.json"),
        "code_sha256": code_hashes(),
        "feature_link": str(feature_link),
        "feature_link_target": str(feature_link.resolve()),
    }


def scenarios() -> list[dict[str, Any]]:
    result = []
    for capital in [20000, 50000, 100000]:
        result.append({
            "scenario": f"O03_c{capital}", "kind": "baseline", "capital": capital,
            "entry_lag_sessions": 3, "limit_rate": 0.10, "take_profit_mode": "intraday",
        })
    for lag in range(1, 16):
        if lag == 3:
            continue
        result.append({
            "scenario": f"O{lag:02d}_c100000", "kind": "timing_sensitivity", "capital": 100000,
            "entry_lag_sessions": lag, "limit_rate": 0.10, "take_profit_mode": "intraday",
        })
    for mode, values in [
        ("limit05", {"limit_rate": 0.05}),
        ("tpnext", {"take_profit_mode": "next_open"}),
        ("liquidity001", {"max_participation": 0.001}),
    ]:
        for capital in [20000, 50000, 100000]:
            result.append({
                "scenario": f"O03_c{capital}_{mode}", "kind": "execution_stress", "capital": capital,
                "entry_lag_sessions": 3, "limit_rate": 0.10, "take_profit_mode": "intraday", **values,
            })
    return result


def export_result(result: Any, scenario: dict[str, Any], execution: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    output = RESULTS / "accounts" / scenario["scenario"]
    output.mkdir(parents=True, exist_ok=True)
    result.daily_equity.to_csv(output / "daily_equity.csv", index=False)
    result.trades.to_csv(output / "trades.csv", index=False)
    result.orders.to_csv(output / "orders.csv", index=False)
    result.corporate_actions.to_csv(output / "corporate_actions.csv", index=False)
    result.positions_daily.to_csv(output / "positions_daily.csv", index=False)
    result.open_positions.to_csv(output / "open_positions.csv", index=False)
    annual_returns(result.daily_equity, scenario["capital"]).to_csv(output / "annual.csv", index=False)
    drawdown_spells(result.daily_equity, scenario["capital"]).to_csv(output / "drawdowns.csv", index=False)
    metrics = dict(result.metrics)
    metrics["closed_plus_open"] = metrics["trade_count"] + metrics["ending_open_positions"]
    metrics.update({key: scenario[key] for key in ["scenario", "kind", "capital", "entry_lag_sessions", "limit_rate", "take_profit_mode"]})
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_config.json", {
        "strategy_id": "flash_panic_reversal_fixed_o03",
        "scenario": scenario,
        "execution": execution,
        "source": source,
        "evaluation_status": "historical_repair_not_independent_holdout",
    })
    write_json(output / "manifest.json", file_manifest(output))
    return metrics


def run_all() -> pd.DataFrame:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    config = load_config()
    if config["selection_policy"] != "frozen_before_repair_no_parameter_search":
        raise RuntimeError("Strategy is not frozen")
    capture_baseline()
    baseline_evidence()
    inputs = input_manifest(config)
    write_json(EVIDENCE / "input_manifest.json", inputs)
    write_json(EVIDENCE / "run_status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    data = load_flash_data(config)
    inputs["source"] = data.source
    write_json(EVIDENCE / "input_manifest.json", inputs)
    candidate_rows = []
    for signal_index, candidates in data.candidates.items():
        for candidate in candidates:
            candidate_rows.append({
                "signal_date": data.dates[candidate.signal_index],
                "thscode": data.symbols[candidate.symbol], "rank": candidate.rank,
                "score": candidate.score, "amount20": candidate.amount20,
            })
    candidate_frame = pd.DataFrame(candidate_rows).sort_values(["signal_date", "rank", "thscode"])
    candidate_frame.to_csv(RESULTS / "entry_candidates.csv", index=False)
    signal_check = {
        "signal_count": data.source["signal_count"],
        "candidate_rows": len(candidate_frame),
        "candidate_symbols": len(data.symbols),
        "candidate_sha256": sha256(RESULTS / "entry_candidates.csv"),
        "last_date": str(data.dates[-1].date()),
        "last_day_top2": candidate_frame[candidate_frame["signal_date"] == data.dates[-1]]["thscode"].head(2).tolist(),
        "up_limit_05_semantics": "price, not a boolean; baseline preserves the effective unfiltered legacy pool",
    }
    write_json(EVIDENCE / "signal_check.json", signal_check)
    rows = []
    annual_frames = []
    drawdown_frames = []
    for scenario in scenarios():
        execution = dict(config["execution"])
        execution.update({
            "initial_capital": scenario["capital"],
            "entry_lag_sessions": scenario["entry_lag_sessions"],
            "limit_rate": scenario["limit_rate"],
            "take_profit_mode": scenario["take_profit_mode"],
        })
        if scenario.get("max_participation") is not None:
            execution["max_participation"] = scenario["max_participation"]
        engine_config = EngineConfig.from_dict(execution)
        result = run_backtest(data, engine_config)
        metrics = export_result(result, scenario, execution, data.source)
        rows.append(metrics)
        annual = annual_returns(result.daily_equity, scenario["capital"])
        annual.insert(0, "scenario", scenario["scenario"])
        annual_frames.append(annual)
        spells = drawdown_spells(result.daily_equity, scenario["capital"])
        spells.insert(0, "scenario", scenario["scenario"])
        drawdown_frames.append(spells)
        print(
            f"{scenario['scenario']:27s} CAGR={metrics['cagr']:.4%} "
            f"DD={metrics['max_drawdown']:.4%} closed={metrics['trade_count']} "
            f"open={metrics['ending_open_positions']}", flush=True,
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    pd.concat(annual_frames, ignore_index=True).to_csv(RESULTS / "annual_returns.csv", index=False)
    pd.concat(drawdown_frames, ignore_index=True).to_csv(RESULTS / "drawdown_spells.csv", index=False)
    original = json.loads((MOTHER / "03_结果" / "实际价_修正版_指标.json").read_text())
    repaired = summary[summary["scenario"] == "O03_c100000"].iloc[0]
    comparison = pd.DataFrame([
        {"version": "original_actual_replay", "cagr": original["cagr"], "max_drawdown": original["dd"], "final_equity": original["final"], "trade_count": original["n"], "ending_open_positions": 2},
        {"version": "strict_dynamic_account", "cagr": repaired["cagr"], "max_drawdown": repaired["max_drawdown"], "final_equity": repaired["final_equity"], "trade_count": repaired["trade_count"], "ending_open_positions": repaired["ending_open_positions"]},
    ])
    comparison.to_csv(RESULTS / "original_vs_repaired_100k.csv", index=False)
    if code_hashes() != inputs["code_sha256"]:
        raise RuntimeError("Code changed while the audit was running")
    manifest = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(rows),
        "scenarios": [item["scenario"] for item in rows],
        "strategy_sha256": inputs["strategy_sha256"],
        "code_sha256": inputs["code_sha256"],
        "input_sha256": inputs["input_sha256"],
        "engine_reconciliation_passed": bool(summary["final_reconciliation_passed"].all()),
        "evaluation_status": "historical_repair_not_independent_holdout",
    }
    write_json(EVIDENCE / "run_manifest.json", manifest)
    write_json(EVIDENCE / "run_status.json", {"status": "completed", **manifest})
    return summary


if __name__ == "__main__":
    run_all()
