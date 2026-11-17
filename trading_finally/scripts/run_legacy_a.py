"""Compute and audit candidate A exclusively from hash-verified frozen Parquet."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from trend_system.legacy_a import audit_legacy_a_entries, compute_legacy_a_signals
from trend_system.legacy_config import load_legacy_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_INPUTS = {
    "full_a_completed": {
        "path": PROJECT_ROOT / "data" / "legacy_full_a_panel_completed.parquet",
        "sha256": "12e92da7600f59c72851da194ac13fca237d593d1cb8c2a0c329e4c28c23d9fb",
    },
    "main": {
        "path": PROJECT_ROOT / "data" / "main_panel.parquet",
        "sha256": "6a51ed0e7958aa1549dcf156e4387badcfced0236fd812d2bc1fd894aa3acc1a",
    },
    "index": {
        "path": PROJECT_ROOT / "data" / "index_panel.parquet",
        "sha256": "8b8763e0a552894eee84407f1b0c5e6efa4ff7eb8640ddb149b637539bef387a",
    },
    "context_audit": {
        "path": PROJECT_ROOT / "results" / "legacy_context_audit.json",
        "sha256": "d2e375ae169f078522d8b979dbea13f457b678d5325d3c77f2a8877d37d56c98",
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_inputs():
    result = {}
    for name, item in FROZEN_INPUTS.items():
        path = item["path"]
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {name}")
        result[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    return result


def _write_parquet(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return {"path": str(path), "rows": len(frame), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_legacy_config(args.config)
    candidate = next(
        item for item in config.candidates if item.name == "three_of_two_low_vol"
    )
    inputs_before = _verify_inputs()
    context_audit = json.loads(
        FROZEN_INPUTS["context_audit"]["path"].read_text(encoding="utf-8")
    )
    full_a_gate = bool(context_audit["coverage"]["full_a"]["passes"])
    task2_status = context_audit["candidate_statuses"]["three_of_two_low_vol"]
    if not full_a_gate or task2_status != "PENDING_DATA_GATE":
        raise ValueError("Task 2 full-A gate does not permit candidate-A signal computation")

    context = pd.read_parquet(FROZEN_INPUTS["full_a_completed"]["path"])
    main_panel = pd.read_parquet(FROZEN_INPUTS["main"]["path"])
    index = pd.read_parquet(
        FROZEN_INPUTS["index"]["path"], columns=["date", "code", "close"]
    )
    signals = compute_legacy_a_signals(context, main_panel, index, config)
    del main_panel, index
    entry_audit, coverage = audit_legacy_a_entries(signals, context, config)

    entry = signals["entry_signal"].eq(True)
    in_window = signals["date"].between(config.start_date, config.end_date)
    checks = {
        "non_main_board_entries": int((entry & ~signals["is_main_board"].eq(True)).sum()),
        "st_entries": int((entry & ~signals["is_st"].eq(0)).sum()),
        "execution_conflict_entries": int(
            (entry & ~signals["execution_conflict"].eq(False)).sum()
        ),
        "outside_window_entries": int((entry & ~in_window).sum()),
        "suspended_or_zero_liquidity_entries": int(
            (
                entry
                & (
                    ~signals["trade_status"].eq(1)
                    | ~signals["volume"].gt(0)
                    | ~signals["amount"].gt(0)
                )
            ).sum()
        ),
    }
    if any(checks.values()):
        raise AssertionError(f"Candidate-A entry permission assertion failed: {checks}")
    if signals.duplicated(["date", "code"]).any():
        raise AssertionError("Candidate-A signals have duplicate date/code keys")
    if len(signals) != 10_776_532:
        raise AssertionError("Candidate-A signals do not align to frozen main-panel row count")
    accepted = signals["accepted_signal"].eq(True)
    accepted_rows = signals.loc[accepted, ["code", "source_row_index"]]
    cooldown_ok = accepted_rows.groupby("code")["source_row_index"].diff().dropna().ge(
        candidate.cooldown
    )
    if not cooldown_ok.all():
        raise AssertionError("Candidate-A accepted signals violate original-row cooldown")

    inputs_after = _verify_inputs()
    if inputs_after != inputs_before:
        raise ValueError("Frozen input changed during candidate-A computation")

    results_dir = PROJECT_ROOT / "results"
    coverage_path = results_dir / "legacy_a_entry_coverage.parquet"
    coverage_output = _write_parquet(entry_audit, coverage_path)
    signal_name = (
        "legacy_a_signals.parquet"
        if coverage["passes"]
        else "legacy_a_signals_not_evaluable.parquet"
    )
    signal_output = _write_parquet(signals, results_dir / signal_name)
    entry_rows = signals.loc[entry]
    proposed = signals["proposed_entry"].eq(True)
    summary = {
        "candidate": "three_of_two_low_vol",
        "candidate_a_status": coverage["candidate_a_status"],
        "ranking_contract": {
            "authoritative_field": "legacy_rank_score",
            "compatibility_alias": "rank_return_120",
            "ordering": "legacy_rank_score DESC, code ASC",
            "donchian_return_recomputed": False,
        },
        "task2_status": task2_status,
        "full_a_context_gate": full_a_gate,
        "account_backtest_allowed": coverage["passes"],
        "config": {
            "path": str(args.config),
            "sha256": _sha256(args.config),
            "candidate": asdict(candidate),
        },
        "input_mode": "frozen hash-verified Parquet only; no source database connection",
        "source_databases_connected": False,
        "inputs_before": inputs_before,
        "inputs_after": inputs_after,
        "window": {"start_date": config.start_date, "end_date": config.end_date},
        "rows": {
            "context": len(context),
            "signals": len(signals),
            "unique_signal_keys": int(signals[["date", "code"]].drop_duplicates().shape[0]),
            "main_keys_without_point_in_time_context": int(
                (~signals["context_key_present"]).sum()
            ),
            "raw_signals": int(signals["raw_signal"].sum()),
            "accepted_signals_full_history": int(accepted.sum()),
            "proposed_entries": int(proposed.sum()),
            "entry_signals": int(entry.sum()),
            "entry_codes": int(entry_rows["code"].nunique()),
            "entry_first_date": str(entry_rows["date"].min()) if len(entry_rows) else None,
            "entry_last_date": str(entry_rows["date"].max()) if len(entry_rows) else None,
        },
        "permission_assertions": checks,
        "cooldown_assertion": {
            "minimum_original_row_gap": candidate.cooldown,
            "violations": int((~cooldown_ok).sum()),
            "passes": bool(cooldown_ok.all()),
        },
        "required_entry_coverage": coverage,
        "outputs": {"signals": signal_output, "entry_coverage": coverage_output},
        "returns_run": False,
        "account_returns_run": False,
    }
    summary_path = results_dir / "legacy_a_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidate_a_status": summary["candidate_a_status"],
                "rows": summary["rows"],
                "required_entry_coverage": coverage,
                "permission_assertions": checks,
                "signals_path": signal_output["path"],
                "summary_path": str(summary_path),
                "returns_run": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
