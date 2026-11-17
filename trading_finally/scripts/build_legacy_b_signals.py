import json
from pathlib import Path

import pandas as pd

from trend_system.legacy_b import compute_legacy_b_signals, load_industry_csv
from trend_system.legacy_b_config import load_legacy_b_config
from trend_system.legacy_runner import PROJECT_ROOT, _sha256


CONFIG = PROJECT_ROOT / "config" / "legacy_b_recovery.json"
FULL_A = PROJECT_ROOT / "data" / "legacy_full_a_panel_completed.parquet"
MAIN = PROJECT_ROOT / "data" / "main_panel.parquet"
INDUSTRY = Path(
    "/Users/admin/WorkSpace/trading/2026-10-18_成交量融合_2010-2025/01_数据/申万行业分类_时点.csv"
)


def main() -> None:
    config = load_legacy_b_config(CONFIG)
    sources = [*config.candidates[0].sources, config.industry_source]
    for source in sources:
        if _sha256(Path(source.path)) != source.sha256:
            raise ValueError(f"B recovery source hash mismatch: {source.path}")
    full_columns = [
        "date", "code", "signal_open", "signal_high", "signal_low", "signal_close",
        "close", "prev_close", "amount", "is_main_board",
    ]
    main_columns = [
        "date", "code", "open", "high", "low", "close", "prev_close",
        "signal_open", "signal_high", "signal_low", "signal_close",
        "volume", "amount", "trade_status", "is_st", "ipo_date",
        "execution_conflict",
    ]
    full = pd.read_parquet(FULL_A, columns=full_columns)
    main_panel = pd.read_parquet(MAIN, columns=main_columns)
    industry = load_industry_csv(INDUSTRY)
    result = compute_legacy_b_signals(full, main_panel, industry, config)

    output = result.signals[[
        "date", "code", "entry_signal", "atr_pct", "amt20",
        "legacy_rank_score", "rank_return_120",
    ]].copy()
    output["stop_pct"] = 0.20
    signals_path = PROJECT_ROOT / "results" / "legacy_b_signals.parquet"
    output.to_parquet(signals_path, index=False)

    full_a_audit = json.loads(
        (PROJECT_ROOT / "results" / "legacy_context_audit.json").read_text(encoding="utf-8")
    )
    full_a_pass = bool(full_a_audit["coverage"]["full_a"]["passes"])
    audit = result.audit
    evaluable = bool(
        full_a_pass
        and audit["entry_count"] > 0
        and audit["industry_daily_min"] >= config.industry_coverage_min
        and audit["industry_entry_coverage"] >= config.required_entry_coverage
        and audit["required_entry_coverage"] >= config.required_entry_coverage
    )
    summary = {
        "candidate": "early_trend_breadth",
        "status": "EVALUABLE" if evaluable else "NOT_EVALUABLE",
        "full_a_context_pass": full_a_pass,
        "industry_daily_min": audit["industry_daily_min"],
        "industry_days_below_min": audit["industry_days_below_min"],
        "industry_entry_coverage": audit["industry_entry_coverage"],
        "required_entry_coverage": audit["required_entry_coverage"],
        "proposed_entries": audit["proposed_entries"],
        "entry_count": audit["entry_count"],
        "entry_codes": audit["entry_codes"],
        "first_entry": audit["first_entry"],
        "last_entry": audit["last_entry"],
        "eligibility_start": audit["eligibility_start"],
        "st_rejected_proposed": audit["st_rejected_proposed"],
        "inputs": {
            "full_a": {"path": str(FULL_A), "sha256": _sha256(FULL_A)},
            "main": {"path": str(MAIN), "sha256": _sha256(MAIN)},
            "industry": {"path": str(INDUSTRY), "sha256": _sha256(INDUSTRY)},
            "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        },
        "outputs": {
            "signals": {
                "path": str(signals_path),
                "rows": int(len(output)),
                "sha256": _sha256(signals_path),
            }
        },
        "account_returns_run": False,
    }
    recovery_audit = {
        "candidate": "early_trend_breadth",
        "restored": True,
        "status": summary["status"],
        "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        "sources": [
            {"path": source.path, "sha256": source.sha256} for source in sources
        ],
        "industry_daily_min": summary["industry_daily_min"],
        "industry_entry_coverage": summary["industry_entry_coverage"],
        "required_entry_coverage": summary["required_entry_coverage"],
    }
    (PROJECT_ROOT / "results" / "legacy_b_recovery_audit.json").write_text(
        json.dumps(recovery_audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (PROJECT_ROOT / "results" / "legacy_b_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("inputs", "outputs")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
