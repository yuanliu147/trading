from __future__ import annotations

import json

import numpy as np
import pandas as pd

from baostock_optimization import OUTPUT, ROOT, write_json
from baostock_replay import SCENARIOS
from verify_audit import audit_account


def main() -> None:
    accounts = sorted((OUTPUT / "accounts").iterdir())
    if len(accounts) != 30:
        raise AssertionError(f"Expected 30 research accounts, found {len(accounts)}")
    reports = [audit_account(account) for account in accounts]
    write_json(OUTPUT / "verification.json", {
        "accounts": len(reports),
        "passed": all(report["passed"] for report in reports),
        "reports": reports,
    })
    failed = [report["account"] for report in reports if not report["passed"]]
    if failed:
        raise AssertionError(f"Independent account audits failed: {failed}")
    rows = []
    annual_frames = []
    for account in accounts:
        metrics = json.loads((account / "metrics.json").read_text())
        rows.append(metrics)
        annual = pd.read_csv(account / "annual.csv")
        annual.insert(0, "account", account.name)
        annual.insert(1, "scenario", metrics["scenario"])
        annual.insert(2, "capital", metrics["capital"])
        annual_frames.append(annual)
    summary = pd.DataFrame(rows)
    baseline = summary[summary["scenario"] == "baseline"].set_index("capital")
    summary["baseline_cagr"] = summary["capital"].map(baseline["cagr"])
    summary["cagr_delta"] = summary["cagr"] - summary["baseline_cagr"]
    summary["baseline_max_drawdown"] = summary["capital"].map(baseline["max_drawdown"])
    summary["max_drawdown_delta"] = summary["max_drawdown"] - summary["baseline_max_drawdown"]
    summary["scenario_order"] = summary["scenario"].map({name: index for index, name in enumerate(SCENARIOS)})
    summary = summary.sort_values(["capital", "scenario_order"]).drop(columns="scenario_order")
    summary.to_csv(OUTPUT / "optimization_summary.csv", index=False)
    pd.concat(annual_frames, ignore_index=True).to_csv(OUTPUT / "annual_returns_long.csv", index=False)
    for capital in [20000, 50000, 100000]:
        research = summary[(summary["scenario"] == "baseline") & (summary["capital"] == capital)].iloc[0]
        expected = json.loads((ROOT / "results" / "accounts" / f"O03_c{capital}" / "metrics.json").read_text())
        for key in ["final_equity", "cagr", "max_drawdown", "trade_count", "ending_open_positions"]:
            if isinstance(research[key], (float, np.floating)):
                if not np.isclose(research[key], expected[key], rtol=1e-10, atol=1e-7):
                    raise AssertionError(f"Baseline {capital} {key}: {research[key]} != {expected[key]}")
            elif research[key] != expected[key]:
                raise AssertionError(f"Baseline {capital} {key}: {research[key]} != {expected[key]}")
    print(json.dumps({
        "independent_accounts_passed": len(reports),
        "baseline_reproduction_capitals": [20000, 50000, 100000],
        "summary": str(OUTPUT / "optimization_summary.csv"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
