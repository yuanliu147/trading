from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def summarize(period: str) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "results" / period / "account_summary.csv")
    keys = ["candidate_id", "holding_sessions", "max_positions"]
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        if set(group["initial_capital"]) != {20000, 50000} or len(group) != 2:
            continue
        rows.append(
            {
                "candidate_id": key[0],
                "holding_sessions": int(key[1]),
                "max_positions": int(key[2]),
                "minimum_cagr": group["cagr"].min(),
                "mean_cagr": group["cagr"].mean(),
                "minimum_calmar": group["calmar"].min(),
                "maximum_drawdown": group["max_drawdown"].max(),
                "minimum_trades": group["trade_count"].min(),
                "maximum_top5_profit_share": group["top_5_profit_share"].max(),
                "mean_holding_time_ratio": group["holding_time_ratio"].mean(),
                "cagr_20000": group.loc[group["initial_capital"] == 20000, "cagr"].iloc[0],
                "cagr_50000": group.loc[group["initial_capital"] == 50000, "cagr"].iloc[0],
                "drawdown_20000": group.loc[group["initial_capital"] == 20000, "max_drawdown"].iloc[0],
                "drawdown_50000": group.loc[group["initial_capital"] == 50000, "max_drawdown"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def eligible(frame: pd.DataFrame, rules: dict) -> pd.DataFrame:
    return frame[
        (frame["minimum_trades"] >= rules["minimum_closed_trades_each_capital"])
        & (frame["maximum_drawdown"] <= rules["maximum_drawdown_each_capital"])
        & (frame["maximum_top5_profit_share"] <= rules["maximum_top5_profit_share_each_capital"])
        & (frame["minimum_cagr"] >= rules["minimum_cagr_each_capital"])
    ].copy()


def make_validation() -> None:
    config = json.loads((ROOT / "config" / "account_selection.json").read_text())["development"]
    summary = summarize("development")
    passed = eligible(summary, config).sort_values(
        ["minimum_cagr", "minimum_calmar", "maximum_drawdown"], ascending=[False, False, True]
    )
    hypotheses = json.loads((ROOT / "config" / "hypotheses.json").read_text())["hypotheses"]
    hypotheses += json.loads((ROOT / "config" / "round2_hypotheses.json").read_text())["hypotheses"]
    by_id = {item["id"]: item for item in hypotheses}
    seen = set()
    finalists = []
    for row in passed.to_dict("records"):
        if row["candidate_id"] in seen:
            continue
        seen.add(row["candidate_id"])
        hypothesis = by_id[row["candidate_id"]]
        finalists.append(
            {
                "id": row["candidate_id"],
                "holding_options": [row["holding_sessions"]],
                "position_options": [row["max_positions"]],
                "development_evidence": row,
                "entry": hypothesis["entry"],
                "exit": hypothesis["exit"],
                "score": hypothesis["score"],
            }
        )
        if len(finalists) >= config["maximum_validation_finalists"]:
            break
    summary.to_csv(ROOT / "results" / "development" / "account_variants.csv", index=False)
    passed.to_csv(ROOT / "results" / "development" / "account_variants_passing.csv", index=False)
    output = {
        "stage": "validation_finalists",
        "holdout_used": False,
        "selection_config_sha256": sha256(ROOT / "config" / "account_selection.json"),
        "development_account_summary_sha256": sha256(ROOT / "results" / "development" / "account_summary.csv"),
        "eligible_variants": len(passed),
        "candidates": finalists,
    }
    if not finalists:
        write_json(ROOT / "config" / "validation_finalists.json", output)
        raise RuntimeError("No development account variant passed the pre-registered gates")
    write_json(ROOT / "config" / "validation_finalists.json", output)
    print(json.dumps({"ok": True, "finalists": [item["id"] for item in finalists]}, ensure_ascii=False))


def lock_final() -> None:
    config = json.loads((ROOT / "config" / "account_selection.json").read_text())["validation"]
    summary = summarize("validation")
    passed = eligible(summary, config).sort_values(["minimum_cagr", "minimum_calmar"], ascending=[False, False])
    summary.to_csv(ROOT / "results" / "validation" / "account_variants.csv", index=False)
    passed.to_csv(ROOT / "results" / "validation" / "account_variants_passing.csv", index=False)
    if passed.empty:
        raise RuntimeError("No validation candidate passed the pre-registered gates; final strategy remains unlocked")
    winner = passed.iloc[0].to_dict()
    hypotheses = json.loads((ROOT / "config" / "hypotheses.json").read_text())["hypotheses"]
    hypotheses += json.loads((ROOT / "config" / "round2_hypotheses.json").read_text())["hypotheses"]
    hypothesis = next(item for item in hypotheses if item["id"] == winner["candidate_id"])
    research = json.loads((ROOT / "config" / "research.json").read_text())
    locked = {
        "schema_version": 1,
        "stage": "locked_before_holdout",
        "locked_before_holdout": True,
        "holdout_used_for_selection": False,
        "candidate_id": winner["candidate_id"],
        "family": hypothesis["family"],
        "entry": hypothesis["entry"],
        "exit": hypothesis["exit"],
        "score": hypothesis["score"],
        "holding_sessions": winner["holding_sessions"],
        "max_positions": winner["max_positions"],
        "slot_fraction": 1.0 / winner["max_positions"],
        "engine_overrides": hypothesis.get("engine_overrides", {}),
        "capital": [20000, 50000],
        "costs": research["costs"],
        "execution": research["execution"],
        "limit_rate": 0.10,
        "slippage": 0.001,
        "validation_evidence": winner,
        "hypotheses_sha256": sha256(ROOT / "config" / "hypotheses.json"),
        "round2_hypotheses_sha256": sha256(ROOT / "config" / "round2_hypotheses.json"),
        "research_config_sha256": sha256(ROOT / "config" / "research.json"),
        "account_selection_sha256": sha256(ROOT / "config" / "account_selection.json"),
        "engine_sha256": sha256(ROOT / "engine.py"),
        "features_sha256": json.loads((ROOT / "data" / "work" / "metadata.json").read_text())["files"]["features"]["sha256"],
        "survivorship_bias": "current_listed_only",
        "historical_st_known": False,
        "return_guarantee": False,
    }
    write_json(ROOT / "config" / "final_strategy.json", locked)
    print(json.dumps({"ok": True, "locked": {key: locked[key] for key in ["candidate_id", "holding_sessions", "max_positions"]}}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["make-validation", "lock-final"])
    args = parser.parse_args()
    if args.action == "make-validation":
        make_validation()
    else:
        lock_final()


if __name__ == "__main__":
    main()
