from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    selection = json.loads((ROOT / "config" / "hypotheses.json").read_text())["selection"]
    summary = pd.read_csv(ROOT / "results" / "development" / "hypothesis_summary.csv")
    yearly = pd.read_csv(ROOT / "results" / "development" / "hypothesis_yearly.csv")
    hypotheses = json.loads((ROOT / "config" / "hypotheses.json").read_text())["hypotheses"]
    round2_path = ROOT / "config" / "round2_hypotheses.json"
    round2 = json.loads(round2_path.read_text())
    round2_hypotheses = round2["hypotheses"]
    hypotheses.extend(round2_hypotheses)
    round2_summary = pd.read_csv(ROOT / "results" / "development" / "round2_summary.csv")
    summary = pd.concat([summary, round2_summary], ignore_index=True)
    summary["family"] = summary["id"].map({item["id"]: item["family"] for item in hypotheses})
    round2_yearly = []
    for hypothesis in round2_hypotheses:
        round2_yearly.append(pd.read_csv(ROOT / "results" / "development" / f"round2_{hypothesis['id']}_yearly.csv"))
    yearly = pd.concat([yearly, *round2_yearly], ignore_index=True)
    positive_years = yearly.assign(positive=yearly["excess20"] > 0).groupby("id")["positive"].sum()
    summary["positive_excess_years"] = summary["id"].map(positive_years).fillna(0).astype(int)
    eligible = summary[
        (summary["observations"] >= selection["minimum_development_signals"])
        & (summary["positive_excess_years"] >= selection["minimum_development_positive_excess_years"])
    ].copy()
    eligible = eligible.sort_values(["net60bp_excess20", "observations"], ascending=[False, False])
    chosen = eligible.groupby("family", as_index=False).head(
        selection["maximum_account_candidates_per_family"]
    ).head(selection["maximum_account_candidates"])
    by_id = {item["id"]: item for item in hypotheses}
    candidates = []
    evidence_columns = [
        "id", "family", "observations", "symbols", "active_days", "positive_excess_years",
        "mean_fwd5", "mean_fwd10", "mean_fwd20", "mean_fwd60", "excess20", "net60bp_excess20",
    ]
    for row in chosen[evidence_columns].to_dict("records"):
        hypothesis = by_id[row["id"]]
        candidates.append(
            {
                **row,
                "holding_options": hypothesis["holding_options"],
                "position_options": [2, 3],
                "capital_options": [20000, 50000],
                "entry": hypothesis["entry"],
                "exit": hypothesis["exit"],
                "score": hypothesis["score"],
                "research_round": 2 if hypothesis in round2_hypotheses else 1,
            }
        )
    output = {
        "schema_version": 1,
        "stage": "development_account_candidates",
        "selection_rule": selection,
        "eligible_hypotheses": int(len(eligible)),
        "rejected_hypotheses": int(len(summary) - len(eligible)),
        "holdout_used": False,
        "hypotheses_sha256": sha256(ROOT / "config" / "hypotheses.json"),
        "round2_hypotheses_sha256": sha256(round2_path),
        "development_summary_sha256": sha256(ROOT / "results" / "development" / "hypothesis_summary.csv"),
        "round2_summary_sha256": sha256(ROOT / "results" / "development" / "round2_summary.csv"),
        "candidates": candidates,
    }
    path = ROOT / "config" / "account_candidates.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"ok": True, "candidates": [item["id"] for item in candidates], "path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
