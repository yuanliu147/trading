from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import holding_exploration

ROOT = Path(__file__).resolve().parent


def main() -> None:
    summary_path = ROOT / "results" / "development" / "holding_trials_summary.csv"
    previous = pd.read_csv(summary_path)
    holding_exploration.HOLDINGS = [16, 17, 18, 19, 21, 22, 23, 24]
    holding_exploration.main()
    current = pd.read_csv(summary_path)
    summary_path.write_text(previous.to_csv(index=False))
    output = ROOT / "results" / "development" / "holding_fine_summary.csv"
    current.to_csv(output, index=False)
    pd.concat([previous, current], ignore_index=True).to_csv(
        ROOT / "results" / "development" / "all_holding_trials_summary.csv", index=False
    )
    print(json.dumps({"ok": True, "runs": len(current), "output": str(output), "holdout_used": False}))


if __name__ == "__main__":
    main()
