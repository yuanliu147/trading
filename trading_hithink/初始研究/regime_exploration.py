from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import gate_exploration

ROOT = Path(__file__).resolve().parent

REGIME_GATES = {
    "only_bear": "breadth20 <= 0.35 AND market_return20 < 0",
    "only_bear_repair": "breadth20 <= 0.35 AND market_return20 < 0 AND breadth20_change5 > 0",
    "only_panic": "market_return20 < -0.10",
    "only_breadth_below20": "breadth20 < 0.20",
    "only_market_below_ma20": "market_close_ma20 < 0",
    "sentiment_below_zero": "sentiment_score < 0",
    "bear_or_highvol": "(breadth20 <= 0.35 AND market_return20 < 0) OR market_volatility20 > 0.25",
    "not_bull": "NOT (breadth20 >= 0.60 AND market_close_ma20 > 0)",
}


def main() -> None:
    summary_path = ROOT / "results" / "development" / "gate_trials_summary.csv"
    if not summary_path.exists():
        raise RuntimeError("Run gate_exploration.py first")
    previous = pd.read_csv(summary_path)
    gate_exploration.GATES = REGIME_GATES
    gate_exploration.main()
    current = pd.read_csv(summary_path)
    summary_path.write_text(previous.to_csv(index=False))
    output = ROOT / "results" / "development" / "regime_trials_summary.csv"
    current.to_csv(output, index=False)
    combined = pd.concat([previous, current], ignore_index=True)
    combined.to_csv(ROOT / "results" / "development" / "all_gate_trials_summary.csv", index=False)
    print(json.dumps({"ok": True, "runs": len(current), "output": str(output), "holdout_used": False}))


if __name__ == "__main__":
    main()
