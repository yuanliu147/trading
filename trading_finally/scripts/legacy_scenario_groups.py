import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Descriptive size/volatility groups from frozen results")
    parser.add_argument("--candidate", choices=["three_of_two_low_vol", "early_trend_breadth"], required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / "legacy_candidates" / args.candidate
    trips = pd.read_parquet(directory / "round_trips.parquet")
    positions = pd.read_parquet(directory / "daily_positions.parquet")
    positions["date"] = pd.to_datetime(positions["date"])
    positions["entry_date"] = pd.to_datetime(positions["entry_date"])
    trips["entry_date"] = pd.to_datetime(trips["entry_date"])

    volatility = []
    for _, trip in trips.iterrows():
        held = positions[
            (positions["code"] == trip["code"]) & (positions["entry_date"] == trip["entry_date"])
        ].sort_values("date")
        volatility.append(held["close"].pct_change().std() if len(held) >= 3 else np.nan)
    trips["holding_volatility"] = volatility
    trips["holding_group"] = np.where(
        trips["holding_volatility"].notna(), "volatility", "short_hold_lt3"
    )

    groups = {}
    for column, name in (("entry_amount", "entry_amount_groups"),
                         ("holding_volatility", "holding_volatility_groups")):
        frame = trips.copy()
        if column == "holding_volatility":
            frame["group"] = frame["holding_group"]
            frame.loc[frame["holding_volatility"].notna(), "group"] = pd.qcut(
                frame.loc[frame["holding_volatility"].notna(), "holding_volatility"],
                3, labels=["low", "mid", "high"], duplicates="drop",
            ).astype(str)
        else:
            frame = frame[frame[column].notna()].copy()
            frame["group"] = pd.qcut(frame[column], 3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
        table = frame.groupby("group", observed=True).agg(
            trades=("net_pnl", "size"),
            net_pnl=("net_pnl", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            win_rate=("net_pnl", lambda values: (values > 0).mean()),
        ).reset_index()
        groups[name] = table.to_dict(orient="records")

    performance = json.loads((directory / "performance_summary.json").read_text(encoding="utf-8"))
    rolling = pd.Series(performance["rolling_3y_returns"]).astype(float)
    payload = {
        "candidate": args.candidate,
        "groups": groups,
        "rolling_3y": {
            "windows": int(len(rolling)),
            "positive_share": float((rolling > 0).mean()),
            "min": float(rolling.min()),
            "median": float(rolling.median()),
            "max": float(rolling.max()),
        },
    }
    output = ROOT / ".sdd-legacy" / f"scenario_groups_{args.candidate}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
