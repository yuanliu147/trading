import argparse
import json

from trend_system.legacy_b_runner import run_legacy_b
from trend_system.legacy_b_config import load_legacy_b_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recovered legacy candidate B account")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_legacy_b(load_legacy_b_config(args.config), config_path=args.config)
    summary = result.summary
    print(
        json.dumps(
            {
                "status": result.status,
                "candidate": "early_trend_breadth",
                "output_dir": str(result.output_dir),
                "start_date": summary.get("start_date"),
                "end_date": summary.get("end_date"),
                "initial_equity": summary.get("initial_equity"),
                "final_equity": summary.get("final_equity"),
                "total_return": summary.get("total_return"),
                "cagr": summary.get("cagr"),
                "max_drawdown": summary.get("max_drawdown"),
                "round_trip_count": summary.get("round_trip_count"),
                "win_rate": summary.get("win_rate"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
