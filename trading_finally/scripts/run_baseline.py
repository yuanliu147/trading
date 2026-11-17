import argparse
import json

from trend_system.config import load_config
from trend_system.metrics import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen local-panel research baseline")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_baseline(load_config(args.config), config_path=args.config)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(result.output_dir),
                "daily_account_rows": len(result.backtest.daily_account),
                "trade_rows": len(result.backtest.trades),
                "round_trip_rows": len(result.backtest.round_trips),
                "start_date": result.summary["start_date"],
                "end_date": result.summary["end_date"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
