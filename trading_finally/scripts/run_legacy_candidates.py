import argparse
import json

from trend_system.legacy_config import load_legacy_config
from trend_system.legacy_runner import CANDIDATE_A, run_legacy_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen legacy candidate A account")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_legacy_config(args.config)
    result = run_legacy_candidate(CANDIDATE_A, config)
    summary = result.summary
    print(
        json.dumps(
            {
                "status": result.status,
                "candidate": result.candidate_id,
                "output_dir": str(result.output_dir),
                "start_date": summary.get("start_date"),
                "end_date": summary.get("end_date"),
                "initial_equity": summary.get("initial_equity"),
                "final_equity": summary.get("final_equity"),
                "total_return": summary.get("total_return"),
                "cagr": summary.get("cagr"),
                "max_drawdown": summary.get("max_drawdown"),
                "round_trips": summary.get("round_trips"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
