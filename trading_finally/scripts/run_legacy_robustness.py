import argparse
import json

from trend_system.legacy_config import load_legacy_config
from trend_system.legacy_robustness import run_legacy_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen candidate-A robustness")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_legacy_robustness(load_legacy_config(args.config))
    print(
        json.dumps(
            {
                "status": result.status,
                "output_dir": str(result.output_dir),
                "main": result.summary["main"],
                "double_cost": result.summary["double_cost"],
                "random_positive_paths": result.summary["random_order"]["positive_paths"],
                "stress_2010_2015": result.summary["stress_2010_2015"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
