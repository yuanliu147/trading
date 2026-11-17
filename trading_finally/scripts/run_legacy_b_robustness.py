import argparse
import json

from trend_system.legacy_b_robustness import run_legacy_b_robustness
from trend_system.legacy_b_config import load_legacy_b_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recovered candidate-B robustness")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_legacy_b_robustness(load_legacy_b_config(args.config), config_path=args.config)
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
