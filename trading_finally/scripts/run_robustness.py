import argparse
import json

from trend_system.config import load_config
from trend_system.robustness import run_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen robustness and stress matrix")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_robustness(load_config(args.config))
    print(
        json.dumps(
            {
                "status": result.summary["run_status"]["status"],
                "verdict": result.verdict,
                "output_dir": str(result.output_dir),
                "stress_output_dir": str(result.stress_output_dir),
                "report_path": str(result.report_path),
                "completed_real_variants": len(result.variants),
                "bootstrap_rows": len(result.bootstrap),
                "elapsed_seconds": result.summary["run_status"]["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
