import argparse
import json

from trend_system.legacy_config import load_legacy_config
from trend_system.legacy_data import build_legacy_context


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build read-only full-A point-in-time context and data-gate audit"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = build_legacy_context(load_legacy_config(args.config))
    print(
        json.dumps(
            {
                "full_a_panel": str(result.full_a_panel_path),
                "source_full_a_panel": str(result.source_full_a_panel_path),
                "industry_panel": str(result.industry_panel_path),
                "audit": str(result.audit_path),
                "full_a_rows": result.full_a_rows,
                "industry_rows": result.industry_rows,
                "candidate_a_status": result.candidate_a_status,
                "candidate_b_status": result.candidate_b_status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
