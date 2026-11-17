import argparse
import json

from trend_system.config import load_config
from trend_system.data import build_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build read-only dual-source research panels")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = build_panel(load_config(args.config))
    print(
        json.dumps(
            {
                "main_panel": str(result.main_panel_path),
                "stress_panel": str(result.stress_panel_path),
                "corporate_actions": str(result.corporate_actions_path),
                "hithink_signal_panel": str(result.hithink_signal_panel_path),
                "index_panel": str(result.index_panel_path),
                "audit": str(result.audit_path),
                "conflicts": str(result.conflicts_path),
                "exclusions": str(result.exclusions_path),
                "main_rows": result.main_rows,
                "stress_rows": result.stress_rows,
                "conflict_rows": result.conflict_rows,
                "exclusion_rows": result.exclusion_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
