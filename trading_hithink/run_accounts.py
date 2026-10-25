from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def run_name(candidate: str, holding: int, positions: int, capital: int, slippage: float, limit_rate: float) -> str:
    return f"{candidate}_h{holding}_p{positions}_c{capital}_s{int(slippage * 10000)}_l{int(limit_rate * 100)}"


def current(period: str, candidate: str, holding: int, positions: int, capital: int, slippage: float, limit_rate: float) -> dict | None:
    path = ROOT / "results" / period / "accounts" / run_name(candidate, holding, positions, capital, slippage, limit_rate) / "metrics.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())
    engine_hash = hashlib.sha256((ROOT / "engine.py").read_bytes()).hexdigest()
    feature_hash = json.loads((ROOT / "data" / "work" / "metadata.json").read_text())["files"]["features"]["sha256"]
    hypothesis_hash = hashlib.sha256((ROOT / "config" / "hypotheses.json").read_bytes()).hexdigest()
    round2_hash = hashlib.sha256((ROOT / "config" / "round2_hypotheses.json").read_bytes()).hexdigest()
    if (
        metrics.get("engine_sha256") != engine_hash
        or metrics.get("features_sha256") != feature_hash
        or metrics.get("hypothesis_sha256") != hypothesis_hash
        or metrics.get("round2_hypotheses_sha256") != round2_hash
    ):
        return None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=["development", "validation"], default="development")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.period == "development":
        candidates = json.loads((ROOT / "config" / "account_candidates.json").read_text())["candidates"]
    else:
        candidates = json.loads((ROOT / "config" / "validation_finalists.json").read_text())["candidates"]
    records = []
    commands = []
    for candidate in candidates:
        for holding in candidate["holding_options"]:
            for positions in candidate.get("position_options", [2, 3]):
                for capital in [20000, 50000]:
                    commands.append((candidate["id"], holding, positions, capital))
    for index, (candidate, holding, positions, capital) in enumerate(commands, start=1):
        cached = current(args.period, candidate, holding, positions, capital, 0.001, 0.10) if args.reuse else None
        if cached is not None:
            records.append(cached)
            continue
        command = [
            sys.executable,
            str(ROOT / "portfolio.py"),
            "--candidate-id",
            candidate,
            "--holding",
            str(holding),
            "--positions",
            str(positions),
            "--capital",
            str(capital),
            "--period",
            args.period,
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"Account run failed: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        metrics = json.loads(result.stdout.strip().splitlines()[-1])
        records.append(metrics)
        print(json.dumps({"completed": index, "total": len(commands), "candidate": candidate, "holding": holding, "positions": positions, "capital": capital}))
    output = ROOT / "results" / args.period / "account_summary.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    print(json.dumps({"ok": True, "runs": len(records), "output": str(output)}))


if __name__ == "__main__":
    main()
