"""Run all generated D0-D4 scenarios sequentially with fixed run IDs."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path, nargs="?", default=Path("scenarios/diversity"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = args.matrix_dir / "matrix_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        scenario = args.matrix_dir / row["scenario"]
        run_id = f"diversity_{row['diversity_level'].lower()}_seed_{row['seed']}"
        command = [sys.executable, "aml_runner.py", str(scenario), "--run-id", run_id]
        if args.dry_run:
            command.append("--dry-run")
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
