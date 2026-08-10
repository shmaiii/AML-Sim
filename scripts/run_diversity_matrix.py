"""Run a budgeted D0-D4 diversity matrix with locked validation/OOS seeds."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aml_sim.scenario import load_scenario


DEFAULT_MATRIX_DIR = ROOT / "scenarios" / "diversity"
DEFAULT_PROTOCOL = ROOT / "experiments" / "diversity_matrix.yaml"
RUNS_DIR = ROOT / ".aml_runs"
LEVELS = ("D0", "D1", "D2", "D3", "D4")


@dataclass(frozen=True)
class MatrixRun:
    level: str
    seed: int
    replicate: int
    phase: str
    scenario_path: Path

    @property
    def run_id(self) -> str:
        phase_prefix = "" if self.phase == "validation" else "oos_"
        return f"diversity_{phase_prefix}{self.level.lower()}_seed_{self.seed}"


def _parse_interval_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    units = {
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "m": 60.0,
        "min": 60.0,
        "mins": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "h": 3600.0,
        "hour": 3600.0,
        "hours": 3600.0,
    }
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            return float(text[: -len(suffix)].strip()) * units[suffix]
    raise ValueError(f"Unsupported interval value: {value!r}")


def _simulation_duration_seconds(raw: dict[str, Any]) -> float:
    simulation = raw["stocksim_config"]["simulation"]
    start = datetime.fromisoformat(str(simulation["start_time"]).replace("Z", "+00:00"))
    end = datetime.fromisoformat(str(simulation["end_time"]).replace("Z", "+00:00"))
    duration = (end - start).total_seconds()
    if duration <= 0:
        raise ValueError("Simulation end_time must be after start_time")
    return duration


def estimate_api_calls(raw: dict[str, Any]) -> tuple[int, int]:
    """Return planned calls and worst-case request attempts including retries."""
    duration = _simulation_duration_seconds(raw)
    llm_defaults = raw.get("aml_config", {}).get("llm", {}) or {}
    planned_calls = 0
    worst_case_attempts = 0
    agents = raw["stocksim_config"].get("agents", {})
    for details in agents.values():
        parameters = details.get("parameters", {}) or {}
        strategist = parameters.get("slow_strategist")
        if not isinstance(strategist, dict):
            continue
        effective_llm = {**llm_defaults, **strategist}
        if str(effective_llm.get("type", "static")).lower() not in {
            "openai",
            "openai_json",
        }:
            continue
        interval = _parse_interval_seconds(
            parameters.get("slow_loop_interval", "1h")
        )
        if interval <= 0:
            raise ValueError("slow_loop_interval must be positive")
        updates = max(1, math.ceil(duration / interval))
        count = int(details.get("count", 1))
        retries = max(0, int(effective_llm.get("max_retries", 2)))
        planned_calls += count * updates
        worst_case_attempts += count * updates * (retries + 1)
    return planned_calls, worst_case_attempts


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return raw


def _write_yaml(path: Path, raw: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)


def _load_manifest(matrix_dir: Path) -> list[dict[str, str]]:
    manifest = matrix_dir / "matrix_manifest.csv"
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _phase_seeds(protocol: dict[str, Any], phase: str) -> list[int]:
    field = "validation_seeds" if phase == "validation" else "out_of_sample_seeds"
    seeds = protocol.get(field)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Protocol must define a non-empty {field} list")
    return [int(seed) for seed in seeds]


def _selected_levels(values: list[str] | None) -> list[str]:
    if not values:
        return list(LEVELS)
    normalized = [value.upper() for value in values]
    unknown = sorted(set(normalized) - set(LEVELS))
    if unknown:
        raise ValueError("Unknown diversity level(s): " + ", ".join(unknown))
    return [level for level in LEVELS if level in normalized]


def _validation_runs(
    manifest_rows: list[dict[str, str]],
    matrix_dir: Path,
    levels: list[str],
    seeds: list[int],
) -> list[MatrixRun]:
    selected: list[MatrixRun] = []
    requested = {(level, seed) for level in levels for seed in seeds}
    found: set[tuple[str, int]] = set()
    for row in manifest_rows:
        level = row["diversity_level"].upper()
        seed = int(row["seed"])
        if (level, seed) not in requested:
            continue
        found.add((level, seed))
        selected.append(
            MatrixRun(
                level=level,
                seed=seed,
                replicate=int(row["replicate"]),
                phase="validation",
                scenario_path=matrix_dir / row["scenario"],
            )
        )
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Manifest is missing validation scenarios: {missing}")
    return selected


def _oos_runs(
    manifest_rows: list[dict[str, str]],
    matrix_dir: Path,
    generated_dir: Path,
    levels: list[str],
    seeds: list[int],
) -> list[MatrixRun]:
    template_by_level: dict[str, Path] = {}
    for row in manifest_rows:
        template_by_level.setdefault(
            row["diversity_level"].upper(),
            matrix_dir / row["scenario"],
        )

    selected: list[MatrixRun] = []
    for replicate, seed in enumerate(seeds, start=1):
        for level in levels:
            template = template_by_level.get(level)
            if template is None:
                raise ValueError(f"No validation template is available for {level}")
            raw = _read_yaml(template)
            raw["name"] = f"aml_diversity_{level.lower()}_oos_seed_{seed}"
            raw["description"] = (
                f"Locked out-of-sample diversity experiment {level} with seed {seed}."
            )
            aml_config = raw.setdefault("aml_config", {})
            aml_config["dataset_split"] = "out_of_sample"
            experiment = aml_config.setdefault("experiment", {})
            experiment["diversity_level"] = level
            experiment["seed"] = seed
            experiment["replicate"] = replicate
            scenario_path = generated_dir / f"{level.lower()}_oos_seed_{seed}.yaml"
            _write_yaml(scenario_path, raw)
            selected.append(
                MatrixRun(
                    level=level,
                    seed=seed,
                    replicate=replicate,
                    phase="out_of_sample",
                    scenario_path=scenario_path,
                )
            )
    return selected


def _run_complete(run_id: str) -> bool:
    return (RUNS_DIR / run_id / "reports" / "research_metrics.json").exists()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected D0-D4 scenarios with an API-call budget."
    )
    parser.add_argument("matrix_dir", type=Path, nargs="?", default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--phase", choices=("validation", "out_of_sample"), default="validation")
    parser.add_argument("--levels", nargs="+", help="Subset such as D0 D2 D4.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Subset of locked phase seeds.")
    parser.add_argument("--max-api-calls", type=int, help="Maximum worst-case API request attempts.")
    parser.add_argument("--resume", action="store_true", help="Skip runs with a complete research_metrics.json.")
    parser.add_argument("--reports", action="store_true", help="Also request StockSim post-simulation reports.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate only; do not create run directories.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    matrix_dir = args.matrix_dir.expanduser().resolve()
    protocol = _read_yaml(args.protocol.expanduser().resolve())
    levels = _selected_levels(args.levels)
    locked_seeds = _phase_seeds(protocol, args.phase)
    seeds = list(args.seeds) if args.seeds else locked_seeds
    unlocked = sorted(set(seeds) - set(locked_seeds))
    if unlocked:
        raise ValueError(
            f"Seeds are not locked for phase {args.phase}: {unlocked}"
        )
    manifest_rows = _load_manifest(matrix_dir)

    with tempfile.TemporaryDirectory(prefix="aml_diversity_oos_") as temp_dir:
        if args.phase == "validation":
            selected = _validation_runs(
                manifest_rows,
                matrix_dir,
                levels,
                seeds,
            )
        else:
            selected = _oos_runs(
                manifest_rows,
                matrix_dir,
                Path(temp_dir),
                levels,
                seeds,
            )

        pending: list[tuple[MatrixRun, int, int]] = []
        for item in selected:
            run_dir = RUNS_DIR / item.run_id
            if args.resume and _run_complete(item.run_id):
                print(f"SKIP complete: {item.run_id}")
                continue
            if run_dir.exists():
                raise FileExistsError(
                    f"Run directory exists but is not resumable: {run_dir}"
                )
            load_scenario(item.scenario_path)
            raw = _read_yaml(item.scenario_path)
            planned, worst_case = estimate_api_calls(raw)
            pending.append((item, planned, worst_case))

        total_planned = sum(item[1] for item in pending)
        total_worst_case = sum(item[2] for item in pending)
        api_policy = protocol.get("api_policy", {}) or {}
        default_budget = int(api_policy.get("pilot_max_calls", 30))
        budget = args.max_api_calls if args.max_api_calls is not None else default_budget
        if budget < 0:
            raise ValueError("--max-api-calls must be non-negative")

        print(f"Phase: {args.phase}")
        print(f"Runs selected: {len(selected)}; pending: {len(pending)}")
        print(f"Planned API calls: {total_planned}")
        print(f"Worst-case API attempts: {total_worst_case}")
        print(f"API attempt budget: {budget}")
        for item, planned, worst_case in pending:
            print(
                f"  {item.run_id}: calls={planned}, "
                f"worst_case={worst_case}, scenario={item.scenario_path.name}"
            )

        if total_worst_case > budget:
            print(
                "Refusing to start: estimated API attempts exceed the budget. "
                "Select fewer levels/seeds or explicitly raise --max-api-calls.",
                file=sys.stderr,
            )
            return 2
        if args.dry_run:
            print("Dry run complete. No run directories or API calls were created.")
            return 0

        env = dict(os.environ)
        for item, _, _ in pending:
            command = [
                sys.executable,
                str(ROOT / "aml_runner.py"),
                str(item.scenario_path),
                "--run-id",
                item.run_id,
            ]
            if args.reports:
                command.append("--reports")
            completed = subprocess.run(command, check=False, cwd=ROOT, env=env)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
