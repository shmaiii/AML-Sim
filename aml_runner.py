"""AML-level runner for StockSim scenarios.

This script keeps AML orchestration outside the StockSim fork. It reads an AML
scenario file, writes the StockSim-compatible config into a run directory, and
optionally launches StockSim with that generated config.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
STOCKSIM_DIR = ROOT / "simulators" / "StockSim"
DEFAULT_SCENARIO = ROOT / "scenarios" / "aml_orderbook_replay.yaml"
RUNS_DIR = ROOT / ".aml_runs"
ENV_FILE = ROOT / ".env"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must contain a YAML mapping: {path}")
    return data


def make_run_dir(scenario_path: Path, run_id: str | None) -> Path:
    scenario_name = scenario_path.stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final_run_id = run_id or f"{scenario_name}_{timestamp}"
    run_dir = RUNS_DIR / final_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    return run_dir


def build_stocksim_config(scenario: dict[str, Any]) -> dict[str, Any]:
    config = scenario.get("stocksim_config")
    if not isinstance(config, dict):
        raise ValueError("Scenario must define a 'stocksim_config' mapping")

    required = ["exchange_mode", "instruments", "exchanges", "agents", "simulation"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"stocksim_config missing required keys: {', '.join(missing)}")

    return config


def write_config(config: dict[str, Any], run_dir: Path) -> Path:
    config_path = run_dir / "stocksim_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def launch_stocksim(config_path: Path, run_dir: Path, rabbitmq_host: str | None) -> int:
    if not STOCKSIM_DIR.exists():
        raise FileNotFoundError(f"StockSim directory not found: {STOCKSIM_DIR}")

    env = {**load_env_file(ENV_FILE), **os.environ}
    env["LOG_DIR"] = str(run_dir / "logs")
    if rabbitmq_host:
        env["RABBITMQ_HOST"] = rabbitmq_host

    command = [sys.executable, "main_launcher.py", str(config_path)]
    print(f"Launching StockSim from: {STOCKSIM_DIR}")
    print(f"Generated config: {config_path}")
    print(f"AML env file: {ENV_FILE if ENV_FILE.exists() else 'not found'}")
    print(f"Logs: {run_dir / 'logs'}")
    return subprocess.run(command, cwd=STOCKSIM_DIR, env=env, check=False).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an AML scenario through StockSim.")
    parser.add_argument(
        "scenario",
        nargs="?",
        default=str(DEFAULT_SCENARIO),
        help="Path to an AML scenario YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the StockSim config but do not launch StockSim.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run directory name under .aml_runs/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario_path = Path(args.scenario).expanduser().resolve()
    scenario = load_yaml(scenario_path)
    config = build_stocksim_config(scenario)
    run_dir = make_run_dir(scenario_path, args.run_id)
    config_path = write_config(config, run_dir)

    print(f"Scenario: {scenario.get('name', scenario_path.stem)}")
    print(f"Run directory: {run_dir}")
    print(f"StockSim config written: {config_path}")

    if args.dry_run:
        print("Dry run only. StockSim was not launched.")
        return 0

    rabbitmq_host = scenario.get("rabbitmq_host")
    return launch_stocksim(config_path, run_dir, rabbitmq_host)


if __name__ == "__main__":
    raise SystemExit(main())
