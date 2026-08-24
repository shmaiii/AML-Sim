"""AML-level runner for StockSim scenarios.

This script keeps AML orchestration outside the StockSim fork. It reads an AML
scenario file, writes the StockSim-compatible config into a run directory, and
optionally launches StockSim with that generated config.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from aml_sim.launcher import launch_stocksim
from aml_sim.runs import create_run
from aml_sim.scenario import load_scenario


ROOT = Path(__file__).resolve().parent
STOCKSIM_DIR = ROOT / "simulators" / "StockSim"
DEFAULT_SCENARIO = ROOT / "scenarios" / "aml_orderbook_replay.yaml"
RUNS_DIR = ROOT / ".aml_runs"
ENV_FILE = ROOT / ".env"


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
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Generate StockSim post-simulation reports into the AML run directory.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        help="Override the ecology master seed and derive fresh per-agent random streams.",
    )
    parser.add_argument(
        "--replicate-id",
        type=int,
        help="Override the ecology replicate identifier recorded with this run.",
    )
    parser.add_argument(
        "--arbitrage-entry-bps",
        type=float,
        help="Override the cross-market arbitrage entry threshold for calibration runs.",
    )
    parser.add_argument(
        "--arbitrage-exit-bps",
        type=float,
        help="Override the cross-market arbitrage exit threshold for calibration runs.",
    )
    parser.add_argument(
        "--slow-strategist",
        choices=("frozen", "openai"),
        help=(
            "Override every trading participant's slow strategist. Use frozen for "
            "a deterministic control or openai for an all-agent adaptive run."
        ),
    )
    return parser.parse_args()


def apply_research_overrides(
    scenario,
    *,
    master_seed: int | None = None,
    replicate_id: int | None = None,
    arbitrage_entry_bps: float | None = None,
    arbitrage_exit_bps: float | None = None,
    slow_strategist: str | None = None,
):
    """Return an isolated, archived scenario variant for a controlled run."""
    seed_override = master_seed is not None or replicate_id is not None
    threshold_override = (
        arbitrage_entry_bps is not None or arbitrage_exit_bps is not None
    )
    strategist_override = slow_strategist is not None
    if not seed_override and not threshold_override and not strategist_override:
        return scenario
    if threshold_override and (
        arbitrage_entry_bps is None or arbitrage_exit_bps is None
    ):
        raise ValueError(
            "Set both --arbitrage-entry-bps and --arbitrage-exit-bps together."
        )
    if threshold_override and (
        arbitrage_entry_bps < 0
        or arbitrage_exit_bps < 0
        or arbitrage_exit_bps >= arbitrage_entry_bps
    ):
        raise ValueError(
            "Arbitrage thresholds must satisfy entry > exit >= 0."
        )

    raw = deepcopy(scenario.raw)
    ecology = raw.get("aml_config", {}).get("ecology")
    if not isinstance(ecology, dict) or not ecology.get("enabled"):
        raise ValueError("Research overrides require an ecology-enabled scenario.")

    experiment = ecology.setdefault("experiment", {})
    if master_seed is not None:
        experiment["master_seed"] = master_seed
    if replicate_id is not None:
        experiment["replicate_id"] = replicate_id

    agents = raw["stocksim_config"]["agents"]
    if seed_override:
        # Explicit scenario seeds would otherwise prevent paired replicates from
        # changing their stochastic streams. The launcher derives one seed per
        # agent from the archived master seed and replicate identifier instead.
        for agent in agents.values():
            parameters = agent.get("parameters") if isinstance(agent, dict) else None
            if isinstance(parameters, dict):
                parameters.pop("random_seed", None)

    if threshold_override:
        arbitrage_agents = [
            agent
            for agent in agents.values()
            if isinstance(agent, dict)
            and agent.get("type") == "AML_Cross_Market_Arbitrageur"
        ]
        if not arbitrage_agents:
            raise ValueError(
                "Arbitrage threshold overrides require an A1 scenario with an "
                "AML_Cross_Market_Arbitrageur."
            )
        for agent in arbitrage_agents:
            parameters = agent.setdefault("parameters", {})
            parameters["entry_threshold_bps"] = arbitrage_entry_bps
            parameters["exit_threshold_bps"] = arbitrage_exit_bps

    if strategist_override:
        for agent in agents.values():
            if not isinstance(agent, dict) or agent.get("type") == "AML_Shock_Agent":
                continue
            parameters = agent.setdefault("parameters", {})
            parameters["slow_strategist"] = {"type": slow_strategist}

    return replace(
        scenario,
        aml_config=raw["aml_config"],
        stocksim_config=raw["stocksim_config"],
        raw=raw,
    )


def main() -> int:
    args = parse_args()
    scenario = load_scenario(Path(args.scenario))
    scenario = apply_research_overrides(
        scenario,
        master_seed=args.master_seed,
        replicate_id=args.replicate_id,
        arbitrage_entry_bps=args.arbitrage_entry_bps,
        arbitrage_exit_bps=args.arbitrage_exit_bps,
        slow_strategist=args.slow_strategist,
    )
    aml_run = create_run(scenario, RUNS_DIR, args.run_id)

    print(f"Scenario: {scenario.name}")
    print(f"Run directory: {aml_run.run_dir}")
    print(f"Scenario archived: {aml_run.scenario_path}")
    print(f"StockSim config written: {aml_run.stocksim_config_path}")
    print(f"Run metadata written: {aml_run.metadata_path}")

    if args.dry_run:
        print("Dry run only. StockSim was not launched.")
        return 0

    return launch_stocksim(
        scenario,
        aml_run,
        STOCKSIM_DIR,
        ENV_FILE,
        generate_reports=args.reports,
    )


if __name__ == "__main__":
    raise SystemExit(main())
