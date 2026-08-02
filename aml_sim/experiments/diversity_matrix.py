"""Generate D0-D4 diversity scenarios with fixed repeated seeds."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml


LEVELS = {
    "D0": {"strategy": False, "profile": False, "prompt": False},
    "D1": {"strategy": True, "profile": False, "prompt": False},
    "D2": {"strategy": True, "profile": True, "prompt": False},
    "D3": {"strategy": True, "profile": True, "prompt": True},
    "D4": {"strategy": True, "profile": True, "prompt": True},
}
DEFAULT_SEEDS = [104729, 130363, 155921, 181081, 206369]

STRATEGY_VARIANTS = [
    {"trade_probability": 0.24, "buy_bias": 0.42, "herding_tendency": 0.15, "panic_level": 0.25},
    {"trade_probability": 0.34, "buy_bias": 0.48, "herding_tendency": 0.35, "panic_level": 0.45},
    {"trade_probability": 0.44, "buy_bias": 0.54, "herding_tendency": 0.55, "panic_level": 0.65},
    {"trade_probability": 0.54, "buy_bias": 0.60, "herding_tendency": 0.75, "panic_level": 0.85},
]
PROFILE_VARIANTS = [
    {"social_sensitivity": 0.2, "panic_sensitivity": 0.25, "herding_tendency": 0.2, "news_reactivity": 0.35, "loss_aversion": 0.8},
    {"social_sensitivity": 0.4, "panic_sensitivity": 0.45, "herding_tendency": 0.4, "news_reactivity": 0.5, "loss_aversion": 0.65},
    {"social_sensitivity": 0.65, "panic_sensitivity": 0.65, "herding_tendency": 0.6, "news_reactivity": 0.7, "loss_aversion": 0.45},
    {"social_sensitivity": 0.85, "panic_sensitivity": 0.85, "herding_tendency": 0.8, "news_reactivity": 0.9, "loss_aversion": 0.3},
]
PROMPT_VARIANTS = [
    {"goal": "preserve capital and trade only on strong evidence", "behavior": "prefer low turnover and resist crowd pressure"},
    {"goal": "trade public information selectively", "behavior": "balance recent momentum with loss control"},
    {"goal": "adapt quickly to news and changing sentiment", "behavior": "increase participation when confidence is high"},
    {"goal": "capture short-lived crowd-driven opportunities", "behavior": "act quickly but respect inventory and risk limits"},
]


def stable_seed(experiment_seed: int, agent_index: int) -> int:
    digest = hashlib.sha256(f"{experiment_seed}:retail:{agent_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def build_condition(base: dict[str, Any], level: str, seed: int) -> dict[str, Any]:
    if level not in LEVELS:
        raise ValueError(f"Unknown diversity level {level!r}")
    scenario = copy.deepcopy(base)
    scenario["name"] = f"aml_diversity_{level.lower()}_seed_{seed}"
    scenario["description"] = (
        f"Diversity experiment {level} with fixed experiment seed {seed}."
    )
    aml_config = scenario.setdefault("aml_config", {})
    aml_config["experiment"] = {
        "diversity_level": level,
        "seed": seed,
        "replicate": DEFAULT_SEEDS.index(seed) + 1 if seed in DEFAULT_SEEDS else None,
    }

    agents = scenario["stocksim_config"]["agents"]
    retail_template = copy.deepcopy(agents.pop("retail_trader"))
    base_parameters = retail_template["parameters"]
    base_profile = copy.deepcopy(base_parameters.get("profile", {}))
    base_prompt = copy.deepcopy(
        base_parameters.get("slow_strategist", {}).get("role_prompt", {})
    )

    for index in range(4):
        group = copy.deepcopy(retail_template)
        params = group["parameters"]
        params["random_seed"] = stable_seed(seed, index)
        params["profile"] = copy.deepcopy(base_profile)
        params["profile"]["name"] = f"{level.lower()}_retail_{index + 1}"
        params["slow_strategist"]["role_prompt"] = copy.deepcopy(base_prompt)

        if LEVELS[level]["strategy"]:
            params.update(STRATEGY_VARIANTS[index])
        if LEVELS[level]["profile"]:
            params["profile"].update(PROFILE_VARIANTS[index])
        if LEVELS[level]["prompt"]:
            prompt_index = index
            if level == "D4":
                prompt_index = 3 - index
                params["profile"]["decision_style"] = (
                    "contrarian" if index % 2 == 0 else "momentum_sensitive"
                )
            params["slow_strategist"]["role_prompt"] = copy.deepcopy(
                PROMPT_VARIANTS[prompt_index]
            )
        agents[f"retail_{index + 1}"] = group
    return scenario


def generate_matrix(base_path: Path, output_dir: Path, seeds: list[int]) -> list[Path]:
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        for replicate, seed in enumerate(seeds, start=1):
            scenario = build_condition(base, level, seed)
            output_path = output_dir / f"{level.lower()}_seed_{seed}.yaml"
            with output_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(scenario, handle, sort_keys=False)
            generated.append(output_path)
            manifest_rows.append({
                "diversity_level": level,
                "seed": seed,
                "replicate": replicate,
                "scenario": output_path.name,
                **LEVELS[level],
            })
    with (output_dir / "matrix_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("scenarios/aml_diversity_api_10min.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("scenarios/diversity"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    generated = generate_matrix(args.base, args.output_dir, args.seeds)
    print(f"Generated {len(generated)} diversity scenarios in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
