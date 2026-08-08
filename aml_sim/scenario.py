"""Scenario loading and validation for AML-Sim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_STOCKSIM_CONFIG_KEYS = [
    "exchange_mode",
    "instruments",
    "exchanges",
    "agents",
    "simulation",
]
ALLOWED_SCENARIO_KEYS = {
    "name", "description", "rabbitmq_host", "aml_config", "stocksim_config",
}
ALLOWED_AML_CONFIG_KEYS = {"dataset_split", "llm", "experiment"}
ALLOWED_AGENT_GROUP_KEYS = {"type", "count", "parameters"}
ALLOWED_STOCKSIM_CONFIG_KEYS = {
    "exchange_mode", "instruments", "exchanges", "agents", "simulation", "environment",
}
ALLOWED_SIMULATION_KEYS = {
    "start_time", "end_time", "tick_interval", "expected_exchange_agent_count",
    "max_wall_time_seconds", "startup_grace_seconds", "barrier_timeout_seconds",
    "inter_tick_delay_seconds",
}
ALLOWED_EXCHANGE_KEYS = {
    "candle_interval", "data_source", "symbol_type", "warmup_candles",
    "warmup_start_date", "warmup_end_date", "warmup_resolution",
    "indicator_kwargs", "news", "spread_factor", "trades_outfile",
}
ALLOWED_ENVIRONMENT_KEYS = {"rabbitmq_host", "log_level"}


@dataclass(frozen=True)
class AMLScenario:
    """Parsed AML scenario with the StockSim-compatible config separated out."""

    path: Path
    name: str
    description: str | None
    rabbitmq_host: str | None
    aml_config: dict[str, Any]
    stocksim_config: dict[str, Any]
    raw: dict[str, Any]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must contain a YAML mapping: {path}")

    return data


def build_stocksim_config(scenario_data: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate the StockSim config embedded in an AML scenario."""
    config = scenario_data.get("stocksim_config")
    if not isinstance(config, dict):
        raise ValueError("Scenario must define a 'stocksim_config' mapping")

    missing = [key for key in REQUIRED_STOCKSIM_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"stocksim_config missing required keys: {', '.join(missing)}")

    unknown = sorted(set(config) - ALLOWED_STOCKSIM_CONFIG_KEYS)
    if unknown:
        raise ValueError("Unknown stocksim_config field(s): " + ", ".join(unknown))

    simulation = config.get("simulation", {})
    if not isinstance(simulation, dict):
        raise ValueError("stocksim_config.simulation must be a mapping")
    unknown_simulation = sorted(set(simulation) - ALLOWED_SIMULATION_KEYS)
    if unknown_simulation:
        raise ValueError(
            "Unknown simulation field(s): " + ", ".join(unknown_simulation)
        )

    exchanges = config.get("exchanges", {})
    if not isinstance(exchanges, dict):
        raise ValueError("stocksim_config.exchanges must be a mapping")
    for instrument, exchange in exchanges.items():
        if not isinstance(exchange, dict):
            raise ValueError(f"Exchange {instrument!r} must be a mapping")
        unknown_exchange = sorted(set(exchange) - ALLOWED_EXCHANGE_KEYS)
        if unknown_exchange:
            raise ValueError(
                f"Unknown exchange field(s) for {instrument!r}: "
                + ", ".join(unknown_exchange)
            )

    environment = config.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("stocksim_config.environment must be a mapping")
    unknown_environment = sorted(set(environment) - ALLOWED_ENVIRONMENT_KEYS)
    if unknown_environment:
        raise ValueError(
            "Unknown environment field(s): " + ", ".join(unknown_environment)
        )

    return config


def load_scenario(path: Path) -> AMLScenario:
    """Load an AML scenario file and return its parsed representation."""
    scenario_path = path.expanduser().resolve()
    data = load_yaml(scenario_path)
    unknown_root = sorted(set(data) - ALLOWED_SCENARIO_KEYS)
    if unknown_root:
        raise ValueError("Unknown scenario field(s): " + ", ".join(unknown_root))
    stocksim_config = build_stocksim_config(data)

    name = data.get("name") or scenario_path.stem
    description = data.get("description")
    rabbitmq_host = data.get("rabbitmq_host")
    aml_config = data.get("aml_config", {})

    if not isinstance(name, str):
        raise ValueError("Scenario 'name' must be a string when provided")
    if description is not None and not isinstance(description, str):
        raise ValueError("Scenario 'description' must be a string when provided")
    if rabbitmq_host is not None and not isinstance(rabbitmq_host, str):
        raise ValueError("Scenario 'rabbitmq_host' must be a string when provided")
    if not isinstance(aml_config, dict):
        raise ValueError("Scenario 'aml_config' must be a mapping when provided")
    unknown_aml = sorted(set(aml_config) - ALLOWED_AML_CONFIG_KEYS)
    if unknown_aml:
        raise ValueError("Unknown aml_config field(s): " + ", ".join(unknown_aml))

    agents = stocksim_config.get("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("stocksim_config.agents must be a mapping")
    for agent_name, agent_group in agents.items():
        if not isinstance(agent_group, dict):
            raise ValueError(f"Agent group {agent_name!r} must be a mapping")
        unknown_group = sorted(set(agent_group) - ALLOWED_AGENT_GROUP_KEYS)
        if unknown_group:
            raise ValueError(
                f"Unknown field(s) for agent group {agent_name!r}: "
                + ", ".join(unknown_group)
            )
        count = agent_group.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Agent group {agent_name!r} count must be a positive integer")
        if not isinstance(agent_group.get("parameters", {}), dict):
            raise ValueError(f"Agent group {agent_name!r} parameters must be a mapping")

    return AMLScenario(
        path=scenario_path,
        name=name,
        description=description,
        rabbitmq_host=rabbitmq_host,
        aml_config=aml_config,
        stocksim_config=stocksim_config,
        raw=data,
    )
