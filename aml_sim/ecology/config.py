"""Configuration and reproducibility metadata for financial-ecology runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentConfig:
    """Stable identity for a treatment and its paired randomization stream."""

    experiment_id: str
    treatment: str
    master_seed: int
    replicate_id: int


@dataclass(frozen=True)
class EcologyConfig:
    """AML-owned extensions needed for an ecology experiment."""

    enabled: bool
    experiment: ExperimentConfig
    relationships: tuple[dict[str, Any], ...]


def load_ecology_config(aml_config: Mapping[str, Any] | None) -> EcologyConfig:
    """Parse the optional ecology block without changing legacy scenarios."""
    aml_config = dict(aml_config or {})
    ecology = aml_config.get("ecology", {})
    if ecology is None:
        ecology = {}
    if not isinstance(ecology, Mapping):
        raise ValueError("aml_config.ecology must be a mapping when provided")

    legacy_experiment = aml_config.get("experiment", {})
    if legacy_experiment is None:
        legacy_experiment = {}
    if not isinstance(legacy_experiment, Mapping):
        raise ValueError("aml_config.experiment must be a mapping when provided")

    configured_experiment = ecology.get("experiment", {})
    if configured_experiment is None:
        configured_experiment = {}
    if not isinstance(configured_experiment, Mapping):
        raise ValueError("aml_config.ecology.experiment must be a mapping")

    experiment = {**legacy_experiment, **configured_experiment}
    relationships_raw = ecology.get("relationships", [])
    if relationships_raw is None:
        relationships_raw = []
    if not isinstance(relationships_raw, list):
        raise ValueError("aml_config.ecology.relationships must be a list")
    relationships = tuple(
        dict(relationship)
        for relationship in relationships_raw
        if isinstance(relationship, Mapping)
    )
    if len(relationships) != len(relationships_raw):
        raise ValueError("Every ecology relationship must be a mapping")

    enabled = bool(ecology.get("enabled", bool(relationships)))
    return EcologyConfig(
        enabled=enabled,
        experiment=ExperimentConfig(
            experiment_id=str(experiment.get("id", experiment.get("name", "aml_ecology"))),
            treatment=str(experiment.get("treatment", "baseline")),
            master_seed=_coerce_int(experiment.get("master_seed", experiment.get("seed", 0))),
            replicate_id=_coerce_int(experiment.get("replicate_id", experiment.get("replicate", 0))),
        ),
        relationships=relationships,
    )


def scenario_fingerprint(data: Mapping[str, Any]) -> str:
    """Return a canonical hash for scenario-level reproducibility checks."""
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an integer seed or replicate id, got {value!r}") from exc
