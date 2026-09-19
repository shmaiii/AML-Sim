"""Stable, independent random streams for paired simulation experiments."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping


def component_seed(
    master_seed: int,
    replicate_id: int,
    component_type: str,
    component_id: str,
    purpose: str = "randomness",
) -> int:
    """Derive one portable 32-bit seed from the declared component identity."""
    material = ":".join(
        (
            str(master_seed),
            str(replicate_id),
            str(component_type),
            str(component_id),
            str(purpose),
        )
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def build_agent_seed_plan(
    agents_config: Mapping[str, Any],
    *,
    master_seed: int,
    replicate_id: int,
) -> dict[str, int]:
    """Plan one seed per actual agent process before a run starts."""
    plan: dict[str, int] = {}
    for group_name, details in agents_config.items():
        if not isinstance(details, Mapping):
            continue
        count = int(details.get("count", 1))
        agent_type = str(details.get("type", "agent"))
        for index in range(max(0, count)):
            agent_id = f"{group_name}_{index + 1}" if count > 1 else str(group_name)
            plan[agent_id] = component_seed(
                master_seed,
                replicate_id,
                agent_type,
                agent_id,
            )
    return plan


def unique_configured_seed(
    configured_seed: Any,
    *,
    count: int,
    agent_type: str,
    agent_id: str,
    replicate_id: int,
) -> int | None:
    """Preserve a singleton seed, but split one repeated seed across replicas."""
    if configured_seed is None:
        return None
    try:
        seed = int(configured_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"random_seed must be an integer, got {configured_seed!r}") from exc
    if count <= 1:
        return seed
    return component_seed(seed, replicate_id, agent_type, agent_id, "replica_randomness")
