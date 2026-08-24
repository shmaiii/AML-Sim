"""Explicit routing for direct and relationship-mediated event information."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from aml_sim.ecology.registry import RelationshipRegistry


_LOCAL_VISIBILITIES = {"affected", "local", "private", "affected_market"}


@dataclass(frozen=True)
class EventDelivery:
    """One agent's informational access to an AML event."""

    agent_id: str
    delivery_type: str
    relationship_ids: tuple[str, ...] = ()


class EcologyInformationRouter:
    """Route an event without turning information access into direct exposure."""

    def __init__(
        self,
        registry: RelationshipRegistry,
        agent_instrument_map: Mapping[str, Mapping[str, Any] | list[str] | tuple[str, ...]],
    ) -> None:
        self.registry = registry
        self.agent_instruments = {
            str(agent_id): _instruments(value)
            for agent_id, value in agent_instrument_map.items()
        }

    def deliveries(
        self,
        event: Mapping[str, Any],
        payload: Mapping[str, Any],
        target_agent_ids: list[str],
    ) -> list[EventDelivery]:
        """Return direct or information-only recipients for one declared event."""
        all_targets = [str(agent_id) for agent_id in target_agent_ids]
        affected = _affected_instruments(event, payload)
        visibility = str(payload.get("visibility", event.get("visibility", "public"))).lower()
        if visibility in _LOCAL_VISIBILITIES and self.agent_instruments:
            direct_ids = {
                agent_id
                for agent_id in all_targets
                if self.agent_instruments.get(agent_id, set()) & affected
            }
        else:
            direct_ids = set(all_targets)

        related: dict[str, set[str]] = {}
        for relationship in self.registry.relationships:
            if not relationship.channel_enabled("information"):
                continue
            if relationship.source in affected:
                related_instrument = relationship.target
            elif relationship.target in affected:
                related_instrument = relationship.source
            else:
                continue
            for agent_id in all_targets:
                if agent_id in direct_ids:
                    continue
                if related_instrument in self.agent_instruments.get(agent_id, set()):
                    related.setdefault(agent_id, set()).add(relationship.relationship_id)

        deliveries = [
            EventDelivery(agent_id, "direct")
            for agent_id in all_targets
            if agent_id in direct_ids
        ]
        deliveries.extend(
            EventDelivery(agent_id, "relationship_information", tuple(sorted(relationship_ids)))
            for agent_id, relationship_ids in sorted(related.items())
        )
        return deliveries


def _instruments(value: Mapping[str, Any] | list[str] | tuple[str, ...]) -> set[str]:
    if isinstance(value, Mapping):
        return {str(instrument) for instrument in value}
    return {str(instrument) for instrument in value}


def _affected_instruments(
    event: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> set[str]:
    affected = payload.get("affected_instruments", event.get("affected_instruments", []))
    if isinstance(affected, str):
        return {affected}
    if isinstance(affected, (list, tuple, set)):
        return {str(instrument) for instrument in affected}
    return set()
