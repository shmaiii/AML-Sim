"""Typed instrument relationships used by AML financial-ecology scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


_KNOWN_RELATIONSHIP_TYPES = {
    "spot_future",
    "underlying_option",
    "basket_etf",
    "bond_rate_future",
    "generic",
}
_KNOWN_CHANNELS = {"information", "valuation", "arbitrage", "shared_risk"}


@dataclass(frozen=True)
class InstrumentRelationship:
    """One explicit economic relationship between two simulated instruments."""

    relationship_id: str
    relationship_type: str
    source: str
    target: str
    channels: frozenset[str]
    parameters: dict[str, Any]

    def channel_enabled(self, channel: str) -> bool:
        return channel in self.channels


class RelationshipRegistry:
    """Validate and query the instrument graph without symbol-name assumptions."""

    def __init__(
        self,
        instrument_metadata: Mapping[str, Mapping[str, Any]],
        relationships: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    ) -> None:
        self.instrument_metadata = {
            str(instrument): dict(metadata)
            for instrument, metadata in instrument_metadata.items()
        }
        self._relationships: dict[str, InstrumentRelationship] = {}
        for index, raw_relationship in enumerate(relationships):
            relationship = self._parse_relationship(raw_relationship, index)
            if relationship.relationship_id in self._relationships:
                raise ValueError(
                    f"Duplicate ecology relationship id {relationship.relationship_id!r}"
                )
            self._relationships[relationship.relationship_id] = relationship

    @property
    def relationships(self) -> tuple[InstrumentRelationship, ...]:
        return tuple(self._relationships.values())

    def get(self, relationship_id: str) -> InstrumentRelationship:
        try:
            return self._relationships[relationship_id]
        except KeyError as exc:
            raise ValueError(f"Unknown ecology relationship {relationship_id!r}") from exc

    def for_instrument(self, instrument: str) -> tuple[InstrumentRelationship, ...]:
        return tuple(
            relationship
            for relationship in self.relationships
            if instrument in {relationship.source, relationship.target}
        )

    def reference_target_price(
        self,
        relationship_id: str,
        source_price: float,
    ) -> float:
        """Calculate a simple, declared fair-value reference for one edge."""
        relationship = self.get(relationship_id)
        if source_price <= 0:
            return 0.0

        parameters = relationship.parameters
        if relationship.relationship_type == "spot_future":
            carry_bps = _as_float(parameters.get("annualized_carry_bps"), 0.0)
            days_to_expiry = max(0.0, _as_float(parameters.get("expiry_days"), 0.0))
            return source_price * math.exp((carry_bps / 10_000.0) * days_to_expiry / 365.0)

        return source_price * _as_float(parameters.get("reference_ratio"), 1.0)

    def hedge_ratio(self, relationship_id: str) -> float:
        relationship = self.get(relationship_id)
        return max(0.0001, _as_float(relationship.parameters.get("hedge_ratio"), 1.0))

    def as_manifest(self) -> list[dict[str, Any]]:
        """Return the normalized graph for run metadata and report artifacts."""
        return [
            {
                "id": relationship.relationship_id,
                "type": relationship.relationship_type,
                "source": relationship.source,
                "target": relationship.target,
                "channels": sorted(relationship.channels),
                "parameters": relationship.parameters,
            }
            for relationship in self.relationships
        ]

    def _parse_relationship(
        self,
        raw_relationship: Mapping[str, Any],
        index: int,
    ) -> InstrumentRelationship:
        if not isinstance(raw_relationship, Mapping):
            raise ValueError("Every ecology relationship must be a mapping")
        relationship_id = str(raw_relationship.get("id", raw_relationship.get("relationship_id", ""))).strip()
        if not relationship_id:
            raise ValueError(f"Ecology relationship at index {index} needs an id")
        relationship_type = str(raw_relationship.get("type", "generic")).strip().lower()
        if relationship_type not in _KNOWN_RELATIONSHIP_TYPES:
            allowed = ", ".join(sorted(_KNOWN_RELATIONSHIP_TYPES))
            raise ValueError(
                f"Relationship {relationship_id!r} has unsupported type {relationship_type!r}; "
                f"use one of {allowed}"
            )
        source = str(raw_relationship.get("source", "")).strip()
        target = str(raw_relationship.get("target", "")).strip()
        if not source or not target or source == target:
            raise ValueError(
                f"Relationship {relationship_id!r} needs distinct source and target instruments"
            )
        missing = [instrument for instrument in (source, target) if instrument not in self.instrument_metadata]
        if missing:
            raise ValueError(
                f"Relationship {relationship_id!r} references instruments not in the scenario: "
                f"{', '.join(missing)}"
            )

        channels = _parse_channels(raw_relationship.get("channels", raw_relationship.get("enabled_channels", [])))
        if "arbitrage" in channels and "valuation" not in channels:
            raise ValueError(
                f"Relationship {relationship_id!r} enables arbitrage without a valuation channel"
            )
        parameters = raw_relationship.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError(f"Relationship {relationship_id!r} parameters must be a mapping")
        return InstrumentRelationship(
            relationship_id=relationship_id,
            relationship_type=relationship_type,
            source=source,
            target=target,
            channels=frozenset(channels),
            parameters=dict(parameters),
        )


def _parse_channels(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        channels = {str(name).lower() for name, enabled in value.items() if bool(enabled)}
    elif value is None:
        channels = set()
    elif isinstance(value, str):
        channels = {value.lower()}
    else:
        channels = {str(channel).lower() for channel in value}
    unknown = channels - _KNOWN_CHANNELS
    if unknown:
        raise ValueError(f"Unsupported ecology channels: {', '.join(sorted(unknown))}")
    return channels


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
