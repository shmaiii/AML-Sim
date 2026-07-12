"""Pluggable alpha strategy protocol and shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class AlphaContext:
    """Input package passed to every alpha strategy's ``generate`` method."""

    prices: list[float] = field(default_factory=list)
    volume_history: list[float] = field(default_factory=list)
    current_position: int = 0
    portfolio_value: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    current_strategy: Any = None
    profile: Any = None           # BehavioralTraits or compatible mapping
    instrument: str = ""
    last_market_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlphaSignal:
    """A structured signal produced by one alpha strategy."""

    direction: float = 0.0          # -1.0 .. +1.0
    strength: float = 0.0           # 0.0 .. 1.0
    confidence: float = 0.0         # 0.0 .. 1.0
    horizon_ticks: int = 5          # expected holding period in ticks
    reason: str = ""
    suggested_order_type: str = ""  # "MARKET" | "LIMIT" | ""
    suggested_limit_offset: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return abs(self.direction) > 0 and self.strength > 0 and self.confidence > 0


class AlphaStrategy(Protocol):
    """Interface for pluggable alpha signal generators."""

    name: str
    description: str

    def generate(self, context: AlphaContext) -> AlphaSignal:
        """Produce a trading signal from the given context."""
        ...
