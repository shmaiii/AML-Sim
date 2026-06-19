"""Slow-loop strategy updaters for AML agents."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Mapping, Protocol


class SlowStrategist(Protocol):
    """Interface for rule-based and future LLM slow strategists."""

    def propose(self, observation: Mapping[str, Any], current_strategy: Any) -> Any:
        """Return a proposed strategy state for validation and application."""


@dataclass
class RuleBasedSlowStrategist:
    """
    Minimal slow-loop strategist used before adding LLM calls.

    For market makers, it widens spread when inventory is too far from target.
    For other strategy states, it returns the current strategy unchanged.
    """

    inventory_widen_threshold: int = 100
    spread_multiplier: float = 1.25
    max_spread: float = 5.0

    def propose(self, observation: Mapping[str, Any], current_strategy: Any) -> Any:
        if not hasattr(current_strategy, "spread"):
            return current_strategy

        max_inventory_gap = self._max_inventory_gap(observation, current_strategy)
        if max_inventory_gap <= self.inventory_widen_threshold:
            return current_strategy

        widened_spread = min(
            self.max_spread,
            current_strategy.spread * self.spread_multiplier,
        )
        if widened_spread == current_strategy.spread:
            return current_strategy

        return self._replace_strategy(
            current_strategy,
            spread=widened_spread,
            reason=(
                "Rule-based slow loop widened spread because inventory "
                f"gap reached {max_inventory_gap}."
            ),
            updated_at=observation.get("current_time"),
        )

    def _max_inventory_gap(
        self,
        observation: Mapping[str, Any],
        current_strategy: Any,
    ) -> int:
        portfolio = observation.get("portfolio", {})
        inventory = portfolio.get("inventory", {})
        target_inventory = getattr(current_strategy, "target_inventory", 0)

        gaps = [
            abs(position.get("net", 0) - target_inventory)
            for position in inventory.values()
        ]
        return max(gaps, default=0)

    def _replace_strategy(self, current_strategy: Any, **changes: Any) -> Any:
        if is_dataclass(current_strategy):
            return replace(current_strategy, **changes)

        for key, value in changes.items():
            setattr(current_strategy, key, value)
        return current_strategy
