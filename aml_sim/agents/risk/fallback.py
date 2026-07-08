"""Strategy fallback chain for graceful degradation on slow-loop failure."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FallbackChain:
    """Degrades to a safe strategy when the slow loop fails repeatedly.

    The fallback chain is role-agnostic: it just snapshots the initial strategy
    state as "safe" and restores it after N consecutive failures. Concrete
    agents can override ``get_fallback_state()`` to return a role-specific
    conservative state if desired.
    """

    max_consecutive_failures: int = 3
    safe_state: Any = None
    failure_count: int = 0
    total_failures: int = 0
    total_successes: int = 0
    active: bool = False
    restored_at: str | None = None

    def seed_safe_state(self, strategy_state: Any) -> None:
        """Capture the initial strategy state as the safe fallback."""
        if self.safe_state is None:
            try:
                self.safe_state = copy.deepcopy(strategy_state)
            except Exception:
                self.safe_state = strategy_state

    def record_result(self, success: bool) -> None:
        """Call after each slow-loop attempt."""
        if success:
            self.failure_count = 0
            self.total_successes += 1
            if self.active:
                self.active = False
                self.restored_at = None
        else:
            self.failure_count += 1
            self.total_failures += 1
            if self.failure_count >= self.max_consecutive_failures:
                self.active = True

    def should_fallback(self) -> bool:
        """Return True if the agent should use the fallback state right now."""
        return self.active and self.safe_state is not None

    def get_fallback_state(self) -> Any | None:
        """Return the safe strategy state, or None if fallback is not active."""
        if not self.active or self.safe_state is None:
            return None
        try:
            return copy.deepcopy(self.safe_state)
        except Exception:
            return self.safe_state

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "failure_count": self.failure_count,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "max_consecutive_failures": self.max_consecutive_failures,
        }
