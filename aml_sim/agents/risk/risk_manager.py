"""Per-agent risk controller with circuit breakers and drawdown protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrderVerdict(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    SIZE_REDUCED = "size_reduced"


class HealthVerdict(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"       # warnings but still able to trade
    HALTED = "halted"            # no trading allowed


@dataclass
class RiskManager:
    """Gates every order and monitors portfolio health for one agent.

    The risk manager is intentionally synchronous and stateless beyond its
    counters so it can be called from either the fast or slow loop without
    async coordination.
    """

    # -- configurable limits -------------------------------------------------
    max_drawdown_pct: float = 0.25
    max_position_pct: float = 1.0
    max_order_rate: int = 50               # max orders per rate window
    max_consecutive_rejections: int = 8
    cooldown_ticks: int = 5
    max_order_value_pct: float = 0.10      # single order ≤ 10 % of portfolio

    # -- tracked state -------------------------------------------------------
    peak_portfolio_value: float = 0.0
    initial_portfolio_value: float = 0.0
    last_portfolio_value: float = 0.0       # updated by check_portfolio_health
    order_count: int = 0
    order_count_reset_tick: int = 0
    consecutive_rejections: int = 0
    circuit_tripped_until_tick: int = 0
    total_rejected: int = 0
    total_approved: int = 0
    last_health: HealthVerdict = HealthVerdict.HEALTHY
    health_reason: str = ""

    def seed_portfolio(self, value: float) -> None:
        """Call once after initial portfolio is known."""
        if value > 0:
            self.peak_portfolio_value = value
        if self.initial_portfolio_value <= 0 and value > 0:
            self.initial_portfolio_value = value

    # -----------------------------------------------------------------------
    # Order gating
    # -----------------------------------------------------------------------

    def check_order(
        self,
        instrument: str,
        side: str,
        quantity: int,
        price: float | None,
        *,
        portfolio_value: float,
        current_position: int,
        current_tick_id: int = 0,
        last_price: float | None = None,
    ) -> tuple[OrderVerdict, int | None]:
        """Return (verdict, adjusted_quantity_or_None).

        adjusted_quantity is only set when the verdict is SIZE_REDUCED.
        ``last_price`` is used as a fallback when ``price`` is None
        (market orders) so that single-order value limits still apply.
        """

        if self.circuit_tripped_until_tick > 0:
            if current_tick_id < self.circuit_tripped_until_tick:
                return OrderVerdict.REJECTED, None
            # Cooldown expired — reset circuit
            self.circuit_tripped_until_tick = 0
            self.consecutive_rejections = 0

        if self.last_health == HealthVerdict.HALTED:
            return OrderVerdict.REJECTED, None

        # Rate limit
        if current_tick_id != self.order_count_reset_tick:
            self.order_count = 0
            self.order_count_reset_tick = current_tick_id
        if self.order_count >= self.max_order_rate:
            return OrderVerdict.REJECTED, None

        # Drawdown check
        if self.peak_portfolio_value > 0 and portfolio_value > 0:
            drawdown = 1.0 - (portfolio_value / self.peak_portfolio_value)
            if drawdown >= self.max_drawdown_pct:
                return OrderVerdict.REJECTED, None

        # Position limit — check projected exposure after the order, not just
        # the current position.  If the order would push past the cap, reduce
        # it to fill only the remaining capacity (or reject if already capped).
        if portfolio_value > 0 and self.max_position_pct < 1.0:
            position_price = price or last_price or 1.0
            max_allowed = int(portfolio_value * self.max_position_pct / position_price)
            # Determine the signed delta: BUY increases position, SELL decreases
            delta = quantity if side.upper() == "BUY" else -quantity
            projected = current_position + delta
            if abs(projected) > max_allowed:
                remaining = max(0, max_allowed - abs(current_position))
                if remaining <= 0:
                    return OrderVerdict.REJECTED, None
                # Reduce quantity to exactly fill remaining capacity
                adjusted = min(quantity, remaining)
                return OrderVerdict.SIZE_REDUCED, adjusted

        # Single-order value limit
        # Use last_price fallback so MARKET orders are still checked.
        effective_price = price or last_price
        if effective_price and effective_price > 0 and portfolio_value > 0:
            order_value = quantity * effective_price
            if order_value > portfolio_value * self.max_order_value_pct:
                adjusted = max(1, int(portfolio_value * self.max_order_value_pct / effective_price))
                return OrderVerdict.SIZE_REDUCED, adjusted

        return OrderVerdict.APPROVED, None

    def record_order_result(self, accepted: bool, *, current_tick_id: int = 0) -> None:
        """Called after an order was submitted (or rejected by the exchange)."""
        self.order_count += 1
        if accepted:
            self.total_approved += 1
            self.consecutive_rejections = 0
        else:
            self.total_rejected += 1
            self.consecutive_rejections += 1
            if self.consecutive_rejections >= self.max_consecutive_rejections:
                self._trip_circuit(tick_id=current_tick_id)

    def _trip_circuit(self, *, tick_id: int = 0, duration_ticks: int | None = None) -> None:
        self.circuit_tripped_until_tick = tick_id + (duration_ticks or self.cooldown_ticks)

    # -----------------------------------------------------------------------
    # Portfolio health
    # -----------------------------------------------------------------------

    def check_portfolio_health(
        self,
        *,
        portfolio_value: float,
        current_tick_id: int = 0,
    ) -> HealthVerdict:
        """Return the current health status.

        Call this at the start of each tick. The agent can use the result to
        decide whether to skip the fast loop entirely.
        """
        self.last_portfolio_value = portfolio_value

        # Circuit check
        if self.circuit_tripped_until_tick > 0 and current_tick_id < self.circuit_tripped_until_tick:
            self.last_health = HealthVerdict.HALTED
            self.health_reason = f"Circuit breaker active until tick {self.circuit_tripped_until_tick}"
            return HealthVerdict.HALTED

        # Drawdown check
        if self.peak_portfolio_value > 0 and portfolio_value > 0:
            drawdown = 1.0 - (portfolio_value / self.peak_portfolio_value)
            if drawdown >= self.max_drawdown_pct:
                self.last_health = HealthVerdict.HALTED
                self.health_reason = (
                    f"Max drawdown exceeded: {drawdown:.1%} >= {self.max_drawdown_pct:.1%}"
                )
                return HealthVerdict.HALTED
            if drawdown >= self.max_drawdown_pct * 0.7:
                self.last_health = HealthVerdict.DEGRADED
                self.health_reason = f"Drawdown warning: {drawdown:.1%}"
                return HealthVerdict.DEGRADED

        # Rejection rate warning
        total = self.total_approved + self.total_rejected
        if total > 20 and self.total_rejected / max(total, 1) > 0.5:
            self.last_health = HealthVerdict.DEGRADED
            self.health_reason = "High order rejection rate"
            return HealthVerdict.DEGRADED

        self.last_health = HealthVerdict.HEALTHY
        self.health_reason = ""
        return HealthVerdict.HEALTHY

    def update_portfolio_peak(self, value: float) -> None:
        if value > self.peak_portfolio_value:
            self.peak_portfolio_value = value

    def is_circuit_open(self, *, current_tick_id: int = 0) -> bool:
        return (
            self.circuit_tripped_until_tick > 0
            and current_tick_id < self.circuit_tripped_until_tick
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of current risk state.

        Drawdown is computed against the peak (not initial) value,
        reflecting the current portfolio decline from its high-water mark.
        """
        peak = max(self.peak_portfolio_value, 1.0)
        current = max(self.last_portfolio_value, 0.0)
        drawdown_pct = (
            round((1.0 - current / peak) * 100, 2)
            if self.peak_portfolio_value > 0
            else 0.0
        )
        return {
            "health": self.last_health.value,
            "health_reason": self.health_reason,
            "drawdown_pct": drawdown_pct,
            "peak_portfolio_value": round(self.peak_portfolio_value, 2),
            "current_portfolio_value": round(self.last_portfolio_value, 2),
            "circuit_open": self.circuit_tripped_until_tick > 0,
            "total_approved": self.total_approved,
            "total_rejected": self.total_rejected,
            "consecutive_rejections": self.consecutive_rejections,
        }
