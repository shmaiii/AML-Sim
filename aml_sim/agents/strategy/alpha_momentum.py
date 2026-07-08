"""Momentum alpha strategy — follows price trends."""

from __future__ import annotations

from dataclasses import dataclass

from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal
from aml_sim.agents.strategy.signals import momentum_signal


@dataclass
class MomentumStrategy:
    """Trade in the direction of recent price movement.

    Positive momentum → buy signal. Negative momentum → sell signal.
    Signal strength scales with the magnitude of the momentum return.
    """

    name: str = "momentum"
    description: str = "Follows price trends: buy when price is rising, sell when falling"
    lookback_ticks: int = 5
    min_signal: float = 0.001    # minimum absolute return to trigger

    def generate(self, context: AlphaContext) -> AlphaSignal:
        prices = context.prices
        if len(prices) < self.lookback_ticks + 1:
            return AlphaSignal(reason="momentum: insufficient price history")

        raw = momentum_signal(prices, self.lookback_ticks)
        if abs(raw) < self.min_signal:
            return AlphaSignal(
                direction=raw,
                strength=0.0,
                confidence=0.0,
                reason=f"momentum: {raw:.5f} below threshold {self.min_signal}",
            )

        direction = 1.0 if raw > 0 else -1.0
        strength = min(1.0, abs(raw) * 100)   # scale roughly, 1% return → 1.0 strength
        confidence = min(1.0, abs(raw) * 80)

        return AlphaSignal(
            direction=direction,
            strength=round(strength, 4),
            confidence=round(confidence, 4),
            horizon_ticks=max(3, self.lookback_ticks),
            reason=f"momentum: {raw:.5f} return over {self.lookback_ticks} ticks",
            suggested_order_type="MARKET",
        )
