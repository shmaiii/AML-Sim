"""Breakout alpha strategy — trades when price breaks through recent range."""

from __future__ import annotations

from dataclasses import dataclass

from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal


@dataclass
class BreakoutStrategy:
    """Trade when price breaks above a recent high or below a recent low.

    Price > recent high → buy signal (breakout up).
    Price < recent low  → sell signal (breakdown).
    """

    name: str = "breakout"
    description: str = (
        "Trades price range breakouts: buys when price exceeds recent high, "
        "sells when price drops below recent low"
    )
    lookback_ticks: int = 10
    confirmation_ticks: int = 1    # ticks price must stay beyond level
    buffer_pct: float = 0.001       # 0.1 % buffer to avoid noise breakouts

    def generate(self, context: AlphaContext) -> AlphaSignal:
        prices = context.prices
        if len(prices) < self.lookback_ticks + self.confirmation_ticks:
            return AlphaSignal(reason="breakout: insufficient price history")

        window = prices[-(self.lookback_ticks + self.confirmation_ticks):]
        recent_high = max(window[:self.lookback_ticks])
        recent_low = min(window[:self.lookback_ticks])
        current = window[-1]

        high_threshold = recent_high * (1.0 + self.buffer_pct)
        low_threshold = recent_low * (1.0 - self.buffer_pct)

        if current > high_threshold:
            strength = min(1.0, (current - recent_high) / (recent_high * 0.01 + 0.0001))
            return AlphaSignal(
                direction=1.0,
                strength=round(strength, 4),
                confidence=min(1.0, 0.4 + strength * 0.6),
                horizon_ticks=self.lookback_ticks,
                reason=f"breakout: {current:.2f} > high {recent_high:.2f} over {self.lookback_ticks}t",
                suggested_order_type="MARKET",
            )

        if current < low_threshold:
            strength = min(1.0, (recent_low - current) / (recent_low * 0.01 + 0.0001))
            return AlphaSignal(
                direction=-1.0,
                strength=round(strength, 4),
                confidence=min(1.0, 0.4 + strength * 0.6),
                horizon_ticks=self.lookback_ticks,
                reason=f"breakout: {current:.2f} < low {recent_low:.2f} over {self.lookback_ticks}t",
                suggested_order_type="MARKET",
            )

        return AlphaSignal(
            reason=(
                f"breakout: {current:.2f} within [{recent_low:.2f}, {recent_high:.2f}]"
            ),
        )
