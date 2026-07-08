"""Mean-reversion alpha strategy — bets against short-term price moves."""

from __future__ import annotations

from dataclasses import dataclass

from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal
from aml_sim.agents.strategy.signals import mean_reversion_signal


@dataclass
class MeanReversionStrategy:
    """Trade against recent price deviations from the moving average.

    Price below mean → buy signal (expect reversion up).
    Price above mean → sell signal (expect reversion down).
    """

    name: str = "mean_reversion"
    description: str = (
        "Bets against short-term deviations: buy dips, sell rallies"
    )
    lookback_ticks: int = 8
    min_signal: float = 0.0005

    def generate(self, context: AlphaContext) -> AlphaSignal:
        prices = context.prices
        if len(prices) < self.lookback_ticks:
            return AlphaSignal(reason="mean_reversion: insufficient price history")

        raw = mean_reversion_signal(prices, self.lookback_ticks)
        if abs(raw) < self.min_signal:
            return AlphaSignal(
                direction=raw,
                strength=0.0,
                confidence=0.0,
                reason=f"mean_reversion: {raw:.5f} below threshold {self.min_signal}",
            )

        direction = 1.0 if raw > 0 else -1.0
        strength = min(1.0, abs(raw) * 200)
        confidence = min(1.0, abs(raw) * 150)

        return AlphaSignal(
            direction=direction,
            strength=round(strength, 4),
            confidence=round(confidence, 4),
            horizon_ticks=max(3, self.lookback_ticks // 2),
            reason=f"mean_reversion: {raw:.5f} deviation over {self.lookback_ticks} ticks",
            suggested_order_type="LIMIT",
        )
