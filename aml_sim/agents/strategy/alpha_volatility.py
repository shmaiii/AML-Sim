"""Volatility-regime alpha strategy — adjusts exposure based on volatility."""

from __future__ import annotations

from dataclasses import dataclass

from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal
from aml_sim.agents.strategy.signals import momentum_signal, realized_volatility


@dataclass
class VolatilityRegimeStrategy:
    """Adjust trading posture based on the current volatility regime.

    In high-volatility regimes this strategy becomes more cautious (lower
    confidence) and may flip to a defensive direction. In low-volatility
    regimes it leans into momentum more aggressively.

    This works best as a *modulator* layered with directional strategies
    rather than a standalone signal.
    """

    name: str = "volatility_regime"
    description: str = (
        "Adjusts exposure based on volatility regime: cautious in high vol, "
        "aggressive in low vol. Best used in composite with directional strategies."
    )
    lookback_ticks: int = 12
    high_vol_threshold: float = 0.01    # 1 % per tick — very high
    low_vol_threshold: float = 0.002    # 0.2 % per tick — very low

    def generate(self, context: AlphaContext) -> AlphaSignal:
        prices = context.prices
        if len(prices) < self.lookback_ticks + 2:
            return AlphaSignal(reason="volatility: insufficient price history")

        vol = realized_volatility(prices, self.lookback_ticks)

        if vol > self.high_vol_threshold:
            # High vol → preference to reduce or go flat
            momentum = momentum_signal(prices, min(3, len(prices) - 1))
            direction = 1.0 if momentum > 0 else -1.0
            return AlphaSignal(
                direction=direction,
                strength=min(0.3, 1.0 / (vol * 100 + 1)),
                confidence=min(0.4, 0.6 / (vol * 50 + 1)),
                horizon_ticks=3,
                reason=f"volatility_regime: HIGH vol {vol:.4f}, cautious",
                suggested_order_type="LIMIT",
                metadata={"volatility": round(vol, 6), "regime": "high"},
            )

        if vol < self.low_vol_threshold:
            # Low vol → lean into momentum
            momentum = momentum_signal(prices, min(5, len(prices) - 1))
            direction = 1.0 if momentum > 0 else -1.0
            strength = min(0.9, abs(momentum) * 80 + 0.3)
            return AlphaSignal(
                direction=direction,
                strength=round(strength, 4),
                confidence=min(0.85, abs(momentum) * 60 + 0.3),
                horizon_ticks=8,
                reason=f"volatility_regime: LOW vol {vol:.4f}, leaning into momentum",
                suggested_order_type="MARKET",
                metadata={"volatility": round(vol, 6), "regime": "low"},
            )

        # Normal vol → neutral, let other strategies drive
        return AlphaSignal(
            reason=f"volatility_regime: normal vol {vol:.4f}, neutral",
            metadata={"volatility": round(vol, 6), "regime": "normal"},
        )
