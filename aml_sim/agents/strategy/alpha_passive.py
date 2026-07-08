"""Passive benchmark alpha strategy — trades minimally, tracks a reference."""

from __future__ import annotations

from dataclasses import dataclass

from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal


@dataclass
class PassiveBenchmarkStrategy:
    """A deliberately inactive strategy that rarely produces signals.

    This serves as a baseline / control group strategy. It can also be used
    as a circuit-breaker fallback: when risk is elevated, replace the active
    strategy set with this one to stop trading.
    """

    name: str = "passive_benchmark"
    description: str = (
        "Produces no signals under normal conditions. Use as a benchmark "
        "or as a safe fallback when risk limits are breached."
    )

    # Extremely low probability of a random trade — useful for adding noise
    # to otherwise identical control-group agents.
    noise_probability: float = 0.0
    noise_strength: float = 0.1

    def generate(self, context: AlphaContext) -> AlphaSignal:
        import random

        if self.noise_probability <= 0:
            return AlphaSignal(reason="passive_benchmark: holding")

        if random.random() < self.noise_probability:
            direction = 1.0 if random.random() < 0.5 else -1.0
            return AlphaSignal(
                direction=direction,
                strength=self.noise_strength,
                confidence=0.1,
                horizon_ticks=1,
                reason="passive_benchmark: random noise trade",
                suggested_order_type="MARKET",
            )

        return AlphaSignal(reason="passive_benchmark: holding")
