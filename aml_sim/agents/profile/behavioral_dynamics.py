"""Behavioral state tracking and dynamics for AML agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BehavioralState:
    """Tracks an agent's emotional / behavioral state over time.

    These values drift each tick based on PnL, fill rate, and market
    conditions, then feed back into profile modulation (e.g. high stress
    temporarily boosts risk_aversion).
    """

    pnl_streak: int = 0               # positive = winning streak, negative = losing
    recent_accuracy: float = 0.5       # rolling hit rate of recent signals
    stress_level: float = 0.0          # 0 = calm, 1 = stressed
    confidence_drift: float = 0.0      # shift from baseline confidence (-0.5 .. +0.5)
    market_regime_belief: str = "neutral"   # trending | ranging | volatile | neutral
    ticks_since_last_trade: int = 0
    total_pnl: float = 0.0
    peak_pnl: float = 0.0
    drawdown_from_peak: float = 0.0    # as a fraction

    def snapshot(self) -> dict[str, Any]:
        return {
            "pnl_streak": self.pnl_streak,
            "recent_accuracy": round(self.recent_accuracy, 3),
            "stress_level": round(self.stress_level, 3),
            "confidence_drift": round(self.confidence_drift, 3),
            "market_regime_belief": self.market_regime_belief,
            "ticks_since_last_trade": self.ticks_since_last_trade,
            "drawdown_from_peak": round(self.drawdown_from_peak, 4),
        }


@dataclass
class BehavioralDynamics:
    """Updates an agent's BehavioralState each tick.

    Usage (inside agent tick):
        dynamics.update(state, pnl=..., had_fill=True, ...)
    """

    # Smoothing factors
    stress_decay: float = 0.05        # how fast stress decays per tick without a loss
    stress_gain: float = 0.12         # how fast stress rises per losing tick
    accuracy_smoothing: float = 0.1   # EMA weight for new accuracy observation
    regime_lookback: int = 20          # ticks to look back for regime detection
    streak_saturation: int = 10        # max absolute streak value

    def update(
        self,
        state: BehavioralState,
        *,
        pnl_delta: float = 0.0,
        had_fill: bool = False,
        signal_correct: bool | None = None,
        prices: list[float] | None = None,
    ) -> None:
        """Advance the behavioral state by one tick."""

        state.total_pnl += pnl_delta
        state.ticks_since_last_trade += 1

        # PnL streak
        if pnl_delta > 0:
            state.pnl_streak = min(self.streak_saturation, max(state.pnl_streak, 0) + 1)
        elif pnl_delta < 0:
            state.pnl_streak = max(-self.streak_saturation, min(state.pnl_streak, 0) - 1)
        # zero delta: streak unchanged

        # Stress — rises on losing ticks, decays otherwise
        if pnl_delta < 0:
            state.stress_level = min(1.0, state.stress_level + self.stress_gain)
        else:
            state.stress_level = max(0.0, state.stress_level - self.stress_decay)

        # Accuracy
        if signal_correct is not None:
            new_accuracy = 1.0 if signal_correct else 0.0
            state.recent_accuracy = (
                state.recent_accuracy * (1 - self.accuracy_smoothing)
                + new_accuracy * self.accuracy_smoothing
            )

        # Confidence drift — winning streak → overconfidence, losing → underconfidence
        state.confidence_drift = max(-0.5, min(0.5, state.pnl_streak * 0.05))

        # Drawdown
        if state.total_pnl > state.peak_pnl:
            state.peak_pnl = state.total_pnl
        if state.peak_pnl != 0:
            state.drawdown_from_peak = max(
                0.0, (state.peak_pnl - state.total_pnl) / abs(state.peak_pnl + 0.01)
            )

        # Reset trade counter on fill
        if had_fill:
            state.ticks_since_last_trade = 0

        # Simple regime detection from prices
        if prices and len(prices) >= self.regime_lookback:
            state.market_regime_belief = _detect_regime(prices[-self.regime_lookback:])


def _detect_regime(prices: list[float]) -> str:
    """Crude regime classifier: trending, ranging, or volatile."""
    if len(prices) < 5:
        return "neutral"

    # Directional consistency
    ups = sum(1 for a, b in zip(prices, prices[1:]) if b > a)
    downs = len(prices) - 1 - ups
    if ups == 0 and downs == 0:
        return "neutral"

    directionality = abs(ups - downs) / max(ups + downs, 1)

    # Volatility
    from aml_sim.agents.strategy.signals import realized_volatility

    vol = realized_volatility(prices, len(prices))

    if vol > 0.008:
        return "volatile"
    if directionality > 0.55:
        return "trending"
    if directionality < 0.25:
        return "ranging"
    return "neutral"
