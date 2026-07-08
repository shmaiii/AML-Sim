"""Composite alpha strategy that blends multiple sub-strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal, AlphaStrategy


@dataclass
class CompositeAlphaStrategy:
    """Combines multiple alpha strategies with configurable blending.

    Blend modes
    -----------
    - ``weighted_sum`` : Each strategy's direction × strength is weighted,
      summed, and normalized.
    - ``vote`` : Majority vote on direction. Strength = avg of agreeing
      strategies' strengths.
    - ``unanimous`` : All strategies must agree on direction, otherwise no
      signal (strength = 0, confidence = 0).
    - ``priority_cascade`` : Try strategies in order; use the first one whose
      signal passes its confidence threshold.
    """

    strategies: list[tuple[AlphaStrategy, float]] = field(default_factory=list)
    blend_mode: str = "weighted_sum"
    min_confidence: float = 0.1

    _ALLOWED_MODES = {"weighted_sum", "vote", "unanimous", "priority_cascade"}

    def __post_init__(self) -> None:
        if self.blend_mode not in self._ALLOWED_MODES:
            raise ValueError(
                f"blend_mode must be one of {self._ALLOWED_MODES}, got {self.blend_mode!r}"
            )

    def generate(self, context: AlphaContext) -> AlphaSignal:
        if not self.strategies:
            return AlphaSignal()

        if self.blend_mode == "priority_cascade":
            return self._blend_cascade(context)
        if self.blend_mode == "unanimous":
            return self._blend_unanimous(context)
        if self.blend_mode == "vote":
            return self._blend_vote(context)
        return self._blend_weighted(context)

    # -------------------------------------------------------------------
    # Blending implementations
    # -------------------------------------------------------------------

    def _blend_weighted(self, context: AlphaContext) -> AlphaSignal:
        total_weight = 0.0
        weighted_dir = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0
        reasons: list[str] = []
        horizons: list[int] = []

        for strategy, weight in self.strategies:
            signal = strategy.generate(context)
            if not signal.is_actionable and signal.confidence < self.min_confidence:
                continue
            weighted_dir += signal.direction * weight
            weighted_strength += signal.strength * weight
            weighted_confidence += signal.confidence * weight
            total_weight += weight
            if signal.reason:
                reasons.append(f"{strategy.name}:{signal.reason}")
            if signal.horizon_ticks:
                horizons.append(signal.horizon_ticks)

        if total_weight <= 0:
            return AlphaSignal()

        return AlphaSignal(
            direction=max(-1.0, min(1.0, weighted_dir / total_weight)),
            strength=min(1.0, weighted_strength / total_weight),
            confidence=min(1.0, weighted_confidence / total_weight),
            horizon_ticks=int(sum(horizons) / max(len(horizons), 1)) if horizons else 5,
            reason=" | ".join(reasons) if reasons else "weighted blend",
        )

    def _blend_vote(self, context: AlphaContext) -> AlphaSignal:
        bullish = 0
        bearish = 0
        strengths: list[float] = []
        confidences: list[float] = []
        reasons: list[str] = []

        for strategy, _weight in self.strategies:
            signal = strategy.generate(context)
            if not signal.is_actionable and signal.confidence < self.min_confidence:
                continue
            if signal.direction > 0:
                bullish += 1
            elif signal.direction < 0:
                bearish += 1
            strengths.append(signal.strength)
            confidences.append(signal.confidence)
            if signal.reason:
                reasons.append(f"{strategy.name}:{signal.reason}")

        total_votes = bullish + bearish
        if total_votes == 0:
            return AlphaSignal()

        direction = 1.0 if bullish > bearish else -1.0
        return AlphaSignal(
            direction=direction,
            strength=sum(strengths) / len(strengths) if strengths else 0.0,
            confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            reason=f"vote {bullish}B/{bearish}S: " + (" | ".join(reasons) if reasons else ""),
        )

    def _blend_unanimous(self, context: AlphaContext) -> AlphaSignal:
        signals: list[AlphaSignal] = []
        reasons: list[str] = []

        for strategy, _weight in self.strategies:
            signal = strategy.generate(context)
            if not signal.is_actionable and signal.confidence < self.min_confidence:
                return AlphaSignal(reason=f"no consensus: {strategy.name} abstained")
            signals.append(signal)
            if signal.reason:
                reasons.append(f"{strategy.name}:{signal.reason}")

        if not signals:
            return AlphaSignal()

        first_dir = 1 if signals[0].direction > 0 else -1
        for sig in signals[1:]:
            sig_dir = 1 if sig.direction > 0 else -1
            if sig_dir != first_dir:
                return AlphaSignal(
                    reason=f"unanimous rejected: direction split ({first_dir} vs {sig_dir})"
                )

        return AlphaSignal(
            direction=signals[0].direction,
            strength=sum(s.strength for s in signals) / len(signals),
            confidence=sum(s.confidence for s in signals) / len(signals),
            reason="unanimous: " + (" | ".join(reasons) if reasons else ""),
        )

    def _blend_cascade(self, context: AlphaContext) -> AlphaSignal:
        for strategy, _weight in self.strategies:
            signal = strategy.generate(context)
            if signal.is_actionable and signal.confidence >= self.min_confidence:
                signal.metadata["cascade_source"] = strategy.name
                return signal

        return AlphaSignal(reason="cascade: no strategy produced an actionable signal")

    def strategy_names(self) -> list[str]:
        return [s.name for s, _ in self.strategies]

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_strategies": self.strategy_names(),
            "blend_mode": self.blend_mode,
            "strategy_count": len(self.strategies),
        }
