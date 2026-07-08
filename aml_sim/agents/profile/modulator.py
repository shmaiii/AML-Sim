"""Profile-to-strategy-parameter modulation for AML agents."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from aml_sim.agents.models.profile import BehavioralTraits
from aml_sim.agents.profile.behavioral_dynamics import BehavioralState


@dataclass
class ProfileModulator:
    """Translates an agent's behavioral traits + state into strategy parameter
    adjustments.

    The modulator takes a strategy state dataclass (or mapping), applies
    trait-driven and state-driven adjustments, and returns a new strategy
    state. The original is never mutated — the caller decides whether to
    adopt the modulated copy.
    """

    # How strongly traits influence parameters (0 = no modulation, 1 = full)
    trait_influence: float = 0.5
    state_influence: float = 0.3

    def modulate(
        self,
        strategy_state: Any,
        traits: BehavioralTraits,
        state: BehavioralState | None = None,
    ) -> Any:
        """Return a modulated copy of the strategy state."""
        modulated = copy.deepcopy(strategy_state)
        self._apply_traits(modulated, traits)
        if state is not None:
            self._apply_state(modulated, traits, state)
        return modulated

    # ------------------------------------------------------------------
    # Trait → parameter mappings
    # ------------------------------------------------------------------

    def _apply_traits(self, strategy: Any, traits: BehavioralTraits) -> None:
        influence = self.trait_influence

        # risk_aversion → position sizing (inverse), confidence threshold
        self._scale_if_present(strategy, "max_position", 1.0 - (traits.risk_aversion - 0.5) * influence)
        self._scale_if_present(strategy, "max_order_size", 1.0 - (traits.risk_aversion - 0.5) * influence * 0.6)
        self._scale_if_present(strategy, "signal_threshold", 1.0 + (traits.risk_aversion - 0.5) * influence * 0.5)
        self._scale_if_present(strategy, "entry_threshold", 1.0 + (traits.risk_aversion - 0.5) * influence * 0.5)

        # patience → order type preference, limit offset
        self._scale_if_present(strategy, "limit_offset", 1.0 + (traits.patience - 0.5) * influence * 0.8)

        # conviction → signal threshold (inverse), confidence
        self._scale_if_present(strategy, "signal_threshold", 1.0 - (traits.conviction - 0.5) * influence * 0.5)
        if hasattr(strategy, "confidence"):
            strategy.confidence = max(0.0, min(1.0, strategy.confidence + (traits.conviction - 0.5) * influence * 0.3))

        # adaptability → shock reactivity
        self._scale_if_present(strategy, "shock_reactivity", 1.0 + (traits.adaptability - 0.5) * influence)
        self._scale_if_present(strategy, "shock_sensitivity", 1.0 + (traits.adaptability - 0.5) * influence)
        self._scale_if_present(strategy, "sentiment_sensitivity", 1.0 + (traits.adaptability - 0.5) * influence * 0.7)

        # discipline → threshold widening
        self._scale_if_present(strategy, "entry_threshold", 1.0 + (traits.discipline - 0.5) * influence * 0.4)

        # aggression → flow_intensity, trade_probability
        self._scale_if_present(strategy, "trade_probability", 1.0 + (traits.aggression - 0.5) * influence * 0.8)
        self._scale_if_present(strategy, "flow_intensity", 1.0 + (traits.aggression - 0.5) * influence * 0.8)

        # social_influence → herding, sentiment sensitivity
        self._scale_if_present(strategy, "herding_tendency", 1.0 + (traits.social_influence - 0.3) * influence * 1.2)
        self._scale_if_present(strategy, "sentiment_sensitivity", 1.0 + (traits.social_influence - 0.3) * influence * 0.6)

        # loss_aversion → exit_threshold on losing positions
        self._scale_if_present(strategy, "exit_threshold", 1.0 - (traits.loss_aversion - 0.5) * influence * 0.6)

        # recency_bias → lookback_ticks
        self._scale_if_present(strategy, "lookback_ticks", 1.0 - (traits.recency_bias - 0.5) * influence * 0.7)

    # ------------------------------------------------------------------
    # State → parameter adjustments (on top of traits)
    # ------------------------------------------------------------------

    def _apply_state(
        self,
        strategy: Any,
        traits: BehavioralTraits,
        state: BehavioralState,
    ) -> None:
        if self.state_influence <= 0:
            return
        si = self.state_influence

        # Stress → temporary risk_aversion boost, reduce position limits
        if state.stress_level > 0.3:
            stress_factor = 1.0 - (state.stress_level * si * 0.4)
            self._scale_if_present(strategy, "max_position", stress_factor)
            self._scale_if_present(strategy, "max_order_size", stress_factor)

        # Winning streak → confidence boost (can be positive or negative)
        if hasattr(strategy, "confidence"):
            strategy.confidence = max(0.0, min(1.0, strategy.confidence + state.confidence_drift * si))

        # Market regime → order type suggestion
        if state.market_regime_belief == "volatile" and hasattr(strategy, "order_type"):
            # Prefer limit orders in volatile markets
            pass  # Signal-level, not strategy-state-level

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_if_present(obj: Any, attr: str, factor: float) -> None:
        """Multiply a numeric attribute by factor, clamping to sensible range."""
        if not hasattr(obj, attr):
            return
        current = getattr(obj, attr)
        if not isinstance(current, (int, float)):
            return
        factor = max(0.05, min(20.0, factor))
        new_value = current * factor
        if isinstance(current, int):
            new_value = max(1, int(round(new_value)))
        setattr(obj, attr, new_value)

    @staticmethod
    def _add_if_present(obj: Any, attr: str, delta: float) -> None:
        if not hasattr(obj, attr):
            return
        current = getattr(obj, attr)
        if not isinstance(current, (int, float)):
            return
        setattr(obj, attr, current + delta)
