"""Time-aware central market state for persistent and temporary shocks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from aml_sim.shocks import safe_float, safe_int


_PERSISTENT_SHOCK_TYPES = {
    "policy_rate_hike",
    "policy_rate_cut",
    "yield_curve_shift",
    "credit_rating_change",
    "regulatory_change",
}
_STATE_EFFECT_KEYS = {
    "policy_rate_bps": "rate_shift_bps",
    "yield_curve_shift_bps": "yield_shift_bps",
    "funding_spread_bps": "funding_spread_bps",
    "credit_spread_bps": "credit_spread_bps",
    "risk_aversion": "risk_aversion_shift",
}


@dataclass(frozen=True)
class StateImpact:
    """One event's contribution to the central state."""

    event_id: str
    start_tick: int
    active_until_tick: int
    persistence: str
    reversion_ticks: int
    additive: dict[str, float]
    absolute: dict[str, Any]
    liquidity_multiplier: float

    def weight(self, tick_id: int) -> float:
        if tick_id < self.start_tick:
            return 0.0
        if self.persistence == "permanent" or tick_id <= self.active_until_tick:
            return 1.0
        if self.reversion_ticks <= 0:
            return 0.0
        elapsed = tick_id - self.active_until_tick
        return max(0.0, 1.0 - (elapsed / self.reversion_ticks))


class MarketStateEngine:
    """Rebuild central state from a baseline and active/persistent impacts."""

    def __init__(self, initial_market_state: Mapping[str, Any] | None = None) -> None:
        self.baseline = deepcopy(dict(initial_market_state or {}))
        self.impacts: list[StateImpact] = []
        self.current_state = deepcopy(self.baseline)

    def register(
        self,
        event_id: str,
        event: Mapping[str, Any],
        shock_payload: Mapping[str, Any],
        *,
        current_tick_id: int,
        default_duration_ticks: int,
    ) -> None:
        """Register a state impact when an event becomes economically active."""
        start_tick = safe_int(shock_payload.get("effective_tick_id"), current_tick_id)
        if start_tick is None:
            start_tick = current_tick_id
        configured_duration = safe_int(
            shock_payload.get("duration_ticks"),
            default_duration_ticks,
        )
        duration = max(
            0,
            configured_duration
            if configured_duration is not None
            else default_duration_ticks,
        )
        persistence = _persistence_for(event, shock_payload)
        reversion_ticks = _reversion_ticks(event, duration, persistence)
        additive, absolute, liquidity_multiplier = _state_contribution(event, shock_payload)

        if not additive and not absolute and liquidity_multiplier == 1.0:
            return

        self.impacts.append(
            StateImpact(
                event_id=event_id,
                start_tick=start_tick,
                active_until_tick=start_tick + duration,
                persistence=persistence,
                reversion_ticks=reversion_ticks,
                additive=additive,
                absolute=absolute,
                liquidity_multiplier=liquidity_multiplier,
            )
        )

    def snapshot(self, tick_id: int | None) -> dict[str, Any]:
        """Return state at a tick, decaying temporary impacts after expiry."""
        if tick_id is None:
            return deepcopy(self.current_state)

        state = deepcopy(self.baseline)
        active_impacts: list[StateImpact] = []
        for impact in self.impacts:
            weight = impact.weight(tick_id)
            if weight <= 0.0:
                continue
            active_impacts.append(impact)
            _apply_absolute(state, impact.absolute, weight)
            _apply_additive(state, impact.additive, weight)
            if impact.liquidity_multiplier != 1.0:
                current_liquidity = safe_float(state.get("liquidity_index", 1.0), 1.0)
                state["liquidity_index"] = current_liquidity * (
                    impact.liquidity_multiplier ** weight
                )

        self.impacts = active_impacts
        self.current_state = _normalize_state(state)
        return deepcopy(self.current_state)


def _persistence_for(event: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    configured = event.get("state_persistence", event.get("market_state_persistence"))
    if configured is not None:
        value = str(configured).strip().lower()
        if value in {"permanent", "persistent"}:
            return "permanent"
        if value in {"temporary", "transient", "reverting"}:
            return "transient"
        raise ValueError(
            "state_persistence must be permanent or transient when configured"
        )

    shock_type = str(
        payload.get("shock_type", event.get("shock_type", event.get("type", "")))
    ).strip().lower()
    return "permanent" if shock_type in _PERSISTENT_SHOCK_TYPES else "transient"


def _reversion_ticks(
    event: Mapping[str, Any],
    duration_ticks: int,
    persistence: str,
) -> int:
    if persistence == "permanent":
        return 0
    configured = safe_int(
        event.get("state_reversion_ticks", event.get("market_state_reversion_ticks"))
    )
    if configured is not None:
        return max(0, configured)
    return max(1, duration_ticks)


def _state_contribution(
    event: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any], float]:
    absolute = event.get("market_state", event.get("state", {}))
    absolute = dict(absolute) if isinstance(absolute, Mapping) else {}

    configured_delta = event.get("market_state_delta", event.get("state_delta", {}))
    additive = {
        str(key): safe_float(value, 0.0)
        for key, value in configured_delta.items()
    } if isinstance(configured_delta, Mapping) else {}

    for state_key, effect_key in _STATE_EFFECT_KEYS.items():
        if state_key in absolute or state_key in additive:
            continue
        delta = safe_float(payload.get(effect_key), 0.0)
        if delta != 0.0:
            additive[state_key] = delta

    liquidity_multiplier = safe_float(payload.get("liquidity_multiplier"), 1.0)
    if "liquidity_index" in absolute or "liquidity_index" in additive:
        liquidity_multiplier = 1.0

    return additive, absolute, max(0.01, liquidity_multiplier)


def _apply_absolute(state: dict[str, Any], absolute: Mapping[str, Any], weight: float) -> None:
    for key, target in absolute.items():
        if isinstance(target, (int, float)):
            current = safe_float(state.get(key), 0.0)
            state[key] = current + ((float(target) - current) * weight)
        elif weight >= 1.0:
            state[key] = deepcopy(target)


def _apply_additive(state: dict[str, Any], additive: Mapping[str, float], weight: float) -> None:
    for key, delta in additive.items():
        state[key] = safe_float(state.get(key), 0.0) + (delta * weight)


def _normalize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, (int, float)):
            normalized[key] = round(float(value), 4)
        else:
            normalized[key] = value
    if "liquidity_index" in normalized:
        normalized["liquidity_index"] = max(0.01, normalized["liquidity_index"])
    if "risk_aversion" in normalized:
        normalized["risk_aversion"] = max(0.0, normalized["risk_aversion"])
    return normalized
