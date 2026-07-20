"""Shared signal helpers for AML trading strategies."""

from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Any, Callable, Mapping


EffectResolver = Callable[[Mapping[str, Any], str], Mapping[str, float]]

_EFFECT_DEFAULTS: dict[str, float] = {
    "fundamental_price_shift": 0.0,
    "order_arrival_multiplier": 1.0,
    "risk_limit_multiplier": 1.0,
    "liquidity_multiplier": 1.0,
    "volatility_multiplier": 1.0,
    "spread_multiplier": 1.0,
    "price_impact_multiplier": 1.0,
    "rate_shift_bps": 0.0,
    "yield_shift_bps": 0.0,
    "funding_spread_bps": 0.0,
    "credit_spread_bps": 0.0,
    "correlation_shift": 0.0,
    "sentiment_shift": 0.0,
    "risk_aversion_shift": 0.0,
}
_EFFECT_KEYS = tuple(_EFFECT_DEFAULTS.keys())


def clamp(value: float, lower: float, upper: float) -> float:
    """Return value clipped to the inclusive lower/upper range."""
    return max(lower, min(upper, value))


def price_series(
    price_history: Mapping[str, list[Mapping[str, Any]]],
    instrument: str,
    fallback_price: float | None = None,
) -> list[float]:
    """Extract clean positive prices for one instrument."""
    values: list[float] = []
    for row in price_history.get(instrument, []):
        try:
            price = float(row.get("price", 0))
        except (TypeError, ValueError):
            continue
        if price > 0:
            values.append(price)

    if not values and fallback_price is not None and fallback_price > 0:
        values.append(float(fallback_price))
    return values


def momentum_signal(prices: list[float], lookback_ticks: int) -> float:
    """Simple lookback return: positive means upward momentum."""
    if len(prices) < 2:
        return 0.0
    lookback = max(1, min(lookback_ticks, len(prices) - 1))
    start = prices[-lookback - 1]
    end = prices[-1]
    if start <= 0:
        return 0.0
    return (end / start) - 1.0


def mean_reversion_signal(prices: list[float], lookback_ticks: int) -> float:
    """Deviation from recent mean: positive means price is below recent mean."""
    if len(prices) < 2:
        return 0.0
    lookback = max(2, min(lookback_ticks, len(prices)))
    window = prices[-lookback:]
    avg_price = mean(window)
    last_price = window[-1]
    if avg_price <= 0:
        return 0.0
    return (avg_price - last_price) / avg_price


def realized_volatility(prices: list[float], lookback_ticks: int) -> float:
    """Estimate realized volatility from recent simple returns."""
    if len(prices) < 3:
        return 0.0
    lookback = max(2, min(lookback_ticks, len(prices) - 1))
    window = prices[-lookback - 1 :]
    returns = []
    for left, right in zip(window, window[1:]):
        if left > 0:
            returns.append((right / left) - 1.0)
    if len(returns) < 2:
        return abs(returns[0]) if returns else 0.0
    avg_return = mean(returns)
    variance = sum((item - avg_return) ** 2 for item in returns) / (len(returns) - 1)
    return sqrt(max(variance, 0.0))


def event_applies_to(event: Mapping[str, Any], instrument: str) -> bool:
    """Check whether an AML event should affect an instrument."""
    affected = event.get("affected_instruments") or event.get("instruments") or []
    if not affected:
        return True
    return instrument in affected


def event_pressure(
    events: list[Mapping[str, Any]],
    instrument: str,
    effect_resolver: EffectResolver | None = None,
    *,
    market_state: Mapping[str, Any] | None = None,
    market_state_baseline: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """
    Aggregate active event pressure for an instrument.

    direction is expected to be -1, 0, or +1. The returned directional_bias is
    severity-weighted and clipped so strategy code can safely add it to signals.
    """
    max_severity = 0.0
    directional_bias = 0.0
    fundamental_price_shift = 0.0
    order_arrival_multiplier = 1.0
    risk_limit_multiplier = 1.0
    liquidity_multiplier = 1.0
    volatility_multiplier = 1.0
    spread_multiplier = 1.0
    price_impact_multiplier = 1.0
    rate_shift_bps = 0.0
    yield_shift_bps = 0.0
    funding_spread_bps = 0.0
    credit_spread_bps = 0.0
    sentiment_shift = 0.0
    risk_aversion_shift = 0.0

    for event in events:
        if not event_applies_to(event, instrument):
            continue
        try:
            severity = clamp(float(event.get("severity", 0.0)), 0.0, 1.0)
        except (TypeError, ValueError):
            severity = 0.0
        try:
            direction = clamp(float(event.get("direction", 0.0)), -1.0, 1.0)
        except (TypeError, ValueError):
            direction = 0.0

        effect = (
            effect_resolver(event, instrument)
            if effect_resolver is not None
            else _resolve_event_effect(event, instrument)
        )
        max_severity = max(max_severity, severity)
        directional_bias += severity * direction
        fundamental_price_shift += _effect_float(effect, "fundamental_price_shift")
        order_arrival_multiplier *= _effect_float(effect, "order_arrival_multiplier")
        risk_limit_multiplier = min(
            risk_limit_multiplier,
            _effect_float(effect, "risk_limit_multiplier"),
        )
        liquidity_multiplier = min(
            liquidity_multiplier,
            _effect_float(effect, "liquidity_multiplier"),
        )
        volatility_multiplier *= _effect_float(effect, "volatility_multiplier")
        spread_multiplier *= _effect_float(effect, "spread_multiplier")
        price_impact_multiplier *= _effect_float(effect, "price_impact_multiplier")
        rate_shift_bps += _effect_float(effect, "rate_shift_bps")
        yield_shift_bps += _effect_float(effect, "yield_shift_bps")
        funding_spread_bps += _effect_float(effect, "funding_spread_bps")
        credit_spread_bps += _effect_float(effect, "credit_spread_bps")
        sentiment_shift += _effect_float(effect, "sentiment_shift")
        risk_aversion_shift += _effect_float(effect, "risk_aversion_shift")

    state_effect = market_state_pressure(
        _state_without_active_event_contributions(
            market_state,
            events,
            instrument,
        ),
        market_state_baseline,
    )
    max_severity = max(max_severity, state_effect["severity"])
    directional_bias += state_effect["directional_bias"]
    fundamental_price_shift += state_effect["fundamental_price_shift"]
    order_arrival_multiplier *= state_effect["order_arrival_multiplier"]
    risk_limit_multiplier = min(
        risk_limit_multiplier,
        state_effect["risk_limit_multiplier"],
    )
    liquidity_multiplier = min(
        liquidity_multiplier,
        state_effect["liquidity_multiplier"],
    )
    volatility_multiplier *= state_effect["volatility_multiplier"]
    spread_multiplier *= state_effect["spread_multiplier"]
    price_impact_multiplier *= state_effect["price_impact_multiplier"]
    rate_shift_bps += state_effect["rate_shift_bps"]
    yield_shift_bps += state_effect["yield_shift_bps"]
    funding_spread_bps += state_effect["funding_spread_bps"]
    credit_spread_bps += state_effect["credit_spread_bps"]
    sentiment_shift += state_effect["sentiment_shift"]
    risk_aversion_shift += state_effect["risk_aversion_shift"]

    return {
        "severity": clamp(max_severity, 0.0, 1.0),
        "directional_bias": clamp(directional_bias, -1.0, 1.0),
        "fundamental_price_shift": fundamental_price_shift,
        "order_arrival_multiplier": clamp(order_arrival_multiplier, 0.05, 5.0),
        "risk_limit_multiplier": clamp(risk_limit_multiplier, 0.05, 2.0),
        "liquidity_multiplier": clamp(liquidity_multiplier, 0.05, 2.0),
        "volatility_multiplier": clamp(volatility_multiplier, 0.05, 5.0),
        "spread_multiplier": clamp(spread_multiplier, 0.05, 5.0),
        "price_impact_multiplier": clamp(price_impact_multiplier, 0.05, 5.0),
        "rate_shift_bps": rate_shift_bps,
        "yield_shift_bps": yield_shift_bps,
        "funding_spread_bps": funding_spread_bps,
        "credit_spread_bps": credit_spread_bps,
        "sentiment_shift": clamp(sentiment_shift, -1.0, 1.0),
        "risk_aversion_shift": clamp(risk_aversion_shift, -1.0, 1.0),
    }


def market_state_pressure(
    market_state: Mapping[str, Any] | None,
    market_state_baseline: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Translate persistent central conditions into bounded trading pressure.

    Event effects model the immediate incident. This helper adds the slower
    background effects of a changed policy rate, funding conditions, credit
    conditions, liquidity, and risk appetite after the event itself expires.
    """
    if not market_state:
        return dict(_EFFECT_DEFAULTS)

    baseline = market_state_baseline or {}
    liquidity = max(0.01, _safe_float(market_state.get("liquidity_index"), 1.0))
    baseline_liquidity = max(
        0.01,
        _safe_float(baseline.get("liquidity_index"), 1.0),
    )
    liquidity_ratio = clamp(liquidity / baseline_liquidity, 0.05, 2.0)

    rate_delta = _state_delta(market_state, baseline, "policy_rate_bps")
    yield_delta = _state_delta(market_state, baseline, "yield_curve_shift_bps")
    funding_delta = _state_delta(market_state, baseline, "funding_spread_bps")
    credit_delta = _state_delta(market_state, baseline, "credit_spread_bps")
    risk_delta = _state_delta(market_state, baseline, "risk_aversion")

    liquidity_stress = max(0.0, 1.0 - liquidity_ratio)
    rate_stress = max(0.0, rate_delta) / 100.0
    yield_stress = max(0.0, yield_delta) / 100.0
    funding_stress = max(0.0, funding_delta) / 100.0
    credit_stress = max(0.0, credit_delta) / 100.0
    risk_stress = max(0.0, risk_delta)
    stress = clamp(
        0.55 * liquidity_stress
        + 0.20 * risk_stress
        + 0.15 * funding_stress
        + 0.10 * credit_stress
        + 0.08 * rate_stress
        + 0.05 * yield_stress,
        0.0,
        1.0,
    )

    return {
        "severity": stress,
        "directional_bias": clamp(
            -0.12 * (rate_delta / 100.0)
            - 0.08 * (yield_delta / 100.0)
            - 0.10 * (funding_delta / 100.0)
            - 0.08 * (credit_delta / 100.0),
            -0.5,
            0.5,
        ),
        "fundamental_price_shift": 0.0,
        "order_arrival_multiplier": clamp(1.0 + (0.25 * stress), 0.5, 1.5),
        "risk_limit_multiplier": clamp(1.0 - (0.55 * stress), 0.25, 1.0),
        "liquidity_multiplier": liquidity_ratio,
        "volatility_multiplier": clamp(1.0 + (0.80 * stress), 1.0, 2.0),
        "spread_multiplier": clamp(1.0 + (1.00 * stress), 1.0, 2.25),
        "price_impact_multiplier": clamp(1.0 + (0.70 * stress), 1.0, 2.0),
        "rate_shift_bps": rate_delta,
        "yield_shift_bps": yield_delta,
        "funding_spread_bps": funding_delta,
        "credit_spread_bps": credit_delta,
        "sentiment_shift": clamp(-0.25 * risk_stress, -0.25, 0.0),
        "risk_aversion_shift": clamp(risk_delta, -1.0, 1.0),
    }


def _state_delta(
    market_state: Mapping[str, Any],
    baseline: Mapping[str, Any],
    key: str,
) -> float:
    return _safe_float(market_state.get(key), 0.0) - _safe_float(
        baseline.get(key),
        0.0,
    )


def _state_without_active_event_contributions(
    market_state: Mapping[str, Any] | None,
    events: list[Mapping[str, Any]],
    instrument: str,
) -> Mapping[str, Any] | None:
    """Remove active effects already represented by the event pressure itself."""
    if not market_state:
        return market_state

    adjusted = dict(market_state)
    state_keys = (
        ("policy_rate_bps", "rate_shift_bps"),
        ("yield_curve_shift_bps", "yield_shift_bps"),
        ("funding_spread_bps", "funding_spread_bps"),
        ("credit_spread_bps", "credit_spread_bps"),
        ("risk_aversion", "risk_aversion_shift"),
    )
    for event in events:
        if not event_applies_to(event, instrument):
            continue
        contribution = event.get("market_state_contribution")
        if not isinstance(contribution, Mapping):
            continue

        for state_key, effect_key in state_keys:
            if _effect_float(event, effect_key) == 0.0:
                continue
            delta = _safe_float(contribution.get(state_key), 0.0)
            if delta != 0.0:
                adjusted[state_key] = _safe_float(adjusted.get(state_key), 0.0) - delta

        liquidity_multiplier = _effect_float(event, "liquidity_multiplier")
        state_multiplier = _safe_float(
            contribution.get("liquidity_multiplier"),
            1.0,
        )
        if liquidity_multiplier != 1.0 and state_multiplier > 0:
            adjusted["liquidity_index"] = (
                _safe_float(adjusted.get("liquidity_index"), 1.0)
                / state_multiplier
            )

    return adjusted


def _resolve_event_effect(event: Mapping[str, Any], instrument: str) -> dict[str, float]:
    """Resolve event effects without requiring the shock module at import time."""
    effects = dict(_EFFECT_DEFAULTS)
    _merge_effects(effects, event)

    nested = event.get("effects")
    if isinstance(nested, Mapping):
        _merge_effects(effects, nested)

    instrument_effects = event.get("instrument_effects")
    if isinstance(instrument_effects, Mapping):
        specific = instrument_effects.get(instrument)
        if isinstance(specific, Mapping):
            _merge_effects(effects, specific)

    return _clamp_effects(effects)


def _merge_effects(target: dict[str, float], source: Mapping[str, Any]) -> None:
    for key in _EFFECT_KEYS:
        if key in source:
            target[key] = _safe_float(source.get(key), target[key])


def _clamp_effects(effects: Mapping[str, float]) -> dict[str, float]:
    clean = dict(effects)
    for key in (
        "order_arrival_multiplier",
        "volatility_multiplier",
        "spread_multiplier",
        "price_impact_multiplier",
    ):
        clean[key] = clamp(
            _safe_float(clean.get(key), _EFFECT_DEFAULTS[key]),
            0.05,
            5.0,
        )
    for key in ("risk_limit_multiplier", "liquidity_multiplier"):
        clean[key] = clamp(
            _safe_float(clean.get(key), _EFFECT_DEFAULTS[key]),
            0.05,
            2.0,
        )
    clean["correlation_shift"] = clamp(
        _safe_float(clean.get("correlation_shift"), 0.0),
        -1.0,
        1.0,
    )
    clean["sentiment_shift"] = clamp(
        _safe_float(clean.get("sentiment_shift"), 0.0),
        -1.0,
        1.0,
    )
    clean["risk_aversion_shift"] = clamp(
        _safe_float(clean.get("risk_aversion_shift"), 0.0),
        -1.0,
        1.0,
    )
    return clean


def _effect_float(effect: Mapping[str, Any], key: str) -> float:
    return _safe_float(effect.get(key), _EFFECT_DEFAULTS[key])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def target_from_signal(
    signal: float,
    *,
    current_target: int,
    entry_threshold: float,
    exit_threshold: float,
    max_position: int,
    min_position: int,
) -> int:
    """Convert an alpha signal into a target inventory."""
    if abs(signal) <= exit_threshold:
        return 0
    if signal >= entry_threshold:
        return max_position
    if signal <= -entry_threshold:
        return min_position
    return current_target
