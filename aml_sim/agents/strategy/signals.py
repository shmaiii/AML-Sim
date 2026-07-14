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
