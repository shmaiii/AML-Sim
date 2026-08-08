"""Explicit, bounded profile-to-fast-loop parameter formulas."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping


def _value(profile: Any, field: str, default: float = 0.5) -> float:
    if is_dataclass(profile):
        raw = getattr(profile, field, default)
    elif isinstance(profile, Mapping):
        raw = profile.get(field, default)
    else:
        raw = default
    return max(0.0, min(1.0, float(raw)))


def _bounded_multiplier(value: float, lower: float = 0.5, upper: float = 1.5) -> float:
    """Keep combined profile effects useful without allowing extreme scaling."""

    return max(lower, min(upper, value))


def market_maker_profile_effects(profile: Any) -> dict[str, float]:
    aggressiveness = _value(profile, "quote_aggressiveness", 0.5)
    adverse = _value(profile, "adverse_selection_sensitivity", 0.6)
    resilience = _value(profile, "liquidity_resilience", 0.5)
    discipline = _value(profile, "inventory_discipline", 0.7)
    return {
        "spread_multiplier": _bounded_multiplier(
            (1.0 - 0.5 * (aggressiveness - 0.5))
            * (1.0 + 0.5 * (adverse - 0.6)),
            0.75,
            1.5,
        ),
        "inventory_skew_multiplier": _bounded_multiplier(1.0 + discipline - 0.7),
        "quote_size_multiplier": _bounded_multiplier(1.0 + 0.5 * (resilience - 0.5)),
        "withdrawal_sensitivity_multiplier": _bounded_multiplier(
            1.0 - (resilience - 0.5)
        ),
    }


def retail_profile_effects(profile: Any) -> dict[str, float]:
    social = _value(profile, "social_sensitivity", 0.5)
    herding = _value(profile, "herding_tendency", 0.4)
    return {
        "participation_multiplier": _bounded_multiplier(
            1.0 + 0.5 * (_value(profile, "news_reactivity", 0.5) - 0.5)
        ),
        "herding_multiplier": _bounded_multiplier(
            1.0 + 0.5 * ((social - 0.5) + (herding - 0.4))
        ),
        "panic_multiplier": _bounded_multiplier(
            1.0 + (_value(profile, "panic_sensitivity", 0.4) - 0.4)
        ),
        "order_size_multiplier": _bounded_multiplier(
            1.0 - 0.5 * (_value(profile, "loss_aversion", 0.5) - 0.5)
        ),
    }


def institutional_profile_effects(profile: Any, urgency: float) -> dict[str, float]:
    patience = _value(profile, "execution_patience", 0.6)
    impact_aversion = _value(profile, "market_impact_aversion", 0.6)
    urgency_multiplier = 0.5 + max(0.0, min(1.0, urgency))
    execution_profile = (
        1.0
        - 0.5 * (patience - 0.6)
        - 0.5 * (impact_aversion - 0.6)
    )
    return {
        "signal_multiplier": _bounded_multiplier(
            1.0 + (_value(profile, "information_sensitivity", 0.5) - 0.5)
        ),
        "child_size_multiplier": _bounded_multiplier(
            urgency_multiplier * execution_profile
        ),
    }


def informed_profile_effects(profile: Any) -> dict[str, float]:
    return {
        "information_edge_multiplier": _bounded_multiplier(
            1.0 + (_value(profile, "information_quality", 0.7) - 0.7)
        ),
        "participation_multiplier": _bounded_multiplier(
            1.0
            + 0.5 * (_value(profile, "conviction", 0.65) - 0.65)
            - 0.5 * (_value(profile, "patience", 0.45) - 0.45)
        ),
        "order_size_multiplier": _bounded_multiplier(
            1.0 + (_value(profile, "conviction", 0.65) - 0.65)
        ),
        "shock_tolerance_multiplier": _bounded_multiplier(
            1.0
            + (_value(profile, "adverse_selection_tolerance", 0.6) - 0.6)
        ),
    }


def liquidity_taker_profile_effects(profile: Any) -> dict[str, float]:
    return {
        "participation_multiplier": _bounded_multiplier(
            1.0
            + 0.5 * (_value(profile, "immediacy_preference", 0.8) - 0.8)
            + 0.5 * (_value(profile, "flow_persistence", 0.5) - 0.5)
        ),
        "order_size_multiplier": _bounded_multiplier(
            1.0
            + (_value(profile, "market_impact_tolerance", 0.6) - 0.6)
        ),
        "aggression_multiplier": _bounded_multiplier(
            1.0 + (_value(profile, "immediacy_preference", 0.8) - 0.8)
        ),
    }
