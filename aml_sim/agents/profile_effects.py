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


def market_maker_profile_effects(profile: Any) -> dict[str, float]:
    aggressiveness = _value(profile, "quote_aggressiveness")
    adverse = _value(profile, "adverse_selection_sensitivity")
    resilience = _value(profile, "liquidity_resilience")
    return {
        "spread_multiplier": (1.25 - 0.5 * aggressiveness) * (0.75 + 0.5 * adverse),
        "inventory_skew_multiplier": 0.5 + _value(profile, "inventory_discipline"),
        "quote_size_multiplier": 0.75 + 0.5 * resilience,
        "withdrawal_sensitivity_multiplier": 1.5 - resilience,
    }


def retail_profile_effects(profile: Any) -> dict[str, float]:
    social = _value(profile, "social_sensitivity")
    herding = _value(profile, "herding_tendency")
    return {
        "participation_multiplier": 0.75 + 0.5 * _value(profile, "news_reactivity"),
        "herding_multiplier": 0.5 + (social + herding) / 2.0,
        "panic_multiplier": 0.5 + _value(profile, "panic_sensitivity"),
        "order_size_multiplier": 1.25 - 0.5 * _value(profile, "loss_aversion"),
    }


def institutional_profile_effects(profile: Any, urgency: float) -> dict[str, float]:
    patience = _value(profile, "execution_patience")
    impact_aversion = _value(profile, "market_impact_aversion")
    return {
        "signal_multiplier": 0.5 + _value(profile, "information_sensitivity"),
        "child_size_multiplier": (0.5 + max(0.0, min(1.0, urgency)))
        * (1.5 - 0.5 * patience - 0.5 * impact_aversion),
    }


def informed_profile_effects(profile: Any) -> dict[str, float]:
    return {
        "information_edge_multiplier": 0.5 + _value(profile, "information_quality"),
        "participation_multiplier": (0.5 + _value(profile, "conviction"))
        * (1.25 - 0.5 * _value(profile, "patience")),
        "order_size_multiplier": 0.5 + _value(profile, "conviction"),
        "shock_tolerance_multiplier": 0.5 + _value(profile, "adverse_selection_tolerance"),
    }


def liquidity_taker_profile_effects(profile: Any) -> dict[str, float]:
    return {
        "participation_multiplier": (0.5 + _value(profile, "immediacy_preference"))
        * (0.75 + 0.5 * _value(profile, "flow_persistence")),
        "order_size_multiplier": 0.5 + _value(profile, "market_impact_tolerance"),
        "aggression_multiplier": 0.5 + _value(profile, "immediacy_preference"),
    }
