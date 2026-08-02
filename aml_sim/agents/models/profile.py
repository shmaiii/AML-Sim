"""Agent profile models for AML behavioral context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping, TypeVar


_RISK_TOLERANCES = {"low", "medium", "high"}
_BOUNDED_PROFILE_FIELDS = {
    "inventory_discipline", "quote_aggressiveness",
    "adverse_selection_sensitivity", "liquidity_resilience",
    "social_sensitivity", "panic_sensitivity", "herding_tendency",
    "news_reactivity", "loss_aversion", "execution_patience",
    "information_sensitivity", "market_impact_aversion", "information_quality",
    "patience", "adverse_selection_tolerance", "conviction",
    "immediacy_preference", "market_impact_tolerance", "flow_persistence",
}


@dataclass
class AgentProfile:
    """Stable identity and behavioral context for an AML agent."""

    role: str
    name: str | None = None
    risk_tolerance: str = "medium"
    decision_style: str = "balanced"
    personality: dict[str, Any] = field(default_factory=dict)
    behavioral_traits: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketMakerProfile(AgentProfile):
    """Profile defaults for a market-making liquidity provider."""

    role: str = "market_maker"
    risk_tolerance: str = "medium"
    decision_style: str = "inventory_aware_quoting"
    inventory_discipline: float = 0.7
    quote_aggressiveness: float = 0.5
    adverse_selection_sensitivity: float = 0.6
    liquidity_resilience: float = 0.5


@dataclass
class RetailProfile(AgentProfile):
    """Profile defaults for a small noisy retail participant."""

    role: str = "retail"
    risk_tolerance: str = "medium"
    decision_style: str = "noisy_sentiment_driven"
    social_sensitivity: float = 0.5
    panic_sensitivity: float = 0.4
    herding_tendency: float = 0.4
    news_reactivity: float = 0.5
    loss_aversion: float = 0.5


@dataclass
class InstitutionalProfile(AgentProfile):
    """Profile defaults for a larger target-execution participant."""

    role: str = "institutional"
    risk_tolerance: str = "medium"
    decision_style: str = "target_execution"
    execution_patience: float = 0.6
    benchmark_focus: str = "arrival_price"
    information_sensitivity: float = 0.5
    alpha_horizon: str = "intraday"
    market_impact_aversion: float = 0.6


@dataclass
class InformedProfile(AgentProfile):
    """Profile defaults for a participant trading from a private value signal."""

    role: str = "informed_trader"
    risk_tolerance: str = "medium"
    decision_style: str = "private_signal_value_trading"
    information_quality: float = 0.7
    patience: float = 0.45
    adverse_selection_tolerance: float = 0.6
    conviction: float = 0.65


@dataclass
class LiquidityTakerProfile(AgentProfile):
    """Profile defaults for a participant that consumes displayed liquidity."""

    role: str = "liquidity_taker"
    risk_tolerance: str = "medium"
    decision_style: str = "aggressive_flow_execution"
    immediacy_preference: float = 0.8
    market_impact_tolerance: float = 0.6
    flow_persistence: float = 0.5


ProfileT = TypeVar("ProfileT", bound=AgentProfile)


def coerce_profile(
    profile: AgentProfile | Mapping[str, Any] | None,
    profile_cls: type[ProfileT],
) -> ProfileT:
    """Create a role-specific profile from YAML/config data."""

    if profile is None:
        return profile_cls()
    if is_dataclass(profile):
        profile = asdict(profile)
    if not isinstance(profile, Mapping):
        raise TypeError(f"profile must be a mapping or {profile_cls.__name__}")

    valid_fields = {field.name for field in fields(profile_cls)}
    known_values: dict[str, Any] = {
        key: value
        for key, value in profile.items()
        if key in valid_fields
    }
    unrecognised = {
        key: value
        for key, value in profile.items()
        if key not in valid_fields
    }
    if unrecognised:
        raise ValueError(
            f"Unknown {profile_cls.__name__} field(s): "
            + ", ".join(sorted(unrecognised))
        )

    result = profile_cls(**known_values)
    if result.role != profile_cls().role:
        raise ValueError(
            f"{profile_cls.__name__}.role must be {profile_cls().role!r}, "
            f"not {result.role!r}"
        )
    if result.risk_tolerance not in _RISK_TOLERANCES:
        raise ValueError(
            "risk_tolerance must be one of: "
            + ", ".join(sorted(_RISK_TOLERANCES))
        )
    for profile_field in fields(result):
        if profile_field.name not in _BOUNDED_PROFILE_FIELDS:
            continue
        value = getattr(result, profile_field.name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"profile.{profile_field.name} must be numeric in [0, 1]")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"profile.{profile_field.name} must be in [0, 1]")
    return result


def profile_to_dict(profile: AgentProfile | Mapping[str, Any] | None) -> dict[str, Any]:
    """Serialize a profile into the dict shape passed to LLM context."""

    if profile is None:
        return {}
    if is_dataclass(profile):
        return asdict(profile)
    return dict(profile)
