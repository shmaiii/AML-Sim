"""Agent profile models for AML behavioral context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping, TypeVar


# ---------------------------------------------------------------------------
# Behavioral traits — the measurable dimensions that modulate strategy
# ---------------------------------------------------------------------------


@dataclass
class BehavioralTraits:
    """Measurable behavioral dimensions that modulate strategy execution.

    Each trait is a float in [0, 1] unless noted otherwise. These are the
    "knobs" that the ProfileModulator uses to translate personality into
    concrete strategy-parameter adjustments.
    """

    risk_aversion: float = 0.5          # 0 = risk-seeking, 1 = risk-averse
    patience: float = 0.5               # 0 = impatient (market orders), 1 = patient (limit)
    conviction: float = 0.5             # how strongly held beliefs are
    adaptability: float = 0.5           # responsiveness to new information
    discipline: float = 0.7             # adherence to strategy vs impulse
    aggression: float = 0.5             # willingness to cross spread
    social_influence: float = 0.3       # sensitivity to other agents' behaviour
    loss_aversion: float = 0.5          # tendency to hold losers / cut winners
    overconfidence: float = 0.3         # bias toward own beliefs
    recency_bias: float = 0.5           # weight on recent vs historical data

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> BehavioralTraits:
        if not data:
            return cls()
        valid = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in valid})


# ---------------------------------------------------------------------------
# Base agent profile — extended with traits
# ---------------------------------------------------------------------------


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

    @property
    def traits(self) -> BehavioralTraits:
        if not hasattr(self, "_cached_traits"):
            self._cached_traits = BehavioralTraits.from_mapping(
                self.behavioral_traits
            )
        return self._cached_traits


# ---------------------------------------------------------------------------
# Role-specific profiles with sensible trait defaults
# ---------------------------------------------------------------------------


@dataclass
class MarketMakerProfile(AgentProfile):
    """Profile for a market-making liquidity provider.

    High discipline, moderate risk aversion, patient execution.
    """

    role: str = "market_maker"
    risk_tolerance: str = "medium"
    decision_style: str = "inventory_aware_quoting"
    inventory_discipline: float = 0.7
    quote_aggressiveness: float = 0.5
    adverse_selection_sensitivity: float = 0.6
    liquidity_resilience: float = 0.5
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.6,
        "patience": 0.7,
        "conviction": 0.5,
        "adaptability": 0.55,
        "discipline": 0.85,
        "aggression": 0.3,
        "social_influence": 0.2,
        "loss_aversion": 0.4,
        "overconfidence": 0.2,
        "recency_bias": 0.6,
    })


@dataclass
class RetailProfile(AgentProfile):
    """Profile for a small noisy retail participant.

    Higher social influence, moderate loss aversion, lower discipline.
    """

    role: str = "retail"
    risk_tolerance: str = "medium"
    decision_style: str = "noisy_sentiment_driven"
    social_sensitivity: float = 0.5
    panic_sensitivity: float = 0.4
    herding_tendency: float = 0.4
    news_reactivity: float = 0.5
    loss_aversion: float = 0.5
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.45,
        "patience": 0.25,
        "conviction": 0.3,
        "adaptability": 0.7,
        "discipline": 0.35,
        "aggression": 0.55,
        "social_influence": 0.65,
        "loss_aversion": 0.6,
        "overconfidence": 0.5,
        "recency_bias": 0.75,
    })


@dataclass
class InstitutionalProfile(AgentProfile):
    """Profile for a larger target-execution participant.

    High discipline, high patience, moderate conviction.
    """

    role: str = "institutional"
    risk_tolerance: str = "medium"
    decision_style: str = "target_execution"
    execution_patience: float = 0.6
    benchmark_focus: str = "arrival_price"
    information_sensitivity: float = 0.5
    alpha_horizon: str = "intraday"
    market_impact_aversion: float = 0.6
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.55,
        "patience": 0.8,
        "conviction": 0.65,
        "adaptability": 0.45,
        "discipline": 0.85,
        "aggression": 0.3,
        "social_influence": 0.2,
        "loss_aversion": 0.35,
        "overconfidence": 0.25,
        "recency_bias": 0.4,
    })


@dataclass
class InformedProfile(AgentProfile):
    """Profile for a participant trading from a private value signal.

    High conviction, moderate risk aversion, lower recency bias.
    """

    role: str = "informed_trader"
    risk_tolerance: str = "medium"
    decision_style: str = "private_signal_value_trading"
    information_quality: float = 0.7
    patience: float = 0.45
    adverse_selection_tolerance: float = 0.6
    conviction: float = 0.65
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.4,
        "patience": 0.5,
        "conviction": 0.8,
        "adaptability": 0.55,
        "discipline": 0.65,
        "aggression": 0.5,
        "social_influence": 0.15,
        "loss_aversion": 0.35,
        "overconfidence": 0.55,
        "recency_bias": 0.35,
    })


@dataclass
class LiquidityTakerProfile(AgentProfile):
    """Profile for aggressive flow that consumes displayed liquidity.

    High aggression, low patience, high adaptability to flow.
    """

    role: str = "liquidity_taker"
    risk_tolerance: str = "medium"
    decision_style: str = "aggressive_flow_execution"
    immediacy_preference: float = 0.8
    market_impact_tolerance: float = 0.6
    flow_persistence: float = 0.5
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.3,
        "patience": 0.15,
        "conviction": 0.4,
        "adaptability": 0.75,
        "discipline": 0.4,
        "aggression": 0.85,
        "social_influence": 0.25,
        "loss_aversion": 0.3,
        "overconfidence": 0.45,
        "recency_bias": 0.7,
    })


# ---------------------------------------------------------------------------
# Additional role profiles
# ---------------------------------------------------------------------------


@dataclass
class TrendFollowerProfile(AgentProfile):
    """Profile for a pure momentum / trend-following participant."""

    role: str = "trend_follower"
    risk_tolerance: str = "medium"
    decision_style: str = "trend_following"
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.4,
        "patience": 0.4,
        "conviction": 0.7,
        "adaptability": 0.5,
        "discipline": 0.6,
        "aggression": 0.6,
        "social_influence": 0.35,
        "loss_aversion": 0.4,
        "overconfidence": 0.5,
        "recency_bias": 0.8,
    })


@dataclass
class ArbitrageurProfile(AgentProfile):
    """Profile for a participant that exploits short-term mispricing."""

    role: str = "arbitrageur"
    risk_tolerance: str = "low"
    decision_style: str = "statistical_arbitrage"
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.75,
        "patience": 0.3,
        "conviction": 0.5,
        "adaptability": 0.9,
        "discipline": 0.9,
        "aggression": 0.7,
        "social_influence": 0.05,
        "loss_aversion": 0.2,
        "overconfidence": 0.15,
        "recency_bias": 0.9,
    })


@dataclass
class HedgeProfile(AgentProfile):
    """Profile for a hedger that prioritizes risk reduction over returns."""

    role: str = "hedger"
    risk_tolerance: str = "conservative"
    decision_style: str = "risk_minimization"
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.9,
        "patience": 0.7,
        "conviction": 0.3,
        "adaptability": 0.6,
        "discipline": 0.95,
        "aggression": 0.1,
        "social_influence": 0.1,
        "loss_aversion": 0.9,
        "overconfidence": 0.1,
        "recency_bias": 0.3,
    })


@dataclass
class PassiveIndexProfile(AgentProfile):
    """Profile for a passive benchmark / control-group participant."""

    role: str = "passive_index"
    risk_tolerance: str = "medium"
    decision_style: str = "passive_benchmark"
    behavioral_traits: dict[str, Any] = field(default_factory=lambda: {
        "risk_aversion": 0.5,
        "patience": 0.9,
        "conviction": 0.1,
        "adaptability": 0.1,
        "discipline": 0.95,
        "aggression": 0.05,
        "social_influence": 0.05,
        "loss_aversion": 0.5,
        "overconfidence": 0.05,
        "recency_bias": 0.1,
    })


# ---------------------------------------------------------------------------
# Profile coercion / serialization helpers
# ---------------------------------------------------------------------------


ProfileT = TypeVar("ProfileT", bound=AgentProfile)


def coerce_profile(
    profile: AgentProfile | Mapping[str, Any] | None,
    profile_cls: type[ProfileT],
) -> ProfileT:
    """Create a role-specific profile from YAML/config data."""

    if profile is None:
        return profile_cls()
    if isinstance(profile, profile_cls):
        return profile
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
        existing_custom = dict(known_values.get("custom", {}))
        existing_custom.setdefault("_unrecognised_fields", {})
        existing_custom["_unrecognised_fields"].update(unrecognised)
        known_values["custom"] = existing_custom

    return profile_cls(**known_values)


def profile_to_dict(profile: AgentProfile | Mapping[str, Any] | None) -> dict[str, Any]:
    """Serialize a profile into the dict shape passed to LLM context."""

    if profile is None:
        return {}
    if is_dataclass(profile):
        result = asdict(profile)
        if hasattr(profile, "traits"):
            result["computed_traits"] = profile.traits.to_dict()
        return result
    return dict(profile)
