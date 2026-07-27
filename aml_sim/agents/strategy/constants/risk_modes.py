"""Allowed AML strategy risk modes and their fast-loop policy modifiers.
"""

from dataclasses import dataclass
from math import sqrt

RISK_MODE_DEFINITIONS = {
    # Strong defensive posture: preserve capital, reduce exposure, shrink
    # order sizes, widen spreads, and avoid adding inventory unless necessary.
    "risk_off": "strong defensive posture; preserve capital and reduce exposure",
    # Cautious participation: stay active, but use smaller orders, lower trade
    # probability, wider spreads, and slower execution.
    "conservative": "cautious participation with reduced size and slower execution",
    # Baseline behavior: use the configured/default strategy parameters.
    "normal": "baseline strategy posture",
    # Selective risk-on behavior: take more risk only when signal or conditions
    # are favorable enough to justify it.
    "opportunistic": "selective risk-on posture when evidence is favorable",
    # High risk appetite: larger orders, higher participation, more urgent
    # execution, or more assertive quoting when role and context justify it.
    "aggressive": "high risk appetite with stronger participation",
}

ALLOWED_RISK_MODES = tuple(RISK_MODE_DEFINITIONS.keys())


@dataclass(frozen=True)
class RiskModePolicy:
    """Risk preference translated into shared fast-loop decision properties.

    ``risk_aversion`` is normalized around 1.0 so normal mode preserves the
    configured strategy. Higher values penalize uncertain exposure more.

    The inverse relationship between risk aversion and risky allocation follows
    Merton-style portfolio choice. Square-root scaling keeps order-level changes
    milder than portfolio-level exposure changes.
    """

    risk_aversion: float = 1.0

    @property
    def participation_multiplier(self) -> float:
        return _clamp(1.0 / self.risk_aversion, 0.25, 1.50)

    @property
    def order_size_multiplier(self) -> float:
        return _clamp(1.0 / sqrt(self.risk_aversion), 0.40, 1.40)

    @property
    def position_limit_multiplier(self) -> float:
        # Configured position limits remain hard ceilings in risk-seeking modes.
        return _clamp(1.0 / self.risk_aversion, 0.25, 1.0)

    @property
    def signal_threshold_multiplier(self) -> float:
        return _clamp(sqrt(self.risk_aversion), 0.70, 2.0)

    @property
    def market_maker_spread_multiplier(self) -> float:
        # Avellaneda-Stoikov makes the inventory-risk component increase with
        # risk aversion. This normalized form preserves the configured baseline.
        return _clamp(sqrt(self.risk_aversion), 0.70, 2.0)

    @property
    def inventory_skew_multiplier(self) -> float:
        return _clamp(self.risk_aversion, 0.50, 3.0)

    def execution_size_multiplier(self, *, reduces_exposure: bool) -> float:
        """Scale institutional execution according to its effect on exposure.

        Almgren-Chriss-style execution trades market impact against inventory
        risk. A more risk-averse agent executes faster when reducing unwanted
        exposure, but adds new exposure more cautiously.
        """

        if reduces_exposure:
            return _clamp(sqrt(self.risk_aversion), 0.70, 2.0)
        return self.order_size_multiplier


RISK_MODE_POLICIES: dict[str, RiskModePolicy] = {
    "risk_off": RiskModePolicy(risk_aversion=3.0),
    "conservative": RiskModePolicy(risk_aversion=1.5),
    "normal": RiskModePolicy(),
    "opportunistic": RiskModePolicy(risk_aversion=0.75),
    "aggressive": RiskModePolicy(risk_aversion=0.50),
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def risk_mode_policy(mode: str) -> RiskModePolicy:
    """Return the fast-loop policy for a validated risk mode, falling back to normal."""

    return RISK_MODE_POLICIES.get(str(mode).lower(), RISK_MODE_POLICIES["normal"])


def format_risk_mode_definitions() -> str:
    """Return a prompt-ready description of allowed risk modes."""

    return "\n".join(
        f"- {mode}: {description}"
        for mode, description in RISK_MODE_DEFINITIONS.items()
    )
