"""Allowed AML strategy risk modes."""

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


def format_risk_mode_definitions() -> str:
    """Return a prompt-ready description of allowed risk modes."""

    return "\n".join(
        f"- {mode}: {description}"
        for mode, description in RISK_MODE_DEFINITIONS.items()
    )
