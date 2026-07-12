"""Event-driven alpha strategy — trades around market shock events."""

from __future__ import annotations

from dataclasses import dataclass

from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal
from aml_sim.agents.strategy.signals import event_pressure


@dataclass
class EventDrivenStrategy:
    """Generate signals in response to active market shock events.

    This strategy converts AML shock events (negative news, liquidity stress,
    recovery news, etc.) into directional trading signals. It is designed to
    work alongside slower strategies like momentum — event signals are short-
    horizon and decay as the event ages.
    """

    name: str = "event_driven"
    description: str = (
        "Generates short-horizon signals from active market shocks: trades "
        "in the opposite direction of adverse events (fade), or follows "
        "positive events (trend)."
    )
    min_severity: float = 0.15       # ignore very mild shocks
    fade_stress: bool = True          # buy into negative shocks (contrarian)
    follow_recovery: bool = True      # follow positive recovery shocks

    def generate(self, context: AlphaContext) -> AlphaSignal:
        events = context.events
        if not events:
            return AlphaSignal(reason="event_driven: no active events")

        pressure = event_pressure(events, context.instrument)
        severity = pressure["severity"]

        if severity < self.min_severity:
            return AlphaSignal(
                reason=f"event_driven: severity {severity:.2f} below threshold",
                metadata={"severity": severity},
            )

        direction_bias = pressure["directional_bias"]
        fundamental_shift = pressure["fundamental_price_shift"]

        # Determine direction from the shock
        if direction_bias < -0.1:
            # Negative shock
            if self.fade_stress:
                # Contrarian: buy the dip
                direction = 1.0
                reasoning = "fading negative shock"
            else:
                direction = -1.0
                reasoning = "following negative shock"
        elif direction_bias > 0.1:
            # Positive shock
            if self.follow_recovery:
                direction = 1.0
                reasoning = "following positive shock"
            else:
                direction = -1.0
                reasoning = "fading positive shock (take profit)"
        else:
            return AlphaSignal(
                reason=f"event_driven: no clear direction (bias={direction_bias:.3f})",
                metadata={"severity": severity},
            )

        # Price shift refines conviction
        if fundamental_shift != 0 and context.prices and context.prices[-1] > 0:
            shift_pct = abs(fundamental_shift / context.prices[-1])
            conviction_boost = min(0.3, shift_pct * 5)
        else:
            conviction_boost = 0.0

        confidence = min(1.0, severity * 0.7 + conviction_boost)

        return AlphaSignal(
            direction=direction,
            strength=round(severity, 4),
            confidence=round(confidence, 4),
            horizon_ticks=min(8, max(3, int(severity * 10))),
            reason=f"event_driven: {reasoning} (severity={severity:.2f}, bias={direction_bias:.2f})",
            suggested_order_type="MARKET",
            metadata={
                "severity": severity,
                "directional_bias": direction_bias,
                "fundamental_shift": fundamental_shift,
            },
        )
