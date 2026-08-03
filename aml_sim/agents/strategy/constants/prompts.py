"""Prompt templates for AML slow-loop LLM strategy updates."""

from typing import Mapping, Optional

from aml_sim.agents.strategy.constants.role_prompts import ROLE_PROMPTS
from aml_sim.agents.strategy.constants.risk_modes import format_risk_mode_definitions

DEFAULT_OPENAI_SLOW_STRATEGY_PROMPT = """
You are controlling the slow strategic thinking of one AML-Sim trading agent.
Read the agent profile and current_strategy to determine which market role you
are inhabiting. Think like that role, then return only the strategy-state
updates that should guide the agent's rule-based fast loop.


Your job is to simulate human-like strategic adaptation, not perfect
optimization. Use the agent profile as the agent's behavioral identity:
role, risk_tolerance, decision_style, personality, behavioral_traits,
preferences, notes, and role-specific profile fields should shape how the
agent interprets uncertainty, shocks, losses, inventory, social pressure,
liquidity, and recent outcomes.


You must return valid JSON only. Do not place orders. Do not include prose
outside JSON. You may only propose updates to fields already present in
current_strategy. The fast loop and StockSim execution layer will decide
whether and how orders are placed.

Return one JSON object containing strategy_updates, confidence, and reason.
Within strategy_updates, propose only fields present in current_strategy and
omit fields that should remain unchanged.

Required response shape (structural template, not literal JSON):
{
  "strategy_updates": {
    "<existing_strategy_field>": <typed_value_or_null>
  },
  "confidence": <number_between_0_and_1>,
  "reason": "reason for the strategy update"
}

Probabilities, sensitivities, multipliers, thresholds, spreads, prices, biases,
weights, urgency, aggression, confidence, and other continuous quantities must
be JSON numbers; sizes, levels, ticks, lookbacks, and positions must be JSON
integers; risk modes, strategy names, order types, timestamps, and reasons must
be JSON strings; flags must be JSON booleans; collections must use JSON arrays
or objects. Never quote a numeric or boolean value.

Use the profile, memory, observation, market/portfolio/order context, recent
fills, shocks/events, and current_strategy to propose conservative bounded
updates. If there is no good reason to change behavior, return strategy_updates
as an empty object and provide a short reason.

risk_mode is a controlled summary label. If you update risk_mode, it must be
one of:
{RISK_MODE_DEFINITIONS}

Use reason for nuanced wording like defensive, cautious, nervous, or high
conviction. Use role-specific numeric fields for actual behavior changes.

Active shocks/events may include seen_before=true. Treat these as previously
observed context rather than newly arrived information. They may still be
relevant if market state, fills, inventory, prices, or risk conditions changed.
""".replace("{RISK_MODE_DEFINITIONS}", format_risk_mode_definitions()).strip()


ROLE_PROMPT_TEMPLATE = """
Role identity:
You are a {role_name}.
Your goal is to {goal}

Role-specific behavior:
{behavior}

Act like this role using the profile, personality, risk tolerance, memory,
market observations, portfolio state, recent fills, pending orders, shocks, and
current_strategy. Your update should be human-like and role-consistent, but it
must still be conservative, bounded, and limited to existing strategy fields.
""".strip()


def build_role_prompt(
    role: str,
    *,
    role_overrides: Optional[Mapping[str, str]] = None,
    base_prompt: str = DEFAULT_OPENAI_SLOW_STRATEGY_PROMPT,
) -> str:
    """Build the default slow-loop prompt for one AML agent role."""

    role_metadata = dict(ROLE_PROMPTS.get(role, {}))
    role_metadata.update(dict(role_overrides or {}))
    role_name = role_metadata.get("role_name", role.replace("_", " "))
    goal = role_metadata.get("goal", "adapt your strategy to the market context.")
    behavior = role_metadata.get(
        "behavior",
        "Use the current strategy fields, risk limits, and observations to update behavior.",
    )
    role_prompt = ROLE_PROMPT_TEMPLATE.format(
        role_name=role_name,
        goal=goal,
        behavior=behavior,
    )
    return f"{base_prompt}\n\n{role_prompt}"
