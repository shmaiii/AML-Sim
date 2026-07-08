"""LLM-backed slow-loop strategist for AML agents."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, fields, is_dataclass, replace
from typing import Any, Mapping, Optional, Protocol

from aml_sim.agents.models.profile import profile_to_dict


class SlowStrategist(Protocol):
    """Interface for AML slow-loop strategists."""

    def propose(
        self,
        observation: Mapping[str, Any],
        current_strategy: Any,
        **kwargs: Any,
    ) -> Any:
        """Return a proposed strategy state for validation and application."""


class LLMStrategistConfigurationError(RuntimeError):
    """Raised when the LLM strategist is used without a configured client."""


class LLMStrategyResponseError(ValueError):
    """Raised when an LLM response cannot be parsed into a strategy proposal."""


class JSONLLMClient(Protocol):
    """Minimal protocol expected from an LLM client adapter."""

    async def complete_json(self, context: Mapping[str, Any]) -> Mapping[str, Any] | str:
        """Return a JSON object or JSON string containing strategy updates."""


# ---------------------------------------------------------------------------
# Enhanced LLM response schema — layered on top of the upstream contract
# ---------------------------------------------------------------------------

LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reason"],
    "properties": {
        "risk_mode": {
            "type": "string",
            "enum": ["conservative", "normal", "aggressive"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
        "strategy_updates": {
            "type": "object",
            "description": "Field-level parameter changes (upstream contract).",
        },
        "strategy_config": {
            "type": "object",
            "description": "Enable/disable strategies and set blending mode.",
            "properties": {
                "active_strategies": {"type": "array", "items": {"type": "string"}},
                "blend_mode": {
                    "type": "string",
                    "enum": ["weighted_sum", "vote", "unanimous", "priority_cascade"],
                },
                "strategy_weights": {"type": "object"},
            },
        },
        "risk_overrides": {
            "type": "object",
            "description": "Temporary risk-limit overrides.",
            "properties": {
                "max_drawdown_pct": {"type": "number"},
                "max_position_pct": {"type": "number"},
                "max_order_rate": {"type": "integer"},
                "max_consecutive_rejections": {"type": "integer"},
                "cooldown_ticks": {"type": "integer"},
            },
        },
    },
}


class LLMStrategist:
    """
    Slow-loop strategist that asks an LLM for strategy-state updates.

    The strategist never places orders. It only proposes an updated strategy
    state, which should be passed through the strategy validator before use.

    Two response contracts are supported:

    1. **Upstream** (simple):  ``{"strategy_updates": {...}, "confidence": 0.7, "reason": "..."}``
    2. **Enhanced**: additionally ``{"strategy_config": {...}, "risk_overrides": {...}}``

    The base agent handles both — ``strategy_config`` enables/disables alpha
    strategies, ``risk_overrides`` adjusts risk-manager limits.
    """

    def __init__(
        self,
        client: Optional[JSONLLMClient] = None,
        *,
        allowed_strategy_fields: Optional[set[str]] = None,
    ) -> None:
        self.client = client
        self.allowed_strategy_fields = allowed_strategy_fields

    async def propose(
        self,
        observation: Mapping[str, Any],
        current_strategy: Any,
        *,
        profile: Optional[Mapping[str, Any]] = None,
        memory: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """
        Ask the LLM to propose a new strategy state from context.

        Returns the full LLM response dict so the caller can extract
        strategy_config, strategy_updates, and risk_overrides.
        """

        if self.client is None:
            raise LLMStrategistConfigurationError(
                "LLMStrategist requires a JSONLLMClient. Provide a client adapter "
                "from config before enabling the LLM slow loop."
            )

        context = self.build_context(
            observation=observation,
            profile=profile,
            memory=memory,
            current_strategy=current_strategy,
        )
        raw_response = await self.client.complete_json(context)
        response = self._parse_response(raw_response)
        return response

    def build_context(
        self,
        *,
        observation: Mapping[str, Any],
        current_strategy: Any,
        profile: Optional[Mapping[str, Any]] = None,
        memory: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build the structured context sent to the LLM client."""

        return {
            "task": (
                "Propose strategy updates for the next tick cycle. "
                "You may adjust: "
                "(1) strategy_updates — specific numeric field changes, "
                "(2) strategy_config — which alpha strategies to use and how to blend them, "
                "(3) risk_overrides — temporary risk-limit adjustments. "
                "You are an agent strategy director. Do NOT place orders."
            ),
            "output_schema": LLM_RESPONSE_SCHEMA,
            "available_strategies": self._available_strategies(observation),
            "profile": profile_to_dict(profile),
            "memory": dict(memory or observation.get("memory", {}) or {}),
            "observation": dict(observation),
            "current_strategy": self._strategy_to_dict(current_strategy),
        }

    @staticmethod
    def _available_strategies(observation: Mapping[str, Any]) -> list[str]:
        try:
            from aml_sim.agents.strategy.registry import StrategyRegistry
            return StrategyRegistry.list_all()
        except Exception:
            return ["momentum", "mean_reversion", "breakout",
                    "volatility_regime", "event_driven", "passive_benchmark"]

    def _parse_response(self, raw_response: Mapping[str, Any] | str) -> dict[str, Any]:
        if isinstance(raw_response, Mapping):
            return dict(raw_response)

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise LLMStrategyResponseError(
                f"LLM response was not valid JSON: {exc}"
            ) from exc

        if not isinstance(parsed, Mapping):
            raise LLMStrategyResponseError("LLM response JSON must be an object.")

        return dict(parsed)

    def _strategy_to_dict(self, strategy: Any) -> dict[str, Any]:
        if is_dataclass(strategy):
            return asdict(strategy)
        if isinstance(strategy, Mapping):
            return dict(strategy)
        return dict(vars(strategy))


# ---------------------------------------------------------------------------
# Static test client
# ---------------------------------------------------------------------------


class StaticJSONLLMClient:
    """Tiny test client that returns a fixed JSON response."""

    def __init__(self, response: Mapping[str, Any] | str) -> None:
        self.response = response
        self.last_context: Optional[Mapping[str, Any]] = None

    async def complete_json(self, context: Mapping[str, Any]) -> Mapping[str, Any] | str:
        self.last_context = context
        if inspect.isawaitable(self.response):
            return await self.response
        return self.response


# ---------------------------------------------------------------------------
# OpenAI JSON LLM client
# ---------------------------------------------------------------------------


class OpenAIJSONLLMClient:
    """OpenAI-backed JSON client for AML slow-loop strategy updates."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.system_prompt = system_prompt or DEFAULT_OPENAI_SLOW_STRATEGY_PROMPT
        self.last_context: Optional[Mapping[str, Any]] = None

    async def complete_json(self, context: Mapping[str, Any]) -> Mapping[str, Any] | str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMStrategistConfigurationError(
                f"Missing OpenAI API key. Set {self.api_key_env} in .env or the process environment."
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LLMStrategistConfigurationError(
                "OpenAI slow strategist requires the 'openai' Python package."
            ) from exc

        self.last_context = context
        client = AsyncOpenAI(
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        response = await client.responses.create(
            model=self.model,
            instructions=self.system_prompt,
            input=json.dumps(context, default=str),
            temperature=self.temperature,
            text={"format": {"type": "json_object"}},
        )
        content = getattr(response, "output_text", None)
        if not content:
            raise LLMStrategyResponseError("OpenAI returned an empty strategy response.")
        return content


DEFAULT_OPENAI_SLOW_STRATEGY_PROMPT = """
You are the slow-loop strategy module for one AML-Sim trading agent.

You must return valid JSON only. Do not place orders. Do not include prose
outside JSON. You may only propose updates to fields already present in
current_strategy. The fast loop and StockSim execution layer will decide
whether and how orders are placed.

Return this shape:
{
  "strategy_updates": {
    "<existing_strategy_field>": "<new_value>"
  },
  "strategy_config": {
    "active_strategies": ["momentum"],
    "blend_mode": "weighted_sum",
    "strategy_weights": {"momentum": 0.5}
  },
  "risk_overrides": {},
  "confidence": 0.0,
  "reason": "brief reason for the strategy update"
}

strategy_config and risk_overrides are optional. Use strategy_updates for
numeric field changes. Use strategy_config to change which alpha strategies
are active (only on institutional agents). Use risk_overrides to temporarily
adjust risk limits.

If there is no good reason to change behavior, return an empty
strategy_updates object with a short reason.
""".strip()


# ---------------------------------------------------------------------------
# Static responses — using the upstream strategy_updates key
# ---------------------------------------------------------------------------


STATIC_MARKET_MAKER_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "spread": 0.25,
        "quote_size": 100,
        "inventory_skew": 0.0015,
    },
    "strategy_config": {},
    "risk_overrides": {},
    "confidence": 0.75,
    "reason": "Static market-maker LLM test response: quote slightly wider and manage inventory conservatively.",
}


STATIC_RETAIL_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "trade_probability": 0.35,
        "buy_bias": 0.52,
        "herding_tendency": 0.15,
        "panic_level": 0.05,
    },
    "strategy_config": {},
    "risk_overrides": {},
    "confidence": 0.7,
    "reason": "Static retail LLM test response: slightly active, mildly bullish, low panic.",
}


STATIC_INSTITUTIONAL_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "child_order_size": 100,
        "execution_style": "sliced",
        "urgency": 0.6,
    },
    "strategy_config": {
        "active_strategies": ["momentum", "mean_reversion"],
        "blend_mode": "weighted_sum",
        "strategy_weights": {"momentum": 0.5, "mean_reversion": 0.5},
    },
    "risk_overrides": {},
    "confidence": 0.78,
    "reason": "Static institutional LLM test response: keep sliced execution with moderate urgency.",
}


STATIC_INFORMED_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "trade_probability": 0.38,
        "information_edge": 0.72,
    },
    "strategy_config": {},
    "risk_overrides": {},
    "confidence": 0.74,
    "reason": "Static informed-trader LLM test response: keep trading only when the private value signal is strong.",
}


STATIC_LIQUIDITY_TAKER_RESPONSE = {
    "strategy_updates": {
        "risk_mode": "normal",
        "flow_intensity": 0.38,
        "aggression": 0.75,
    },
    "strategy_config": {},
    "risk_overrides": {},
    "confidence": 0.7,
    "reason": "Static liquidity-taker LLM test response: maintain steady aggressive flow with bounded size.",
}


STATIC_RESPONSES_BY_ROLE = {
    "market_maker": STATIC_MARKET_MAKER_RESPONSE,
    "retail": STATIC_RETAIL_RESPONSE,
    "institutional": STATIC_INSTITUTIONAL_RESPONSE,
    "informed": STATIC_INFORMED_RESPONSE,
    "liquidity_taker": STATIC_LIQUIDITY_TAKER_RESPONSE,
}


def create_llm_strategist(
    role: str,
    config: Optional[Mapping[str, Any]] = None,
) -> LLMStrategist:
    """Create a slow-loop strategist from role-specific config."""
    config = dict(config or {})
    strategist_type = str(config.get("type", "static")).lower()

    if not config.get("enabled", True):
        strategist_type = "static"

    if strategist_type in {"static", "fixed", "test"}:
        response = STATIC_RESPONSES_BY_ROLE[role]
        return LLMStrategist(client=StaticJSONLLMClient(response))

    if strategist_type in {"openai", "openai_json"}:
        client = OpenAIJSONLLMClient(
            model=str(config.get("model", "gpt-4o-mini")),
            api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
            temperature=float(config.get("temperature", 0.2)),
            timeout_seconds=float(config.get("timeout_seconds", 30.0)),
            max_retries=int(config.get("max_retries", 2)),
            system_prompt=config.get("system_prompt") or config.get("prompt"),
        )
        allowed_fields = config.get("allowed_strategy_fields")
        return LLMStrategist(
            client=client,
            allowed_strategy_fields=set(allowed_fields) if allowed_fields else None,
        )

    raise LLMStrategistConfigurationError(
        f"Unsupported slow_strategist type {strategist_type!r}. "
        "Use 'static' or 'openai'."
    )
