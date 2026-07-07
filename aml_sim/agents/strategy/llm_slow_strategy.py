"""LLM-backed slow-loop strategist for AML agents."""

from __future__ import annotations

import inspect
import json
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
# Enhanced LLM response schema
# ---------------------------------------------------------------------------

LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["reason"],
    "properties": {
        "risk_mode": {
            "type": "string",
            "enum": ["conservative", "normal", "aggressive"],
            "description": "Overall risk posture for this update cycle.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How confident the model is in this proposal.",
        },
        "reason": {
            "type": "string",
            "description": "Short explanation of the proposed changes.",
        },
        "strategy_config": {
            "type": "object",
            "description": "Enable/disable strategies and set blending mode.",
            "properties": {
                "active_strategies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of strategy names to activate.",
                },
                "blend_mode": {
                    "type": "string",
                    "enum": ["weighted_sum", "vote", "unanimous", "priority_cascade"],
                },
                "strategy_weights": {
                    "type": "object",
                    "description": "Per-strategy weight mapping.",
                },
            },
        },
        "parameter_updates": {
            "type": "object",
            "description": "Field-level parameter changes to merge into the current strategy state.",
        },
        "risk_overrides": {
            "type": "object",
            "description": "Temporary risk-limit overrides for the risk manager.",
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


# ---------------------------------------------------------------------------
# LLM Strategist
# ---------------------------------------------------------------------------


class LLMStrategist:
    """
    Slow-loop strategist that asks an LLM for strategy-state updates.

    The strategist never places orders. It only proposes an updated strategy
    state, which should be passed through the strategy validator before use.

    The LLM response can now include three categories of directives:

    1. **strategy_config** — enable/disable alpha strategies, set blend mode
       and per-strategy weights.
    2. **parameter_updates** — tweak individual strategy-state fields (same
       as before).
    3. **risk_overrides** — temporarily adjust risk manager limits.
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

        Returns the full LLM response dict (not just a dataclass).
        The caller (BaseAMLAgent.run_slow_loop) handles the three-part
        response shape.
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

        # Always return the full response dict so the base agent can extract
        # strategy_config, parameter_updates, and risk_overrides.
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
                "You may adjust three categories: "
                "(1) strategy_config — which alpha strategies to use and how to blend them, "
                "(2) parameter_updates — specific numeric field changes, "
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
        """List strategy names available from registry or configured set."""
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
# Static test client (unchanged contract — used during development)
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
# Static responses — now using the enhanced contract
# ---------------------------------------------------------------------------

STATIC_MARKET_MAKER_RESPONSE = {
    "risk_mode": "normal",
    "confidence": 0.75,
    "reason": "Static market-maker LLM test response: quote slightly wider and manage inventory conservatively.",
    "strategy_config": {},
    "parameter_updates": {
        "risk_mode": "normal",
        "spread": 0.25,
        "quote_size": 100,
        "inventory_skew": 0.0015,
    },
    "risk_overrides": {},
}

STATIC_RETAIL_RESPONSE = {
    "risk_mode": "normal",
    "confidence": 0.7,
    "reason": "Static retail LLM test response: slightly active, mildly bullish, low panic.",
    "strategy_config": {},
    "parameter_updates": {
        "risk_mode": "normal",
        "trade_probability": 0.35,
        "buy_bias": 0.52,
        "herding_tendency": 0.15,
        "panic_level": 0.05,
    },
    "risk_overrides": {},
}

STATIC_INSTITUTIONAL_RESPONSE = {
    "risk_mode": "normal",
    "confidence": 0.78,
    "reason": "Static institutional LLM test response: keep sliced execution with moderate urgency.",
    "strategy_config": {
        "active_strategies": ["momentum", "mean_reversion"],
        "blend_mode": "weighted_sum",
        "strategy_weights": {"momentum": 0.5, "mean_reversion": 0.5},
    },
    "parameter_updates": {
        "risk_mode": "normal",
        "child_order_size": 100,
        "execution_style": "sliced",
        "urgency": 0.6,
    },
    "risk_overrides": {},
}

STATIC_INFORMED_RESPONSE = {
    "risk_mode": "normal",
    "confidence": 0.74,
    "reason": "Static informed-trader LLM test response: keep trading only when the private value signal is strong.",
    "strategy_config": {},
    "parameter_updates": {
        "risk_mode": "normal",
        "trade_probability": 0.38,
        "information_edge": 0.72,
    },
    "risk_overrides": {},
}

STATIC_LIQUIDITY_TAKER_RESPONSE = {
    "risk_mode": "normal",
    "confidence": 0.7,
    "reason": "Static liquidity-taker LLM test response: maintain steady aggressive flow with bounded size.",
    "strategy_config": {},
    "parameter_updates": {
        "risk_mode": "normal",
        "flow_intensity": 0.38,
        "aggression": 0.75,
    },
    "risk_overrides": {},
}


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_static_market_maker_llm_strategist() -> LLMStrategist:
    return LLMStrategist(client=StaticJSONLLMClient(STATIC_MARKET_MAKER_RESPONSE))


def create_static_retail_llm_strategist() -> LLMStrategist:
    return LLMStrategist(client=StaticJSONLLMClient(STATIC_RETAIL_RESPONSE))


def create_static_institutional_llm_strategist() -> LLMStrategist:
    return LLMStrategist(client=StaticJSONLLMClient(STATIC_INSTITUTIONAL_RESPONSE))


def create_static_informed_llm_strategist() -> LLMStrategist:
    return LLMStrategist(client=StaticJSONLLMClient(STATIC_INFORMED_RESPONSE))


def create_static_liquidity_taker_llm_strategist() -> LLMStrategist:
    return LLMStrategist(client=StaticJSONLLMClient(STATIC_LIQUIDITY_TAKER_RESPONSE))
