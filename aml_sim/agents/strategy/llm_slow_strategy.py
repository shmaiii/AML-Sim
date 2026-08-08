"""LLM-backed slow-loop strategist for AML agents."""

from __future__ import annotations

import inspect
import json
import os
import copy
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol

from aml_sim.agents.models.profile import profile_to_dict
from aml_sim.agents.models.state import (
    InformedStrategyState,
    InstitutionalStrategyState,
    LiquidityTakerStrategyState,
    MarketMakerStrategyState,
    RetailStrategyState,
)
from aml_sim.agents.strategy.constants import (
    DEFAULT_OPENAI_SLOW_STRATEGY_PROMPT,
    STATIC_RESPONSES_BY_ROLE,
    build_role_prompt,
)


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


_STRATEGY_STATE_BY_ROLE = {
    "market_maker": MarketMakerStrategyState,
    "retail": RetailStrategyState,
    "institutional": InstitutionalStrategyState,
    "informed": InformedStrategyState,
    "liquidity_taker": LiquidityTakerStrategyState,
}
_SLOW_STRATEGIST_CONFIG_FIELDS = {
    "enabled", "type", "provider", "model", "api_key_env", "temperature",
    "timeout_seconds", "max_retries", "role_prompt", "role_prompts",
    "allowed_strategy_fields", "max_output_tokens",
}


class JSONLLMClient(Protocol):
    """Minimal protocol expected from an LLM client adapter."""

    async def complete_json(self, context: Mapping[str, Any]) -> Mapping[str, Any] | str:
        """Return a JSON object or JSON string containing strategy updates."""


class LLMStrategist:
    """
    Slow-loop strategist that asks an LLM for strategy-state updates.

    The strategist never places orders. It only proposes an updated strategy
    state, which should be passed through the strategy validator before use.
    """

    def __init__(
        self,
        client: Optional[JSONLLMClient] = None,
        *,
        allowed_strategy_fields: Optional[set[str]] = None,
    ) -> None:
        self.client = client
        self.allowed_strategy_fields = allowed_strategy_fields
        self.last_update_audit: dict[str, Any] = {}

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

        Expected LLM JSON shape:
            {
              "strategy_updates": {"risk_mode": "conservative"},
              "confidence": 0.7,
              "reason": "Inventory is elevated after recent fills."
            }
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
        updates = self._extract_strategy_updates(response)

        return self._apply_updates(
            current_strategy=current_strategy,
            updates=updates,
            observation=observation,
            response=response,
        )

    def build_context(
        self,
        *,
        observation: Mapping[str, Any],
        current_strategy: Any,
        profile: Optional[Mapping[str, Any]] = None,
        memory: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build the structured context sent to the LLM client."""

        compact_observation = self._compact_observation(observation)
        compact_memory = self._compact_memory(
            memory or observation.get("memory", {}) or {}
        )
        return {
            "task": "Propose strategy_state updates only. Do not place orders.",
            "output_contract": {
                "strategy_updates": "object containing only existing strategy fields",
                "confidence": "float between 0 and 1",
                "reason": "short explanation",
            },
            "profile": profile_to_dict(profile),
            "memory": compact_memory,
            "observation": compact_observation,
            "current_strategy": self._strategy_to_dict(current_strategy),
        }

    @staticmethod
    def _compact_memory(memory: Mapping[str, Any]) -> dict[str, Any]:
        compact = copy.deepcopy(dict(memory))
        events = compact.get("recent_events")
        if isinstance(events, list):
            compact["recent_events"] = events[-5:]
        return compact

    @staticmethod
    def _compact_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        """Bound LLM context without changing live simulation/report state."""

        compact = copy.deepcopy(dict(observation))
        compact.pop("memory", None)

        market = compact.get("market")
        if isinstance(market, dict):
            history = market.get("price_history")
            if isinstance(history, dict):
                market["price_history"] = {
                    instrument: values[-5:] if isinstance(values, list) else values
                    for instrument, values in history.items()
                }

        orders = compact.get("orders")
        if isinstance(orders, dict):
            pending = orders.get("pending")
            if isinstance(pending, dict) and len(pending) > 10:
                orders["pending"] = dict(list(pending.items())[-10:])

        fills = compact.get("recent_fills")
        if isinstance(fills, list):
            compact["recent_fills"] = fills[-5:]
        return compact

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

    def _extract_strategy_updates(self, response: Mapping[str, Any]) -> dict[str, Any]:
        updates = response.get("strategy_updates", response)
        if not isinstance(updates, Mapping):
            raise LLMStrategyResponseError("strategy_updates must be a JSON object.")
        return dict(updates)

    def _apply_updates(
        self,
        *,
        current_strategy: Any,
        updates: Mapping[str, Any],
        observation: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> Any:
        all_fields = self._strategy_fields(current_strategy)
        allowed_fields = self._allowed_fields(current_strategy)
        clean_updates: dict[str, Any] = {}
        rejected_updates: dict[str, dict[str, Any]] = {}
        for key, value in updates.items():
            if key not in all_fields:
                rejected_updates[key] = {
                    "value": value,
                    "reason": "unknown_strategy_field",
                }
            elif key not in allowed_fields:
                rejected_updates[key] = {
                    "value": value,
                    "reason": "not_allowed_by_configuration",
                }
            elif value is None:
                rejected_updates[key] = {
                    "value": value,
                    "reason": "null_value_ignored",
                }
            else:
                clean_updates[key] = value

        if "confidence" in response and "confidence" in allowed_fields:
            clean_updates["confidence"] = response["confidence"]
        if "reason" in response and "reason" in allowed_fields:
            clean_updates["reason"] = response["reason"]
        if "updated_at" in allowed_fields:
            clean_updates.setdefault("updated_at", observation.get("current_time"))

        bounded_numeric_fields = {
            "trade_probability": 1.0,
            "buy_bias": 1.0,
            "flow_intensity": 1.0,
            "information_edge": 1.0,
            "urgency": 1.0,
            "size_decay": 1.0,
            "confidence": 1.0,
            "liquidity_withdrawal_sensitivity": 2.0,
            "shock_sensitivity": 2.0,
            "sentiment_sensitivity": 2.0,
            "shock_reactivity": 2.0,
            "herding_tendency": 2.0,
            "panic_level": 2.0,
            "aggression": 2.0,
            "momentum_weight": 2.0,
        }
        adjusted_updates: dict[str, dict[str, Any]] = {}
        for field_name in bounded_numeric_fields.keys() & clean_updates.keys():
            value = clean_updates[field_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                rejected_updates[field_name] = {
                    "value": value,
                    "reason": "invalid_numeric_type",
                }
                clean_updates.pop(field_name)
                continue
            adjusted_value = max(
                0.0,
                min(bounded_numeric_fields[field_name], float(value)),
            )
            clean_updates[field_name] = adjusted_value
            if adjusted_value != value:
                adjusted_updates[field_name] = {
                    "original": value,
                    "applied": adjusted_value,
                    "reason": "clamped_to_valid_range",
                }

        self.last_update_audit = {
            "proposed_updates": dict(updates),
            "eligible_updates": dict(clean_updates),
            "adjusted_updates": adjusted_updates,
            "rejected_updates": rejected_updates,
        }

        if is_dataclass(current_strategy):
            return replace(current_strategy, **clean_updates)

        # Non-dataclass: copy first so a later validation failure does not
        # corrupt the live strategy object.
        import copy

        proposed = copy.deepcopy(current_strategy)
        for key, value in clean_updates.items():
            setattr(proposed, key, value)
        return proposed

    def _allowed_fields(self, current_strategy: Any) -> set[str]:
        if self.allowed_strategy_fields is not None:
            return set(self.allowed_strategy_fields) & self._strategy_fields(current_strategy)
        return self._strategy_fields(current_strategy)

    def _strategy_fields(self, current_strategy: Any) -> set[str]:
        if is_dataclass(current_strategy):
            return {field.name for field in fields(current_strategy)}
        if isinstance(current_strategy, Mapping):
            return set(current_strategy)
        return set(vars(current_strategy).keys())

    def _strategy_to_dict(self, strategy: Any) -> dict[str, Any]:
        if is_dataclass(strategy):
            return asdict(strategy)
        if isinstance(strategy, Mapping):
            return dict(strategy)
        return dict(vars(strategy))


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


class OpenAIJSONLLMClient:
    """OpenAI-backed JSON client for AML slow-loop strategy updates."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        max_output_tokens: int | None = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
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
        client_options: dict[str, Any] = {"api_key": api_key}
        if self.timeout_seconds is not None:
            client_options["timeout"] = self.timeout_seconds
        if self.max_retries is not None:
            client_options["max_retries"] = self.max_retries
        client = AsyncOpenAI(**client_options)

        request: dict[str, Any] = {
            "model": self.model,
            "instructions": self.system_prompt,
            "input": (
                "Return JSON only using the requested strategy update contract.\n\n"
                f"Context JSON:\n{json.dumps(context, default=str)}"
            ),
            "text": {"format": {"type": "json_object"}},
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens
        response = await client.responses.create(**request)
        content = getattr(response, "output_text", None)
        if not content:
            raise LLMStrategyResponseError("OpenAI returned an empty strategy response.")
        self._write_response_log(context=context, content=content)
        return content

    def _write_response_log(self, *, context: Mapping[str, Any], content: str) -> None:
        decision_context_dir = os.getenv("DECISION_CONTEXT_DIR")
        if not decision_context_dir:
            return

        observation = context.get("observation", {})
        if not isinstance(observation, Mapping):
            observation = {}
        agent_context = observation.get("agent", {})
        if not isinstance(agent_context, Mapping):
            agent_context = {}
        agent_id = str(agent_context.get("agent_id") or observation.get("agent_id") or "unknown_agent")
        safe_agent_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in agent_id
        )
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": "openai",
            "model": self.model,
            "agent_id": agent_id,
            "simulation_time": observation.get("current_time"),
            "response": content,
        }
        agent_dir = os.path.join(decision_context_dir, safe_agent_id)
        os.makedirs(agent_dir, exist_ok=True)
        agent_output_path = os.path.join(agent_dir, "llm_responses.jsonl")
        with open(agent_output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


def create_llm_strategist(
    role: str,
    config: Optional[Mapping[str, Any]] = None,
) -> LLMStrategist:
    """Create a slow-loop strategist from role-specific config."""
    config = dict(config or {})
    unknown_config = sorted(set(config) - _SLOW_STRATEGIST_CONFIG_FIELDS)
    if unknown_config:
        raise LLMStrategistConfigurationError(
            "Unknown slow_strategist field(s): " + ", ".join(unknown_config)
        )
    strategist_type = str(config.get("type", "static")).lower()

    if not config.get("enabled", True):
        strategist_type = "static"

    if strategist_type in {"static", "fixed", "test"}:
        response = STATIC_RESPONSES_BY_ROLE[role]
        return LLMStrategist(client=StaticJSONLLMClient(response))

    if strategist_type in {"openai", "openai_json"}:
        provider = str(config.get("provider", "openai")).lower()
        if provider != "openai":
            raise LLMStrategistConfigurationError(
                f"OpenAI strategist requires provider='openai', not {provider!r}."
            )
        role_overrides = config.get("role_prompt") or config.get("role_prompts")
        if role_overrides is not None and not isinstance(role_overrides, Mapping):
            raise LLMStrategistConfigurationError(
                "slow_strategist role_prompt must be a mapping when provided."
            )
        system_prompt = build_role_prompt(role, role_overrides=role_overrides)

        temperature = config.get("temperature")
        timeout_seconds = config.get("timeout_seconds")
        max_retries = config.get("max_retries")
        max_output_tokens = config.get("max_output_tokens")
        client = OpenAIJSONLLMClient(
            model=str(config.get("model", "gpt-5.4")),
            api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
            temperature=(
                float(temperature) if temperature is not None else None
            ),
            timeout_seconds=(
                float(timeout_seconds)
                if timeout_seconds is not None
                else None
            ),
            max_retries=(
                int(max_retries) if max_retries is not None else None
            ),
            max_output_tokens=(
                int(max_output_tokens) if max_output_tokens is not None else None
            ),
            system_prompt=system_prompt,
        )
        allowed_fields = config.get("allowed_strategy_fields")
        if allowed_fields is not None and not isinstance(allowed_fields, list):
            raise LLMStrategistConfigurationError(
                "allowed_strategy_fields must be a YAML list."
            )
        strategy_cls = _STRATEGY_STATE_BY_ROLE.get(role)
        valid_fields = (
            {item.name for item in fields(strategy_cls)} if strategy_cls else set()
        )
        unknown_allowed = sorted(set(allowed_fields or []) - valid_fields)
        if unknown_allowed:
            raise LLMStrategistConfigurationError(
                f"Unknown {role} allowed_strategy_fields: "
                + ", ".join(unknown_allowed)
            )
        return LLMStrategist(
            client=client,
            allowed_strategy_fields=set(allowed_fields) if allowed_fields else None,
        )

    raise LLMStrategistConfigurationError(
        f"Unsupported slow_strategist type {strategist_type!r}. "
        "Use 'static' or 'openai'."
    )
