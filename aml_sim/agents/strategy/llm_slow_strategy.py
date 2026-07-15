"""LLM-backed slow-loop strategist for AML agents."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Protocol

from aml_sim.agents.models.profile import profile_to_dict
from aml_sim.agents.strategy.constants import (
    DEFAULT_OPENAI_SLOW_STRATEGY_PROMPT,
    STATIC_RESPONSES_BY_ROLE,
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

        return {
            "task": "Propose strategy_state updates only. Do not place orders.",
            "output_contract": {
                "strategy_updates": "object containing only existing strategy fields",
                "confidence": "optional float",
                "reason": "optional short explanation",
            },
            "profile": profile_to_dict(profile),
            "memory": dict(memory or observation.get("memory", {}) or {}),
            "observation": dict(observation),
            "current_strategy": self._strategy_to_dict(current_strategy),
        }

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
        allowed_fields = self._allowed_fields(current_strategy)
        clean_updates = {
            key: value
            for key, value in updates.items()
            if key in allowed_fields
        }

        if "confidence" in response and "confidence" in allowed_fields:
            clean_updates["confidence"] = response["confidence"]
        if "reason" in response and "reason" in allowed_fields:
            clean_updates["reason"] = response["reason"]
        if "updated_at" in allowed_fields:
            clean_updates.setdefault("updated_at", observation.get("current_time"))

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
            return set(self.allowed_strategy_fields)
        if is_dataclass(current_strategy):
            return {field.name for field in fields(current_strategy)}
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
            input=(
                "Return JSON only using the requested strategy update contract.\n\n"
                f"Context JSON:\n{json.dumps(context, default=str)}"
            ),
            temperature=self.temperature,
            text={"format": {"type": "json_object"}},
        )
        content = getattr(response, "output_text", None)
        if not content:
            raise LLMStrategyResponseError("OpenAI returned an empty strategy response.")
        self._write_response_log(context=context, content=content)
        return content

    def _write_response_log(self, *, context: Mapping[str, Any], content: str) -> None:
        log_dir = os.getenv("LOG_DIR")
        if not log_dir:
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

        output_dir = os.path.join(log_dir, "llm_responses")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{safe_agent_id}.jsonl")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": "openai",
            "model": self.model,
            "agent_id": agent_id,
            "simulation_time": observation.get("current_time"),
            "response": content,
        }
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


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
