"""AML shock/event injector agent."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Mapping, Optional

from aml_sim.shocks import (
    build_shock_payload,
    safe_float,
    safe_int,
    stable_seed,
)
from agents.agent import Agent
from utils.messages import MessageType
from utils.time_utils import parse_datetime_utc


class AMLShockAgent(Agent):
    """
    Emits synthetic market shocks to AML agents during a simulation.

    Shock messages use StockSim's existing STATUS_UPDATE message type so the
    StockSim submodule does not need a new enum value for AML-owned events.
    """

    def __init__(
        self,
        scheduled_events: Optional[list[Mapping[str, Any]]] = None,
        random_events: Optional[Mapping[str, Any]] = None,
        target_agent_ids: Optional[list[str]] = None,
        instrument_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
        initial_market_state: Optional[Mapping[str, Any]] = None,
        default_duration_ticks: int = 10,
        random_seed: Optional[int] = None,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = "localhost",
        **_: Any,
    ) -> None:
        super().__init__(agent_id=agent_id, rabbitmq_host=rabbitmq_host)
        self.scheduled_events = [dict(event) for event in (scheduled_events or [])]
        self.random_events_config = dict(random_events or {})
        self.target_agent_ids = list(target_agent_ids or [])
        self.instrument_metadata = {
            str(key): dict(value)
            for key, value in (instrument_metadata or {}).items()
            if isinstance(value, Mapping)
        }
        self.market_state = dict(initial_market_state or {})
        self.default_duration_ticks = default_duration_ticks
        self.emitted_event_ids: set[str] = set()
        self.announced_event_ids: set[str] = set()
        self.random_event_count = 0
        self.random = random.Random(
            random_seed
            if random_seed is not None
            else stable_seed(self.agent_id or "shock_agent", len(self.scheduled_events))
        )

    async def _handle_regular_message(self, msg: dict[str, Any]) -> None:
        self.logger.debug(f"AMLShockAgent ignored message: {msg.get('type')}")

    async def handle_time_tick(self, payload: dict[str, Any]) -> None:
        await super().handle_time_tick(payload)
        for index, event in enumerate(self.scheduled_events):
            event_id = str(event.get("id") or event.get("shock_id") or f"shock_{index}")
            if (
                event_id not in self.announced_event_ids
                and event_id not in self.emitted_event_ids
                and self._announcement_due(event, payload)
            ):
                await self._emit_event(
                    event_id,
                    event,
                    payload,
                    phase="announcement",
                    trigger_type=str(event.get("trigger_type", "scheduled")),
                )
                self.announced_event_ids.add(event_id)

            if event_id in self.emitted_event_ids:
                continue
            if self._event_due(event, payload):
                await self._emit_event(
                    event_id,
                    event,
                    payload,
                    phase="active",
                    trigger_type=str(event.get("trigger_type", "scheduled")),
                )

        for event_id, event in self._random_events_due(payload):
            await self._emit_event(
                event_id,
                event,
                payload,
                phase="active",
                trigger_type=str(event.get("trigger_type", "unexpected")),
            )

    def _event_due(self, event: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
        tick_id = payload.get("tick_id")
        event_tick = event.get("tick", event.get("tick_id"))
        if event_tick is not None and tick_id is not None:
            try:
                return int(tick_id) >= int(event_tick)
            except (TypeError, ValueError):
                return False

        event_time = event.get("at") or event.get("time")
        current_time = payload.get("current_time")
        if event_time and current_time:
            try:
                return parse_datetime_utc(str(current_time)) >= parse_datetime_utc(str(event_time))
            except ValueError:
                return False

        return False

    def _announcement_due(
        self,
        event: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> bool:
        if self._event_due(event, payload):
            return False

        tick_id = safe_int(payload.get("tick_id"))
        announce_tick = safe_int(event.get("announce_tick", event.get("announcement_tick")))
        if tick_id is not None and announce_tick is not None:
            return tick_id >= announce_tick

        notice_ticks = safe_int(event.get("notice_ticks"))
        event_tick = safe_int(event.get("tick", event.get("tick_id")))
        if tick_id is not None and notice_ticks is not None and event_tick is not None:
            return tick_id >= max(0, event_tick - notice_ticks)

        announce_time = event.get("announce_at") or event.get("announcement_time")
        current_time = payload.get("current_time")
        if announce_time and current_time:
            try:
                return parse_datetime_utc(str(current_time)) >= parse_datetime_utc(str(announce_time))
            except ValueError:
                return False

        return False

    async def _emit_event(
        self,
        event_id: str,
        event: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        phase: str,
        trigger_type: str,
    ) -> None:
        shock_payload = build_shock_payload(
            event_id,
            event,
            payload,
            default_duration_ticks=self.default_duration_ticks,
            instrument_metadata=self.instrument_metadata,
            market_state=self.market_state,
            phase=phase,
            trigger_type=trigger_type,
        )
        if phase == "active":
            self._apply_market_state_update(event, shock_payload)
            shock_payload["market_state"] = deepcopy(self.market_state)

        if not self.target_agent_ids:
            self.logger.warning(f"AMLShockAgent has no targets for shock {event_id}")
            if phase == "active":
                self.emitted_event_ids.add(event_id)
            return

        for target_agent_id in self.target_agent_ids:
            await self.send_message(
                target_agent_id,
                MessageType.STATUS_UPDATE,
                shock_payload,
            )

        if phase == "active":
            self.emitted_event_ids.add(event_id)
        self.logger.info(
            f"AMLShockAgent emitted {phase} {event_id} to {len(self.target_agent_ids)} targets: "
            f"type={shock_payload['shock_type']}, class={shock_payload['shock_class']}, "
            f"severity={shock_payload['severity']}, direction={shock_payload['direction']}"
        )

    def _random_events_due(
        self,
        payload: Mapping[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        config = self.random_events_config
        if not config or not bool(config.get("enabled", False)):
            return []

        max_events = safe_int(config.get("max_events"), 0)
        if max_events is not None and max_events > 0 and self.random_event_count >= max_events:
            return []

        tick_id = safe_int(payload.get("tick_id"))
        start_tick = safe_int(config.get("start_tick"))
        end_tick = safe_int(config.get("end_tick"))
        if tick_id is not None:
            if start_tick is not None and tick_id < start_tick:
                return []
            if end_tick is not None and tick_id > end_tick:
                return []

        probability = safe_float(
            config.get("probability_per_tick", config.get("probability", 0.0)),
            0.0,
        )
        if probability <= 0 or self.random.random() > min(1.0, probability):
            return []

        template = self._pick_random_template(config.get("templates", []))
        if template is None:
            return []

        event = deepcopy(template)
        event.setdefault("trigger_type", "unexpected")
        event.setdefault("scheduled", False)
        event.setdefault("surprise", 1.0)
        self._materialize_random_ranges(event)
        event["tick"] = tick_id
        event_id = str(
            event.get("id")
            or event.get("shock_id")
            or f"random_{event.get('shock_type', 'shock')}_{tick_id}_{self.random_event_count + 1}"
        )
        self.random_event_count += 1
        return [(event_id, event)]

    def _pick_random_template(self, templates: Any) -> dict[str, Any] | None:
        if not isinstance(templates, list) or not templates:
            return None
        clean_templates = [template for template in templates if isinstance(template, Mapping)]
        if not clean_templates:
            return None
        weights = [max(0.0, safe_float(template.get("weight", 1.0), 1.0)) for template in clean_templates]
        if sum(weights) <= 0:
            return dict(self.random.choice(clean_templates))
        return dict(self.random.choices(clean_templates, weights=weights, k=1)[0])

    def _materialize_random_ranges(self, event: dict[str, Any]) -> None:
        severity_range = event.pop("severity_range", None)
        if isinstance(severity_range, (list, tuple)) and len(severity_range) >= 2:
            low = safe_float(severity_range[0], 0.0)
            high = safe_float(severity_range[1], low)
            event["severity"] = self.random.uniform(min(low, high), max(low, high))

        direction_choices = event.pop("direction_choices", None)
        if isinstance(direction_choices, (list, tuple)) and direction_choices:
            event["direction"] = self.random.choice(direction_choices)

    def _apply_market_state_update(
        self,
        event: Mapping[str, Any],
        shock_payload: Mapping[str, Any],
    ) -> None:
        absolute_state = event.get("market_state", event.get("state", {}))
        if isinstance(absolute_state, Mapping):
            self.market_state.update(deepcopy(dict(absolute_state)))

        state_delta = event.get("market_state_delta", event.get("state_delta", {}))
        if isinstance(state_delta, Mapping):
            for key, value in state_delta.items():
                self._add_market_state_delta(str(key), value)

        self._add_market_state_delta("policy_rate_bps", shock_payload.get("rate_shift_bps", 0.0))
        self._add_market_state_delta("yield_curve_shift_bps", shock_payload.get("yield_shift_bps", 0.0))
        self._add_market_state_delta("funding_spread_bps", shock_payload.get("funding_spread_bps", 0.0))
        self._add_market_state_delta("credit_spread_bps", shock_payload.get("credit_spread_bps", 0.0))
        self._add_market_state_delta("risk_aversion", shock_payload.get("risk_aversion_shift", 0.0))

        liquidity_multiplier = safe_float(shock_payload.get("liquidity_multiplier", 1.0), 1.0)
        if liquidity_multiplier != 1.0:
            current_liquidity = safe_float(self.market_state.get("liquidity_index", 1.0), 1.0)
            self.market_state["liquidity_index"] = round(current_liquidity * liquidity_multiplier, 4)

    def _add_market_state_delta(self, key: str, value: Any) -> None:
        delta = safe_float(value, 0.0)
        if delta == 0.0:
            return
        current = safe_float(self.market_state.get(key, 0.0), 0.0)
        self.market_state[key] = round(current + delta, 4)
