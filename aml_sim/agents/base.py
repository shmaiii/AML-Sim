"""Base AML trading agent with shared fast/slow loop orchestration."""

from __future__ import annotations

import inspect
import json
import os
from abc import abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from typing import Any, Callable, Mapping, Optional

from aml_sim.agents.context.memory import LocalAgentMemory, MemoryBackend
from aml_sim.agents.context.observation import ObservationProcessor
from aml_sim.agents.models.profile import AgentProfile
from aml_sim.agents.strategy.llm_slow_strategy import SlowStrategist, create_llm_strategist
from aml_sim.agents.strategy.signals import event_pressure
from aml_sim.agents.strategy.validator import StrategyValidationError, validate_strategy_state
from aml_sim.serialization import serialize_value
from agents.benchmark_traders.trader import TraderAgent
from utils.messages import MessageType


class BaseAMLAgent(TraderAgent):
    """
    Shared AML agent loop on top of StockSim's TraderAgent.

    StockSim owns messaging, execution, accounting, portfolio state, and order
    state. AML owns profile, memory, observation packaging, strategy updates,
    validation, and role-specific fast execution policy.
    """

    LLM_STRATEGY_ROLE: str | None = None

    def __init__(
        self,
        instrument_exchange_map: dict[str, str],
        strategy_state: Any,
        *,
        profile: Optional[AgentProfile | Mapping[str, Any]] = None,
        memory: Optional[MemoryBackend] = None,
        observation_processor: Optional[ObservationProcessor] = None,
        slow_strategist: Optional[SlowStrategist | Mapping[str, Any]] = None,
        strategy_validator: Optional[Callable[[Any], Any]] = None,
        slow_loop_interval_seconds: Optional[int] = None,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = "localhost",
        **trader_kwargs: Any,
    ) -> None:
        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **trader_kwargs,
        )

        self.profile = profile or {}
        self.memory = memory or LocalAgentMemory()
        self.observation_processor = observation_processor or ObservationProcessor()
        self.slow_strategist = self._build_slow_strategist(slow_strategist)
        self.strategy_validator = strategy_validator or validate_strategy_state
        self.strategy_state = self._validate_or_keep(strategy_state, is_initial=True)

        if slow_loop_interval_seconds is None:
            self.logger.warning(
                f"{self.agent_id} slow_loop_interval not configured; "
                f"defaulting to action_interval ({self.action_interval}). "
                f"Set slow_loop_interval in the scenario YAML to control strategy update cadence."
            )
        self.slow_loop_interval = timedelta(
            seconds=slow_loop_interval_seconds or self.action_interval.total_seconds()
        )
        self.next_slow_loop_time = None
        self.action_events: list[dict[str, Any]] = []
        self.recent_events: list[dict[str, Any]] = []

        self.slow_loop_seen_event_ids: set[Any] = set()
        self.market_state: dict[str, Any] = {}
        self.market_state_baseline: dict[str, Any] = {}

        self.price_history: dict[str, list[dict[str, Any]]] = {
            instrument: [] for instrument in self.instrument_exchange_map.keys()
        }
        self.logger.info(
            f"{self.agent_id} slow loop configured: "
            f"strategist={type(self.slow_strategist).__name__}, "
            f"interval={self.slow_loop_interval}"
        )

    def _build_slow_strategist(
        self,
        slow_strategist: Optional[SlowStrategist | Mapping[str, Any]],
    ) -> SlowStrategist:
        if isinstance(slow_strategist, Mapping):
            if not self.LLM_STRATEGY_ROLE:
                raise ValueError(
                    f"{type(self).__name__} must define LLM_STRATEGY_ROLE to build "
                    "a slow strategist from config."
                )
            return create_llm_strategist(self.LLM_STRATEGY_ROLE, slow_strategist)
        if slow_strategist is not None:
            return slow_strategist
        if not self.LLM_STRATEGY_ROLE:
            raise ValueError("BaseAMLAgent requires a slow_strategist.")
        return create_llm_strategist(self.LLM_STRATEGY_ROLE)

    async def handle_time_tick(self, payload: dict[str, Any]) -> None:
        await super().handle_time_tick(payload)

        current_time = self.current_time
        if current_time is None:
            return

        if self.next_action_time is None:
            self.next_action_time = current_time
        if self.next_slow_loop_time is None:
            self.next_slow_loop_time = current_time

        observation = self.build_observation()

        if self.slow_loop_due():
            await self.run_slow_loop(observation)
            self.next_slow_loop_time = current_time + self.slow_loop_interval
            observation = self.build_observation()

        if current_time >= self.next_action_time:
            await self.run_fast_loop(observation)
            self.next_action_time = current_time + self.action_interval

    def build_observation(self) -> dict[str, Any]:
        active_events = self._active_events()
        known_events = self._known_events()
        fresh_observation = self.observation_processor.build_context(
            self,
            profile=self.profile,
            events=active_events,
            known_events=known_events,
        )
        memory_context = self.memory.retrieve_context(
            self.agent_id,
            observation=fresh_observation,
        )
        return self.observation_processor.build_context(
            self,
            profile=self.profile,
            memory=memory_context,
            events=active_events,
            known_events=known_events,
        )

    def slow_loop_due(self) -> bool:
        return (
            self.current_time is not None
            and self.next_slow_loop_time is not None
            and self.current_time >= self.next_slow_loop_time
        )

    async def run_slow_loop(self, observation: Mapping[str, Any]) -> None:
        before = self._strategy_snapshot(self.strategy_state)
        try:
            proposal = self.slow_strategist.propose(
                observation,
                self.strategy_state,
                profile=self.profile,
                memory=observation.get("memory", {}),
            )
            if inspect.isawaitable(proposal):
                proposal = await proposal

            self.strategy_state = self._validate_or_keep(proposal)
        except StrategyValidationError:
            # Already logged in _validate_or_keep; propagate only if initial.
            # For the slow loop, the previous state is kept inside _validate_or_keep.
            pass
        except Exception as exc:
            self.logger.error(
                f"{self.agent_id} slow loop failed; keeping current strategy: {exc}"
            )

        after = self._strategy_snapshot(self.strategy_state)
        self._remember_slow_loop_decision(
            observation=observation,
            before=before,
            after=after,
        )
        self._mark_observed_events_seen_by_slow_loop(observation)
        self.logger.info(
            f"{self.agent_id} slow loop completed: "
            f"strategist={type(self.slow_strategist).__name__}, "
            f"before={before}, after={after}"
        )

    def _remember_slow_loop_decision(
        self,
        *,
        observation: Mapping[str, Any],
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> None:
        event_context = observation.get("event_context", {})
        if not isinstance(event_context, Mapping):
            event_context = {}

        active_events = self._as_event_list(event_context.get("active"))
        known_events = self._as_event_list(event_context.get("known"))
        changed = dict(before) != dict(after)
        confidence = self._numeric_strategy_field(after, "confidence")
        low_confidence = confidence is not None and confidence <= 0.3
        unseen_active_events = [
            event
            for event in active_events
            if not bool(event.get("seen_before", False))
        ]

        if not changed and not unseen_active_events and not low_confidence:
            return

        primary_event = self._primary_memory_event(active_events, known_events)
        payload = {
            "strategy_changed": changed,
            "event_context_present": bool(active_events or known_events),
            "possible_event_influence": changed and bool(active_events or known_events),
            "strategy_before": dict(before),
            "strategy_after": dict(after),
            "confidence": confidence,
            "reason": after.get("reason"),
            "active_event_ids": [
                self._event_memory_id(event) for event in active_events
            ],
            "unseen_active_event_ids": [
                self._event_memory_id(event) for event in unseen_active_events
            ],
            "known_event_ids": [
                self._event_memory_id(event) for event in known_events
            ],
            "primary_event": primary_event,
        }
        self.memory.add_event(
            self.agent_id,
            "slow_loop_decision",
            payload,
            timestamp=self.current_time,
        )

    def _as_event_list(self, events: Any) -> list[dict[str, Any]]:
        if not isinstance(events, list):
            return []
        return [
            dict(event)
            for event in events
            if isinstance(event, Mapping)
        ]

    def _mark_observed_events_seen_by_slow_loop(
        self,
        observation: Mapping[str, Any],
    ) -> None:
        event_context = observation.get("event_context", {})
        if not isinstance(event_context, Mapping):
            return

        for event in self._as_event_list(event_context.get("active")):
            event_id = self._event_memory_id(event)
            if event_id is not None:
                self.slow_loop_seen_event_ids.add(event_id)
                self._mark_recent_event_seen(event_id)

    def _mark_recent_event_seen(self, event_id: Any) -> None:
        for event in self.recent_events:
            if self._event_memory_id(event) == event_id:
                event["seen_before"] = True

    def _primary_memory_event(
        self,
        active_events: list[dict[str, Any]],
        known_events: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if active_events:
            event = active_events[-1]
        elif known_events:
            event = known_events[-1]
        else:
            event = None
        if event is None:
            return None
        return {
            "event_id": self._event_memory_id(event),
            "shock_type": event.get("shock_type"),
            "shock_class": event.get("shock_class"),
            "scope": event.get("scope"),
            "phase": event.get("phase"),
            "severity": event.get("severity"),
            "direction": event.get("direction"),
            "message": event.get("message"),
        }

    def _event_memory_id(self, event: Mapping[str, Any]) -> Any:
        return (
            event.get("shock_id")
            or event.get("id")
            or event.get("event_id")
            or event.get("emitted_tick_id")
            or event.get("tick_id")
        )

    def _numeric_strategy_field(
        self,
        strategy_snapshot: Mapping[str, Any],
        field_name: str,
    ) -> Optional[float]:
        try:
            value = strategy_snapshot.get(field_name)
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _validate_or_keep(self, proposal: Any, *, is_initial: bool = False) -> Any:
        try:
            validator = self.strategy_validator
            if hasattr(validator, "validate"):
                return validator.validate(proposal)
            return validator(proposal)
        except StrategyValidationError as exc:
            if is_initial:
                raise StrategyValidationError(
                    f"Initial strategy state for {self.agent_id} is invalid and "
                    f"cannot be accepted: {exc}"
                ) from exc
            self.logger.warning(
                f"Rejected strategy proposal for {self.agent_id}; keeping previous state: {exc}"
            )
            return self.strategy_state

    def _strategy_snapshot(self, strategy_state: Any) -> dict[str, Any]:
        if is_dataclass(strategy_state):
            return asdict(strategy_state)
        if isinstance(strategy_state, Mapping):
            return dict(strategy_state)
        return dict(vars(strategy_state))

    async def _handle_regular_message(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == MessageType.STATUS_UPDATE.value:
            payload = msg.get("payload", {}) or {}
            if payload.get("event_type") in {"AML_SHOCK", "AML_MARKET_EVENT"}:
                self._handle_aml_event(payload)
                return
            if payload.get("event_type") == "AML_MARKET_STATE":
                self._handle_market_state_update(payload)
                return

        await super()._handle_regular_message(msg)

    async def _handle_portfolio_update(self, payload: dict[str, Any]) -> None:
        await super()._handle_portfolio_update(payload)
        self._record_price(payload.get("instrument"), payload.get("close_price"))

    async def place_order(
        self,
        instrument: str,
        side: str,
        quantity: int,
        order_type: str,
        price: Optional[float] = None,
        oco_group: Optional[str] = None,
        explanation: Optional[str] = None,
        is_short: bool = False,
        is_short_cover: bool = False,
    ) -> Optional[str]:
        before = self._portfolio_snapshot()
        order_id = await super().place_order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            oco_group=oco_group,
            explanation=explanation,
            is_short=is_short,
            is_short_cover=is_short_cover,
        )
        self._record_action_event(
            {
                "event_type": "order_submitted" if order_id else "order_rejected",
                "order_id": order_id,
                "instrument": instrument,
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
                "explanation": explanation,
                "strategy_state": self._strategy_snapshot(self.strategy_state),
                "portfolio_before": before,
                "portfolio_after": self._portfolio_snapshot(),
            }
        )
        return order_id

    async def on_trade_execution(self, trade_data: dict[str, Any]) -> None:
        before = self._portfolio_snapshot()
        await super().on_trade_execution(trade_data)
        self._cleanup_completed_market_order(
            trade_data.get("order_id"),
            trade_data.get("order_status"),
        )
        self._record_price(trade_data.get("instrument"), trade_data.get("price"))
        self._record_action_event(
            {
                "event_type": "trade_executed",
                "order_id": trade_data.get("order_id"),
                "instrument": trade_data.get("instrument"),
                "role": trade_data.get("role"),
                "quantity": trade_data.get("quantity"),
                "price": trade_data.get("price"),
                "order_type": trade_data.get("order_type"),
                "order_status": trade_data.get("order_status"),
                "explanation": trade_data.get("explanation"),
                "raw_trade": dict(trade_data),
                "strategy_state": self._strategy_snapshot(self.strategy_state),
                "portfolio_before": before,
                "portfolio_after": self._portfolio_snapshot(),
            }
        )

    async def _handle_order_confirmation(self, payload: dict[str, Any]) -> None:
        await super()._handle_order_confirmation(payload)
        self._cleanup_completed_market_order(
            payload.get("order_id"),
            payload.get("status"),
        )

    def _cleanup_completed_market_order(self, order_id: Any, status: Any) -> None:
        if not order_id:
            return
        pending = self.pending_orders.get(str(order_id))
        if not pending:
            return
        if str(pending.get("order_type", "")).upper() != "MARKET":
            return
        normalized_status = str(status or "").upper()
        if normalized_status in {"FILLED", "PARTIALLY_FILLED", "CANCELED", "REJECTED"}:
            self.pending_orders.pop(str(order_id), None)

    def _record_action_event(self, event: dict[str, Any]) -> None:
        event.setdefault("agent_id", self.agent_id)
        event.setdefault("timestamp", self.current_time.isoformat() if self.current_time else None)
        self.action_events.append(serialize_value(event))

    def _handle_aml_event(self, event: Mapping[str, Any]) -> None:
        observed = serialize_value(dict(event))
        observed.setdefault("observed_at", self.current_time.isoformat() if self.current_time else None)
        observed.setdefault("observed_tick_id", self.current_tick_id)

        self._update_market_state_from_payload(observed)
        event_id = self._event_memory_id(observed)
        observed.setdefault(
            "seen_before",
            event_id is not None and event_id in self.slow_loop_seen_event_ids,
        )

        self.recent_events.append(observed)
        self.recent_events = self.recent_events[-50:]
        self._record_action_event(
            {
                "event_type": "event_observed",
                "shock_id": observed.get("shock_id"),
                "shock_type": observed.get("shock_type"),
                "shock_class": observed.get("shock_class"),
                "scope": observed.get("scope"),
                "phase": observed.get("phase"),
                "trigger_type": observed.get("trigger_type"),
                "visibility": observed.get("visibility"),
                "severity": observed.get("severity"),
                "direction": observed.get("direction"),
                "fundamental_price_shift": observed.get("fundamental_price_shift"),
                "order_arrival_multiplier": observed.get("order_arrival_multiplier"),
                "risk_limit_multiplier": observed.get("risk_limit_multiplier"),
                "liquidity_multiplier": observed.get("liquidity_multiplier"),
                "volatility_multiplier": observed.get("volatility_multiplier"),
                "spread_multiplier": observed.get("spread_multiplier"),
                "rate_shift_bps": observed.get("rate_shift_bps"),
                "yield_shift_bps": observed.get("yield_shift_bps"),
                "affected_instruments": observed.get("affected_instruments"),
                "affected_asset_classes": observed.get("affected_asset_classes"),
                "market_state": observed.get("market_state"),
                "market_state_baseline": observed.get("market_state_baseline"),
            }
        )
        self.logger.info(
            f"{self.agent_id} observed AML shock: "
            f"type={observed.get('shock_type')}, phase={observed.get('phase')}, "
            f"severity={observed.get('severity')}, "
            f"direction={observed.get('direction')}"
        )

    def _handle_market_state_update(self, payload: Mapping[str, Any]) -> None:
        previous_state = dict(self.market_state)
        self._update_market_state_from_payload(payload)
        if self.market_state == previous_state:
            return
        self._record_action_event(
            {
                "event_type": "market_state_updated",
                "market_state": self.market_state,
                "market_state_baseline": self.market_state_baseline,
            }
        )

    def _update_market_state_from_payload(self, payload: Mapping[str, Any]) -> None:
        state = payload.get("market_state")
        if isinstance(state, Mapping):
            self.market_state = dict(serialize_value(state))
        baseline = payload.get("market_state_baseline")
        if isinstance(baseline, Mapping):
            self.market_state_baseline = dict(serialize_value(baseline))

    def _active_events(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        current_tick = self.current_tick_id

        for event in self.recent_events:
            if str(event.get("phase", "active")).lower() not in {"active", "shock"}:
                continue
            duration = event.get("duration_ticks")
            start_tick = event.get("effective_tick_id", event.get("tick_id", event.get("emitted_tick_id")))
            expiry_tick = event.get("expiry_tick_id")
            if current_tick is None or duration is None or start_tick is None:
                active.append(event)
                continue
            try:
                current_tick_int = int(current_tick)
                start_tick_int = int(start_tick)
                expiry_tick_int = (
                    int(expiry_tick)
                    if expiry_tick is not None
                    else start_tick_int + int(duration)
                )
                if start_tick_int <= current_tick_int <= expiry_tick_int:
                    active.append(event)
            except (TypeError, ValueError):
                active.append(event)

        return active[-20:]

    def _market_pressure(self, instrument: str) -> dict[str, float]:
        """Combine live shock effects with the current central market state."""
        return event_pressure(
            self._active_events(),
            instrument,
            market_state=self.market_state,
            market_state_baseline=self.market_state_baseline,
        )

    def _known_events(self) -> list[dict[str, Any]]:
        return self.recent_events[-50:]

    def _record_price(self, instrument: Any, price: Any) -> None:
        if not instrument:
            return
        try:
            clean_price = float(price)
        except (TypeError, ValueError):
            return
        if clean_price <= 0:
            return

        series = self.price_history.setdefault(str(instrument), [])
        series.append(
            {
                "timestamp": self.current_time.isoformat() if self.current_time else None,
                "tick_id": self.current_tick_id,
                "price": clean_price,
            }
        )
        if len(series) > 250:
            del series[:-250]

    def _portfolio_snapshot(self) -> dict[str, Any]:
        instruments = list(self.instrument_exchange_map.keys())
        unrealized_pnl = self.get_unrealized_pnl()
        exposure = self.get_portfolio_exposure()
        return {
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "unrealized_pnl": round(sum(unrealized_pnl.values()), 2),
            "total_pnl": round(
                sum(self.realized_pnl.values()) + sum(unrealized_pnl.values()),
                2,
            ),
            **exposure,
            "positions": {
                instrument: {
                    "long": self.long_qty[instrument],
                    "short": self.short_qty[instrument],
                    "net": self.long_qty[instrument] - self.short_qty[instrument],
                    "last_price": self.prices[instrument],
                    "realized_pnl": self.realized_pnl[instrument],
                    "unrealized_pnl": unrealized_pnl.get(instrument, 0.0),
                    "total_pnl": round(
                        self.realized_pnl[instrument]
                        + unrealized_pnl.get(instrument, 0.0),
                        2,
                    ),
                    "market_value": round(
                        (self.long_qty[instrument] - self.short_qty[instrument])
                        * self.prices[instrument],
                        2,
                    ),
                }
                for instrument in instruments
            },
        }

    def stop(self) -> None:
        self._export_action_events()
        super().stop()

    def _export_action_events(self) -> None:
        output_dir = os.getenv("METRICS_OUTPUT_DIR", "metrics")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"trader_actions_{self.agent_id}.json")
        try:
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(self.action_events, handle, indent=2)
            self.logger.info(f"AML trader actions exported to {output_file}")
        except Exception as exc:
            self.logger.error(f"Failed to export AML trader actions: {exc}")

    @abstractmethod
    async def run_fast_loop(self, observation: Mapping[str, Any]) -> None:
        """Execute role-specific trading behavior from the current strategy."""
