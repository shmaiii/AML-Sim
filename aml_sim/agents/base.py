"""Base AML trading agent with shared fast/slow loop orchestration."""

from __future__ import annotations

import copy
import inspect
import json
import os
from abc import abstractmethod
from dataclasses import asdict, fields as dc_fields, is_dataclass, replace
from datetime import timedelta
from typing import Any, Callable, Mapping, Optional

from aml_sim.agents.context.memory import LocalAgentMemory, MemoryBackend
from aml_sim.agents.context.observation import ObservationProcessor
from aml_sim.agents.models.profile import AgentProfile, BehavioralTraits
from aml_sim.agents.profile.behavioral_dynamics import BehavioralDynamics, BehavioralState
from aml_sim.agents.profile.modulator import ProfileModulator
from aml_sim.agents.risk.fallback import FallbackChain
from aml_sim.agents.risk.risk_manager import HealthVerdict, OrderVerdict, RiskManager
from aml_sim.agents.strategy.llm_slow_strategy import SlowStrategist
from aml_sim.agents.strategy.performance import (
    StrategyPerformance,
    create_performance_tracker,
)
from aml_sim.agents.strategy.validator import StrategyValidationError, validate_strategy_state
from aml_sim.serialization import serialize_value
from agents.benchmark_traders.trader import TraderAgent
from utils.messages import MessageType


class BaseAMLAgent(TraderAgent):
    """
    Shared AML agent loop on top of StockSim's TraderAgent.

    StockSim owns messaging, execution, accounting, portfolio state, and order
    state. AML owns profile, memory, observation packaging, strategy updates,
    validation, risk management, fallback, behavioral dynamics, and role-
    specific fast execution policy.
    """

    def __init__(
        self,
        instrument_exchange_map: dict[str, str],
        strategy_state: Any,
        *,
        profile: Optional[AgentProfile | Mapping[str, Any]] = None,
        memory: Optional[MemoryBackend] = None,
        observation_processor: Optional[ObservationProcessor] = None,
        slow_strategist: Optional[SlowStrategist] = None,
        strategy_validator: Optional[Callable[[Any], Any]] = None,
        slow_loop_interval_seconds: Optional[int] = None,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = "localhost",
        # --- NEW: robustness ---
        risk_manager: Optional[RiskManager] = None,
        fallback_chain: Optional[FallbackChain] = None,
        # --- NEW: profile ---
        profile_modulator: Optional[ProfileModulator] = None,
        behavioral_dynamics: Optional[BehavioralDynamics] = None,
        # --- NEW: strategy ---
        strategy_performance: Optional[dict[str, StrategyPerformance]] = None,
        **trader_kwargs: Any,
    ) -> None:
        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **trader_kwargs,
        )

        # --- Profile ---------------------------------------------------------
        self.profile = profile or {}
        self._traits = BehavioralTraits()
        if isinstance(self.profile, AgentProfile):
            self._traits = self.profile.traits
        elif isinstance(self.profile, Mapping) and self.profile.get("behavioral_traits"):
            self._traits = BehavioralTraits.from_mapping(self.profile["behavioral_traits"])

        # --- Memory / observation ---------------------------------------------
        self.memory = memory or LocalAgentMemory()
        self.observation_processor = observation_processor or ObservationProcessor()

        # --- Strategy ---------------------------------------------------------
        if slow_strategist is None:
            raise ValueError("BaseAMLAgent requires a slow_strategist.")
        self.slow_strategist = slow_strategist
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

        # --- Robustness -------------------------------------------------------
        self.risk_manager = risk_manager or RiskManager()
        self.fallback_chain = fallback_chain or FallbackChain()
        self.fallback_chain.seed_safe_state(self.strategy_state)

        # --- Profile modulation -----------------------------------------------
        self.profile_modulator = profile_modulator or ProfileModulator()
        self.behavioral_dynamics = behavioral_dynamics or BehavioralDynamics()
        self.behavioral_state = BehavioralState()

        # --- Strategy performance tracking ------------------------------------
        strategy_names = (
            list(strategy_performance.keys())
            if strategy_performance
            else ["default"]
        )
        self.strategy_performance = strategy_performance or create_performance_tracker(strategy_names)

        # --- State ------------------------------------------------------------
        self.next_slow_loop_time = None
        self.action_events: list[dict[str, Any]] = []
        self.recent_events: list[dict[str, Any]] = []
        self.price_history: dict[str, list[dict[str, Any]]] = {
            instrument: [] for instrument in self.instrument_exchange_map.keys()
        }
        self.rejected_order_count: int = 0
        self._last_portfolio_value: float = 0.0

        self.logger.info(
            f"{self.agent_id} slow loop configured: "
            f"strategist={type(self.slow_strategist).__name__}, "
            f"interval={self.slow_loop_interval}"
        )

    # ------------------------------------------------------------------
    # Main tick loop
    # ------------------------------------------------------------------

    async def handle_time_tick(self, payload: dict[str, Any]) -> None:
        await super().handle_time_tick(payload)

        current_time = self.current_time
        if current_time is None:
            return

        if self.next_action_time is None:
            self.next_action_time = current_time
        if self.next_slow_loop_time is None:
            self.next_slow_loop_time = current_time

        # Seed risk manager with initial portfolio value
        portfolio_value = self.portfolio_value or self._last_portfolio_value
        self.risk_manager.seed_portfolio(portfolio_value)

        # --- Pre-tick health check ---
        health = self._pre_tick_health_check()
        if health == HealthVerdict.HALTED:
            self.logger.warning(
                f"{self.agent_id} risk health HALTED: {self.risk_manager.health_reason}"
            )
            return  # Skip fast AND slow loops this tick

        # --- Fallback check ---
        if self.fallback_chain.should_fallback():
            fallback_state = self.fallback_chain.get_fallback_state()
            if fallback_state is not None:
                self.logger.warning(
                    f"{self.agent_id} activating fallback strategy after "
                    f"{self.fallback_chain.failure_count} consecutive slow-loop failures"
                )
                self.strategy_state = fallback_state
                self.fallback_chain.restored_at = (
                    self.current_time.isoformat() if self.current_time else None
                )

        # --- Behavioral dynamics update ---
        self._update_behavioral_state()

        # --- Observation ---
        observation = self.build_observation()

        # --- Slow loop (LLM strategy update) ---
        if self.slow_loop_due():
            await self.run_slow_loop(observation)
            self.next_slow_loop_time = current_time + self.slow_loop_interval
            observation = self.build_observation()

        # --- Profile modulation ---
        mod_state = self.profile_modulator.modulate(
            self.strategy_state,
            self._traits,
            self.behavioral_state,
        )

        # --- Fast loop (execution) ---
        if health != HealthVerdict.HALTED and current_time >= self.next_action_time:
            await self.run_fast_loop(observation)
            self.next_action_time = current_time + self.action_interval

        # --- Update peak ---
        self.risk_manager.update_portfolio_peak(
            self.portfolio_value or self._last_portfolio_value
        )

    # ------------------------------------------------------------------
    # Health & behavioral
    # ------------------------------------------------------------------

    def _pre_tick_health_check(self) -> HealthVerdict:
        portfolio_value = self.portfolio_value or self._last_portfolio_value
        return self.risk_manager.check_portfolio_health(
            portfolio_value=portfolio_value,
            current_tick_id=self.current_tick_id or 0,
        )

    def _update_behavioral_state(self) -> None:
        pnl_delta = 0.0
        portfolio_value = self.portfolio_value or 0.0
        if portfolio_value > 0 and self._last_portfolio_value > 0:
            pnl_delta = portfolio_value - self._last_portfolio_value
        self._last_portfolio_value = portfolio_value

        # Check if any recent action event was a fill (trade execution)
        recent_actions = self.action_events[-10:] if self.action_events else []
        had_fill = any(
            a.get("event_type") == "trade_executed" for a in recent_actions
        )

        prices: list[float] = []
        for instrument in self.instrument_exchange_map.keys():
            series = self.price_history.get(str(instrument), [])
            if series:
                prices = [float(p.get("price", 0)) for p in series if p.get("price")]
                break

        self.behavioral_dynamics.update(
            self.behavioral_state,
            pnl_delta=pnl_delta,
            had_fill=had_fill,
            prices=prices,
        )

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def build_observation(self) -> dict[str, Any]:
        fresh_observation = self.observation_processor.build_context(
            self,
            profile=self.profile,
            events=self._active_events(),
        )
        memory_context = self.memory.retrieve_context(
            self.agent_id,
            observation=fresh_observation,
        )
        observation = self.observation_processor.build_context(
            self,
            profile=self.profile,
            memory=memory_context,
            events=self._active_events(),
        )
        # Inject new context fields
        observation.setdefault("risk_status", self.risk_manager.snapshot())
        observation.setdefault(
            "strategy_performance",
            {k: v.snapshot() for k, v in self.strategy_performance.items()},
        )
        observation.setdefault(
            "behavioral_state",
            self.behavioral_state.snapshot(),
        )
        return observation

    # ------------------------------------------------------------------
    # Slow loop
    # ------------------------------------------------------------------

    def slow_loop_due(self) -> bool:
        return (
            self.current_time is not None
            and self.next_slow_loop_time is not None
            and self.current_time >= self.next_slow_loop_time
        )

    async def run_slow_loop(self, observation: Mapping[str, Any]) -> None:
        before = self._strategy_snapshot(self.strategy_state)
        success = False
        try:
            proposal = self.slow_strategist.propose(
                observation,
                self.strategy_state,
                profile=self.profile,
                memory=observation.get("memory", {}),
            )
            if inspect.isawaitable(proposal):
                proposal = await proposal

            # --- Handle enhanced LLM response ---
            if isinstance(proposal, Mapping):
                # Apply strategy config changes (enable/disable strategies, blend mode)
                self._apply_strategy_config(proposal)
                # Apply parameter updates
                self.strategy_state = self._validate_or_keep(proposal)
                # Apply risk overrides
                self._apply_risk_overrides(proposal)
            else:
                # Legacy: proposal is a strategy state object/dataclass
                self.strategy_state = self._validate_or_keep(proposal)

            success = True
        except StrategyValidationError:
            pass
        except Exception as exc:
            self.logger.error(
                f"{self.agent_id} slow loop failed; keeping current strategy: {exc}"
            )
        finally:
            self.fallback_chain.record_result(success)

        after = self._strategy_snapshot(self.strategy_state)
        self.logger.info(
            f"{self.agent_id} slow loop completed: "
            f"strategist={type(self.slow_strategist).__name__}, "
            f"before={before}, after={after}"
        )

    def _apply_strategy_config(self, proposal: Mapping[str, Any]) -> None:
        """Apply strategy selection/composition directives from LLM response."""
        strategy_config = proposal.get("strategy_config")
        if not isinstance(strategy_config, Mapping):
            return
        # These are stored on the strategy_state for the composite strategy to use
        active = strategy_config.get("active_strategies")
        if isinstance(active, list):
            state = self.strategy_state
            if hasattr(state, "alpha_strategies"):
                state.alpha_strategies = active
            if hasattr(state, "alpha_strategy"):
                state.alpha_strategy = active[0] if active else state.alpha_strategy
        blend = strategy_config.get("blend_mode")
        if isinstance(blend, str) and hasattr(self.strategy_state, "blend_mode"):
            self.strategy_state.blend_mode = blend
        weights = strategy_config.get("strategy_weights")
        if isinstance(weights, Mapping) and hasattr(self.strategy_state, "strategy_weights"):
            self.strategy_state.strategy_weights = dict(weights)

    def _apply_risk_overrides(self, proposal: Mapping[str, Any]) -> None:
        """Apply risk parameter overrides from LLM response."""
        overrides = proposal.get("risk_overrides")
        if not isinstance(overrides, Mapping):
            return
        for key in ("max_drawdown_pct", "max_position_pct", "max_order_rate",
                     "max_consecutive_rejections", "cooldown_ticks"):
            if key in overrides and hasattr(self.risk_manager, key):
                try:
                    setattr(self.risk_manager, key, float(overrides[key]))
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_or_keep(self, proposal: Any, *, is_initial: bool = False) -> Any:
        # If proposal is a mapping (LLM response), extract parameter updates
        if isinstance(proposal, Mapping) and "parameter_updates" in proposal:
            param_updates = proposal["parameter_updates"]
            current = self.strategy_state
            if is_dataclass(current):
                allowed = {f.name for f in dc_fields(current)}
                clean = {k: v for k, v in param_updates.items() if k in allowed}
                if "confidence" not in clean and "confidence" in proposal:
                    clean["confidence"] = proposal["confidence"]
                if "reason" not in clean and "reason" in proposal:
                    clean["reason"] = proposal["reason"]
                if "updated_at" not in clean:
                    clean["updated_at"] = (
                        self.current_time.isoformat() if self.current_time else None
                    )
                proposal = replace(current, **clean)
            elif isinstance(current, Mapping):
                merged = dict(current)
                allowed = set(merged.keys())
                for k, v in param_updates.items():
                    if k in allowed:
                        merged[k] = v
                if "confidence" in proposal and "confidence" in allowed:
                    merged["confidence"] = proposal["confidence"]
                if "reason" in proposal and "reason" in allowed:
                    merged["reason"] = proposal["reason"]
                if "updated_at" in allowed:
                    merged.setdefault(
                        "updated_at",
                        self.current_time.isoformat() if self.current_time else None,
                    )
                proposal = merged
            else:
                # Non-dataclass, non-mapping: copy and set attributes
                proposal = copy.deepcopy(current)
                for k, v in param_updates.items():
                    if hasattr(proposal, k):
                        setattr(proposal, k, v)

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

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_regular_message(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == MessageType.STATUS_UPDATE.value:
            payload = msg.get("payload", {}) or {}
            if payload.get("event_type") == "AML_SHOCK":
                self._handle_aml_event(payload)
                return

        await super()._handle_regular_message(msg)

    async def _handle_portfolio_update(self, payload: dict[str, Any]) -> None:
        await super()._handle_portfolio_update(payload)
        self._record_price(payload.get("instrument"), payload.get("close_price"))

    # ------------------------------------------------------------------
    # Order placement — gated by risk manager
    # ------------------------------------------------------------------

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
        portfolio_value = self.portfolio_value or self._last_portfolio_value
        current_position = (
            self.long_qty.get(instrument, 0) - self.short_qty.get(instrument, 0)
        )

        verdict, adjusted_qty = self.risk_manager.check_order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            price=price,
            portfolio_value=portfolio_value,
            current_position=current_position,
            current_tick_id=self.current_tick_id or 0,
        )

        if verdict == OrderVerdict.REJECTED:
            self.rejected_order_count += 1
            self.risk_manager.record_order_result(False, current_tick_id=self.current_tick_id or 0)
            self._record_action_event(
                {
                    "event_type": "order_rejected",
                    "reject_reason": f"risk_manager: {self.risk_manager.health_reason or 'circuit_or_limit'}",
                    "instrument": instrument,
                    "side": side,
                    "quantity": quantity,
                    "order_type": order_type,
                    "price": price,
                    "strategy_state": self._strategy_snapshot(self.strategy_state),
                    "portfolio_before": self._portfolio_snapshot(),
                    "portfolio_after": self._portfolio_snapshot(),
                }
            )
            self.logger.debug(
                f"{self.agent_id} order rejected by risk manager: "
                f"{self.risk_manager.health_reason}"
            )
            return None

        if verdict == OrderVerdict.SIZE_REDUCED and adjusted_qty is not None:
            quantity = adjusted_qty

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
        self.risk_manager.record_order_result(order_id is not None, current_tick_id=self.current_tick_id or 0)
        return order_id

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Action recording
    # ------------------------------------------------------------------

    def _record_action_event(self, event: dict[str, Any]) -> None:
        event.setdefault("agent_id", self.agent_id)
        event.setdefault("timestamp", self.current_time.isoformat() if self.current_time else None)
        event.setdefault("risk_snapshot", self.risk_manager.snapshot())
        event.setdefault("fallback_snapshot", self.fallback_chain.snapshot())
        event.setdefault("behavioral_state", self.behavioral_state.snapshot())
        self.action_events.append(serialize_value(event))

    def _handle_aml_event(self, event: Mapping[str, Any]) -> None:
        observed = serialize_value(dict(event))
        observed.setdefault("observed_at", self.current_time.isoformat() if self.current_time else None)
        observed.setdefault("observed_tick_id", self.current_tick_id)
        self.recent_events.append(observed)
        self.recent_events = self.recent_events[-50:]
        self._record_action_event(
            {
                "event_type": "event_observed",
                "shock_id": observed.get("shock_id"),
                "shock_type": observed.get("shock_type"),
                "severity": observed.get("severity"),
                "direction": observed.get("direction"),
                "fundamental_price_shift": observed.get("fundamental_price_shift"),
                "order_arrival_multiplier": observed.get("order_arrival_multiplier"),
                "risk_limit_multiplier": observed.get("risk_limit_multiplier"),
                "liquidity_multiplier": observed.get("liquidity_multiplier"),
                "affected_instruments": observed.get("affected_instruments"),
            }
        )
        self.logger.info(
            f"{self.agent_id} observed AML shock: "
            f"type={observed.get('shock_type')}, severity={observed.get('severity')}, "
            f"direction={observed.get('direction')}"
        )

    def _active_events(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        current_tick = self.current_tick_id

        for event in self.recent_events:
            duration = event.get("duration_ticks")
            start_tick = event.get("tick_id", event.get("emitted_tick_id"))
            if current_tick is None or duration is None or start_tick is None:
                active.append(event)
                continue
            try:
                if int(current_tick) <= int(start_tick) + int(duration):
                    active.append(event)
            except (TypeError, ValueError):
                active.append(event)

        return active[-20:]

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
        return {
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "positions": {
                instrument: {
                    "long": self.long_qty[instrument],
                    "short": self.short_qty[instrument],
                    "net": self.long_qty[instrument] - self.short_qty[instrument],
                    "last_price": self.prices[instrument],
                    "realized_pnl": self.realized_pnl[instrument],
                }
                for instrument in instruments
            },
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._export_action_events()
        super().stop()

    def _export_action_events(self) -> None:
        output_dir = os.getenv("METRICS_OUTPUT_DIR", "metrics")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"trader_actions_{self.agent_id}.json")
        try:
            with open(output_file, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "agent_id": self.agent_id,
                        "risk_summary": self.risk_manager.snapshot(),
                        "fallback_summary": self.fallback_chain.snapshot(),
                        "behavioral_summary": self.behavioral_state.snapshot(),
                        "strategy_performance": {
                            k: v.snapshot() for k, v in self.strategy_performance.items()
                        },
                        "actions": self.action_events,
                    },
                    handle,
                    indent=2,
                )
            self.logger.info(f"AML trader actions exported to {output_file}")
        except Exception as exc:
            self.logger.error(f"Failed to export AML trader actions: {exc}")

    @abstractmethod
    async def run_fast_loop(self, observation: Mapping[str, Any]) -> None:
        """Execute role-specific trading behavior from the current strategy."""
