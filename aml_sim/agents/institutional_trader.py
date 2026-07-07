"""
AML Institutional Trader.

Models a larger participant that uses pluggable alpha strategies (momentum,
mean_reversion, breakout, volatility_regime, event_driven, etc.) to build or
reduce target positions in child orders over time.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from aml_sim.agents.base import BaseAMLAgent
from aml_sim.agents.context.memory import MemoryBackend
from aml_sim.agents.context.observation import ObservationProcessor
from aml_sim.agents.models.profile import InstitutionalProfile, coerce_profile
from aml_sim.agents.strategy.alpha import AlphaContext, AlphaSignal
from aml_sim.agents.strategy.composite import CompositeAlphaStrategy
from aml_sim.agents.strategy.llm_slow_strategy import (
    SlowStrategist,
    create_static_institutional_llm_strategist,
)
from aml_sim.agents.strategy.performance import (
    StrategyPerformance,
    create_performance_tracker,
)
from aml_sim.agents.strategy.registry import StrategyRegistry
from aml_sim.agents.models.state import InstitutionalStrategyState
from aml_sim.agents.strategy.signals import (
    clamp,
    event_pressure,
    price_series,
    target_from_signal,
)
from utils.orders import OrderType, Side


class AMLInstitutionalTrader(BaseAMLAgent):
    """
    Institutional participant with pluggable alpha strategies.

    Instead of hardcoded if/elif for momentum/mean_reversion, this agent
    builds a CompositeAlphaStrategy from the registry using the strategy
    names and weights in its strategy state. The LLM slow-loop can add,
    remove, and re-weight strategies at runtime.
    """

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        target_positions: Optional[Dict[str, int]] = None,
        child_order_size: int = 100,
        order_type: str = OrderType.MARKET.value,
        limit_price: Optional[float] = None,
        alpha_strategy: str = "target_execution",
        alpha_strategies: Optional[list[str]] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
        blend_mode: str = "weighted_sum",
        lookback_ticks: int = 5,
        entry_threshold: float = 0.002,
        exit_threshold: float = 0.0005,
        max_position: int = 500,
        min_position: int = 0,
        shock_reactivity: float = 0.5,
        profile: Optional[InstitutionalProfile | Mapping[str, Any]] = None,
        memory: Optional[MemoryBackend] = None,
        observation_processor: Optional[ObservationProcessor] = None,
        slow_loop_interval_seconds: Optional[int] = None,
        slow_strategist: Optional[SlowStrategist] = None,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = "localhost",
        **kwargs,
    ) -> None:
        trader_kwargs = {}
        for param in [
            "initial_cash",
            "initial_positions",
            "initial_cost_basis",
            "action_interval_seconds",
        ]:
            if param in kwargs:
                trader_kwargs[param] = kwargs[param]

        normalized_strategies = self._normalize_alpha_strategies(
            alpha_strategy, alpha_strategies
        )
        normalized_weights = dict(strategy_weights or {})

        strategy_performance = create_performance_tracker(
            normalized_strategies + ["target_execution"]
        )

        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            strategy_state=InstitutionalStrategyState(
                target_positions=dict(target_positions or {}),
                child_order_size=child_order_size,
                order_type=order_type.upper(),
                limit_price=limit_price,
                alpha_strategy=alpha_strategy,
                alpha_strategies=normalized_strategies,
                strategy_weights=normalized_weights,
                blend_mode=blend_mode,
                lookback_ticks=lookback_ticks,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                max_position=max_position,
                min_position=min_position,
                shock_reactivity=shock_reactivity,
            ),
            profile=coerce_profile(profile, InstitutionalProfile),
            memory=memory,
            observation_processor=observation_processor,
            slow_strategist=slow_strategist or create_static_institutional_llm_strategist(),
            slow_loop_interval_seconds=slow_loop_interval_seconds,
            strategy_performance=strategy_performance,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **trader_kwargs,
        )

        self.logger.info(
            f"AMLInstitutionalTrader {self.agent_id} initialized: "
            f"strategy_state={self.strategy_state}"
        )

    # ------------------------------------------------------------------
    # Fast loop
    # ------------------------------------------------------------------

    async def run_fast_loop(self, observation: Mapping[str, Any]) -> None:
        for instrument in self.instrument_exchange_map.keys():
            context = self._build_alpha_context(instrument, observation)
            signal = self._generate_composite_signal(context)
            self._update_target_from_signal(instrument, signal, context)
            await self._execute_toward_target(instrument)

    # ------------------------------------------------------------------
    # Alpha context & signal generation
    # ------------------------------------------------------------------

    def _build_alpha_context(
        self, instrument: str, observation: Mapping[str, Any]
    ) -> AlphaContext:
        fallback_price = self.prices.get(instrument, 0)
        prices = price_series(self.price_history, instrument, fallback_price)
        events = list(observation.get("events", []))
        volume_history: list[float] = []

        # Try to extract volume from market snapshots
        market = observation.get("market", {})
        last_snapshot = market.get("last_market_snapshot", {})
        inst_data = last_snapshot.get(instrument, {})
        if isinstance(inst_data, dict):
            vol = inst_data.get("volume")
            if isinstance(vol, (int, float)):
                volume_history = [float(vol)]

        return AlphaContext(
            prices=prices,
            volume_history=volume_history,
            current_position=(
                self.long_qty[instrument] - self.short_qty[instrument]
            ),
            portfolio_value=self.portfolio_value or 0.0,
            events=events,
            current_strategy=self.strategy_state,
            profile=self._traits,
            instrument=instrument,
            last_market_snapshot=last_snapshot,
        )

    def _generate_composite_signal(self, context: AlphaContext) -> AlphaSignal:
        """Build a CompositeAlphaStrategy from the registry and generate a signal."""
        strategy = self.strategy_state
        active_names = list(strategy.alpha_strategies or [strategy.alpha_strategy])
        active_names = [
            name for name in active_names
            if name.lower() != "target_execution"
        ]

        if not active_names:
            return AlphaSignal(reason="institutional: no active alpha strategies")

        # Build composite strategy from registry
        strategies: list[tuple[Any, float]] = []
        for name in active_names:
            strategy_cls = StrategyRegistry.get(name)
            if strategy_cls is None:
                self.logger.warning(
                    f"Strategy {name!r} not found in registry; skipping."
                )
                continue
            weight = float(strategy.strategy_weights.get(name, 1.0))
            # Instantiate — each strategy class is a dataclass with defaults
            try:
                instance = strategy_cls()
            except TypeError:
                # Some strategies need constructor params — try default
                instance = strategy_cls(lookback_ticks=strategy.lookback_ticks) if name in ("momentum", "mean_reversion", "breakout", "volatility_regime") else strategy_cls()
            strategies.append((instance, weight))

        if not strategies:
            return AlphaSignal(reason="institutional: no strategies could be loaded")

        composite = CompositeAlphaStrategy(
            strategies=strategies,
            blend_mode=getattr(strategy, "blend_mode", "weighted_sum"),
        )

        signal = composite.generate(context)

        # Record performance
        if signal.is_actionable and signal.strength > 0:
            for name in active_names:
                tracker = self.strategy_performance.get(name)
                if tracker:
                    tracker.record_signal(
                        signal.direction,
                        self.current_time.isoformat() if self.current_time else None,
                    )

        return signal

    def _update_target_from_signal(
        self,
        instrument: str,
        signal: AlphaSignal,
        context: AlphaContext,
    ) -> None:
        """Convert the composite alpha signal into a target position."""
        strategy = self.strategy_state

        # Apply shock pressure on top of the alpha signal
        pressure = event_pressure(context.events, instrument)
        combined_signal = signal.direction * signal.strength
        combined_signal += pressure["directional_bias"] * strategy.shock_reactivity * strategy.entry_threshold
        if context.prices and context.prices[-1] > 0:
            combined_signal += (
                pressure["fundamental_price_shift"]
                / context.prices[-1]
                * strategy.shock_reactivity
            )

        effective_max = max(
            strategy.min_position,
            int(strategy.max_position * pressure["risk_limit_multiplier"]),
        )
        effective_min = min(strategy.min_position, effective_max)

        current_target = strategy.target_positions.get(instrument, 0)
        active_names = [
            n for n in (strategy.alpha_strategies or [])
            if n.lower() != "target_execution"
        ]

        if "target_execution" in (strategy.alpha_strategies or []) and abs(combined_signal) <= strategy.exit_threshold:
            next_target = min(current_target, effective_max)
        else:
            next_target = target_from_signal(
                combined_signal,
                current_target=current_target,
                entry_threshold=strategy.entry_threshold,
                exit_threshold=strategy.exit_threshold,
                max_position=effective_max,
                min_position=effective_min,
            )

        strategy.signal_strength = combined_signal
        strategy.target_positions[instrument] = next_target

        if next_target != current_target:
            self.logger.info(
                f"AMLInstitutionalTrader {self.agent_id} alpha target update for {instrument}: "
                f"signal={combined_signal:.6f}, reason={signal.reason}, "
                f"target {current_target} -> {next_target}"
            )

        # Record acted for active strategies
        if signal.is_actionable:
            for name in active_names:
                tracker = self.strategy_performance.get(name)
                if tracker:
                    tracker.record_acted()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute_toward_target(self, instrument: str) -> None:
        strategy = self.strategy_state
        target = strategy.target_positions.get(instrument, 0)
        current = self.long_qty[instrument] - self.short_qty[instrument]
        gap = target - current

        if gap == 0:
            return

        side = Side.BUY.value if gap > 0 else Side.SELL.value
        pressure = event_pressure(self._active_events(), instrument)
        child_size = strategy.child_order_size
        child_size *= clamp(pressure["order_arrival_multiplier"], 0.25, 2.0)
        child_size *= pressure["risk_limit_multiplier"]
        quantity = min(abs(gap), max(1, int(child_size)))

        if side == Side.SELL.value:
            held = self.long_qty[instrument]
            if held <= 0:
                self.logger.debug(f"Institutional trader skipped SELL for {instrument}: no inventory")
                return
            quantity = min(quantity, held)

        price = strategy.limit_price if str(strategy.order_type).upper() == "LIMIT" else None
        order_id = await self.place_order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=strategy.order_type,
            price=price,
            explanation=f"AML institutional child order toward target {target}",
        )
        if order_id:
            self.logger.info(
                f"AMLInstitutionalTrader {self.agent_id} placed {side} "
                f"{strategy.order_type} order for {quantity} {instrument} "
                f"(current={current}, target={target})"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_alpha_strategies(
        alpha_strategy: str,
        alpha_strategies: Optional[list[str]],
    ) -> list[str]:
        strategies = [
            str(item).lower()
            for item in (alpha_strategies or [alpha_strategy])
            if str(item).strip()
        ]
        primary = str(alpha_strategy or "target_execution").lower()
        if primary and primary not in strategies:
            strategies.insert(0, primary)
        return strategies or ["target_execution"]
