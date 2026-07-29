"""AML liquidity taker participant."""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Optional

from aml_sim.agents.base import BaseAMLAgent
from aml_sim.agents.context.memory import MemoryBackend
from aml_sim.agents.context.observation import ObservationProcessor
from aml_sim.agents.models.profile import LiquidityTakerProfile, coerce_profile
from aml_sim.agents.models.state import LiquidityTakerStrategyState
from aml_sim.agents.strategy.llm_slow_strategy import SlowStrategist
from aml_sim.agents.strategy.signals import clamp
from utils.orders import OrderType, Side


class AMLLiquidityTaker(BaseAMLAgent):
    """Aggressive flow participant that consumes available displayed liquidity."""

    LLM_STRATEGY_ROLE = "liquidity_taker"

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        flow_intensity: float = 0.35,
        buy_bias: float = 0.5,
        max_order_size: int = 40,
        inventory_limit: int = 500,
        shock_sensitivity: float = 0.7,
        aggression: float = 0.75,
        risk_mode: str = "normal",
        random_seed: Optional[int] = None,
        profile: Optional[LiquidityTakerProfile | Mapping[str, Any]] = None,
        memory: Optional[MemoryBackend] = None,
        observation_processor: Optional[ObservationProcessor] = None,
        slow_loop_interval_seconds: Optional[int] = None,
        slow_strategist: Optional[SlowStrategist | Mapping[str, Any]] = None,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = "localhost",
        **kwargs: Any,
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

        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            strategy_state=LiquidityTakerStrategyState(
                risk_mode=risk_mode,
                flow_intensity=flow_intensity,
                buy_bias=buy_bias,
                max_order_size=max(1, max_order_size),
                inventory_limit=inventory_limit,
                shock_sensitivity=shock_sensitivity,
                aggression=aggression,
            ),
            profile=coerce_profile(profile, LiquidityTakerProfile),
            memory=memory,
            observation_processor=observation_processor,
            slow_strategist=self._build_slow_strategist(slow_strategist),
            slow_loop_interval_seconds=slow_loop_interval_seconds,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **trader_kwargs,
        )
        self.random = random.Random(random_seed)
        self.logger.info(
            f"AMLLiquidityTaker {self.agent_id} initialized: "
            f"strategy_state={self.strategy_state}"
        )

    async def run_fast_loop(self, observation: Mapping[str, Any]) -> None:
        for instrument in self.instrument_exchange_map.keys():
            await self._maybe_take_liquidity(instrument)

    async def _maybe_take_liquidity(self, instrument: str) -> None:
        strategy = self.strategy_state
        pressure = self._market_pressure(instrument)
        risk_policy = self._risk_policy()
        participation = strategy.flow_intensity
        participation *= pressure["order_arrival_multiplier"]
        participation += pressure["severity"] * strategy.shock_sensitivity * 0.25
        participation *= 0.5 + (strategy.aggression * 0.5)
        participation *= risk_policy.participation_multiplier

        effective_participation = clamp(participation, 0.0, 1.0)
        buy_bias = strategy.buy_bias
        buy_bias += pressure["directional_bias"] * strategy.shock_sensitivity * 0.35
        effective_buy_bias = clamp(buy_bias, 0.0, 1.0)
        effective_order_cap = strategy.max_order_size
        effective_order_cap *= pressure["risk_limit_multiplier"]
        effective_order_cap *= clamp(
            pressure["order_arrival_multiplier"],
            0.25,
            2.0,
        )
        effective_order_cap *= risk_policy.order_size_multiplier
        effective_order_cap = max(1, int(effective_order_cap))
        current_position = (
            self.long_qty[instrument] - self.short_qty[instrument]
        )
        effective_limit = max(
            0,
            int(
                strategy.inventory_limit
                * pressure["risk_limit_multiplier"]
                * risk_policy.position_limit_multiplier
            ),
        )
        self._update_fast_loop_state(
            instrument,
            effective_participation_probability=effective_participation,
            effective_buy_probability=effective_buy_bias,
            effective_order_size_cap=effective_order_cap,
            effective_position_limit=effective_limit,
            position_limit_utilization=self._position_limit_utilization(
                current_position,
                effective_limit,
            ),
            buy_constrained=current_position >= effective_limit,
            sell_constrained=self.long_qty[instrument] <= 0,
            aggression=strategy.aggression,
            preferred_order_type=OrderType.MARKET.value,
        )
        if self.random.random() > effective_participation:
            return

        side = (
            Side.BUY.value
            if self.random.random() < effective_buy_bias
            else Side.SELL.value
        )
        quantity = self._order_quantity(
            instrument,
            side,
            max_size=effective_order_cap,
            inventory_limit=effective_limit,
        )
        if quantity <= 0:
            return

        order_id = await self.place_order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET.value,
            explanation="AML liquidity taker market flow",
        )
        if order_id:
            self.logger.info(
                f"AMLLiquidityTaker {self.agent_id} placed {side} "
                f"market order for {quantity} {instrument}"
            )

    def _order_quantity(
        self,
        instrument: str,
        side: str,
        *,
        max_size: int,
        inventory_limit: int,
    ) -> int:
        quantity = self.random.randint(1, max_size)

        current_position = self.long_qty[instrument] - self.short_qty[instrument]
        if side == Side.BUY.value:
            return min(quantity, max(0, inventory_limit - current_position))

        held = self.long_qty[instrument]
        return min(quantity, held)
