"""
AML Institutional Trader.

Models a larger participant that tries to build or reduce a target position in
child orders over time. Later this can become the LLM-directed strategy agent.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Optional

from aml_sim.agents.observation import build_observation_context
from aml_sim.agents.slow_strategist import RuleBasedSlowStrategist, SlowStrategist
from aml_sim.agents.state import BaseStrategyState
from aml_sim.agents.strategy_validator import validate_strategy_state
from agents.benchmark_traders.trader import TraderAgent
from utils.orders import OrderType, Side


@dataclass
class InstitutionalStrategyState(BaseStrategyState):
    """Role-specific strategy state for an AML institutional trader."""

    strategy_type: str = "target_execution"
    target_positions: Dict[str, int] = field(default_factory=dict)
    child_order_size: int = 100
    order_type: str = OrderType.MARKET.value
    limit_price: Optional[float] = None
    execution_style: str = "sliced"
    urgency: float = 0.5


class AMLInstitutionalTrader(TraderAgent):
    """
    Basic institutional-style participant for synthetic AML markets.

    Institutional here means larger capital, slower cadence, target inventory,
    and sliced execution rather than one giant order.
    """

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        target_positions: Optional[Dict[str, int]] = None,
        child_order_size: int = 100,
        order_type: str = OrderType.MARKET.value,
        limit_price: Optional[float] = None,
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

        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **trader_kwargs,
        )

        self.strategy_state = validate_strategy_state(
            InstitutionalStrategyState(
                target_positions=dict(target_positions or {}),
                child_order_size=child_order_size,
                order_type=order_type.upper(),
                limit_price=limit_price,
            )
        )
        self.slow_strategist = slow_strategist or RuleBasedSlowStrategist()
        self.slow_loop_interval = timedelta(
            seconds=slow_loop_interval_seconds or self.action_interval.total_seconds()
        )
        self.next_slow_loop_time = None

        self.logger.info(
            f"AMLInstitutionalTrader {self.agent_id} initialized: "
            f"strategy_state={self.strategy_state}"
        )

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        await super().handle_time_tick(payload)

        current_time = self.current_time
        if self.next_action_time is None:
            self.next_action_time = current_time

        if self.next_slow_loop_time is None:
            self.next_slow_loop_time = current_time

        if current_time >= self.next_slow_loop_time:
            self._run_slow_loop()
            self.next_slow_loop_time = current_time + self.slow_loop_interval

        if current_time >= self.next_action_time:
            for instrument in self.instrument_exchange_map.keys():
                await self._execute_toward_target(instrument)
            self.next_action_time = current_time + self.action_interval

    def _run_slow_loop(self) -> None:
        observation = build_observation_context(self)
        proposed_strategy = self.slow_strategist.propose(
            observation,
            self.strategy_state,
        )
        self.strategy_state = validate_strategy_state(proposed_strategy)

    async def _execute_toward_target(self, instrument: str) -> None:
        strategy = self.strategy_state
        target = strategy.target_positions.get(instrument, 0)
        current = self.long_qty[instrument]
        gap = target - current

        if gap == 0:
            return

        side = Side.BUY.value if gap > 0 else Side.SELL.value
        quantity = min(abs(gap), strategy.child_order_size)

        if side == Side.SELL.value:
            held = self.long_qty[instrument]
            if held <= 0:
                self.logger.debug(f"Institutional trader skipped SELL for {instrument}: no inventory")
                return
            quantity = min(quantity, held)

        price = strategy.limit_price if strategy.order_type == OrderType.LIMIT.value else None
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
