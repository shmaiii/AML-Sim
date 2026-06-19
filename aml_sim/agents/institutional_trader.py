"""
AML Institutional Trader.

Models a larger participant that tries to build or reduce a target position in
child orders over time. Later this can become the LLM-directed strategy agent.
"""

from typing import Any, Dict, Optional

from agents.benchmark_traders.trader import TraderAgent
from utils.orders import OrderType, Side


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

        self.target_positions = target_positions or {}
        self.child_order_size = max(1, child_order_size)
        self.order_type = order_type.upper()
        self.limit_price = limit_price

        self.logger.info(
            f"AMLInstitutionalTrader {self.agent_id} initialized: "
            f"target_positions={self.target_positions}, "
            f"child_order_size={self.child_order_size}, order_type={self.order_type}"
        )

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        await super().handle_time_tick(payload)

        current_time = self.current_time
        if self.next_action_time is None:
            self.next_action_time = current_time

        if current_time >= self.next_action_time:
            for instrument in self.instrument_exchange_map.keys():
                await self._execute_toward_target(instrument)
            self.next_action_time = current_time + self.action_interval

    async def _execute_toward_target(self, instrument: str) -> None:
        target = self.target_positions.get(instrument, 0)
        current = self.long_qty[instrument]
        gap = target - current

        if gap == 0:
            return

        side = Side.BUY.value if gap > 0 else Side.SELL.value
        quantity = min(abs(gap), self.child_order_size)

        if side == Side.SELL.value:
            held = self.long_qty[instrument]
            if held <= 0:
                self.logger.debug(f"Institutional trader skipped SELL for {instrument}: no inventory")
                return
            quantity = min(quantity, held)

        price = self.limit_price if self.order_type == OrderType.LIMIT.value else None
        order_id = await self.place_order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=self.order_type,
            price=price,
            explanation=f"AML institutional child order toward target {target}",
        )
        if order_id:
            self.logger.info(
                f"AMLInstitutionalTrader {self.agent_id} placed {side} "
                f"{self.order_type} order for {quantity} {instrument} "
                f"(current={current}, target={target})"
            )
