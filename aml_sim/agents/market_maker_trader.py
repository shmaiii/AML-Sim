"""
AML Market Maker Trader.

Posts both bid and ask limit orders around a configurable fair price so an
orderbook scenario has synthetic liquidity without historical replay data.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from aml_sim.agents.state import BaseStrategyState
from aml_sim.agents.strategy_validator import validate_strategy_state
from agents.benchmark_traders.trader import TraderAgent
from utils.orders import OrderType, Side


@dataclass
class MarketMakerStrategyState(BaseStrategyState):
    """Role-specific strategy state for an AML market maker."""

    strategy_type: str = "market_making"
    fair_price: float = 100.0
    spread: float = 0.2
    quote_size: int = 100
    target_inventory: int = 0
    inventory_skew: float = 0.001


class AMLMarketMakerTrader(TraderAgent):
    """
    Simple synthetic market maker for AML simulations.

    On each action interval it:
    - cancels outstanding quotes from previous ticks,
    - posts a buy limit order below fair price,
    - posts a sell limit order above fair price,
    - nudges quotes based on inventory so it does not accumulate forever.
    """

    def __init__(
        self,
        instrument_exchange_map: Dict[str, str],
        fair_price: float = 100.0,
        spread: float = 0.2,
        quote_size: int = 100,
        inventory_skew: float = 0.001,
        target_inventory: int = 0,
        allow_short_selling: bool = False,
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
            MarketMakerStrategyState(
                fair_price=fair_price,
                spread=spread,
                quote_size=quote_size,
                target_inventory=target_inventory,
                inventory_skew=inventory_skew,
            )
        )
        self.allow_short_selling = allow_short_selling
        self.quote_order_ids: set[str] = set()

        self.logger.info(
            f"AMLMarketMakerTrader {self.agent_id} initialized: "
            f"strategy_state={self.strategy_state}"
        )

    async def handle_time_tick(self, payload: Dict[str, Any]) -> None:
        await super().handle_time_tick(payload)

        current_time = self.current_time
        if self.next_action_time is None:
            self.next_action_time = current_time

        if current_time >= self.next_action_time:
            await self._refresh_quotes()
            self.next_action_time = current_time + self.action_interval

    async def _refresh_quotes(self) -> None:
        await self._cancel_existing_quotes()

        for instrument in self.instrument_exchange_map.keys():
            bid_price, ask_price = self._quote_prices(instrument)
            await self._place_bid(instrument, bid_price)
            await self._place_ask(instrument, ask_price)

    async def _cancel_existing_quotes(self) -> None:
        for order_id in list(self.quote_order_ids):
            if order_id in self.pending_orders:
                await self.cancel_order(order_id)
            self.quote_order_ids.discard(order_id)

    def _quote_prices(self, instrument: str) -> tuple[float, float]:
        strategy = self.strategy_state
        inventory = self.long_qty[instrument] - self.short_qty[instrument]
        inventory_gap = inventory - strategy.target_inventory
        skew = inventory_gap * strategy.inventory_skew

        midpoint = max(0.01, strategy.fair_price - skew)
        half_spread = max(0.01, strategy.spread / 2)
        bid = max(0.01, midpoint - half_spread)
        ask = max(bid + 0.01, midpoint + half_spread)
        return round(bid, 2), round(ask, 2)

    async def _place_bid(self, instrument: str, price: float) -> None:
        order_id = await self.place_order(
            instrument=instrument,
            side=Side.BUY.value,
            quantity=self.strategy_state.quote_size,
            order_type=OrderType.LIMIT.value,
            price=price,
            explanation="AML market maker bid quote",
        )
        if order_id:
            self.quote_order_ids.add(order_id)

    async def _place_ask(self, instrument: str, price: float) -> None:
        held = self.long_qty[instrument]
        quote_size = self.strategy_state.quote_size
        quantity = min(quote_size, held) if not self.allow_short_selling else quote_size
        if quantity <= 0:
            self.logger.debug(f"Skipping ask for {instrument}: no inventory to sell")
            return

        order_id = await self.place_order(
            instrument=instrument,
            side=Side.SELL.value,
            quantity=quantity,
            order_type=OrderType.LIMIT.value,
            price=price,
            explanation="AML market maker ask quote",
            is_short=self.allow_short_selling and held <= 0,
        )
        if order_id:
            self.quote_order_ids.add(order_id)

    async def on_trade_execution(self, msg: Dict[str, Any]) -> None:
        await super().on_trade_execution(msg)
        self.logger.debug(
            f"AMLMarketMakerTrader {self.agent_id} inventory after trade: "
            f"{dict(self.long_qty)}"
        )
