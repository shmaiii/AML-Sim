"""Cross-market relative-value participant for financial-ecology experiments."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Mapping, Optional

from aml_sim.agents.base import BaseAMLAgent
from aml_sim.agents.context.memory import MemoryBackend
from aml_sim.agents.context.observation import ObservationProcessor
from aml_sim.agents.models.state import CrossMarketArbitrageStrategyState
from aml_sim.agents.strategy.llm_slow_strategy import SlowStrategist
from aml_sim.ecology.registry import RelationshipRegistry
from utils.orders import OrderType, Side
from utils.messages import MessageType


class AMLCrossMarketArbitrageur(BaseAMLAgent):
    """Trades a declared relationship while retaining both-leg causal provenance."""

    LLM_STRATEGY_ROLE = "cross_market_arbitrage"

    def __init__(
        self,
        instrument_exchange_map: dict[str, str],
        relationships: list[Mapping[str, Any]],
        instrument_metadata: Mapping[str, Mapping[str, Any]],
        relationship_id: str,
        entry_threshold_bps: float = 15.0,
        exit_threshold_bps: float = 5.0,
        child_order_size: int = 20,
        max_position: int = 500,
        hedge_ratio: Optional[float] = None,
        order_type: str = OrderType.MARKET.value,
        risk_mode: str = "normal",
        profile: Optional[Mapping[str, Any]] = None,
        memory: Optional[MemoryBackend] = None,
        observation_processor: Optional[ObservationProcessor] = None,
        slow_loop_interval_seconds: Optional[int] = None,
        slow_strategist: Optional[SlowStrategist | Mapping[str, Any]] = None,
        agent_id: Optional[str] = None,
        rabbitmq_host: str = "localhost",
        **kwargs: Any,
    ) -> None:
        trader_kwargs = {
            name: kwargs[name]
            for name in (
                "initial_cash",
                "initial_positions",
                "initial_cost_basis",
                "action_interval_seconds",
                "enable_ecology",
            )
            if name in kwargs
        }
        self.relationships = RelationshipRegistry(instrument_metadata, relationships)
        relationship = self.relationships.get(relationship_id)
        if relationship.source not in instrument_exchange_map or relationship.target not in instrument_exchange_map:
            raise ValueError(
                "A cross-market arbitrageur must be routed to both relationship instruments"
            )
        resolved_hedge_ratio = (
            float(hedge_ratio)
            if hedge_ratio is not None
            else self.relationships.hedge_ratio(relationship_id)
        )
        super().__init__(
            instrument_exchange_map=instrument_exchange_map,
            strategy_state=CrossMarketArbitrageStrategyState(
                risk_mode=risk_mode,
                relationship_id=relationship_id,
                entry_threshold_bps=max(0.0, float(entry_threshold_bps)),
                exit_threshold_bps=max(0.0, float(exit_threshold_bps)),
                child_order_size=max(1, int(child_order_size)),
                max_position=max(1, int(max_position)),
                hedge_ratio=max(0.0001, resolved_hedge_ratio),
                order_type=str(order_type).upper(),
            ),
            profile=profile or {
                "role": "cross_market_arbitrageur",
                "decision_style": "basis_arbitrage",
            },
            memory=memory,
            observation_processor=observation_processor,
            slow_strategist=slow_strategist or {"type": "frozen"},
            slow_loop_interval_seconds=slow_loop_interval_seconds,
            agent_id=agent_id,
            rabbitmq_host=rabbitmq_host,
            **trader_kwargs,
        )
        self.logger.info(
            "%s initialized for relationship %s (%s -> %s)",
            self.agent_id,
            relationship.relationship_id,
            relationship.source,
            relationship.target,
        )
        self.order_books: dict[str, dict[str, float]] = {}

    async def run_fast_loop(self, observation: Mapping[str, Any]) -> None:
        del observation
        relationship = self.relationships.get(self.strategy_state.relationship_id)
        await self._request_market_snapshots(relationship.source, relationship.target)
        if not relationship.channel_enabled("valuation"):
            return
        opportunity = self._executable_opportunity(relationship)
        if opportunity is None:
            return
        decision_id = f"{self.agent_id}:{self.current_tick_id}:{relationship.relationship_id}"
        active_event_ids = [
            self._event_memory_id(event)
            for event in self._active_events()
            if self._event_memory_id(event) is not None
        ]
        provenance = {
            "decision_id": decision_id,
            "relationship_id": relationship.relationship_id,
            "relationship_type": relationship.relationship_type,
            "linkage_channel": "arbitrage",
            "source_instrument": relationship.source,
            "target_instrument": relationship.target,
            "reference_target_price": round(opportunity["reference_target_price"], 6),
            "observed_basis_bps": round(opportunity["signed_edge_bps"], 4),
            "executable_edge_bps": round(opportunity["best_edge_bps"], 4),
            "overpriced_edge_bps": round(opportunity["overpriced_edge_bps"], 4),
            "underpriced_edge_bps": round(opportunity["underpriced_edge_bps"], 4),
            "active_event_ids": active_event_ids,
        }
        self._record_action_event(
            {
                "event_type": "relationship_observed",
                "ecology": {
                    "decision_id": decision_id,
                    "relationship_id": relationship.relationship_id,
                    "relationship_type": relationship.relationship_type,
                    "linkage_channel": "arbitrage",
                    "source_instrument": relationship.source,
                    "target_instrument": relationship.target,
                    "source_bid": opportunity["source_bid"],
                    "source_ask": opportunity["source_ask"],
                    "target_bid": opportunity["target_bid"],
                    "target_ask": opportunity["target_ask"],
                    "reference_target_price": opportunity["reference_target_price"],
                    "basis_bps": round(opportunity["signed_edge_bps"], 4),
                    "executable_edge_bps": round(opportunity["best_edge_bps"], 4),
                    "overpriced_edge_bps": round(opportunity["overpriced_edge_bps"], 4),
                    "underpriced_edge_bps": round(opportunity["underpriced_edge_bps"], 4),
                    "active_event_ids": active_event_ids,
                },
            }
        )

        if not relationship.channel_enabled("arbitrage"):
            return
        if self._has_open_relationship_order():
            return
        if opportunity["best_edge_bps"] <= self.strategy_state.exit_threshold_bps:
            await self._unwind_positions(
                relationship.source,
                relationship.target,
                decision_id,
                provenance,
            )
            return
        threshold = self.strategy_state.entry_threshold_bps
        if opportunity["best_edge_bps"] < threshold:
            return

        quantity = self._trade_quantity(relationship.source, relationship.target)
        if quantity <= 0:
            return

        if opportunity["direction"] == "future_overpriced":
            source_side, target_side = Side.BUY.value, Side.SELL.value
            source_short, target_short = False, True
        else:
            source_side, target_side = Side.SELL.value, Side.BUY.value
            source_short, target_short = True, False

        source_order_id = await self.place_order(
            instrument=relationship.source,
            side=source_side,
            quantity=quantity,
            order_type=self.strategy_state.order_type,
            explanation=f"AML ecology basis decision {decision_id}",
            is_short=source_short,
            provenance={**provenance, "leg": "source"},
        )
        target_quantity = max(1, round(quantity * self.strategy_state.hedge_ratio))
        target_order_id = await self.place_order(
            instrument=relationship.target,
            side=target_side,
            quantity=target_quantity,
            order_type=self.strategy_state.order_type,
            explanation=f"AML ecology basis decision {decision_id}",
            is_short=target_short,
            provenance={**provenance, "leg": "target"},
        )
        self._record_action_event(
            {
                "event_type": "cross_market_decision",
                "ecology": {
                    **provenance,
                    "source_order_id": source_order_id,
                    "target_order_id": target_order_id,
                    "source_side": source_side,
                    "target_side": target_side,
                    "source_quantity": quantity,
                    "target_quantity": target_quantity,
                },
            }
        )

    async def _unwind_positions(
        self,
        source: str,
        target: str,
        decision_id: str,
        provenance: Mapping[str, Any],
    ) -> None:
        orders: dict[str, Optional[str]] = {}
        for leg, instrument in (("source", source), ("target", target)):
            position = self.long_qty[instrument] - self.short_qty[instrument]
            if position == 0:
                continue
            side = Side.SELL.value if position > 0 else Side.BUY.value
            orders[leg] = await self.place_order(
                instrument=instrument,
                side=side,
                quantity=abs(position),
                order_type=self.strategy_state.order_type,
                explanation=f"AML ecology basis unwind {decision_id}",
                is_short_cover=position < 0,
                provenance={
                    **dict(provenance),
                    "leg": leg,
                    "linkage_channel": "arbitrage_unwind",
                },
            )
        if orders:
            self._record_action_event(
                {
                    "event_type": "cross_market_unwind",
                    "ecology": {**dict(provenance), "orders": orders},
                }
            )

    def _current_price(self, instrument: str) -> float:
        try:
            price = float(self.prices.get(instrument, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return price if price > 0 else 0.0

    async def on_market_data_update(
        self,
        instrument: str,
        snapshot: Mapping[str, Any],
    ) -> None:
        """Store public marks and the executable top of book for both legs."""
        if instrument not in self.instrument_exchange_map:
            return
        book = self._book_prices(snapshot)
        if book is not None:
            self.order_books[instrument] = book
        price = self._snapshot_price(snapshot)
        if price <= 0:
            return
        self.prices[instrument] = price
        self._record_price(instrument, price)

    def _executable_opportunity(self, relationship: Any) -> Optional[dict[str, float | str]]:
        """Return the better two-leg executable edge, net of displayed spread."""
        source_book = self.order_books.get(relationship.source)
        target_book = self.order_books.get(relationship.target)
        if source_book is None or target_book is None:
            return None

        source_bid = source_book["bid"]
        source_ask = source_book["ask"]
        target_bid = target_book["bid"]
        target_ask = target_book["ask"]
        if min(source_bid, source_ask, target_bid, target_ask) <= 0:
            return None

        # A cash-and-carry trade buys the stock at its ask and sells the future
        # at its bid. Reverse cash-and-carry does the opposite. Comparing these
        # executable prices prevents an apparent edge that disappears on fills.
        overpriced_reference = self.relationships.reference_target_price(
            relationship.relationship_id,
            source_ask,
        )
        underpriced_reference = self.relationships.reference_target_price(
            relationship.relationship_id,
            source_bid,
        )
        if overpriced_reference <= 0 or underpriced_reference <= 0:
            return None

        overpriced_edge_bps = (
            (target_bid - overpriced_reference) / overpriced_reference
        ) * 10_000
        underpriced_edge_bps = (
            (underpriced_reference - target_ask) / underpriced_reference
        ) * 10_000

        if overpriced_edge_bps >= underpriced_edge_bps:
            return {
                "direction": "future_overpriced",
                "source_bid": source_bid,
                "source_ask": source_ask,
                "target_bid": target_bid,
                "target_ask": target_ask,
                "reference_target_price": overpriced_reference,
                "signed_edge_bps": overpriced_edge_bps,
                "best_edge_bps": overpriced_edge_bps,
                "overpriced_edge_bps": overpriced_edge_bps,
                "underpriced_edge_bps": underpriced_edge_bps,
            }
        return {
            "direction": "future_underpriced",
            "source_bid": source_bid,
            "source_ask": source_ask,
            "target_bid": target_bid,
            "target_ask": target_ask,
            "reference_target_price": underpriced_reference,
            "signed_edge_bps": -underpriced_edge_bps,
            "best_edge_bps": underpriced_edge_bps,
            "overpriced_edge_bps": overpriced_edge_bps,
            "underpriced_edge_bps": underpriced_edge_bps,
        }

    async def _request_market_snapshots(self, *instruments: str) -> None:
        """Request both public books; responses are used from the next action onward."""
        if self.current_time is None:
            return
        interval_seconds = max(30.0, self.action_interval.total_seconds())
        window_start = self.current_time - timedelta(seconds=min(interval_seconds, 300.0))
        payload = {
            "window_start": window_start.isoformat(),
            "window_end": self.current_time.isoformat(),
        }
        for instrument in instruments:
            exchange_id = self.instrument_exchange_map.get(instrument)
            if exchange_id:
                await self.send_message(
                    exchange_id,
                    MessageType.MARKET_DATA_SNAPSHOT_REQUEST,
                    payload,
                )

    @staticmethod
    def _snapshot_price(snapshot: Mapping[str, Any]) -> float:
        data = snapshot.get("data", {})
        if isinstance(data, Mapping):
            for key in ("close", "vwap"):
                try:
                    price = float(data.get(key, 0.0))
                except (TypeError, ValueError):
                    price = 0.0
                if price > 0:
                    return price

        try:
            bid = float(snapshot.get("best_bid", 0.0) or 0.0)
            ask = float(snapshot.get("best_ask", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return bid if bid > 0 else ask

    @staticmethod
    def _book_prices(snapshot: Mapping[str, Any]) -> Optional[dict[str, float]]:
        try:
            bid = float(snapshot.get("best_bid", 0.0) or 0.0)
            ask = float(snapshot.get("best_ask", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0 or bid > ask:
            return None
        return {"bid": bid, "ask": ask}

    def _trade_quantity(self, source: str, target: str) -> int:
        risk_policy = self._risk_policy()
        max_position = max(
            1,
            int(self.strategy_state.max_position * risk_policy.position_limit_multiplier),
        )
        source_position = abs(self.long_qty[source] - self.short_qty[source])
        target_position = abs(self.long_qty[target] - self.short_qty[target])
        remaining_capacity = max(0, max_position - max(source_position, target_position))
        size = int(self.strategy_state.child_order_size * risk_policy.order_size_multiplier)
        return max(0, min(max(1, size), remaining_capacity))

    def _has_open_relationship_order(self) -> bool:
        return any(
            order.get("instrument") in {
                self.relationships.get(self.strategy_state.relationship_id).source,
                self.relationships.get(self.strategy_state.relationship_id).target,
            }
            for order in self.pending_orders.values()
        )
