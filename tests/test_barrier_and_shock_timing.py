from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
STOCKSIM_DIR = ROOT / "simulators" / "StockSim"
if str(STOCKSIM_DIR) not in sys.path:
    sys.path.insert(0, str(STOCKSIM_DIR))

if importlib.util.find_spec("aio_pika") is None:
    aio_pika = types.ModuleType("aio_pika")
    aio_pika.RobustConnection = object
    aio_pika.RobustChannel = object
    aio_pika.Exchange = object
    aio_pika.Queue = object
    aio_pika.Message = object
    aio_pika.connect_robust = None
    aio_pika.ExchangeType = SimpleNamespace(DIRECT="direct", TOPIC="topic")
    aio_pika.DeliveryMode = SimpleNamespace(PERSISTENT=2)
    aio_pika_abc = types.ModuleType("aio_pika.abc")
    aio_pika_abc.AbstractIncomingMessage = object
    aio_pika_exceptions = types.ModuleType("aio_pika.exceptions")
    aio_pika_exceptions.AMQPConnectionError = RuntimeError
    aio_pika_exceptions.AMQPChannelError = RuntimeError
    sys.modules["aio_pika"] = aio_pika
    sys.modules["aio_pika.abc"] = aio_pika_abc
    sys.modules["aio_pika.exceptions"] = aio_pika_exceptions

from aml_sim.agents.shock_agent import AMLShockAgent
from aml_sim.agents.base import BaseAMLAgent
from simulation.simulation_clock import SimulationClock


class PacketTestAgent(BaseAMLAgent):
    async def run_fast_loop(self, observation):
        return None


class BarrierAndShockTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_barrier_deduplicates_agent_acknowledgements(self) -> None:
        clock = SimulationClock(
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
            tick_interval_seconds=30,
            barrier_timeout_seconds=0.2,
        )
        await clock.trader_response_queue.put(
            {"tick_id": 4, "agent_id": "retail_1", "phase": "trader"}
        )
        await clock.trader_response_queue.put(
            {"tick_id": 4, "agent_id": "retail_1", "phase": "trader"}
        )
        await clock.trader_response_queue.put(
            {"tick_id": 4, "agent_id": "retail_2", "phase": "trader"}
        )
        responses = await clock._wait_for_decision_responses(
            4, expected_count=2, phase="trader"
        )
        self.assertEqual({"retail_1", "retail_2"}, {row["agent_id"] for row in responses})

    async def test_barrier_timeout_fails_instead_of_advancing_clock(self) -> None:
        clock = SimulationClock(
            start_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2025, 1, 1, 0, 1, tzinfo=timezone.utc),
            tick_interval_seconds=30,
            barrier_timeout_seconds=0.01,
        )
        with self.assertRaises(asyncio.TimeoutError):
            await clock._wait_for_decision_responses(
                0, expected_count=1, phase="shock"
            )

    def test_shock_agent_has_dedicated_phase_and_exact_tick_timing(self) -> None:
        self.assertEqual("shock.#", AMLShockAgent.TIME_ROUTING_GROUP)
        agent = object.__new__(AMLShockAgent)
        event = {"tick": 5}
        self.assertFalse(agent._event_due(event, {"tick_id": 4}))
        self.assertTrue(agent._event_due(event, {"tick_id": 5}))

    def test_shock_packet_is_ingested_once_before_agent_decision(self) -> None:
        packet = {
            "tick_id": 5,
            "agent_id": "shock_agent",
            "phase": "shock",
            "events": [{"event_type": "AML_SHOCK", "effective_tick_id": 5}],
        }
        packet["events"][0].update({
            "shock_id": "shock_at_5",
            "phase": "active",
            "tick_id": 5,
            "severity": 0.8,
            "direction": -1,
        })
        agent = object.__new__(PacketTestAgent)
        agent.current_time = datetime(2025, 1, 1, 0, 2, 30, tzinfo=timezone.utc)
        agent.current_tick_id = 5
        agent.agent_id = "retail_1"
        agent.profile = {"role": "retail"}
        agent.market_state = {}
        agent.market_state_baseline = {}
        agent.slow_loop_seen_event_ids = set()
        agent._observed_event_keys = set()
        agent.recent_events = []
        agent.action_events = []
        agent.logger = SimpleNamespace(info=lambda *_: None)
        agent._ingest_shock_packets([packet])
        agent._ingest_shock_packets([packet])
        self.assertEqual(1, len(agent.recent_events))
        self.assertEqual(5, agent.recent_events[0]["observed_tick_id"])
        self.assertEqual("shock_at_5", agent.action_events[0]["shock_id"])
