"""Regression tests for LLM decision and shock-event lifecycle handling."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCKSIM_ROOT = PROJECT_ROOT / "simulators" / "StockSim"
if str(STOCKSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STOCKSIM_ROOT))

from aml_sim.agents.base import BaseAMLAgent
from aml_sim.agents.context.observation import build_observation_context
from aml_sim.agents.models.state import InstitutionalStrategyState
from aml_sim.agents.strategy.llm_slow_strategy import OpenAIJSONLLMClient
from aml_sim.agents.strategy.signals import event_pressure
from aml_sim.agents.strategy.validator import StrategyValidationError, validate_strategy_state


class _Logger:
    def error(self, *args, **kwargs):
        del args, kwargs

    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs


class _Memory:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def add_event(self, agent_id, event_type, payload, *, timestamp=None):
        self.events.append(
            {
                "agent_id": agent_id,
                "event_type": event_type,
                "payload": payload,
                "timestamp": timestamp,
            }
        )


class _RejectedProposal:
    def propose(self, *args, **kwargs):
        del args, kwargs
        return {"risk_mode": "invalid"}


class _TestAgent(BaseAMLAgent):
    async def run_fast_loop(self, observation):
        del observation


class LLMShockLifecycleTests(unittest.TestCase):
    def test_observation_separates_active_historical_and_anticipated_events(self) -> None:
        agent = type(
            "Agent",
            (),
            {
                "instrument_exchange_map": {"AAPL": "exchange_aapl"},
                "current_time": "2025-03-01T10:06:00+00:00",
            },
        )()
        observation = build_observation_context(
            agent,
            events=[{"shock_id": "active"}],
            known_events=[
                {"shock_id": "expired", "context_status": "historical"},
                {"shock_id": "earnings", "context_status": "anticipated"},
            ],
        )

        events = observation["event_context"]
        self.assertTrue(events["active"][0]["is_active"])
        self.assertEqual(events["historical"][0]["shock_id"], "expired")
        self.assertEqual(events["calendar"][0]["shock_id"], "earnings")

    def test_expired_active_phase_is_historical_in_agent_context(self) -> None:
        agent = object.__new__(_TestAgent)
        agent.current_tick_id = 10
        agent.recent_events = [
            {
                "shock_id": "stock_news",
                "phase": "active",
                "effective_tick_id": 2,
                "duration_ticks": 4,
            }
        ]

        known = agent._known_events()

        self.assertEqual(known[0]["context_status"], "historical")
        self.assertFalse(known[0]["is_active"])

    def test_strategy_change_excludes_decision_metadata(self) -> None:
        before = {
            "risk_mode": "normal",
            "confidence": 0.8,
            "reason": "old",
            "updated_at": "09:30",
        }
        after = {
            "risk_mode": "normal",
            "confidence": 0.7,
            "reason": "new",
            "updated_at": "09:33",
        }

        self.assertEqual(BaseAMLAgent._strategy_changed_fields(before, after), [])
        self.assertEqual(
            BaseAMLAgent._strategy_metadata_changes(before, after),
            ["confidence", "reason", "updated_at"],
        )

    def test_limit_order_strategy_requires_a_positive_limit_price(self) -> None:
        with self.assertRaisesRegex(StrategyValidationError, "limit_price"):
            validate_strategy_state(
                InstitutionalStrategyState(order_type="LIMIT", limit_price=None)
            )
        valid = InstitutionalStrategyState(order_type="LIMIT", limit_price=100.0)
        self.assertIs(validate_strategy_state(valid), valid)

    def test_rejected_slow_loop_keeps_active_event_unseen_and_records_status(self) -> None:
        agent = object.__new__(_TestAgent)
        agent.agent_id = "test_agent"
        agent.strategy_state = {"risk_mode": "normal"}
        agent.slow_strategist = _RejectedProposal()
        agent.strategy_validator = lambda proposal: (_ for _ in ()).throw(
            StrategyValidationError("invalid risk mode")
        )
        agent.profile = {}
        agent.memory = _Memory()
        agent.current_time = datetime(2025, 3, 1, 10, 0, tzinfo=timezone.utc)
        agent.slow_loop_seen_event_ids = set()
        agent.recent_events = [{"shock_id": "news"}]
        agent.logger = _Logger()
        agent._last_strategy_rejection_reason = None

        observation = {
            "memory": {},
            "event_context": {"active": [{"shock_id": "news"}], "known": []},
        }
        asyncio.run(agent.run_slow_loop(observation))

        self.assertEqual(agent.slow_loop_seen_event_ids, set())
        payload = agent.memory.events[0]["payload"]
        self.assertEqual(payload["slow_loop_status"], "rejected")
        self.assertEqual(payload["failure_reason"], "invalid risk mode")

    def test_neutral_market_state_pressure_has_the_complete_contract(self) -> None:
        pressure = event_pressure([], "AAPL")

        self.assertEqual(pressure["severity"], 0.0)
        self.assertEqual(pressure["directional_bias"], 0.0)
        self.assertEqual(pressure["liquidity_multiplier"], 1.0)

    def test_llm_response_log_keeps_the_exact_prompt_and_context(self) -> None:
        with TemporaryDirectory() as directory:
            previous = os.environ.get("DECISION_CONTEXT_DIR")
            os.environ["DECISION_CONTEXT_DIR"] = directory
            try:
                client = OpenAIJSONLLMClient(
                    model="gpt-test",
                    system_prompt="system contract",
                )
                context = {
                    "observation": {
                        "agent": {"agent_id": "maker_1"},
                        "current_time": "2025-03-01T10:00:00+00:00",
                    }
                }
                client._write_response_log(context=context, content='{"strategy_updates": {}}')
            finally:
                if previous is None:
                    os.environ.pop("DECISION_CONTEXT_DIR", None)
                else:
                    os.environ["DECISION_CONTEXT_DIR"] = previous

            record_path = Path(directory) / "maker_1" / "llm_responses.jsonl"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["schema_version"], 2)
            self.assertEqual(record["system_prompt"], "system contract")
            self.assertEqual(record["context"], context)


if __name__ == "__main__":
    unittest.main()
