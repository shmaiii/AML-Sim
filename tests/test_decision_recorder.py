from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

from aml_sim.agents.base import BaseAMLAgent
from aml_sim.agents.informed_trader import AMLInformedTrader
from aml_sim.agents.institutional_trader import AMLInstitutionalTrader
from aml_sim.agents.liquidity_taker import AMLLiquidityTaker
from aml_sim.agents.market_maker_trader import AMLMarketMakerTrader
from aml_sim.agents.retail_trader import AMLRetailTrader
from aml_sim.agents.strategy.signals import market_state_pressure


class TestAMLAgent(BaseAMLAgent):
    async def run_fast_loop(self, observation):
        return None


class AgentDecisionRecorderTests(unittest.TestCase):
    def make_agent(self) -> BaseAMLAgent:
        agent = object.__new__(TestAMLAgent)
        agent.agent_id = "informed_1"
        agent.instrument_exchange_map = {"AAPL": "exchange_aapl"}
        agent.current_time = datetime(2025, 3, 1, 9, 35, tzinfo=timezone.utc)
        agent.strategy_state = SimpleNamespace(
            signal_strength=0.01,
            signal_threshold=0.005,
            confidence=1.2,
        )
        agent.action_events = []
        agent.decision_records = []
        agent.interval_outcomes = []
        agent._pending_interval_outcomes = []
        agent._outcome_observations = []
        agent.dataset_split = "validation"
        agent.decision_action_threshold = 0.1
        agent.action_interval = timedelta(minutes=1)
        agent.assigned_risk_budgets = {"AAPL": 25000.0}
        agent.portfolio_value = 100000.0
        agent.test_snapshot = {
            "portfolio_value": 100000.0,
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
        }
        agent._portfolio_snapshot = lambda: dict(agent.test_snapshot)
        return agent

    def test_records_recommendation_and_bounds_score_and_confidence(self) -> None:
        agent = self.make_agent()
        observation = {
            "market": {
                "last_market_snapshot": {
                    "AAPL": {"window_end": "2025-03-01T09:34:30+00:00"}
                },
                "price_history": {
                    "AAPL": [
                        {
                            "source_timestamp": "2025-03-01T09:35:00+00:00",
                            "price": 100.0,
                        }
                    ]
                },
            }
        }

        agent._record_fast_loop_decisions(
            observation=observation,
            action_event_start=0,
        )

        self.assertEqual(1, len(agent.decision_records))
        row = agent.decision_records[0]
        self.assertEqual("BUY", row["action"])
        self.assertEqual("HOLD", row["submitted_action"])
        self.assertTrue(row["decision_id"].startswith("dec_"))
        self.assertEqual(36, len(row["decision_id"]))
        self.assertEqual(
            "2025-03-01T09:35:00+00:00",
            row["data_cutoff_timestamp"],
        )
        self.assertEqual(1.0, row["prediction_score"])
        self.assertEqual(1.0, row["confidence"])
        self.assertEqual("2025-03-01T09:35:00+00:00", row["latest_data_date_used"])
        self.assertEqual("price_history", row["data_timestamp_source"])
        self.assertEqual("validation", row["split"])
        self.assertEqual(1, len(agent._pending_interval_outcomes))

    def test_records_successful_single_side_submission(self) -> None:
        agent = self.make_agent()
        agent.action_events.append(
            {
                "event_type": "order_submitted",
                "instrument": "AAPL",
                "side": "BUY",
            }
        )

        agent._record_fast_loop_decisions(
            observation={"market": {}},
            action_event_start=0,
        )

        self.assertEqual("BUY", agent.decision_records[0]["action"])
        self.assertEqual("BUY", agent.decision_records[0]["submitted_action"])
        self.assertEqual(
            "2025-03-01T09:35:00+00:00",
            agent.decision_records[0]["latest_data_date_used"],
        )
        self.assertEqual(
            "simulation_clock",
            agent.decision_records[0]["data_timestamp_source"],
        )

    def test_recommendation_can_differ_from_execution(self) -> None:
        agent = self.make_agent()
        agent.strategy_state.signal_strength = -0.01
        agent.action_events.append(
            {
                "event_type": "order_submitted",
                "instrument": "AAPL",
                "side": "BUY",
            }
        )

        agent._record_fast_loop_decisions(
            observation={"market": {}},
            action_event_start=0,
        )

        row = agent.decision_records[0]
        self.assertEqual("SELL", row["action"])
        self.assertEqual("BUY", row["submitted_action"])

    def test_empty_market_state_has_complete_pressure_defaults(self) -> None:
        pressure = market_state_pressure(None)

        self.assertEqual(0.0, pressure["severity"])
        self.assertEqual(0.0, pressure["directional_bias"])

    def test_completed_interval_outcome_calculates_requested_metrics(self) -> None:
        agent = self.make_agent()
        agent._record_fast_loop_decisions(
            observation={"market": {}},
            action_event_start=0,
        )
        decision_id = agent.decision_records[0]["decision_id"]
        agent.current_time = datetime(2025, 3, 1, 9, 36, tzinfo=timezone.utc)
        agent.portfolio_value = 101000.0
        agent.test_snapshot = {
            "portfolio_value": 101000.0,
            "gross_exposure": 12000.0,
            "net_exposure": 8000.0,
        }
        agent._capture_outcome_observation()
        agent._finalize_due_interval_outcomes()

        self.assertEqual(1, len(agent.interval_outcomes))
        outcome = agent.interval_outcomes[0]
        self.assertEqual(decision_id, outcome["decision_id"])
        self.assertEqual("completed", outcome["outcome_status"])
        self.assertEqual(0.01, outcome["interval_return"])
        self.assertEqual(1000.0, outcome["interval_pnl"])
        self.assertEqual(101000.0, outcome["portfolio_value"])
        self.assertEqual(0.01, outcome["realized_volatility"])
        self.assertEqual(0.0, outcome["drawdown"])
        self.assertEqual(12000.0, outcome["gross_exposure"])
        self.assertEqual(8000.0, outcome["net_exposure"])
        self.assertEqual(25000.0, outcome["assigned_risk_budget"])

    def test_flat_hold_interval_is_inactive(self) -> None:
        agent = self.make_agent()
        agent.strategy_state.signal_strength = 0.0
        agent._record_fast_loop_decisions(
            observation={"market": {}},
            action_event_start=0,
        )
        agent.current_time = datetime(2025, 3, 1, 9, 36, tzinfo=timezone.utc)
        agent._capture_outcome_observation()
        agent._finalize_due_interval_outcomes()

        self.assertEqual("inactive", agent.interval_outcomes[0]["outcome_status"])
        self.assertEqual(0.0, agent.interval_outcomes[0]["interval_return"])

    def test_unfinished_interval_is_marked_missing(self) -> None:
        agent = self.make_agent()
        agent._record_fast_loop_decisions(
            observation={"market": {}},
            action_event_start=0,
        )

        agent._mark_unresolved_outcomes_missing()

        outcome = agent.interval_outcomes[0]
        self.assertEqual("missing", outcome["outcome_status"])
        self.assertIsNone(outcome["result_available_timestamp"])
        self.assertIsNone(outcome["interval_return"])

    def test_concrete_agents_forward_governance_parameters(self) -> None:
        agent_classes = (
            AMLMarketMakerTrader,
            AMLRetailTrader,
            AMLInstitutionalTrader,
            AMLInformedTrader,
            AMLLiquidityTaker,
        )

        with patch("agents.agent.setup_logger") as setup_logger:
            for agent_class in agent_classes:
                with self.subTest(agent_class=agent_class.__name__):
                    agent = agent_class(
                        instrument_exchange_map={"AAPL": "exchange_aapl"},
                        dataset_split={"label": "validation"},
                        decision_action_threshold=0.25,
                        assigned_risk_budget=50000,
                    )
                    self.assertEqual("validation", agent.dataset_split)
                    self.assertEqual(0.25, agent.decision_action_threshold)
                    self.assertEqual(
                        {"AAPL": 50000.0},
                        agent.assigned_risk_budgets,
                    )

        self.assertEqual(len(agent_classes), setup_logger.call_count)


if __name__ == "__main__":
    unittest.main()
