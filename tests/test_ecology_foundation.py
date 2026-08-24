"""Focused checks for AML's reusable financial-ecology substrate."""

from __future__ import annotations

import sys
import unittest
from json import dumps, loads
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STOCKSIM_ROOT = PROJECT_ROOT / "simulators" / "StockSim"
if str(STOCKSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(STOCKSIM_ROOT))

from aml_sim.agents.strategy.llm_slow_strategy import (
    FrozenStrategist,
    LLMStrategist,
    create_llm_strategist,
)
from aml_sim.ecology.information import EcologyInformationRouter
from aml_sim.ecology.registry import RelationshipRegistry
from aml_sim.ecology.reporting import generate_ecology_reports
from aml_sim.ecology.seeding import build_agent_seed_plan, component_seed
from aml_sim.ecology.state import ScopedMarketStateEngine, effective_market_state
from aml_sim.agents.cross_market_arbitrageur import AMLCrossMarketArbitrageur
from aml_sim.agents.context.observation import build_observation_context
from aml_sim.agents.base import BaseAMLAgent
from aml_sim.agents.models.state import InstitutionalStrategyState
from aml_sim.agents.strategy.validator import StrategyValidationError, validate_strategy_state
from aml_sim.agents.strategy.signals import event_pressure
from aml_sim.runs import create_run
from aml_sim.scenario import load_scenario
from aml_sim.shocks import build_shock_payload, normalize_instrument_metadata
from aml_runner import apply_research_overrides


class EcologyFoundationTests(unittest.TestCase):
    def test_neutral_market_state_pressure_has_the_complete_contract(self):
        pressure = event_pressure([], "AAPL")

        self.assertEqual(pressure["severity"], 0.0)
        self.assertEqual(pressure["directional_bias"], 0.0)
        self.assertEqual(pressure["liquidity_multiplier"], 1.0)

    def test_cross_market_mark_prefers_close_then_book_midpoint(self):
        self.assertEqual(
            AMLCrossMarketArbitrageur._snapshot_price(
                {"data": {"close": 101.25}, "best_bid": 101.0, "best_ask": 101.5}
            ),
            101.25,
        )
        self.assertEqual(
            AMLCrossMarketArbitrageur._snapshot_price(
                {"data": {}, "best_bid": 101.0, "best_ask": 101.5}
            ),
            101.25,
        )

    def test_executable_basis_requires_a_two_leg_bid_ask_edge(self) -> None:
        registry = RelationshipRegistry(
            self.metadata,
            [
                {
                    "id": "aapl_future",
                    "type": "spot_future",
                    "source": "AAPL",
                    "target": "AAPL_FUT",
                    "channels": ["valuation", "arbitrage"],
                    "parameters": {"annualized_carry_bps": 0, "expiry_days": 30},
                }
            ],
        )
        agent = object.__new__(AMLCrossMarketArbitrageur)
        agent.relationships = registry
        agent.order_books = {
            "AAPL": {"bid": 99.9, "ask": 100.1},
            "AAPL_FUT": {"bid": 100.05, "ask": 100.25},
        }

        opportunity = agent._executable_opportunity(registry.get("aapl_future"))

        self.assertIsNotNone(opportunity)
        self.assertEqual(opportunity["direction"], "future_overpriced")
        self.assertLess(float(opportunity["best_edge_bps"]), 0.0)

        agent.order_books["AAPL_FUT"] = {"bid": 100.5, "ask": 100.7}
        opportunity = agent._executable_opportunity(registry.get("aapl_future"))

        self.assertEqual(opportunity["direction"], "future_overpriced")
        self.assertGreater(float(opportunity["best_edge_bps"]), 0.0)

    def setUp(self) -> None:
        self.metadata = {
            "AAPL": {"asset_class": "stock"},
            "AAPL_FUT": {"asset_class": "future"},
        }

    def test_stock_future_reference_and_channels_are_explicit(self) -> None:
        registry = RelationshipRegistry(
            self.metadata,
            [
                {
                    "id": "aapl_future",
                    "type": "spot_future",
                    "source": "AAPL",
                    "target": "AAPL_FUT",
                    "channels": ["valuation", "arbitrage"],
                    "parameters": {"annualized_carry_bps": 500, "expiry_days": 30},
                }
            ],
        )
        relationship = registry.get("aapl_future")
        self.assertTrue(relationship.channel_enabled("arbitrage"))
        self.assertFalse(relationship.channel_enabled("shared_risk"))
        self.assertGreater(registry.reference_target_price("aapl_future", 100.0), 100.0)

    def test_arbitrage_requires_an_explicit_valuation_link(self) -> None:
        with self.assertRaisesRegex(ValueError, "without a valuation channel"):
            RelationshipRegistry(
                self.metadata,
                [
                    {
                        "id": "invalid",
                        "type": "spot_future",
                        "source": "AAPL",
                        "target": "AAPL_FUT",
                        "channels": ["arbitrage"],
                    }
                ],
            )

    def test_local_shock_can_reach_related_market_as_information_only(self) -> None:
        registry = RelationshipRegistry(
            self.metadata,
            [
                {
                    "id": "aapl_future",
                    "type": "spot_future",
                    "source": "AAPL",
                    "target": "AAPL_FUT",
                    "channels": ["information", "valuation"],
                }
            ],
        )
        router = EcologyInformationRouter(
            registry,
            {
                "stock_agent": ["AAPL"],
                "future_agent": ["AAPL_FUT"],
                "cross_agent": ["AAPL", "AAPL_FUT"],
            },
        )
        deliveries = router.deliveries(
            {"visibility": "affected", "affected_instruments": ["AAPL"]},
            {"visibility": "affected", "affected_instruments": ["AAPL"]},
            ["stock_agent", "future_agent", "cross_agent"],
        )
        by_agent = {delivery.agent_id: delivery for delivery in deliveries}
        self.assertEqual(by_agent["stock_agent"].delivery_type, "direct")
        self.assertEqual(by_agent["cross_agent"].delivery_type, "direct")
        self.assertEqual(
            by_agent["future_agent"].delivery_type,
            "relationship_information",
        )
        self.assertEqual(by_agent["future_agent"].relationship_ids, ("aapl_future",))

    def test_micro_state_and_direct_effect_do_not_leak_to_other_instrument(self) -> None:
        engine = ScopedMarketStateEngine(
            {"liquidity_index": 1.0, "risk_aversion": 0.0},
            self.metadata,
        )
        event = {
            "scope": "micro",
            "affected_instruments": ["AAPL"],
            "effects": {"liquidity_multiplier": 0.5},
            "duration_ticks": 2,
        }
        payload = {
            "scope": "micro",
            "shock_class": "non_systematic",
            "affected_instruments": ["AAPL"],
            "duration_ticks": 2,
            "effective_tick_id": 1,
            "liquidity_multiplier": 0.5,
        }
        engine.register("stock_news", event, payload, current_tick_id=1, default_duration_ticks=2)
        snapshot = engine.snapshot(1)
        self.assertEqual(effective_market_state(snapshot, "AAPL")["liquidity_index"], 0.5)
        self.assertEqual(effective_market_state(snapshot, "AAPL_FUT")["liquidity_index"], 1.0)
        self.assertEqual(event_pressure([payload], "AAPL_FUT")["severity"], 0.0)

    def test_component_seeds_are_stable_and_agent_specific(self) -> None:
        first = component_seed(17, 2, "AML_Retail_Trader", "retail_1")
        self.assertEqual(first, component_seed(17, 2, "AML_Retail_Trader", "retail_1"))
        self.assertNotEqual(first, component_seed(17, 2, "AML_Retail_Trader", "retail_2"))
        plan = build_agent_seed_plan(
            {"retail": {"type": "AML_Retail_Trader", "count": 2}},
            master_seed=17,
            replicate_id=2,
        )
        self.assertNotEqual(plan["retail_1"], plan["retail_2"])

    def test_research_overrides_create_a_clean_paired_replicate(self) -> None:
        scenario = load_scenario(
            PROJECT_ROOT
            / "scenarios"
            / "research"
            / "financial_ecology_stock_future_foundation.yaml"
        )
        overridden = apply_research_overrides(
            scenario,
            master_seed=90210,
            replicate_id=4,
            arbitrage_entry_bps=10.0,
            arbitrage_exit_bps=3.0,
        )

        experiment = overridden.aml_config["ecology"]["experiment"]
        self.assertEqual(experiment["master_seed"], 90210)
        self.assertEqual(experiment["replicate_id"], 4)
        for agent in overridden.stocksim_config["agents"].values():
            self.assertNotIn("random_seed", agent.get("parameters", {}))
        parameters = overridden.stocksim_config["agents"]["basis_arbitrageur"][
            "parameters"
        ]
        self.assertEqual(parameters["entry_threshold_bps"], 10.0)
        self.assertEqual(parameters["exit_threshold_bps"], 3.0)
        self.assertEqual(
            scenario.stocksim_config["agents"]["basis_arbitrageur"]["parameters"][
                "entry_threshold_bps"
            ],
            10,
        )

    def test_final_ecology_scenarios_align_cadence_and_paired_shock(self) -> None:
        filenames = (
            "financial_ecology_stock_future_d0_disconnected.yaml",
            "financial_ecology_stock_future_i1_information.yaml",
            "financial_ecology_stock_future_foundation.yaml",
            "financial_ecology_stock_future_d0_disconnected_llm.yaml",
            "financial_ecology_stock_future_i1_information_llm.yaml",
            "financial_ecology_stock_future_foundation_llm.yaml",
        )
        for filename in filenames:
            scenario = load_scenario(PROJECT_ROOT / "scenarios" / "research" / filename)
            agents = scenario.stocksim_config["agents"]
            for details in agents.values():
                if details["type"] == "AML_Shock_Agent":
                    continue
                parameters = details["parameters"]
                self.assertEqual(parameters["action_interval"], "30s", filename)
                self.assertEqual(parameters["slow_loop_interval"], "3m", filename)

            shock = agents["shock_agent"]["parameters"]
            random_events = shock["random_events"]
            self.assertTrue(random_events["enabled"], filename)
            self.assertEqual(random_events["max_events"], 1, filename)
            self.assertEqual(random_events["start_tick"], 58, filename)
            self.assertEqual(random_events["end_tick"], 58, filename)
            template = random_events["templates"][0]
            self.assertEqual(template["duration_ticks"], 8, filename)
            self.assertEqual(template["severity_range"], [0.45, 0.75], filename)

            if filename.endswith("_llm.yaml"):
                self.assertEqual(
                    scenario.stocksim_config["simulation"]["max_wall_time_seconds"],
                    3600,
                    filename,
                )

    def test_three_market_two_shock_c0_and_c2_scenarios(self) -> None:
        filenames = (
            "financial_ecology_stock_future_bond_c0_control.yaml",
            "financial_ecology_stock_future_bond_c2_connected.yaml",
        )
        scenarios = {}
        for filename in filenames:
            scenario = load_scenario(PROJECT_ROOT / "scenarios" / "research" / filename)
            scenarios[filename] = scenario
            config = scenario.stocksim_config
            self.assertEqual(config["instruments"], ["AAPL", "AAPL_FUT", "UST10Y"])
            self.assertEqual(config["exchanges"]["UST10Y"]["asset_class"], "bond")
            self.assertEqual(config["exchanges"]["UST10Y"]["duration"], 8.0)
            self.assertEqual(config["simulation"]["expected_exchange_agent_count"], 3)

            agents = config["agents"]
            for details in agents.values():
                if details["type"] == "AML_Shock_Agent":
                    continue
                parameters = details["parameters"]
                self.assertEqual(parameters["action_interval"], "30s")
                self.assertEqual(parameters["slow_loop_interval"], "3m")
                self.assertEqual(parameters["slow_strategist"]["type"], "frozen")

            shock = agents["shock_agent"]["parameters"]
            micro = shock["random_events"]
            self.assertEqual(micro["start_tick"], 30)
            self.assertEqual(micro["end_tick"], 30)
            self.assertEqual(micro["max_events"], 1)
            self.assertEqual(micro["templates"][0]["affected_instruments"], ["AAPL"])

            macro = shock["scheduled_events"][0]
            self.assertEqual(macro["announce_tick"], 54)
            self.assertEqual(macro["tick"], 60)
            self.assertEqual(macro["state_scope"], "global")
            self.assertEqual(macro["affected_instruments"], ["AAPL", "AAPL_FUT", "UST10Y"])
            self.assertEqual(
                macro["per_instrument_effects"]["UST10Y"]["fundamental_price_shift"],
                -1.44,
            )

        self.assertEqual(
            scenarios[filenames[0]].aml_config["ecology"]["relationships"],
            [],
        )
        c2_relationships = scenarios[filenames[1]].aml_config["ecology"]["relationships"]
        self.assertEqual(len(c2_relationships), 1)
        self.assertTrue(c2_relationships[0]["channels"]["arbitrage"])
        self.assertIn(
            "basis_arbitrageur",
            scenarios[filenames[1]].stocksim_config["agents"],
        )

    def test_slow_strategist_override_applies_to_every_trading_agent(self) -> None:
        scenario = load_scenario(
            PROJECT_ROOT
            / "scenarios"
            / "research"
            / "financial_ecology_stock_future_bond_c2_connected.yaml"
        )
        overridden = apply_research_overrides(scenario, slow_strategist="openai")

        for details in overridden.stocksim_config["agents"].values():
            if details["type"] == "AML_Shock_Agent":
                continue
            self.assertEqual(details["parameters"]["slow_strategist"], {"type": "openai"})
        self.assertEqual(
            scenario.stocksim_config["agents"]["bond_market_maker"]["parameters"]
            ["slow_strategist"],
            {"type": "frozen"},
        )

    def test_c0_macro_event_prices_the_bond_and_reverts_global_state(self) -> None:
        scenario = load_scenario(
            PROJECT_ROOT
            / "scenarios"
            / "research"
            / "financial_ecology_stock_future_bond_c0_control.yaml"
        )
        config = scenario.stocksim_config
        shock = config["agents"]["shock_agent"]["parameters"]
        event = shock["scheduled_events"][0]
        metadata = normalize_instrument_metadata(
            config["instruments"],
            config["exchanges"],
        )
        payload = build_shock_payload(
            event["id"],
            event,
            {"tick_id": 60, "current_time": "2025-03-01T10:00:00+00:00"},
            default_duration_ticks=shock["default_duration_ticks"],
            instrument_metadata=metadata,
            market_state=shock["initial_market_state"],
        )
        engine = ScopedMarketStateEngine(shock["initial_market_state"], metadata)
        engine.register(
            event["id"],
            event,
            payload,
            current_tick_id=60,
            default_duration_ticks=shock["default_duration_ticks"],
        )

        active_state = effective_market_state(engine.snapshot(60), "UST10Y")
        recovered_state = effective_market_state(engine.snapshot(80), "UST10Y")
        self.assertEqual(payload["instrument_effects"]["UST10Y"]["fundamental_price_shift"], -1.44)
        self.assertEqual(active_state["policy_rate_bps"], 450.0)
        self.assertLess(active_state["liquidity_index"], 1.0)
        self.assertEqual(recovered_state["policy_rate_bps"], 425.0)
        self.assertEqual(recovered_state["liquidity_index"], 1.0)

    def test_observation_separates_active_and_historical_events(self) -> None:
        agent = type(
            "Agent",
            (),
            {
                "instrument_exchange_map": {"AAPL": "exchange_aapl"},
                "current_time": "2025-03-01T10:06:00+00:00",
            },
        )()
        historical_event = {
            "shock_id": "stock_news",
            "phase": "active",
            "context_status": "historical",
            "is_active": False,
        }
        observation = build_observation_context(
            agent,
            events=[],
            known_events=[historical_event],
        )
        event_context = observation["event_context"]
        self.assertEqual(event_context["active"], [])
        self.assertEqual(event_context["historical"][0]["shock_id"], "stock_news")
        self.assertFalse(event_context["known"][0]["is_active"])

    def test_expired_active_phase_is_historical_in_agent_context(self) -> None:
        class TestAgent(BaseAMLAgent):
            async def run_fast_loop(self, observation):
                del observation

        agent = object.__new__(TestAgent)
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

    def test_strategy_change_ignores_decision_metadata(self) -> None:
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

    def test_frozen_slow_strategist_is_a_true_control(self) -> None:
        current = {"risk_mode": "normal", "confidence": 1.0}
        strategist = create_llm_strategist("retail", {"type": "frozen"})
        self.assertIsInstance(strategist, FrozenStrategist)
        self.assertIs(strategist.propose({}, current), current)

    def test_disabled_slow_strategist_keeps_legacy_static_behavior(self) -> None:
        strategist = create_llm_strategist("retail", {"enabled": False})
        self.assertIsInstance(strategist, LLMStrategist)

    def test_legacy_scenario_does_not_receive_ecology_metadata(self) -> None:
        scenario = load_scenario(PROJECT_ROOT / "scenarios" / "aml_agent_infra_smoke.yaml")
        with TemporaryDirectory() as directory:
            run = create_run(scenario, Path(directory), run_id="legacy")
            metadata = loads(run.metadata_path.read_text(encoding="utf-8"))
        self.assertNotIn("ecology", metadata)

    def test_reporter_deduplicates_two_sides_of_one_trade(self) -> None:
        with TemporaryDirectory() as directory:
            reports_dir = Path(directory) / "reports"
            reports_dir.mkdir()
            metadata_path = Path(directory) / "metadata.json"
            metadata_path.write_text(dumps({"run_id": "test"}), encoding="utf-8")
            trade = {
                "event_type": "trade_executed",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "instrument": "AAPL",
                "price": 100.0,
                "quantity": 10,
                "raw_trade": {"instrument": "AAPL", "seq": 1, "price": 100.0, "quantity": 10},
            }
            (reports_dir / "trader_actions.json").write_text(
                dumps(
                    {
                        "actions": [
                            trade,
                            {**trade, "agent_id": "other_side"},
                            {
                                "event_type": "cross_market_decision",
                                "timestamp": "2025-01-01T00:00:00+00:00",
                                "ecology": {
                                    "decision_id": "basis:1:aapl_future",
                                    "relationship_id": "aapl_future",
                                    "linkage_channel": "arbitrage",
                                },
                            },
                            {
                                "event_type": "order_submitted",
                                "order_id": "source-order",
                                "instrument": "AAPL",
                                "side": "BUY",
                                "quantity": 10,
                                "ecology": {
                                    "decision_id": "basis:1:aapl_future",
                                    "relationship_id": "aapl_future",
                                    "linkage_channel": "arbitrage",
                                    "leg": "source",
                                },
                            },
                            {
                                "event_type": "order_submitted",
                                "order_id": "target-order",
                                "instrument": "AAPL_FUT",
                                "side": "SELL",
                                "quantity": 10,
                                "ecology": {
                                    "decision_id": "basis:1:aapl_future",
                                    "relationship_id": "aapl_future",
                                    "linkage_channel": "arbitrage",
                                    "leg": "target",
                                },
                            },
                            {
                                "event_type": "trade_executed",
                                "timestamp": "2025-01-01T00:00:00+00:00",
                                "instrument": "AAPL",
                                "price": 100.0,
                                "quantity": 10,
                                "raw_trade": dict(trade["raw_trade"]),
                                "ecology": {
                                    "decision_id": "basis:1:aapl_future",
                                    "relationship_id": "aapl_future",
                                    "linkage_channel": "arbitrage",
                                    "leg": "source",
                                },
                            },
                            {
                                "event_type": "trade_executed",
                                "instrument": "AAPL_FUT",
                                "price": 101.0,
                                "quantity": 10,
                                "ecology": {
                                    "decision_id": "basis:1:aapl_future",
                                    "relationship_id": "aapl_future",
                                    "linkage_channel": "arbitrage",
                                    "leg": "target",
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            generate_ecology_reports(
                reports_dir=reports_dir,
                metadata_path=metadata_path,
                aml_config={
                    "ecology": {
                        "enabled": True,
                        "relationships": [
                            {
                                "id": "aapl_future",
                                "type": "spot_future",
                                "source": "AAPL",
                                "target": "AAPL_FUT",
                                "channels": ["valuation", "arbitrage"],
                            }
                        ],
                    }
                },
                stocksim_config={
                    "instruments": ["AAPL", "AAPL_FUT"],
                    "exchanges": {"AAPL": {}, "AAPL_FUT": {}},
                },
            )
            summary = loads((reports_dir / "ecology_market_summary.json").read_text())
            ledger = loads((reports_dir / "ecology_channel_ledger.json").read_text())
            decisions = loads((reports_dir / "ecology_decision_summary.json").read_text())
            self.assertEqual(summary["markets"]["AAPL"]["trade_count"], 1)
            self.assertEqual(ledger["channel_event_counts"]["arbitrage"], 5)
            self.assertEqual(decisions["decisions"][0]["execution_outcome"], "fully_hedged")

    def test_reporter_exports_delivery_and_slow_loop_evidence(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            reports_dir = run_dir / "reports"
            reports_dir.mkdir()
            metadata_path = run_dir / "metadata.json"
            metadata_path.write_text(dumps({"run_id": "test"}), encoding="utf-8")
            (reports_dir / "trader_actions.json").write_text(
                dumps(
                    {
                        "actions": [
                            {
                                "event_type": "event_observed",
                                "timestamp": "2025-01-01T00:00:00+00:00",
                                "agent_id": "future_agent",
                                "shock_id": "stock_news",
                                "phase": "active",
                                "delivery_type": "relationship_information",
                                "information_relationship_ids": ["aapl_future"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            context_dir = run_dir / "decision_context" / "future_agent"
            context_dir.mkdir(parents=True)
            (context_dir / "memory.json").write_text(
                dumps(
                    {
                        "agent_id": "future_agent",
                        "events": [
                            {
                                "event_type": "slow_loop_decision",
                                "timestamp": "2025-01-01T00:01:00+00:00",
                                "payload": {
                                    "strategy_changed": True,
                                    "slow_loop_status": "completed",
                                    "strategy_changed_fields": ["risk_mode"],
                                    "metadata_changed_fields": ["reason"],
                                    "confidence_changed": False,
                                    "event_context_present": True,
                                    "possible_event_influence": True,
                                    "active_event_ids": ["stock_news"],
                                    "known_event_ids": ["stock_news"],
                                    "primary_event": {"event_id": "stock_news"},
                                    "strategy_before": {
                                        "risk_mode": "normal",
                                        "confidence": 0.5,
                                    },
                                    "strategy_after": {
                                        "risk_mode": "risk_off",
                                        "confidence": 0.8,
                                    },
                                    "reason": "Observed linked stock news.",
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            generate_ecology_reports(
                reports_dir=reports_dir,
                metadata_path=metadata_path,
                aml_config={"ecology": {"enabled": True, "relationships": []}},
                stocksim_config={"instruments": [], "exchanges": {}},
            )
            ledger = loads((reports_dir / "ecology_channel_ledger.json").read_text())
            responses = loads(
                (reports_dir / "ecology_agent_response_summary.json").read_text()
            )
            self.assertEqual(
                ledger["event_delivery_counts"][
                    "stock_news:active:relationship_information"
                ],
                1,
            )
            self.assertEqual(responses["active_event_response_count"], 1)
            self.assertEqual(responses["failed_slow_loop_count"], 0)
            self.assertEqual(responses["responses"][0]["risk_mode_after"], "risk_off")


if __name__ == "__main__":
    unittest.main()
