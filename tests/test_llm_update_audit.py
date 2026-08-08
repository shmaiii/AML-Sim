from __future__ import annotations

import unittest

from aml_sim.agents.models.state import RetailStrategyState
from aml_sim.agents.strategy.llm_slow_strategy import (
    LLMStrategist,
    OpenAIJSONLLMClient,
    StaticJSONLLMClient,
    create_llm_strategist,
)


class LLMUpdateAuditTests(unittest.IsolatedAsyncioTestCase):
    def test_openai_factory_preserves_max_output_tokens(self) -> None:
        strategist = create_llm_strategist(
            "retail",
            {
                "type": "openai",
                "model": "gpt-5.4",
                "max_output_tokens": 512,
                "allowed_strategy_fields": ["trade_probability"],
            },
        )

        self.assertIsInstance(strategist.client, OpenAIJSONLLMClient)
        self.assertEqual(512, strategist.client.max_output_tokens)

    async def test_records_applied_and_rejected_fields_with_reasons(self) -> None:
        strategist = LLMStrategist(
            client=StaticJSONLLMClient({
                "strategy_updates": {
                    "buy_bias": 0.7,
                    "trade_probability": 0.9,
                    "misspelled_field": 123,
                }
            }),
            allowed_strategy_fields={"buy_bias"},
        )
        proposal = await strategist.propose(
            {"current_time": "2025-03-01T09:30:00+00:00"},
            RetailStrategyState(),
        )
        self.assertEqual(0.7, proposal.buy_bias)
        self.assertEqual(0.3, proposal.trade_probability)
        audit = strategist.last_update_audit
        self.assertEqual(0.7, audit["eligible_updates"]["buy_bias"])
        self.assertEqual(
            "not_allowed_by_configuration",
            audit["rejected_updates"]["trade_probability"]["reason"],
        )
        self.assertEqual(
            "unknown_strategy_field",
            audit["rejected_updates"]["misspelled_field"]["reason"],
        )

    async def test_clamps_numeric_values_without_hiding_the_original(self) -> None:
        strategist = LLMStrategist(
            client=StaticJSONLLMClient({
                "strategy_updates": {
                    "trade_probability": 1.4,
                    "buy_bias": "0.7",
                }
            }),
            allowed_strategy_fields={"trade_probability", "buy_bias"},
        )
        proposal = await strategist.propose(
            {"current_time": "2025-03-01T09:30:00+00:00"},
            RetailStrategyState(),
        )

        self.assertEqual(1.0, proposal.trade_probability)
        self.assertEqual(0.5, proposal.buy_bias)
        audit = strategist.last_update_audit
        self.assertEqual(
            {
                "original": 1.4,
                "applied": 1.0,
                "reason": "clamped_to_valid_range",
            },
            audit["adjusted_updates"]["trade_probability"],
        )
        self.assertEqual(
            "invalid_numeric_type",
            audit["rejected_updates"]["buy_bias"]["reason"],
        )
