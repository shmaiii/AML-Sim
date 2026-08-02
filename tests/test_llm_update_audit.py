from __future__ import annotations

import unittest

from aml_sim.agents.models.state import RetailStrategyState
from aml_sim.agents.strategy.llm_slow_strategy import LLMStrategist, StaticJSONLLMClient


class LLMUpdateAuditTests(unittest.IsolatedAsyncioTestCase):
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
