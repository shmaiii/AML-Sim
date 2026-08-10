from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_openai_response_log_records_identity_and_token_usage(self) -> None:
        client = OpenAIJSONLLMClient(model="gpt-test")
        response = SimpleNamespace(
            id="resp_123",
            model="gpt-test-2026-01-01",
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
            ),
        )
        context = {
            "observation": {
                "agent": {"agent_id": "retail_1"},
                "current_time": "2025-03-01T09:30:00+00:00",
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"DECISION_CONTEXT_DIR": temp_dir},
        ):
            client._write_response_log(
                context=context,
                content='{"strategy_updates": {}}',
                response=response,
            )
            record = json.loads(
                (Path(temp_dir) / "retail_1" / "llm_responses.jsonl")
                .read_text(encoding="utf-8")
            )

        self.assertEqual("resp_123", record["response_id"])
        self.assertEqual("gpt-test-2026-01-01", record["response_model"])
        self.assertEqual(150, record["usage"]["total_tokens"])

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

    async def test_top_level_confidence_and_reason_are_the_audited_proposal(self) -> None:
        strategist = LLMStrategist(
            client=StaticJSONLLMClient({
                "strategy_updates": {
                    "trade_probability": 0.55,
                    "confidence": 0.45,
                    "reason": "nested values",
                },
                "confidence": 0.74,
                "reason": "top-level values",
            }),
            allowed_strategy_fields={"trade_probability", "confidence", "reason"},
        )

        proposal = await strategist.propose(
            {"current_time": "2025-03-01T09:30:00+00:00"},
            RetailStrategyState(),
        )

        self.assertEqual(0.74, proposal.confidence)
        self.assertEqual("top-level values", proposal.reason)
        audit = strategist.last_update_audit
        self.assertEqual(0.74, audit["proposed_updates"]["confidence"])
        self.assertEqual("top-level values", audit["proposed_updates"]["reason"])
        self.assertEqual(
            "top_level_override",
            audit["adjusted_updates"]["confidence"]["reason"],
        )
