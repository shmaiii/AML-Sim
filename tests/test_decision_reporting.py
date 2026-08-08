from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from aml_sim.reporting import (
    AGENT_DECISION_FIELDS,
    INTERVAL_OUTCOME_FIELDS,
    generate_agent_decision_csv,
    generate_interval_outcome_csv,
)


class AgentDecisionReportingTests(unittest.TestCase):
    def test_keeps_latest_daily_decision_per_agent_and_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            agent_dir = root / "agents"
            reports_dir = root / "reports"
            agent_dir.mkdir()
            rows = [
                {
                    "decision_id": "dec_early",
                    "agent_name": "retail_1",
                    "asset": "AAPL",
                    "decision_date": "2025-03-01",
                    "decision_timestamp": "2025-03-01T09:30:00+00:00",
                    "data_cutoff_timestamp": "2025-03-01T09:29:30+00:00",
                    "prediction_score": -0.2,
                    "confidence": 0.6,
                    "action": "HOLD",
                    "submitted_action": "HOLD",
                    "latest_data_date_used": "2025-03-01T09:29:30+00:00",
                    "data_timestamp_source": "market_snapshot",
                    "split": "validation",
                },
                {
                    "decision_id": "dec_late",
                    "agent_name": "retail_1",
                    "asset": "AAPL",
                    "decision_date": "2025-03-01",
                    "decision_timestamp": "2025-03-01T09:35:00+00:00",
                    "data_cutoff_timestamp": "2025-03-01T09:34:30+00:00",
                    "prediction_score": 0.4,
                    "confidence": 0.8,
                    "action": "BUY",
                    "submitted_action": "BUY",
                    "latest_data_date_used": "2025-03-01T09:34:30+00:00",
                    "data_timestamp_source": "market_snapshot",
                    "split": "validation",
                },
            ]
            decision_file = agent_dir / "agent_decisions_retail_1.json"
            decision_file.write_text(json.dumps(rows), encoding="utf-8")

            generate_agent_decision_csv(agent_dir, reports_dir)

            with (reports_dir / "agent_decisions.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                output_rows = list(csv.DictReader(handle))

            self.assertEqual(1, len(output_rows))
            self.assertEqual(AGENT_DECISION_FIELDS, list(output_rows[0]))
            self.assertEqual("2025-03-01T09:35:00+00:00", output_rows[0]["decision_timestamp"])
            self.assertEqual("BUY", output_rows[0]["action"])
            self.assertEqual("BUY", output_rows[0]["submitted_action"])
            self.assertEqual("validation", output_rows[0]["split"])
            with (reports_dir / "agent_decisions_detailed.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                detailed_rows = list(csv.DictReader(handle))
            self.assertEqual(2, len(detailed_rows))
            self.assertEqual("dec_early", detailed_rows[0]["decision_id"])
            self.assertEqual("dec_late", detailed_rows[1]["decision_id"])

    def test_compares_equivalent_timestamp_offsets_chronologically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            agent_dir = root / "agents"
            agent_dir.mkdir()
            rows = [
                {
                    "decision_id": "dec_offset",
                    "agent_name": "retail_1",
                    "asset": "AAPL",
                    "decision_date": "2025-03-01",
                    "decision_timestamp": "2025-03-01T10:00:00+01:00",
                    "action": "HOLD",
                },
                {
                    "decision_id": "dec_utc",
                    "agent_name": "retail_1",
                    "asset": "AAPL",
                    "decision_date": "2025-03-01",
                    "decision_timestamp": "2025-03-01T09:30:00+00:00",
                    "action": "BUY",
                },
            ]
            (agent_dir / "agent_decisions_retail_1.json").write_text(
                json.dumps(rows),
                encoding="utf-8",
            )

            generate_agent_decision_csv(agent_dir, root / "reports")

            with (root / "reports" / "agent_decisions.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                output_rows = list(csv.DictReader(handle))
            self.assertEqual("BUY", output_rows[0]["action"])

    def test_combines_interval_outcomes_by_decision_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            agent_dir = root / "agents"
            reports_dir = root / "reports"
            agent_dir.mkdir()
            outcomes = [
                {
                    "decision_id": "dec_1",
                    "agent_name": "retail_1",
                    "asset": "AAPL",
                    "interval_start": "2025-03-01T09:30:00+00:00",
                    "interval_end": "2025-03-01T09:31:00+00:00",
                    "result_available_timestamp": "2025-03-01T09:31:00+00:00",
                    "interval_return": 0.01,
                    "interval_pnl": 100.0,
                    "portfolio_value": 10100.0,
                    "realized_volatility": 0.01,
                    "drawdown": 0.0,
                    "gross_exposure": 1000.0,
                    "net_exposure": 1000.0,
                    "assigned_risk_budget": 5000.0,
                    "outcome_status": "completed",
                    "split": "validation",
                }
            ]
            (agent_dir / "interval_outcomes_retail_1.json").write_text(
                json.dumps(outcomes), encoding="utf-8"
            )

            generate_interval_outcome_csv(agent_dir, reports_dir)

            with (reports_dir / "interval_outcomes.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                output_rows = list(csv.DictReader(handle))
            self.assertEqual(1, len(output_rows))
            self.assertEqual(INTERVAL_OUTCOME_FIELDS, list(output_rows[0]))
            self.assertEqual("dec_1", output_rows[0]["decision_id"])
            self.assertEqual("completed", output_rows[0]["outcome_status"])

    def test_empty_input_still_writes_csv_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generate_agent_decision_csv(root / "agents", root / "reports")

            header = (root / "reports" / "agent_decisions.csv").read_text(
                encoding="utf-8"
            ).strip()
            self.assertEqual(",".join(AGENT_DECISION_FIELDS), header)
            detailed_header = (
                root / "reports" / "agent_decisions_detailed.csv"
            ).read_text(encoding="utf-8").strip()
            self.assertEqual(",".join(AGENT_DECISION_FIELDS), detailed_header)

            generate_interval_outcome_csv(root / "agents", root / "reports")
            outcome_header = (
                root / "reports" / "interval_outcomes.csv"
            ).read_text(encoding="utf-8").strip()
            self.assertEqual(",".join(INTERVAL_OUTCOME_FIELDS), outcome_header)


if __name__ == "__main__":
    unittest.main()
