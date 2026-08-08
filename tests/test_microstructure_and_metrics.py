from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
STOCKSIM_DIR = ROOT / "simulators" / "StockSim"
if str(STOCKSIM_DIR) not in sys.path:
    sys.path.insert(0, str(STOCKSIM_DIR))

from aml_sim.research_metrics import build_research_metrics
from exchanges.exchange_agent import ExchangeAgent
from utils.orders import Order, OrderType, Side


class MicrostructureAndMetricsTests(unittest.TestCase):
    def test_microstructure_export_contains_full_depth_and_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bid = Order(uuid.uuid4(), "AAPL", Side.BUY, 30, OrderType.LIMIT, "mm_1", 99.9)
            ask = Order(uuid.uuid4(), "AAPL", Side.SELL, 40, OrderType.LIMIT, "mm_2", 100.1)
            agent = object.__new__(ExchangeAgent)
            agent.order_book = SimpleNamespace(
                bids={99.9: deque([bid])},
                asks={100.1: deque([ask])},
                trade_history=[],
                trade_seq_counter=0,
                cumulative_traded_volume=0,
                cumulative_turnover=0.0,
            )
            agent.agent_id = "exchange_aapl"
            agent.instrument = "AAPL"
            agent.current_time = datetime(2025, 3, 1, 9, 30, tzinfo=timezone.utc)
            agent.microstructure_output_dir = temp_dir
            agent._tick_submitted_buy_qty = 30
            agent._tick_submitted_sell_qty = 10
            agent._tick_cancelled_buy_qty = 0
            agent._tick_cancelled_sell_qty = 0
            agent._last_microstructure_trade_seq = 0
            agent._last_microstructure_midpoint = None
            agent.logger = SimpleNamespace(error=lambda *_: None)
            agent._export_microstructure_tick(7)

            path = Path(temp_dir) / "order_book_microstructure_AAPL.jsonl"
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(0.2, row["spread"])
            self.assertEqual(30, row["bid_depth"])
            self.assertEqual(40, row["ask_depth"])
            self.assertEqual(20, row["signed_order_flow"])
            self.assertEqual(1, row["bid_levels"][0]["order_count"])

    def test_research_metrics_use_signed_order_flow_and_liquidity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            agents = root / "agents"
            reports = root / "reports"
            agents.mkdir()
            reports.mkdir()
            for agent_id, side in (("a", "BUY"), ("b", "BUY")):
                (agents / f"trader_actions_{agent_id}.json").write_text(
                    json.dumps([{
                        "event_type": "order_submitted",
                        "timestamp": "2025-03-01T09:30:00+00:00",
                        "instrument": "AAPL",
                        "agent_id": agent_id,
                        "agent_role": "retail",
                        "side": side,
                        "quantity": 10,
                    }]),
                    encoding="utf-8",
                )
            with (reports / "order_book_microstructure.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["spread", "bid_depth", "ask_depth", "fill_rate"])
                writer.writeheader()
                writer.writerow({"spread": 0.2, "bid_depth": 100, "ask_depth": 120, "fill_rate": 0.5})
            metrics = build_research_metrics(agents, reports)
            self.assertEqual(1.0, metrics["synchrony"]["mean_signed_flow_herding_index"])
            self.assertEqual(0.2, metrics["liquidity"]["mean_spread"])
            self.assertTrue((reports / "signed_order_flow.csv").exists())
