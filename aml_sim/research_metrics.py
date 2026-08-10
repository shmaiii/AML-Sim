"""Reproducible diversity, synchrony, and liquidity metrics for AML runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0 else None


def build_research_metrics(agent_reports_dir: Path, reports_dir: Path) -> dict[str, Any]:
    """Compute signed-flow synchrony and order-book liquidity statistics."""
    flow_by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    role_by_agent: dict[str, str] = {}

    for action_file in sorted(agent_reports_dir.glob("trader_actions_*.json")):
        agent_id = action_file.stem.removeprefix("trader_actions_")
        with action_file.open("r", encoding="utf-8") as handle:
            actions = json.load(handle)
        role_by_agent.setdefault(agent_id, "unspecified")
        for action in actions if isinstance(actions, list) else []:
            if isinstance(action, dict) and action.get("agent_role"):
                role_by_agent[agent_id] = str(action["agent_role"])
            if not isinstance(action, dict) or action.get("event_type") != "order_submitted":
                continue
            side = str(action.get("side", "")).upper()
            if side not in {"BUY", "SELL"}:
                continue
            timestamp = str(action.get("timestamp") or "")
            instrument = str(action.get("instrument") or "")
            if not timestamp or not instrument:
                continue
            quantity = float(action.get("quantity") or 0.0)
            flow_by_key[(timestamp, instrument)][agent_id] += (
                quantity if side == "BUY" else -quantity
            )
            role_by_agent[agent_id] = str(action.get("agent_role") or "unspecified")

    agents = sorted(role_by_agent)
    keys = sorted(flow_by_key)
    flow_rows: list[dict[str, Any]] = []
    herding_values: list[float] = []
    same_direction_values: list[float] = []
    for timestamp, instrument in keys:
        flows = flow_by_key[(timestamp, instrument)]
        gross = sum(abs(value) for value in flows.values())
        net = sum(flows.values())
        active = [value for value in flows.values() if value != 0]
        herding = abs(net) / gross if gross > 0 and len(active) >= 2 else None
        same_direction = (
            max(sum(value > 0 for value in active), sum(value < 0 for value in active))
            / len(active)
            if len(active) >= 2 else None
        )
        if herding is not None:
            herding_values.append(herding)
        if same_direction is not None:
            same_direction_values.append(same_direction)
        for agent_id in agents:
            flow_rows.append({
                "timestamp": timestamp,
                "instrument": instrument,
                "agent_id": agent_id,
                "agent_role": role_by_agent.get(agent_id, "unspecified"),
                "signed_order_flow": flows.get(agent_id, 0.0),
                "tick_herding_index": herding,
                "tick_same_direction_share": same_direction,
            })

    pairwise: list[dict[str, Any]] = []
    within_role_correlations: list[float] = []
    cross_role_correlations: list[float] = []
    for index, left_agent in enumerate(agents):
        for right_agent in agents[index + 1:]:
            left = [flow_by_key[key].get(left_agent, 0.0) for key in keys]
            right = [flow_by_key[key].get(right_agent, 0.0) for key in keys]
            correlation = _pearson(left, right)
            left_role = role_by_agent.get(left_agent, "unspecified")
            right_role = role_by_agent.get(right_agent, "unspecified")
            if "unspecified" in {left_role, right_role}:
                role_relation = "unknown"
            elif left_role == right_role:
                role_relation = "within_role"
                if correlation is not None:
                    within_role_correlations.append(abs(correlation))
            else:
                role_relation = "cross_role"
                if correlation is not None:
                    cross_role_correlations.append(abs(correlation))
            pairwise.append({
                "left_agent": left_agent,
                "left_role": left_role,
                "right_agent": right_agent,
                "right_role": right_role,
                "role_relation": role_relation,
                "correlation": correlation,
            })

    microstructure_rows: list[dict[str, str]] = []
    microstructure_file = reports_dir / "order_book_microstructure.csv"
    if microstructure_file.exists():
        with microstructure_file.open("r", encoding="utf-8", newline="") as handle:
            microstructure_rows = list(csv.DictReader(handle))

    def has_value(row: dict[str, str], field: str) -> bool:
        return row.get(field) not in {None, "", "None"}

    analysis_rows: list[dict[str, str]] = []
    book_started = False
    for row in microstructure_rows:
        try:
            active_order_count = float(row.get("active_order_count") or 0.0)
        except (TypeError, ValueError):
            active_order_count = 0.0
        if (
            active_order_count > 0
            or has_value(row, "best_bid")
            or has_value(row, "best_ask")
            or has_value(row, "spread")
        ):
            book_started = True
        if book_started:
            analysis_rows.append(row)

    def numeric(field: str, rows: list[dict[str, str]]) -> list[float]:
        values: list[float] = []
        for row in rows:
            raw = row.get(field)
            if raw in {None, "", "None"}:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        return values

    liquidity = {}
    for field in (
        "spread", "relative_spread_bps", "bid_depth", "ask_depth",
        "top5_bid_depth", "top5_ask_depth", "fill_rate",
        "signed_price_impact_bps",
    ):
        values = numeric(field, analysis_rows)
        liquidity[f"mean_{field}"] = mean(values) if values else None
    top5_total_depth = [
        bid_depth + ask_depth
        for row in analysis_rows
        for bid_depth in numeric("top5_bid_depth", [row])
        for ask_depth in numeric("top5_ask_depth", [row])
    ]
    liquidity["mean_top5_total_depth"] = (
        mean(top5_total_depth) if top5_total_depth else None
    )
    trade_counts = numeric("trade_count", analysis_rows)
    liquidity["trade_active_tick_rate"] = (
        sum(value > 0 for value in trade_counts) / len(analysis_rows)
        if analysis_rows else None
    )
    liquidity["valid_book_observation_count"] = len(analysis_rows)

    metrics = {
        "synchrony": {
            "agent_count": len(agents),
            "observation_count": len(keys),
            "mean_signed_flow_herding_index": mean(herding_values) if herding_values else None,
            "mean_same_direction_share": mean(same_direction_values) if same_direction_values else None,
            "mean_absolute_within_role_signed_flow_correlation": (
                mean(within_role_correlations)
                if within_role_correlations else None
            ),
            "mean_absolute_cross_role_signed_flow_correlation": (
                mean(cross_role_correlations)
                if cross_role_correlations else None
            ),
            "pairwise_signed_flow_correlations": pairwise,
        },
        "liquidity": liquidity,
        "definitions": {
            "signed_order_flow": "BUY submitted quantity minus SELL submitted quantity per agent/timestamp/asset",
            "signed_flow_herding_index": "absolute net signed flow divided by gross absolute signed flow",
            "within_role_signed_flow_correlation": "mean absolute Pearson correlation of signed order flow for agent pairs with the same role",
            "cross_role_signed_flow_correlation": "mean absolute Pearson correlation of signed order flow for agent pairs with different roles",
            "top5_total_depth": "sum of displayed bid and ask quantity across the top five price levels",
            "fill_rate": "executed quantity divided by submitted quantity between exchange ticks",
            "signed_price_impact_bps": "midpoint return aligned to the sign of aggregate submitted order flow",
            "trade_active_tick_rate": "share of post-initialization exchange ticks with one or more executed trades",
        },
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "signed_order_flow.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "timestamp", "instrument", "agent_id", "agent_role",
            "signed_order_flow", "tick_herding_index", "tick_same_direction_share",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flow_rows)
    with (reports_dir / "research_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
