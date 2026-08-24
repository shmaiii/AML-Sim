"""Research artifacts for causal cross-market simulation analysis."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from aml_sim.ecology.config import load_ecology_config
from aml_sim.ecology.registry import RelationshipRegistry


def generate_ecology_reports(
    *,
    reports_dir: Path,
    metadata_path: Path,
    aml_config: Mapping[str, Any],
    stocksim_config: Mapping[str, Any],
) -> None:
    """Export explicit graph, event, market, and channel artifacts for one run."""
    ecology_config = load_ecology_config(aml_config)
    if not ecology_config.enabled:
        return

    instruments = list(stocksim_config.get("instruments", []))
    exchanges = stocksim_config.get("exchanges", {})
    instrument_metadata = {
        instrument: dict(exchanges.get(instrument, {}))
        for instrument in instruments
    }
    registry = RelationshipRegistry(instrument_metadata, ecology_config.relationships)
    action_report = _load_json(reports_dir / "trader_actions.json", {})
    actions = action_report.get("actions", []) if isinstance(action_report, Mapping) else []
    if not isinstance(actions, list):
        actions = []

    metadata = _load_json(metadata_path, {})
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "id": ecology_config.experiment.experiment_id,
            "treatment": ecology_config.experiment.treatment,
            "master_seed": ecology_config.experiment.master_seed,
            "replicate_id": ecology_config.experiment.replicate_id,
        },
        "relationships": registry.as_manifest(),
        "run_id": metadata.get("run_id") if isinstance(metadata, Mapping) else None,
    }
    _write_json(reports_dir / "ecology_manifest.json", manifest)

    event_index: dict[tuple[Any, Any], dict[str, Any]] = {}
    market_trades: dict[str, list[dict[str, Any]]] = defaultdict(list)
    observed_trade_keys: set[tuple[Any, ...]] = set()
    channel_events: list[dict[str, Any]] = []
    channel_counts: Counter[str] = Counter()
    decision_events: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    event_deliveries: list[dict[str, Any]] = []
    delivery_counts: Counter[str] = Counter()

    for action in actions:
        if not isinstance(action, Mapping):
            continue
        event_type = action.get("event_type")
        if event_type == "event_observed":
            key = (action.get("shock_id"), action.get("phase"))
            event_index.setdefault(key, dict(action))
            shock_id = action.get("shock_id")
            if shock_id:
                phase = str(action.get("phase", "active"))
                delivery_type = str(action.get("delivery_type", "direct"))
                delivery_counts[f"{shock_id}:{phase}:{delivery_type}"] += 1
                event_deliveries.append(
                    {
                        "timestamp": action.get("timestamp"),
                        "agent_id": action.get("agent_id"),
                        "shock_id": shock_id,
                        "phase": phase,
                        "delivery_type": delivery_type,
                        "information_relationship_ids": action.get(
                            "information_relationship_ids", []
                        ),
                        "affected_instruments": action.get(
                            "affected_instruments", []
                        ),
                    }
                )
        if event_type == "trade_executed":
            raw_trade = action.get("raw_trade", {})
            if not isinstance(raw_trade, Mapping):
                raw_trade = {}
            key = (
                raw_trade.get("instrument", action.get("instrument")),
                raw_trade.get("seq"),
                raw_trade.get("timestamp", action.get("timestamp")),
                raw_trade.get("price", action.get("price")),
                raw_trade.get("quantity", action.get("quantity")),
            )
            if key not in observed_trade_keys:
                observed_trade_keys.add(key)
                instrument = str(action.get("instrument", ""))
                if instrument:
                    market_trades[instrument].append(
                        {
                            "timestamp": action.get("timestamp"),
                            "tick_id": raw_trade.get("tick_id"),
                            "price": action.get("price"),
                            "quantity": action.get("quantity"),
                        }
                    )
        ecology = action.get("ecology")
        if isinstance(ecology, Mapping):
            channel = str(ecology.get("linkage_channel", "unclassified"))
            channel_counts[channel] += 1
            channel_events.append(
                {
                    "timestamp": action.get("timestamp"),
                    "agent_id": action.get("agent_id"),
                    "event_type": event_type,
                    "ecology": dict(ecology),
                }
            )
            decision_id = ecology.get("decision_id")
            if decision_id:
                decision_events[str(decision_id)].append(action)

    market_summary = {
        instrument: _market_summary(trades)
        for instrument, trades in sorted(market_trades.items())
    }
    _write_json(
        reports_dir / "ecology_market_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "markets": market_summary,
            "events": list(event_index.values()),
        },
    )
    _write_json(
        reports_dir / "ecology_channel_ledger.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "channel_event_counts": dict(channel_counts),
            "events": channel_events,
            "event_delivery_counts": dict(delivery_counts),
            "event_deliveries": event_deliveries,
        },
    )
    _write_json(
        reports_dir / "ecology_decision_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decisions": [
                _summarize_decision(decision_id, events)
                for decision_id, events in sorted(decision_events.items())
            ],
        },
    )
    _write_json(
        reports_dir / "ecology_agent_response_summary.json",
        _agent_response_summary(reports_dir.parent / "decision_context"),
    )


def _market_summary(trades: list[Mapping[str, Any]]) -> dict[str, Any]:
    clean = [trade for trade in trades if _as_float(trade.get("price")) > 0]
    if not clean:
        return {"trade_count": 0, "volume": 0, "start_price": None, "end_price": None, "return": None}
    start_price = _as_float(clean[0].get("price"))
    end_price = _as_float(clean[-1].get("price"))
    return {
        "trade_count": len(clean),
        "volume": sum(int(_as_float(trade.get("quantity"))) for trade in clean),
        "start_price": start_price,
        "end_price": end_price,
        "return": ((end_price / start_price) - 1.0) if start_price else None,
        "trades": clean,
    }


def _summarize_decision(
    decision_id: str,
    events: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate submitted and filled legs without inferring an atomic fill."""
    first = events[0] if events else {}
    first_ecology = first.get("ecology", {}) if isinstance(first, Mapping) else {}
    if not isinstance(first_ecology, Mapping):
        first_ecology = {}
    summary: dict[str, Any] = {
        "decision_id": decision_id,
        "timestamp": first.get("timestamp") if isinstance(first, Mapping) else None,
        "agent_id": first.get("agent_id") if isinstance(first, Mapping) else None,
        "relationship_id": first_ecology.get("relationship_id"),
        "relationship_type": first_ecology.get("relationship_type"),
        "linkage_channel": first_ecology.get("linkage_channel"),
        "active_event_ids": first_ecology.get("active_event_ids", []),
        "source_instrument": first_ecology.get("source_instrument"),
        "target_instrument": first_ecology.get("target_instrument"),
        "observed_basis_bps": first_ecology.get(
            "basis_bps",
            first_ecology.get("observed_basis_bps"),
        ),
        "reference_target_price": first_ecology.get("reference_target_price"),
        "orders": {},
        "fills": {},
    }
    submitted = False
    for event in events:
        ecology = event.get("ecology", {})
        if not isinstance(ecology, Mapping):
            continue
        leg = ecology.get("leg")
        event_type = event.get("event_type")
        if event_type == "order_submitted" and leg:
            submitted = True
            summary["orders"][str(leg)] = {
                "order_id": event.get("order_id"),
                "instrument": event.get("instrument"),
                "side": event.get("side"),
                "requested_quantity": int(_as_float(event.get("quantity"))),
            }
        elif event_type == "trade_executed" and leg:
            fill = summary["fills"].setdefault(
                str(leg),
                {"filled_quantity": 0, "notional": 0.0, "fill_count": 0},
            )
            quantity = int(_as_float(event.get("quantity")))
            price = _as_float(event.get("price"))
            fill["filled_quantity"] += quantity
            fill["notional"] = round(fill["notional"] + (quantity * price), 6)
            fill["fill_count"] += 1

    summary["execution_outcome"] = _execution_outcome(
        summary["orders"],
        summary["fills"],
        submitted,
    )
    return summary


def _execution_outcome(
    orders: Mapping[str, Mapping[str, Any]],
    fills: Mapping[str, Mapping[str, Any]],
    submitted: bool,
) -> str:
    if not submitted:
        return "observed_only"
    if not orders:
        return "submitted_without_order_record"
    requested_legs = set(orders)
    filled_legs = {
        leg
        for leg, order in orders.items()
        if _as_float(fills.get(leg, {}).get("filled_quantity"))
        >= _as_float(order.get("requested_quantity"))
    }
    if requested_legs and filled_legs == requested_legs:
        return "fully_hedged"
    if any(_as_float(fill.get("filled_quantity")) > 0 for fill in fills.values()):
        return "partial_or_unhedged"
    return "unfilled"


def _agent_response_summary(decision_context_dir: Path) -> dict[str, Any]:
    """Export recorded slow-loop decisions without exposing raw prompt content."""
    responses: list[dict[str, Any]] = []
    if decision_context_dir.is_dir():
        for memory_path in sorted(decision_context_dir.glob("*/memory.json")):
            memory = _load_json(memory_path, {})
            events = memory.get("events", []) if isinstance(memory, Mapping) else []
            if not isinstance(events, list):
                continue
            agent_id = str(memory.get("agent_id", memory_path.parent.name))
            for event in events:
                if not isinstance(event, Mapping) or event.get("event_type") != "slow_loop_decision":
                    continue
                payload = event.get("payload", {})
                if not isinstance(payload, Mapping):
                    continue
                before = payload.get("strategy_before", {})
                after = payload.get("strategy_after", {})
                primary_event = payload.get("primary_event")
                responses.append(
                    {
                        "agent_id": agent_id,
                        "timestamp": event.get("timestamp"),
                        "slow_loop_status": payload.get(
                            "slow_loop_status", "completed"
                        ),
                        "strategy_changed": bool(payload.get("strategy_changed")),
                        "strategy_changed_fields": payload.get(
                            "strategy_changed_fields", []
                        ),
                        "metadata_changed_fields": payload.get(
                            "metadata_changed_fields", []
                        ),
                        "confidence_changed": bool(
                            payload.get("confidence_changed")
                        ),
                        "event_context_present": bool(
                            payload.get("event_context_present")
                        ),
                        "possible_event_influence": bool(
                            payload.get("possible_event_influence")
                        ),
                        "active_event_ids": payload.get("active_event_ids", []),
                        "known_event_ids": payload.get("known_event_ids", []),
                        "primary_event": primary_event,
                        "risk_mode_before": _mapping_value(before, "risk_mode"),
                        "risk_mode_after": _mapping_value(after, "risk_mode"),
                        "confidence_before": _mapping_value(before, "confidence"),
                        "confidence_after": _mapping_value(after, "confidence"),
                        "reason": payload.get("reason"),
                    }
                )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recorded_slow_loop_decision_count": len(responses),
        "successful_slow_loop_count": sum(
            1
            for response in responses
            if response["slow_loop_status"] == "completed"
        ),
        "failed_slow_loop_count": sum(
            1
            for response in responses
            if response["slow_loop_status"] == "failed"
        ),
        "rejected_slow_loop_count": sum(
            1
            for response in responses
            if response["slow_loop_status"] == "rejected"
        ),
        "event_context_response_count": sum(
            1 for response in responses if response["event_context_present"]
        ),
        "active_event_response_count": sum(
            1 for response in responses if response["active_event_ids"]
        ),
        "known_event_response_count": sum(
            1 for response in responses if response["known_event_ids"]
        ),
        "strategy_change_count": sum(
            1 for response in responses if response["strategy_changed"]
        ),
        "confidence_change_count": sum(
            1 for response in responses if response["confidence_changed"]
        ),
        "responses": responses,
    }


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _load_json(path: Path, fallback: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write("\n")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
