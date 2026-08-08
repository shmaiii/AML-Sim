"""AML-Sim report helpers that reuse StockSim chart/report utilities."""

from __future__ import annotations

import csv
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGENT_DECISION_FIELDS = [
    "decision_id",
    "agent_name",
    "asset",
    "decision_date",
    "decision_timestamp",
    "data_cutoff_timestamp",
    "prediction_score",
    "confidence",
    "action",
    "submitted_action",
    "latest_data_date_used",
    "data_timestamp_source",
    "split",
]

INTERVAL_OUTCOME_FIELDS = [
    "decision_id",
    "agent_name",
    "asset",
    "interval_start",
    "interval_end",
    "result_available_timestamp",
    "interval_return",
    "interval_pnl",
    "portfolio_value",
    "realized_volatility",
    "drawdown",
    "gross_exposure",
    "net_exposure",
    "assigned_risk_budget",
    "outcome_status",
    "split",
]


def generate_order_book_microstructure_report(
    agent_reports_dir: Path,
    reports_dir: Path,
) -> None:
    """Combine exchange JSONL snapshots into a research-friendly CSV."""
    rows: list[dict[str, Any]] = []
    for source in sorted(agent_reports_dir.glob("order_book_microstructure_*.jsonl")):
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Skipping malformed microstructure row {source}:{line_number}: {exc}")
                    continue
                if isinstance(row, dict):
                    rows.append(row)

    rows.sort(key=lambda row: (row.get("timestamp") or "", row.get("instrument") or ""))
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_file = reports_dir / "order_book_microstructure.csv"
    scalar_fields = sorted({key for row in rows for key in row if key not in {"bid_levels", "ask_levels"}})
    fieldnames = scalar_fields + ["bid_levels", "ask_levels"]
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["bid_levels"] = json.dumps(row.get("bid_levels", []), separators=(",", ":"))
            output["ask_levels"] = json.dumps(row.get("ask_levels", []), separators=(",", ":"))
            writer.writerow(output)
    print(f"Generated order-book microstructure CSV: {output_file}")


def generate_llm_update_report(agent_reports_dir: Path, reports_dir: Path) -> None:
    """Combine per-agent LLM strategy update audit files."""
    records: list[dict[str, Any]] = []
    for source in sorted(agent_reports_dir.glob("llm_strategy_updates_*.json")):
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))
    records.sort(key=lambda row: (row.get("timestamp") or "", row.get("agent_id") or ""))
    output_file = reports_dir / "llm_strategy_updates.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    print(f"Generated LLM strategy update audit: {output_file}")


def _parse_report_datetime(value: str):
    """Parse scenario datetimes, including common trailing-Z UTC notation."""
    from utils.time_utils import parse_datetime_utc

    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return parse_datetime_utc(value)


def _decision_timestamp_sort_key(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_simulation_duration(start_str: str, end_str: str) -> str:
    """Return a human-readable simulation duration string.

    Uses hours:minutes for sub-day windows and days for longer runs.
    """
    try:
        start = _parse_report_datetime(start_str)
        end = _parse_report_datetime(end_str)
        delta = end - start
        total_seconds = delta.total_seconds()
        if total_seconds < 0:
            return "0m (end before start)"
        if total_seconds < 3600:
            return f"{total_seconds / 60:.0f}m"
        if total_seconds < 86400:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            return f"{hours}h{minutes:02d}m"
        return f"{delta.days}d"
    except Exception:
        return "unknown"


def _uses_openai_slow_loop(agent_config: dict[str, Any]) -> bool:
    """Return whether an AML agent is configured with an enabled OpenAI strategist."""
    strategist = agent_config.get("parameters", {}).get("slow_strategist", {})
    if not isinstance(strategist, dict) or not strategist.get("enabled", True):
        return False
    return str(strategist.get("type", "")).lower() in {"openai", "openai_json"}


def generate_post_simulation_artifacts(config: dict[str, Any]) -> None:
    """
    Generate StockSim-style artifacts from AML-Sim.

    This intentionally lives in AML-Sim so StockSim can remain an engine/library
    dependency. It still reuses StockSim's data clients and chart/report helpers.
    """
    try:
        from utils.alpha_vantage_client import AlphaVantageClient
        from utils.plot_charts import (
            ensure_output_directories,
            generate_demo_report,
            make_chart_dropdown,
        )
        from utils.polygon_client import PolygonClient

        print("Generating StockSim-style post-simulation artifacts...")

        charts_dir, reports_dir = ensure_output_directories()

        instruments = config.get("instruments", [])
        exchanges_config = config.get("exchanges", {})
        simulation_config = config.get("simulation", {})

        simulation_start_str = simulation_config["start_time"]
        simulation_end_str = simulation_config["end_time"]

        for instrument in instruments:
            try:
                inst_cfg = exchanges_config.get(instrument, {})
                data_source = inst_cfg.get("data_source", "polygon").lower()
                symbol_type = inst_cfg.get("symbol_type", "stock")
                interval = inst_cfg.get("candle_interval", "1d")
                indicator_kwargs = inst_cfg.get("indicator_kwargs", {})

                print(f"Generating artifacts for {instrument} ({symbol_type})...")

                if data_source == "synthetic":
                    print(f"Skipping external chart/report generation for synthetic instrument {instrument}.")
                    continue

                client = AlphaVantageClient() if data_source == "alpha_vantage" else PolygonClient()

                if symbol_type == "crypto":
                    candles = client.load_crypto_aggregates(
                        symbol=instrument,
                        interval=interval,
                        start_date=simulation_start_str,
                        end_date=simulation_end_str,
                        market="USD",
                        sort="asc",
                        limit=10000,
                        use_cache=True,
                    )
                else:
                    candles = client.load_aggregates(
                        symbol=instrument,
                        interval=interval,
                        start_date=simulation_start_str,
                        end_date=simulation_end_str,
                        adjusted=True,
                        sort="asc",
                        limit=10000,
                        use_cache=True,
                    )

                if candles:
                    chart_filename = f"{instrument}_demo_chart.html"
                    make_chart_dropdown(
                        candles=candles,
                        instrument=instrument,
                        scales_seconds=[3600, 14400, 86400],
                        out_html=chart_filename,
                        indicator_kwargs=indicator_kwargs,
                        symbol_type=symbol_type,
                    )

                    report = generate_demo_report(instrument, candles, indicator_kwargs)
                    if report:
                        report_filename = os.path.join(reports_dir, f"{instrument}_demo_report.json")
                        with open(report_filename, "w", encoding="utf-8") as handle:
                            json.dump(report, handle, indent=2)
                        print(f"Generated report: {report_filename}")

            except Exception as exc:
                print(f"Failed to generate artifacts for {instrument}: {exc}")
                continue

        summary_report = {
            "simulation_info": {
                "start_time": simulation_start_str,
                "end_time": simulation_end_str,
                "duration": _format_simulation_duration(
                    simulation_start_str, simulation_end_str
                ),
                "instruments": instruments,
                "total_agents": sum(
                    agent_config.get("count", 1)
                    for agent_config in config.get("agents", {}).values()
                ),
                "exchange_mode": config.get("exchange_mode", "candle"),
            },
            "generated_artifacts": {
                "charts_directory": charts_dir,
                "reports_directory": reports_dir,
                "timestamp": datetime.now().isoformat(),
            },
            "research_metrics": {
                "llm_agents": sum(
                    agent.get("count", 1)
                    for agent in config.get("agents", {}).values()
                    if _uses_openai_slow_loop(agent)
                ),
                "benchmark_agents": sum(
                    agent.get("count", 1)
                    for agent in config.get("agents", {}).values()
                    if not _uses_openai_slow_loop(agent)
                ),
                "multi_market": len(
                    {
                        exchanges_config.get(instrument, {}).get("symbol_type", "stock")
                        for instrument in instruments
                    }
                )
                > 1,
            },
        }

        summary_file = os.path.join(reports_dir, "simulation_summary.json")
        with open(summary_file, "w", encoding="utf-8") as handle:
            json.dump(summary_report, handle, indent=2)

        print("Post-simulation artifacts generated successfully.")
        print(f"Charts available in: {charts_dir}")
        print(f"Reports available in: {reports_dir}")

    except Exception as exc:
        print(f"Failed to generate post-simulation artifacts: {exc}")
        print(traceback.format_exc())


def generate_trader_action_report(agent_reports_dir: Path, reports_dir: Path) -> None:
    """Combine per-agent AML action ledgers into one report JSON file."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    agent_reports_dir.mkdir(parents=True, exist_ok=True)

    action_files = sorted(agent_reports_dir.glob("trader_actions_*.json"))
    actions: list[dict[str, Any]] = []
    by_agent: dict[str, dict[str, Any]] = {}

    for action_file in action_files:
        agent_id = action_file.stem.removeprefix("trader_actions_")
        try:
            with action_file.open("r", encoding="utf-8") as handle:
                agent_actions = json.load(handle)
        except Exception as exc:
            print(f"Failed to read trader action file {action_file}: {exc}")
            continue

        if not isinstance(agent_actions, list):
            print(f"Skipping trader action file with non-list payload: {action_file}")
            continue

        by_agent[agent_id] = {
            "action_count": len(agent_actions),
            "submitted_orders": sum(
                1 for action in agent_actions if action.get("event_type") == "order_submitted"
            ),
            "rejected_orders": sum(
                1 for action in agent_actions if action.get("event_type") == "order_rejected"
            ),
            "executed_trades": sum(
                1 for action in agent_actions if action.get("event_type") == "trade_executed"
            ),
        }
        actions.extend(agent_actions)

    actions.sort(key=lambda action: (action.get("timestamp") or "", action.get("agent_id") or ""))
    report = {
        "generated_at": datetime.now().isoformat(),
        "source_directory": str(agent_reports_dir),
        "action_count": len(actions),
        "agents": by_agent,
        "actions": actions,
    }

    output_file = reports_dir / "trader_actions.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Generated AML trader action report: {output_file}")


def generate_agent_decision_csv(
    agent_reports_dir: Path,
    reports_dir: Path,
) -> None:
    """Build detailed and latest-daily decision datasets."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    agent_reports_dir.mkdir(parents=True, exist_ok=True)

    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    detailed_rows: list[dict[str, Any]] = []
    for decision_file in sorted(agent_reports_dir.glob("agent_decisions_*.json")):
        try:
            with decision_file.open("r", encoding="utf-8") as handle:
                agent_rows = json.load(handle)
        except Exception as exc:
            print(f"Failed to read agent decision file {decision_file}: {exc}")
            continue

        if not isinstance(agent_rows, list):
            print(f"Skipping agent decision file with non-list payload: {decision_file}")
            continue

        for raw_row in agent_rows:
            if not isinstance(raw_row, dict):
                continue
            row = {field: raw_row.get(field) for field in AGENT_DECISION_FIELDS}
            key = (
                str(row.get("agent_name") or ""),
                str(row.get("decision_date") or ""),
                str(row.get("asset") or ""),
            )
            if not all(key):
                continue

            detailed_rows.append(row)

            previous = rows_by_key.get(key)
            if previous is None or _decision_timestamp_sort_key(
                row.get("decision_timestamp")
            ) >= _decision_timestamp_sort_key(previous.get("decision_timestamp")):
                rows_by_key[key] = row

    detailed_rows.sort(
        key=lambda row: (
            _decision_timestamp_sort_key(row.get("decision_timestamp")),
            str(row.get("agent_name") or ""),
            str(row.get("asset") or ""),
        )
    )
    detailed_output_file = reports_dir / "agent_decisions_detailed.csv"
    with detailed_output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGENT_DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(detailed_rows)

    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (
            str(row.get("decision_date") or ""),
            str(row.get("agent_name") or ""),
            str(row.get("asset") or ""),
        ),
    )
    output_file = reports_dir / "agent_decisions.csv"
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGENT_DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated AML agent decision CSV: {output_file}")
    print(f"Generated detailed AML decision CSV: {detailed_output_file}")


def generate_interval_outcome_csv(
    agent_reports_dir: Path,
    reports_dir: Path,
) -> None:
    """Combine per-agent interval outcomes into one decision-linked CSV."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    agent_reports_dir.mkdir(parents=True, exist_ok=True)

    rows_by_decision_id: dict[str, dict[str, Any]] = {}
    for outcome_file in sorted(agent_reports_dir.glob("interval_outcomes_*.json")):
        try:
            with outcome_file.open("r", encoding="utf-8") as handle:
                agent_rows = json.load(handle)
        except Exception as exc:
            print(f"Failed to read interval outcome file {outcome_file}: {exc}")
            continue

        if not isinstance(agent_rows, list):
            print(f"Skipping interval outcome file with non-list payload: {outcome_file}")
            continue

        for raw_row in agent_rows:
            if not isinstance(raw_row, dict):
                continue
            row = {field: raw_row.get(field) for field in INTERVAL_OUTCOME_FIELDS}
            decision_id = str(row.get("decision_id") or "")
            if not decision_id:
                continue
            rows_by_decision_id[decision_id] = row

    rows = sorted(
        rows_by_decision_id.values(),
        key=lambda row: (
            _decision_timestamp_sort_key(row.get("interval_start")),
            str(row.get("agent_name") or ""),
            str(row.get("asset") or ""),
        ),
    )
    output_file = reports_dir / "interval_outcomes.csv"
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERVAL_OUTCOME_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated AML interval outcome CSV: {output_file}")
