"""Local dashboard server with a small API for launching AML simulations."""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
import math
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENARIO = Path("scenarios/aml_one_hour_live.yaml")
DEFAULT_RUN_ID = "local_multi_agent"
DOCKER_COMPOSE_FILE = ROOT / "simulators" / "StockSim" / "docker-compose.yml"
RUNS_DIR = ROOT / ".aml_runs"
DASHBOARD_DIR = ROOT / "dashboard" / "dist"
DASHBOARD_DRAFTS_DIR = RUNS_DIR / "dashboard_drafts"

ORDER_ADDED_RE = re.compile(
    r"Adding order to OrderBook: Order\(id=([^,]+), agent=([^,]+), "
    r"side=(BUY|SELL), type=([^,]+), qty=(\d+)/(\d+), price=([^,\)]+)"
)
ORDER_CANCELLED_RE = re.compile(r"Order canceled: Order\(id=([^,]+)")
ORDER_REMOVED_RE = re.compile(
    r"(?:Order|Bid order|Ask order) ([^ ]+) (?:removed from (?:asks|bids)|fully filled)"
)
TRADE_EXECUTED_RE = re.compile(
    r"Trade executed: (.+?) bought (\d+) shares from (.+?) at \$([\d.]+)"
)
ORDER_TIME_RE = re.compile(r"time=(\d{4}-\d{2}-\d{2}T[\d:.+\-]+)")
SHOCK_EMITTED_RE = re.compile(
    r"emitted (announcement|active) ([\w-]+).*?type=([^,]+), class=([^,]+), severity=([\d.]+)"
)

# The dashboard intentionally exposes a small, validated composition surface.
# It is a scenario editor for supported AML roles, not an arbitrary-code runner.
DASHBOARD_AGENT_TYPES = {
    "AML_Market_Maker": "Market maker",
    "AML_Retail_Trader": "Retail participant",
    "AML_Institutional_Trader": "Institutional participant",
    "AML_Informed_Trader": "Informed participant",
    "AML_Liquidity_Taker": "Liquidity taker",
}

RISK_MODE_OPTIONS = [
    {"value": "risk_off", "label": "Risk off"},
    {"value": "conservative", "label": "Conservative"},
    {"value": "normal", "label": "Normal"},
    {"value": "opportunistic", "label": "Opportunistic"},
    {"value": "aggressive", "label": "Aggressive"},
]


def number_parameter(
    key: str,
    label: str,
    default: int | float,
    minimum: int | float,
    maximum: int | float,
    step: int | float,
    *,
    integer: bool = False,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "kind": "integer" if integer else "number",
        "default": default,
        "min": minimum,
        "max": maximum,
        "step": step,
    }


def select_parameter(
    key: str,
    label: str,
    default: str,
    options: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "kind": "select",
        "default": default,
        "options": options,
    }


COMMON_AGENT_PARAMETERS = [
    select_parameter("risk_mode", "Risk mode", "normal", RISK_MODE_OPTIONS),
]

DASHBOARD_AGENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "AML_Market_Maker": {
        "description": "Two-sided liquidity provider with inventory-aware quotes.",
        "parameters": [
            number_parameter("initial_cash", "Initial cash", 100_000, 1_000, 10_000_000, 1_000),
            *COMMON_AGENT_PARAMETERS,
            number_parameter("fair_price", "Fair price", 100.0, 1.0, 1_000_000.0, 0.01),
            number_parameter("spread", "Base spread", 0.20, 0.01, 100.0, 0.01),
            number_parameter("quote_size", "Quote size", 25, 1, 10_000, 1, integer=True),
            number_parameter("quote_levels", "Quote levels", 3, 1, 20, 1, integer=True),
            number_parameter("level_spacing", "Level spacing", 0.05, 0.01, 100.0, 0.01),
            number_parameter("min_inventory", "Minimum inventory", -500, -1_000_000, 1_000_000, 10, integer=True),
            number_parameter("max_inventory", "Maximum inventory", 500, -1_000_000, 1_000_000, 10, integer=True),
        ],
    },
    "AML_Retail_Trader": {
        "description": "Small noisy participant with sentiment and herding responses.",
        "parameters": [
            number_parameter("initial_cash", "Initial cash", 25_000, 1_000, 10_000_000, 1_000),
            *COMMON_AGENT_PARAMETERS,
            number_parameter("trade_probability", "Trade probability", 0.55, 0.0, 1.0, 0.05),
            number_parameter("max_order_size", "Maximum order size", 8, 1, 1_000, 1, integer=True),
            number_parameter("buy_bias", "Buy bias", 0.50, 0.0, 1.0, 0.05),
            number_parameter("herding_tendency", "Herding tendency", 0.0, 0.0, 2.0, 0.05),
            number_parameter("panic_level", "Panic level", 0.0, 0.0, 2.0, 0.05),
            number_parameter("sentiment_sensitivity", "Sentiment sensitivity", 0.4, 0.0, 2.0, 0.05),
        ],
    },
    "AML_Institutional_Trader": {
        "description": "Large participant using sliced execution and selectable alpha logic.",
        "parameters": [
            number_parameter("initial_cash", "Initial cash", 300_000, 1_000, 10_000_000, 1_000),
            *COMMON_AGENT_PARAMETERS,
            select_parameter(
                "alpha_strategy",
                "Primary strategy",
                "target_execution",
                [
                    {"value": "target_execution", "label": "Target execution"},
                    {"value": "momentum", "label": "Momentum"},
                    {"value": "mean_reversion", "label": "Mean reversion"},
                ],
            ),
            number_parameter("child_order_size", "Child order size", 12, 1, 10_000, 1, integer=True),
            number_parameter("lookback_ticks", "Lookback ticks", 5, 2, 500, 1, integer=True),
            number_parameter("entry_threshold", "Entry threshold", 0.002, 0.0, 0.25, 0.0005),
            number_parameter("exit_threshold", "Exit threshold", 0.0005, 0.0, 0.25, 0.0005),
            number_parameter("min_position", "Minimum position", -500, -1_000_000, 1_000_000, 10, integer=True),
            number_parameter("max_position", "Maximum position", 500, -1_000_000, 1_000_000, 10, integer=True),
            number_parameter("shock_reactivity", "Shock reactivity", 0.7, 0.0, 2.0, 0.05),
        ],
    },
    "AML_Informed_Trader": {
        "description": "Value-oriented participant trading from a private fair-value estimate.",
        "parameters": [
            number_parameter("initial_cash", "Initial cash", 100_000, 1_000, 10_000_000, 1_000),
            *COMMON_AGENT_PARAMETERS,
            number_parameter("fair_value_anchor", "Fair-value anchor", 100.0, 1.0, 1_000_000.0, 0.01),
            number_parameter("information_edge", "Information edge", 0.70, 0.0, 1.0, 0.05),
            number_parameter("trade_probability", "Trade probability", 0.35, 0.0, 1.0, 0.05),
            number_parameter("max_order_size", "Maximum order size", 12, 1, 10_000, 1, integer=True),
            number_parameter("signal_threshold", "Signal threshold", 0.002, 0.0, 0.25, 0.0005),
            number_parameter("shock_reactivity", "Shock reactivity", 0.8, 0.0, 2.0, 0.05),
        ],
    },
    "AML_Liquidity_Taker": {
        "description": "Aggressive flow participant that consumes resting liquidity.",
        "parameters": [
            number_parameter("initial_cash", "Initial cash", 100_000, 1_000, 10_000_000, 1_000),
            *COMMON_AGENT_PARAMETERS,
            number_parameter("flow_intensity", "Flow intensity", 0.35, 0.0, 1.0, 0.05),
            number_parameter("buy_bias", "Buy bias", 0.50, 0.0, 1.0, 0.05),
            number_parameter("max_order_size", "Maximum order size", 12, 1, 10_000, 1, integer=True),
            number_parameter("inventory_limit", "Inventory limit", 500, 1, 1_000_000, 10, integer=True),
            number_parameter("shock_sensitivity", "Shock sensitivity", 0.7, 0.0, 2.0, 0.05),
            number_parameter("aggression", "Aggression", 0.75, 0.0, 2.0, 0.05),
        ],
    },
}


@dataclass
class RunJob:
    run_id: str
    scenario: str
    status: str = "starting"
    command: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    return_code: int | None = None
    log_lines: list[str] = field(default_factory=list)
    error: str | None = None

    def append_log(self, line: str) -> None:
        self.log_lines.append(line.rstrip())
        if len(self.log_lines) > 400:
            del self.log_lines[:-400]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "status": self.status,
            "command": self.command,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "error": self.error,
            "log_tail": self.log_lines[-40:],
            "dashboard_url": f"/?run={self.run_id}",
        }


class DashboardState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_job: RunJob | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.current_job is None:
                return {"status": "idle"}
            return self.current_job.to_dict()

    def start_job(
        self,
        scenario: str,
        run_id: str,
        reports: bool,
        agent_additions: list[dict[str, Any]] | None = None,
    ) -> RunJob:
        with self.lock:
            if self.current_job and self.current_job.status == "running":
                raise RuntimeError(
                    f"Run {self.current_job.run_id!r} is already in progress."
                )

            clean_run_id = next_available_run_id(validate_run_id(run_id))
            scenario_path = validate_scenario_path(scenario)
            effective_scenario_path = scenario_path
            if agent_additions:
                effective_scenario_path = materialize_dashboard_scenario(
                    scenario_path,
                    clean_run_id,
                    agent_additions,
                )
            command = [
                runner_python(),
                str(ROOT / "aml_runner.py"),
                str(effective_scenario_path),
                "--run-id",
                clean_run_id,
            ]
            if reports:
                command.append("--reports")

            job = RunJob(
                run_id=clean_run_id,
                scenario=str(scenario_path.relative_to(ROOT)),
                status="running",
                command=command,
            )
            self.current_job = job

        thread = threading.Thread(
            target=run_simulation_process,
            args=(job,),
            daemon=True,
        )
        thread.start()
        return job


STATE = DashboardState()


class LiveOrderBook:
    """Incrementally reconstruct the visible book without streaming raw logs."""

    def __init__(self) -> None:
        self.offset = 0
        self.partial_line = ""
        self.current_time = ""
        self.orders: dict[str, dict[str, Any]] = {}
        self.trades: deque[dict[str, Any]] = deque(maxlen=800)

    def refresh(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self.offset:
            self.__init__()
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except OSError:
            return
        if not chunk:
            return
        lines = (self.partial_line + chunk).splitlines(keepends=True)
        self.partial_line = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.partial_line = lines.pop()
        for line in lines:
            self.consume(line)

    def consume(self, line: str) -> None:
        time_match = ORDER_TIME_RE.search(line)
        if time_match:
            self.current_time = time_match.group(1)

        added = ORDER_ADDED_RE.search(line)
        if added:
            order_id, agent_id, side, order_type, quantity, _, raw_price = added.groups()
            try:
                price = float(raw_price)
            except ValueError:
                price = 0.0
            if order_type == "LIMIT" and price > 0:
                self.orders[order_id] = {
                    "agent_id": agent_id,
                    "side": side,
                    "quantity": int(quantity),
                    "price": price,
                }
            return

        cancelled = ORDER_CANCELLED_RE.search(line)
        if cancelled:
            self.orders.pop(cancelled.group(1), None)
            return

        removed = ORDER_REMOVED_RE.search(line)
        if removed:
            self.orders.pop(removed.group(1), None)
            return

        trade = TRADE_EXECUTED_RE.search(line)
        if not trade:
            return
        buyer, raw_quantity, seller, raw_price = trade.groups()
        quantity = int(raw_quantity)
        price = float(raw_price)
        self.trades.append(
            {
                "timestamp": self.current_time,
                "price": price,
                "quantity": quantity,
                "buyer": buyer,
                "seller": seller,
            }
        )
        # Market orders are not stored in the book. Reducing a matching resting
        # counterparty keeps the reconstructed depth aligned with partial fills.
        self.reduce_resting_order("SELL", seller, price, quantity)
        self.reduce_resting_order("BUY", buyer, price, quantity)

    def reduce_resting_order(
        self, side: str, agent_id: str, price: float, quantity: int
    ) -> None:
        remaining = quantity
        for order_id, order in list(self.orders.items()):
            if remaining <= 0:
                break
            if (
                order["side"] != side
                or order["agent_id"] != agent_id
                or abs(float(order["price"]) - price) > 1e-8
            ):
                continue
            filled = min(int(order["quantity"]), remaining)
            order["quantity"] = int(order["quantity"]) - filled
            remaining -= filled
            if order["quantity"] <= 0:
                self.orders.pop(order_id, None)

    def snapshot(self) -> dict[str, Any]:
        def levels(side: str) -> list[dict[str, Any]]:
            aggregated: dict[float, dict[str, Any]] = {}
            for order in self.orders.values():
                if order["side"] != side or order["quantity"] <= 0:
                    continue
                price = float(order["price"])
                level = aggregated.setdefault(
                    price, {"price": price, "quantity": 0, "orderCount": 0}
                )
                level["quantity"] += int(order["quantity"])
                level["orderCount"] += 1
            return sorted(
                aggregated.values(),
                key=lambda level: float(level["price"]),
                reverse=side == "BUY",
            )[:20]

        return {
            "bids": levels("BUY"),
            "asks": levels("SELL"),
            "trades": list(self.trades),
        }


LIVE_BOOKS: dict[tuple[str, str], LiveOrderBook] = {}
LIVE_BOOKS_LOCK = threading.Lock()


def live_book_snapshot(run_id: str, instrument: str, path: Path) -> dict[str, Any]:
    key = (run_id, instrument)
    with LIVE_BOOKS_LOCK:
        book = LIVE_BOOKS.setdefault(key, LiveOrderBook())
        book.refresh(path)
        return book.snapshot()


def validate_run_id(run_id: str | None) -> str:
    value = (run_id or DEFAULT_RUN_ID).strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if not value or any(char not in allowed for char in value):
        raise ValueError("run_id may only contain letters, numbers, '_' and '-'.")
    return value


def validate_scenario_path(scenario: str | None) -> Path:
    value = (scenario or str(DEFAULT_SCENARIO)).strip()
    path = (ROOT / value).resolve()
    if ROOT not in path.parents and path != ROOT:
        raise ValueError("scenario must be inside the AML-Sim workspace.")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("scenario must be a YAML file.")
    if not path.exists():
        raise FileNotFoundError(f"scenario not found: {path}")
    return path


def next_available_run_id(run_id: str) -> str:
    candidate = run_id
    suffix = 2
    while (RUNS_DIR / candidate).exists():
        candidate = f"{run_id}_{suffix}"
        suffix += 1
    return candidate


def validate_agent_id(agent_id: Any) -> str:
    value = str(agent_id or "").strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,47}", value):
        raise ValueError(
            "Agent ids must start with a lowercase letter and contain only "
            "lowercase letters, numbers, and '_' (3-48 characters)."
        )
    return value


def validate_agent_parameters(agent_type: str, raw_parameters: Any) -> dict[str, Any]:
    """Validate dashboard overrides against the server-owned role schema."""
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, dict):
        raise ValueError("Participant parameters must be an object.")
    schema = DASHBOARD_AGENT_SCHEMAS.get(agent_type)
    if schema is None:
        raise ValueError(f"Unsupported dashboard agent type: {agent_type}")
    definitions = {item["key"]: item for item in schema["parameters"]}
    unknown = sorted(set(raw_parameters) - set(definitions))
    if unknown:
        raise ValueError(f"Unsupported participant parameters: {', '.join(unknown)}")

    validated: dict[str, Any] = {}
    for key, definition in definitions.items():
        value = raw_parameters.get(key, definition["default"])
        if definition["kind"] == "select":
            allowed = {option["value"] for option in definition["options"]}
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{definition['label']} must be one of {sorted(allowed)}.")
            validated[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{definition['label']} must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{definition['label']} must be finite.")
        if numeric < definition["min"] or numeric > definition["max"]:
            raise ValueError(
                f"{definition['label']} must be between {definition['min']} and "
                f"{definition['max']}."
            )
        if definition["kind"] == "integer":
            if not numeric.is_integer():
                raise ValueError(f"{definition['label']} must be an integer.")
            validated[key] = int(numeric)
        else:
            validated[key] = numeric

    minimum_key = "min_inventory" if agent_type == "AML_Market_Maker" else "min_position"
    maximum_key = "max_inventory" if agent_type == "AML_Market_Maker" else "max_position"
    if minimum_key in validated and maximum_key in validated:
        if validated[minimum_key] >= validated[maximum_key]:
            raise ValueError(f"{minimum_key} must be lower than {maximum_key}.")
    if agent_type == "AML_Institutional_Trader":
        if validated["exit_threshold"] > validated["entry_threshold"]:
            raise ValueError("Exit threshold cannot exceed entry threshold.")
    return validated


def agent_template_spec(
    agent_type: str,
    instrument: str,
    slow_strategist: str,
    parameter_overrides: Any = None,
) -> dict[str, Any]:
    """Return conservative defaults for a dashboard-added participant."""
    exchange_id = f"exchange_{instrument.lower()}"
    common = {
        "instrument_exchange_map": {instrument: exchange_id},
        "initial_cash": 100_000,
        "initial_positions": {instrument: 0},
        "initial_cost_basis": {instrument: 100.0},
        "action_interval": "30s",
        "slow_loop_interval": "3m",
        "slow_strategist": {"type": slow_strategist},
    }
    if agent_type == "AML_Market_Maker":
        parameters = {
            **common,
            "fair_price": 100.0,
            "spread": 0.20,
            "quote_size": 25,
            "quote_levels": 3,
            "level_spacing": 0.05,
            "target_inventory": 0,
            "min_inventory": -500,
            "max_inventory": 500,
        }
    elif agent_type == "AML_Retail_Trader":
        parameters = {
            **common,
            "initial_cash": 25_000,
            "trade_probability": 0.55,
            "max_order_size": 8,
            "buy_bias": 0.50,
        }
    elif agent_type == "AML_Institutional_Trader":
        parameters = {
            **common,
            "initial_cash": 300_000,
            "target_positions": {instrument: 0},
            "child_order_size": 12,
            "order_type": "MARKET",
            "alpha_strategy": "target_execution",
            "alpha_strategies": ["target_execution", "momentum", "mean_reversion"],
            "strategy_weights": {"momentum": 0.5, "mean_reversion": 0.5},
            "lookback_ticks": 5,
            "entry_threshold": 0.002,
            "exit_threshold": 0.0005,
            "max_position": 500,
            "min_position": -500,
            "shock_reactivity": 0.7,
        }
    elif agent_type == "AML_Informed_Trader":
        parameters = {
            **common,
            "fair_value_anchor": 100.0,
            "information_edge": 0.70,
            "max_order_size": 12,
            "trade_probability": 0.35,
            "max_position": 500,
            "min_position": -500,
            "signal_threshold": 0.002,
            "shock_reactivity": 0.8,
        }
    elif agent_type == "AML_Liquidity_Taker":
        parameters = {
            **common,
            "flow_intensity": 0.35,
            "buy_bias": 0.50,
            "max_order_size": 12,
            "inventory_limit": 500,
            "shock_sensitivity": 0.7,
            "aggression": 0.75,
        }
    else:  # Kept defensive in case the catalog and builder drift.
        raise ValueError(f"Unsupported dashboard agent type: {agent_type}")
    overrides = validate_agent_parameters(agent_type, parameter_overrides)
    parameters.update(overrides)
    if agent_type == "AML_Institutional_Trader":
        parameters["alpha_strategies"] = [parameters["alpha_strategy"]]
        parameters["strategy_weights"] = {parameters["alpha_strategy"]: 1.0}
    return {"type": agent_type, "parameters": parameters}


def materialize_dashboard_scenario(
    source_path: Path,
    run_id: str,
    additions: list[dict[str, Any]],
) -> Path:
    """Write a validated, run-local scenario variant for the UI composer."""
    if len(additions) > 12:
        raise ValueError("A dashboard scenario may add at most 12 participant groups.")
    try:
        source = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse source scenario: {exc}") from exc
    if not isinstance(source, dict):
        raise ValueError("Source scenario must be a YAML mapping.")

    variant = deepcopy(source)
    config = variant.get("stocksim_config")
    if not isinstance(config, dict):
        raise ValueError("Source scenario has no StockSim configuration.")
    agents = config.get("agents")
    instruments = config.get("instruments")
    if not isinstance(agents, dict) or not isinstance(instruments, list):
        raise ValueError("Source scenario has no editable agents or instruments.")
    allowed_instruments = {str(instrument) for instrument in instruments}

    for addition in additions:
        if not isinstance(addition, dict):
            raise ValueError("Each added participant must be an object.")
        agent_id = validate_agent_id(addition.get("id"))
        agent_type = str(addition.get("type") or "")
        instrument = str(addition.get("instrument") or "")
        slow_strategist = str(addition.get("slow_strategist") or "frozen")
        count = addition.get("count", 1)
        if agent_type not in DASHBOARD_AGENT_TYPES:
            raise ValueError(f"Unsupported dashboard agent type: {agent_type}")
        if instrument not in allowed_instruments:
            raise ValueError(f"Instrument '{instrument}' is not in the selected scenario.")
        if slow_strategist not in {"frozen", "openai"}:
            raise ValueError("slow_strategist must be either 'frozen' or 'openai'.")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 25:
            raise ValueError("Participant count must be an integer between 1 and 25.")
        if agent_id in agents:
            raise ValueError(f"An agent named '{agent_id}' already exists in this scenario.")

        spec = agent_template_spec(
            agent_type,
            instrument,
            slow_strategist,
            addition.get("parameters"),
        )
        if count > 1:
            spec["count"] = count
        agents[agent_id] = spec

    variant["name"] = f"{source.get('name', source_path.stem)}_dashboard_variant"
    variant["description"] = (
        f"Dashboard-composed variant of {source_path.name}; archived with run {run_id}."
    )
    DASHBOARD_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    variant_path = DASHBOARD_DRAFTS_DIR / f"{run_id}.yaml"
    with variant_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(variant, handle, sort_keys=False)
    return variant_path


def runner_python() -> str:
    """Prefer the repo venv so dashboard launches use installed AML deps."""
    candidates = [
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_simulation_process(job: RunJob) -> None:
    try:
        process = subprocess.Popen(
            job.command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            job.append_log(line)
        job.return_code = process.wait()
        job.finished_at = time.time()
        job.status = "completed" if job.return_code == 0 else "failed"
        if job.return_code != 0:
            job.error = f"aml_runner.py exited with code {job.return_code}"
    except Exception as exc:  # pragma: no cover - defensive local server path
        job.status = "failed"
        job.finished_at = time.time()
        job.error = str(exc)
        job.append_log(str(exc))


def rabbitmq_reachable(
    host: str = "127.0.0.1",
    port: int = 5672,
    timeout_seconds: float = 1.0,
) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def load_run_artifacts(run_id: str | None) -> dict[str, Any]:
    clean_run_id = validate_run_id(run_id)
    run_dir = RUNS_DIR / clean_run_id
    current_job = STATE.snapshot()
    payload: dict[str, Any] = {
        "run_id": clean_run_id,
        "exists": run_dir.exists(),
        "missing": [],
        "actions_report": None,
        "summary": None,
        "metadata": None,
        "configured_agents": [],
        "agent_reports": {},
        "order_log": "",
        "order_logs": {},
        "live_markets": {},
        "shock_log": "",
        "shock_events": [],
        "analysis_reports": {},
        "available_reports": [],
        "slow_loop_status": {
            "recorded": 0,
            "completed": 0,
            "failed": 0,
            "rejected": 0,
            "recent_failures": [],
        },
        "job": current_job if current_job.get("run_id") == clean_run_id else {"status": "idle"},
    }
    if not run_dir.exists():
        return payload

    payload["metadata"] = read_json_if_exists(run_dir / "metadata.json", payload)
    payload["configured_agents"] = load_configured_agents(run_dir)
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    # `trader_actions.json` can be several megabytes. It remains on disk for
    # research export, but is intentionally not pushed through the one-second
    # live stream because the UI uses the smaller specialized report artifacts.
    payload["summary"] = read_json_if_exists(
        reports_dir / "simulation_summary.json",
        payload,
    )
    if (
        current_job.get("run_id") != clean_run_id
        and isinstance(payload["summary"], dict)
    ):
        payload["job"] = {
            "run_id": clean_run_id,
            "status": "completed",
            "finished_at": run_dir.stat().st_mtime,
        }
    elif current_job.get("run_id") != clean_run_id:
        clock_log = logs_dir / "simulation_clock.log"
        if clock_log.exists() and time.time() - clock_log.stat().st_mtime < 20:
            payload["job"] = {
                "run_id": clean_run_id,
                "status": "running",
                "started_at": run_dir.stat().st_mtime,
            }
    payload["agent_reports"] = load_agent_reports(reports_dir / "agents")
    payload["slow_loop_status"] = load_slow_loop_status(run_dir)
    payload["analysis_reports"] = load_analysis_reports(reports_dir)
    payload["available_reports"] = sorted(
        path.name for path in reports_dir.glob("*.json") if path.is_file()
    )

    instruments = (
        payload.get("summary", {})
        .get("simulation_info", {})
        .get("instruments", [])
        if isinstance(payload.get("summary"), dict)
        else []
    )
    preferred_logs = [logs_dir / f"order_book_{instrument}.log" for instrument in instruments]
    known_log_paths = {path for path in preferred_logs if path.exists()}
    known_log_paths.update(logs_dir.glob("order_book_*.log"))
    for path in sorted(known_log_paths):
        instrument = path.stem.removeprefix("order_book_")
        payload["live_markets"][instrument] = live_book_snapshot(
            clean_run_id, instrument, path
        )
    if payload["live_markets"]:
        first_instrument = next(iter(instruments), next(iter(payload["live_markets"])))
        payload["order_log"] = first_instrument
    else:
        payload["missing"].append(str((logs_dir / "order_book_<instrument>.log").relative_to(ROOT)))

    shock_log = logs_dir / "agents" / "agent_shock_agent.log"
    if shock_log.exists():
        payload["shock_events"] = load_shock_events(shock_log)

    return payload


def load_analysis_reports(reports_dir: Path) -> dict[str, Any]:
    """Expose the small, structured research reports used by the dashboard."""
    report_names = (
        "ecology_manifest",
        "ecology_market_summary",
        "ecology_channel_ledger",
        "ecology_decision_summary",
        "ecology_agent_response_summary",
    )
    return {
        name: value
        for name in report_names
        if (value := read_json_file(reports_dir / f"{name}.json")) is not None
    }


def load_shock_events(path: Path) -> list[dict[str, Any]]:
    """Return compact shock records for the live timeline."""
    events: list[dict[str, Any]] = []
    current_time = ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return events
    for line in lines:
        time_match = re.search(r"TIME_TICK message: ([\dT:+\-]+)", line)
        if time_match:
            current_time = time_match.group(1)
        emitted = SHOCK_EMITTED_RE.search(line)
        if not emitted:
            continue
        phase, event_id, event_type, classification, severity = emitted.groups()
        events.append(
            {
                "timestamp": current_time,
                "id": event_id,
                "phase": phase,
                "type": event_type,
                "classification": classification,
                "severity": float(severity),
            }
        )
    return events[-80:]


def load_slow_loop_status(run_dir: Path) -> dict[str, Any]:
    """Summarize persisted slow-loop decisions while a run is still live."""
    summary: dict[str, Any] = {
        "recorded": 0,
        "completed": 0,
        "failed": 0,
        "rejected": 0,
        "recent_failures": [],
    }
    decision_context_dir = run_dir / "decision_context"
    if not decision_context_dir.is_dir():
        return summary

    failures: list[dict[str, str]] = []
    for memory_path in sorted(decision_context_dir.glob("*/memory.json")):
        memory = read_json_file(memory_path)
        if not isinstance(memory, dict):
            continue
        agent_id = str(memory.get("agent_id", memory_path.parent.name))
        events = memory.get("events", [])
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict) or event.get("event_type") != "slow_loop_decision":
                continue
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("slow_loop_status", "completed"))
            if status not in {"completed", "failed", "rejected"}:
                status = "failed"
            summary["recorded"] += 1
            summary[status] += 1
            if status == "failed":
                failures.append(
                    {
                        "agent_id": agent_id,
                        "timestamp": str(event.get("timestamp", "")),
                        "reason": str(payload.get("failure_reason", "Unknown slow-loop failure.")),
                    }
                )
    summary["recent_failures"] = failures[-8:]
    return summary


def load_agent_reports(agent_reports_dir: Path) -> dict[str, dict[str, Any]]:
    if not agent_reports_dir.exists():
        return {}

    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(agent_reports_dir.glob("metrics_*.json")):
        agent_id = path.stem.removeprefix("metrics_")
        reports.setdefault(agent_id, {})["metrics"] = read_json_file(path)

    for path in sorted(agent_reports_dir.glob("portfolio_timeseries_*.json")):
        agent_id = path.stem.removeprefix("portfolio_timeseries_")
        series = read_json_file(path)
        if isinstance(series, list):
            reports.setdefault(agent_id, {})["portfolio_timeseries"] = series[-250:]
            reports[agent_id]["last_portfolio_value"] = (
                series[-1].get("value") if series and isinstance(series[-1], dict) else None
            )

    for path in sorted(agent_reports_dir.glob("pending_orders_*.json")):
        agent_id = path.stem.removeprefix("pending_orders_")
        pending = read_json_file(path)
        reports.setdefault(agent_id, {})["pending_order_count"] = (
            len(pending) if isinstance(pending, dict) else 0
        )

    for path in sorted(agent_reports_dir.glob("executed_orders_*.json")):
        agent_id = path.stem.removeprefix("executed_orders_")
        executed = read_json_file(path)
        reports.setdefault(agent_id, {})["executed_order_count"] = (
            len(executed) if isinstance(executed, list) else 0
        )

    return reports


def read_json_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def load_configured_agents(run_dir: Path) -> list[dict[str, Any]]:
    config_path = run_dir / "stocksim_config.yaml"
    if not config_path.exists():
        return []
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return []

    agents = config.get("agents", {})
    if not isinstance(agents, dict):
        return []

    configured = []
    for agent_id, spec in agents.items():
        if not isinstance(spec, dict):
            continue
        configured.append(
            {
                "id": str(agent_id),
                "type": str(spec.get("type", "Agent")),
                "count": int(spec.get("count", 1) or 1),
                "instruments": sorted(
                    str(instrument)
                    for instrument in (
                        spec.get("parameters", {}).get("instrument_exchange_map", {})
                        if isinstance(spec.get("parameters"), dict)
                        else {}
                    )
                ),
                "slow_strategist": (
                    spec.get("parameters", {}).get("slow_strategist", {}).get("type")
                    if isinstance(spec.get("parameters"), dict)
                    and isinstance(spec.get("parameters", {}).get("slow_strategist"), dict)
                    else "frozen"
                ),
            }
        )
    return configured


def list_scenarios() -> list[str]:
    scenarios_dir = ROOT / "scenarios"
    return [
        str(path.relative_to(ROOT))
        for path in sorted(scenarios_dir.rglob("*.yaml"))
        if path.is_file()
    ]


def list_scenario_details() -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for relative_path in list_scenarios():
        path = ROOT / relative_path
        try:
            source = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            config = source.get("stocksim_config", {})
            instruments = config.get("instruments", []) if isinstance(config, dict) else []
        except (OSError, yaml.YAMLError):
            instruments = []
        details.append(
            {
                "path": relative_path,
                "instruments": [
                    str(instrument) for instrument in instruments
                ] if isinstance(instruments, list) else [],
            }
        )
    return details


def list_runs(limit: int = 80) -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    candidates = [
        path
        for path in RUNS_DIR.iterdir()
        if path.is_dir() and path.name != DASHBOARD_DRAFTS_DIR.name
    ]
    results: list[dict[str, Any]] = []
    for run_dir in sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[:limit]:
        metadata = read_json_file(run_dir / "metadata.json")
        summary = read_json_file(run_dir / "reports" / "simulation_summary.json")
        results.append(
            {
                "run_id": run_dir.name,
                "created_at": metadata.get("created_at") if isinstance(metadata, dict) else None,
                "scenario": (
                    metadata.get("scenario", {}).get("name")
                    if isinstance(metadata, dict)
                    and isinstance(metadata.get("scenario"), dict)
                    else None
                ),
                "completed": isinstance(summary, dict),
                "instruments": (
                    summary.get("simulation_info", {}).get("instruments", [])
                    if isinstance(summary, dict)
                    and isinstance(summary.get("simulation_info"), dict)
                    else []
                ),
            }
        )
    return results


def read_json_if_exists(path: Path, payload: dict[str, Any]) -> Any:
    if not path.exists():
        payload["missing"].append(str(path.relative_to(ROOT)))
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        payload["missing"].append(f"{path.relative_to(ROOT)} (not ready)")
        return None


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    server_version = "AMLDashboardServer/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        static_dir = DASHBOARD_DIR if DASHBOARD_DIR.exists() else ROOT
        super().__init__(*args, directory=str(static_dir), **kwargs)

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            if not DASHBOARD_DIR.exists():
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", f"/dashboard.html?run={DEFAULT_RUN_ID}")
                self.end_headers()
                return
            self.path = "/index.html"
            return super().do_GET()
        if parsed.path == "/dashboard.html" and DASHBOARD_DIR.exists():
            query = f"?{parsed.query}" if parsed.query else ""
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", f"/{query}")
            self.end_headers()
            return
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if parsed.path == "/api/status":
            self.write_json(STATE.snapshot())
            return
        if parsed.path == "/api/scenarios":
            self.write_json({"scenarios": list_scenarios()})
            return
        if parsed.path == "/api/catalog":
            self.write_json(
                {
                    "scenarios": list_scenarios(),
                    "scenario_details": list_scenario_details(),
                    "runs": list_runs(),
                    "agent_types": [
                        {
                            "type": agent_type,
                            "label": label,
                            **DASHBOARD_AGENT_SCHEMAS[agent_type],
                        }
                        for agent_type, label in DASHBOARD_AGENT_TYPES.items()
                    ],
                    "rabbitmq_reachable": rabbitmq_reachable(),
                }
            )
            return
        if parsed.path == "/api/runs":
            self.write_json({"runs": list_runs()})
            return
        if parsed.path == "/api/artifacts":
            params = parse_qs(parsed.query)
            try:
                payload = load_run_artifacts(
                    params.get("run_id", [DEFAULT_RUN_ID])[0],
                )
            except Exception as exc:
                self.write_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.write_json(payload)
            return
        if parsed.path == "/api/live":
            params = parse_qs(parsed.query)
            try:
                run_id = validate_run_id(params.get("run_id", [DEFAULT_RUN_ID])[0])
            except Exception as exc:
                self.write_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.stream_live_run(run_id)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_json_body()
            job = STATE.start_job(
                scenario=str(payload.get("scenario") or DEFAULT_SCENARIO),
                run_id=str(payload.get("run_id") or DEFAULT_RUN_ID),
                reports=bool(payload.get("reports", True)),
                agent_additions=(
                    payload.get("agent_additions")
                    if isinstance(payload.get("agent_additions"), list)
                    else None
                ),
            )
        except Exception as exc:
            self.write_json({"status": "error", "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        self.write_json(job.to_dict(), HTTPStatus.ACCEPTED)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def write_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream_live_run(self, run_id: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(b"retry: 1000\n\n")
        self.wfile.flush()

        while True:
            try:
                payload = load_run_artifacts(run_id)
                payload["server_time"] = time.time()
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.wfile.write(b"event: snapshot\n")
                self.wfile.write(b"data: " + body + b"\n\n")
                self.wfile.flush()
                time.sleep(1)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                error_body = json.dumps({"error": str(exc)}).encode("utf-8")
                try:
                    self.wfile.write(b"event: error\n")
                    self.wfile.write(b"data: " + error_body + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                time.sleep(1)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def start_rabbitmq() -> None:
    if rabbitmq_reachable():
        print("RabbitMQ already reachable on 127.0.0.1:5672.")
        return

    docker_cmd = shutil.which("docker")
    if docker_cmd is None:
        raise RuntimeError(
            "Docker was not found on PATH, and RabbitMQ is not reachable on "
            "127.0.0.1:5672. Install/start Docker Desktop, or start RabbitMQ "
            "manually, then rerun without --start-rabbitmq: "
            "python dashboard_server.py --run --run-id local_four_agent"
        )

    subprocess.run(
        [
            docker_cmd,
            "compose",
            "-f",
            str(DOCKER_COMPOSE_FILE),
            "up",
            "-d",
            "rabbitmq",
        ],
        cwd=ROOT,
        check=True,
    )
    wait_for_rabbitmq(docker_cmd=docker_cmd)


def wait_for_rabbitmq(
    docker_cmd: str | None = None,
    timeout_seconds: int = 60,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if rabbitmq_reachable():
            print("RabbitMQ is reachable on 127.0.0.1:5672.")
            return
        if docker_cmd is not None:
            result = subprocess.run(
                [
                    docker_cmd,
                    "inspect",
                    "-f",
                    "{{.State.Health.Status}}",
                    "stocksim-rabbitmq",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            status = result.stdout.strip()
            if status == "healthy":
                print("RabbitMQ container is healthy.")
                return
            if status == "unhealthy":
                raise RuntimeError("RabbitMQ container reported unhealthy status.")
        time.sleep(2)
    raise TimeoutError("Timed out waiting for RabbitMQ to become healthy.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the AML dashboard and optionally launch a simulation.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run", action="store_true", help="Start the simulation immediately.")
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Do not pass --reports to aml_runner.py.",
    )
    parser.add_argument(
        "--start-rabbitmq",
        action="store_true",
        help="Start the StockSim RabbitMQ container before serving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.start_rabbitmq:
        print("Starting RabbitMQ via Docker Compose...", flush=True)
        try:
            start_rabbitmq()
        except Exception as exc:
            print(f"Could not start RabbitMQ: {exc}", file=sys.stderr)
            return 1

    if args.run:
        if not args.start_rabbitmq and not rabbitmq_reachable():
            print(
                "Warning: RabbitMQ is not reachable on 127.0.0.1:5672. "
                "The simulation may fail unless RabbitMQ is already running "
                "somewhere reachable by the scenario.",
                file=sys.stderr,
            )
        job = STATE.start_job(
            scenario=args.scenario,
            run_id=args.run_id,
            reports=not args.no_reports,
        )
        print(f"Started simulation run: {job.run_id}")
        args.run_id = job.run_id

    server = ThreadingHTTPServer((args.host, args.port), DashboardRequestHandler)
    url = f"http://{args.host}:{args.port}/?run={args.run_id}"
    print(f"Serving AML dashboard at {url}", flush=True)
    print("Press Ctrl+C to stop the dashboard server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
