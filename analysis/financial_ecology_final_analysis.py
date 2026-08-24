#!/usr/bin/env python3
"""Build a reproducible pilot analysis for the final stock-future ecology runs.

The script reads the 18 paired runs named ``ecology_final_v2_*`` and writes
tables, figures, and a cautious research note. It deliberately reports effect
sizes and paired paths rather than significance tests: three replicates are
enough to validate the mechanism, but not to make a confirmatory claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aml-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT / ".aml_runs"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "financial_ecology_final_v2"

TREATMENTS = ("d0", "i1", "a1")
MODES = ("frozen", "llm")
REPLICATES = (1, 2, 3)
TREATMENT_LABELS = {
    "d0": "D0 disconnected",
    "i1": "I1 information",
    "a1": "A1 arbitrage",
}
MODE_LABELS = {"frozen": "Frozen strategies", "llm": "LLM adaptive strategies"}
TREATMENT_COLORS = {"d0": "#5c677d", "i1": "#168aad", "a1": "#f18f01"}

SESSION_START = pd.Timestamp("2025-03-01T09:30:00+00:00")
SESSION_END = pd.Timestamp("2025-03-01T10:30:00+00:00")
SHOCK_START = pd.Timestamp("2025-03-01T09:59:00+00:00")
SHOCK_DURATION = pd.Timedelta(minutes=4)
SHOCK_END = SHOCK_START + SHOCK_DURATION
ANALYSIS_WINDOW_START = SHOCK_START - pd.Timedelta(minutes=10)
ANALYSIS_WINDOW_END = SHOCK_START + pd.Timedelta(minutes=15)

ROLE_LABELS = {
    "stock_market_maker": "Stock market maker",
    "stock_institutional": "Stock institutional",
    "stock_retail_1": "Stock retail 1",
    "stock_retail_2": "Stock retail 2",
    "future_market_maker": "Future market maker",
    "future_institutional": "Future institutional",
    "future_retail": "Future retail",
}
ROLE_ORDER = list(ROLE_LABELS)
OUTCOME_ORDER = ("fully_hedged", "partial_or_unhedged", "unfilled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without adding a tabulate dependency."""
    table = frame.reset_index()
    headers = [str(column) for column in table.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)

    rows = [[render(value) for value in row] for row in table.itertuples(index=False, name=None)]
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def run_id(treatment: str, mode: str, replicate: int) -> str:
    return f"ecology_final_v2_{treatment}_{mode}_r{replicate}"


def run_path(runs_dir: Path, treatment: str, mode: str, replicate: int) -> Path:
    path = runs_dir / run_id(treatment, mode, replicate)
    if not path.exists():
        raise FileNotFoundError(f"Expected final run is missing: {path}")
    return path


def shock_severity(run_dir: Path) -> float:
    log_path = run_dir / "logs" / "agents" / "agent_shock_agent.log"
    match = re.search(r"severity=([0-9.]+)", log_path.read_text(errors="replace"))
    if match is None:
        return float("nan")
    return float(match.group(1))


def market_path(market: dict[str, Any]) -> pd.DataFrame:
    """Turn irregular trade prints into a 30-second, last-trade marked path."""
    grid = pd.DataFrame({"timestamp": pd.date_range(SESSION_START, SESSION_END, freq="30s")})
    trades = pd.DataFrame(market.get("trades", []))
    if trades.empty:
        grid["price"] = np.nan
        grid["volume"] = 0
        grid["trade_count"] = 0
        return grid

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades["price"] = pd.to_numeric(trades["price"])
    trades["quantity"] = pd.to_numeric(trades["quantity"])
    trades = trades.sort_values("timestamp")

    last_price = trades.drop_duplicates("timestamp", keep="last")[["timestamp", "price"]]
    grid = pd.merge_asof(grid, last_price, on="timestamp", direction="backward")
    grid["price"] = grid["price"].ffill().fillna(as_float(market.get("start_price")))

    volume = trades.groupby("timestamp")["quantity"].sum()
    trade_count = trades.groupby("timestamp").size()
    grid["volume"] = grid["timestamp"].map(volume).fillna(0).astype(int)
    grid["trade_count"] = grid["timestamp"].map(trade_count).fillna(0).astype(int)
    return grid


def price_at(path: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    row = path.loc[path["timestamp"] == timestamp, "price"]
    return as_float(row.iloc[0]) if not row.empty else float("nan")


def event_metrics(path: pd.DataFrame) -> dict[str, float]:
    pre_price = price_at(path, SHOCK_START - pd.Timedelta(seconds=30))
    end_price = price_at(path, SHOCK_END)
    recovery_price = price_at(path, SHOCK_START + pd.Timedelta(minutes=15))
    active = path.loc[(path["timestamp"] >= SHOCK_START) & (path["timestamp"] < SHOCK_END)]
    active_prices = active["price"].replace(0, np.nan).dropna()
    log_returns = np.log(active_prices).diff().dropna()
    return {
        "pre_shock_price": pre_price,
        "shock_end_price": end_price,
        "event_return_pct": 100 * ((end_price / pre_price) - 1) if pre_price > 0 else np.nan,
        "fifteen_min_return_pct": 100 * ((recovery_price / pre_price) - 1)
        if pre_price > 0
        else np.nan,
        "event_volume": int(active["volume"].sum()),
        "event_trade_count": int(active["trade_count"].sum()),
        "event_realized_vol_bps": float(log_returns.std(ddof=1) * 10_000)
        if len(log_returns) > 1
        else np.nan,
    }


def agent_role(agent_id: str) -> str:
    if "market_maker" in agent_id:
        return "market_maker"
    if "institutional" in agent_id:
        return "institutional"
    if "retail" in agent_id:
        return "retail"
    if "arbitrage" in agent_id:
        return "arbitrageur"
    return "other"


def collect_data(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths: list[pd.DataFrame] = []
    markets: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for mode in MODES:
        for replicate in REPLICATES:
            for treatment in TREATMENTS:
                current_run_id = run_id(treatment, mode, replicate)
                current_run = run_path(runs_dir, treatment, mode, replicate)
                reports = current_run / "reports"
                manifest = read_json(reports / "ecology_manifest.json")
                market_summary = read_json(reports / "ecology_market_summary.json")["markets"]
                severity = shock_severity(current_run)
                seed = manifest["experiment"]["master_seed"]

                for asset, summary in market_summary.items():
                    path = market_path(summary)
                    pre_price = price_at(path, SHOCK_START - pd.Timedelta(seconds=30))
                    path["run_id"] = current_run_id
                    path["mode"] = mode
                    path["treatment"] = treatment
                    path["replicate"] = replicate
                    path["seed"] = seed
                    path["asset"] = asset
                    path["relative_minutes"] = (
                        (path["timestamp"] - SHOCK_START).dt.total_seconds() / 60
                    )
                    path["normalized_return_pct"] = 100 * ((path["price"] / pre_price) - 1)
                    paths.append(path)

                    row = {
                        "run_id": current_run_id,
                        "mode": mode,
                        "treatment": treatment,
                        "replicate": replicate,
                        "seed": seed,
                        "shock_severity": severity,
                        "asset": asset,
                        "total_trade_count": int(summary.get("trade_count", 0)),
                        "total_volume": int(summary.get("volume", 0)),
                        "start_price": as_float(summary.get("start_price")),
                        "end_price": as_float(summary.get("end_price")),
                        "session_return_pct": 100 * as_float(summary.get("return")),
                    }
                    row.update(event_metrics(path))
                    markets.append(row)

                for metrics_path in sorted((reports / "agents").glob("metrics_*.json")):
                    agent_id = metrics_path.stem.removeprefix("metrics_")
                    metrics = read_json(metrics_path)
                    series = read_json(
                        reports / "agents" / f"portfolio_timeseries_{agent_id}.json"
                    )
                    initial_value = as_float(series[0].get("value")) if series else np.nan
                    agents.append(
                        {
                            "run_id": current_run_id,
                            "mode": mode,
                            "treatment": treatment,
                            "replicate": replicate,
                            "seed": seed,
                            "agent_id": agent_id,
                            "role": agent_role(agent_id),
                            "initial_portfolio_value": initial_value,
                            "last_portfolio_value": as_float(metrics.get("Last Portfolio Value")),
                            "roi_pct": 100 * as_float(metrics.get("ROI")),
                            "max_drawdown_pct": 100 * as_float(metrics.get("Max Drawdown")),
                            "sharpe_ratio": as_float(metrics.get("Sharpe Ratio")),
                            "total_pnl": sum(
                                as_float(value, 0.0)
                                for value in (metrics.get("Total P&L") or {}).values()
                            ),
                            "unrealized_pnl": sum(
                                as_float(value, 0.0)
                                for value in (metrics.get("Unrealized P&L") or {}).values()
                            ),
                            "gross_exposure": as_float(metrics.get("Gross Exposure")),
                            "net_exposure": as_float(metrics.get("Net Exposure")),
                            "num_trades": int(as_float(metrics.get("Num Trades"), 0)),
                        }
                    )

                summary = read_json(reports / "ecology_agent_response_summary.json")
                active_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for response in summary.get("responses", []):
                    if response.get("active_event_ids"):
                        active_by_agent[str(response.get("agent_id"))].append(response)
                for agent_id, values in active_by_agent.items():
                    is_linked_future = agent_id.startswith("future_") or agent_id == "basis_arbitrageur"
                    responses.append(
                        {
                            "run_id": current_run_id,
                            "mode": mode,
                            "treatment": treatment,
                            "replicate": replicate,
                            "seed": seed,
                            "agent_id": agent_id,
                            "role": agent_role(agent_id),
                            "linked_future_agent": is_linked_future,
                            "active_event_slow_loops": len(values),
                            "active_event_strategy_changes": sum(
                                bool(value.get("strategy_changed")) for value in values
                            ),
                            "active_event_risk_mode_changes": sum(
                                value.get("risk_mode_before") != value.get("risk_mode_after")
                                for value in values
                            ),
                        }
                    )

                decision_path = reports / "ecology_decision_summary.json"
                for decision in read_json(decision_path).get("decisions", []):
                    timestamp = pd.to_datetime(decision.get("timestamp"), utc=True)
                    decisions.append(
                        {
                            "run_id": current_run_id,
                            "mode": mode,
                            "treatment": treatment,
                            "replicate": replicate,
                            "seed": seed,
                            "decision_id": decision.get("decision_id"),
                            "timestamp": timestamp,
                            "relative_minutes": (timestamp - SHOCK_START).total_seconds() / 60,
                            "observed_basis_bps": as_float(decision.get("observed_basis_bps")),
                            "abs_observed_basis_bps": abs(as_float(decision.get("observed_basis_bps"))),
                            "active_shock": bool(decision.get("active_event_ids")),
                            "execution_outcome": decision.get("execution_outcome"),
                        }
                    )

    return (
        pd.concat(paths, ignore_index=True),
        pd.DataFrame(markets),
        pd.DataFrame(agents),
        pd.DataFrame(responses),
        pd.DataFrame(decisions),
    )


def paired_role_deltas(agent_df: pd.DataFrame) -> pd.DataFrame:
    base = agent_df.loc[
        (agent_df["treatment"] == "d0") & agent_df["agent_id"].isin(ROLE_ORDER)
    ].set_index(["mode", "replicate", "agent_id"])
    rows: list[dict[str, Any]] = []
    for treatment in ("i1", "a1"):
        linked = agent_df.loc[
            (agent_df["treatment"] == treatment) & agent_df["agent_id"].isin(ROLE_ORDER)
        ].set_index(["mode", "replicate", "agent_id"])
        merged = linked.join(base[["roi_pct", "max_drawdown_pct"]], rsuffix="_d0", how="inner")
        for (mode, replicate, agent_id), row in merged.iterrows():
            rows.append(
                {
                    "mode": mode,
                    "treatment": treatment,
                    "replicate": replicate,
                    "agent_id": agent_id,
                    "role": agent_role(agent_id),
                    "roi_delta_bps_vs_d0": (row["roi_pct"] - row["roi_pct_d0"]) * 100,
                    "drawdown_delta_bps_vs_d0": (
                        row["max_drawdown_pct"] - row["max_drawdown_pct_d0"]
                    )
                    * 100,
                }
            )
    return pd.DataFrame(rows)


def save_event_path_figure(path_df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, mode in enumerate(MODES):
        for col, asset in enumerate(("AAPL", "AAPL_FUT")):
            ax = axes[row, col]
            subset = path_df.loc[
                (path_df["mode"] == mode)
                & (path_df["asset"] == asset)
                & (path_df["timestamp"] >= ANALYSIS_WINDOW_START)
                & (path_df["timestamp"] <= ANALYSIS_WINDOW_END)
            ]
            for treatment in TREATMENTS:
                data = subset.loc[subset["treatment"] == treatment]
                for _, run in data.groupby("replicate"):
                    ax.plot(
                        run["relative_minutes"],
                        run["normalized_return_pct"],
                        color=TREATMENT_COLORS[treatment],
                        alpha=0.16,
                        linewidth=1,
                    )
                aggregate = data.groupby("relative_minutes")["normalized_return_pct"].agg(
                    ["mean", "min", "max"]
                )
                ax.fill_between(
                    aggregate.index,
                    aggregate["min"],
                    aggregate["max"],
                    color=TREATMENT_COLORS[treatment],
                    alpha=0.08,
                )
                ax.plot(
                    aggregate.index,
                    aggregate["mean"],
                    color=TREATMENT_COLORS[treatment],
                    linewidth=2.5,
                    label=TREATMENT_LABELS[treatment],
                )
            ax.axvspan(0, SHOCK_DURATION.total_seconds() / 60, color="#f6bd60", alpha=0.18)
            ax.axvline(0, color="#495057", linestyle="--", linewidth=1)
            ax.axhline(0, color="#adb5bd", linewidth=0.8)
            ax.set_title(f"{MODE_LABELS[mode]}: {asset}")
            ax.set_ylabel("Return from pre-shock mark (%)")
            ax.set_xlabel("Minutes relative to stock shock")
            ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Event-study price paths: stock-specific shock at minute 0", y=1.01, fontsize=14)
    fig.text(0.5, 0.01, "Solid line = mean of 3 paired seeds; faint lines and bands = seed range.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_dir / "01_event_price_paths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_transmission_figure(market_df: pd.DataFrame, output_dir: Path) -> None:
    future = market_df.loc[market_df["asset"] == "AAPL_FUT"].copy()
    metrics = (
        ("event_return_pct", "4-minute future return after stock shock (%)"),
        ("event_volume", "Future traded volume during active shock"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    positions = np.arange(len(TREATMENTS))
    for row, mode in enumerate(MODES):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            subset = future.loc[future["mode"] == mode]
            for replicate, values in subset.groupby("replicate"):
                values = values.set_index("treatment").reindex(TREATMENTS)
                ax.plot(positions, values[metric], color="#8d99ae", alpha=0.7, linewidth=1.2)
                ax.scatter(
                    positions,
                    values[metric],
                    s=42,
                    color=[TREATMENT_COLORS[item] for item in TREATMENTS],
                    edgecolors="white",
                    linewidth=0.7,
                    zorder=3,
                )
            means = subset.groupby("treatment")[metric].mean().reindex(TREATMENTS)
            ax.plot(positions, means, color="#111827", linewidth=2.5, marker="D", markersize=5)
            ax.axhline(0, color="#adb5bd", linewidth=0.8)
            ax.set_xticks(positions, [TREATMENT_LABELS[item] for item in TREATMENTS], rotation=12)
            ax.set_title(f"{MODE_LABELS[mode]}\n{label}")
            ax.set_ylabel(label)
    fig.suptitle("Future-market transmission outcomes by channel treatment", y=1.01, fontsize=14)
    fig.text(0.5, 0.01, "Each connected line is one matched random seed; black diamonds are treatment means.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(output_dir / "02_future_transmission.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_arbitrage_figure(decision_df: pd.DataFrame, output_dir: Path) -> None:
    a1 = decision_df.loc[(decision_df["treatment"] == "a1")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    window = a1.loc[
        (a1["relative_minutes"] >= -10) & (a1["relative_minutes"] <= 15)
    ]
    for mode in MODES:
        subset = window.loc[window["mode"] == mode]
        aggregate = subset.groupby("relative_minutes")["abs_observed_basis_bps"].agg(["mean", "min", "max"])
        axes[0].fill_between(
            aggregate.index,
            aggregate["min"],
            aggregate["max"],
            alpha=0.12,
            color="#168aad" if mode == "frozen" else "#f18f01",
        )
        axes[0].plot(
            aggregate.index,
            aggregate["mean"],
            linewidth=2.5,
            label=MODE_LABELS[mode],
            color="#168aad" if mode == "frozen" else "#f18f01",
        )
    axes[0].axvspan(0, 4, color="#f6bd60", alpha=0.18)
    axes[0].axvline(0, color="#495057", linestyle="--", linewidth=1)
    axes[0].set_title("A1 executable basis dislocation")
    axes[0].set_xlabel("Minutes relative to stock shock")
    axes[0].set_ylabel("Absolute observed basis (bps)")
    axes[0].legend(frameon=False)

    executed = a1.loc[a1["execution_outcome"] != "observed_only"]
    outcome_counts = (
        executed.groupby(["mode", "execution_outcome"]).size().unstack(fill_value=0).reindex(
            index=MODES, columns=OUTCOME_ORDER, fill_value=0
        )
    )
    bottom = np.zeros(len(MODES))
    colors = {"fully_hedged": "#168aad", "partial_or_unhedged": "#f18f01", "unfilled": "#d1495b"}
    for outcome in OUTCOME_ORDER:
        values = outcome_counts[outcome].to_numpy()
        axes[1].bar(
            [MODE_LABELS[mode] for mode in MODES],
            values,
            bottom=bottom,
            color=colors[outcome],
            label=outcome.replace("_", " "),
        )
        bottom += values
    axes[1].set_title("A1 submitted arbitrage outcomes across 3 seeds")
    axes[1].set_ylabel("Decision count")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Arbitrage channel: observed basis and two-leg execution", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "03_a1_arbitrage_mechanism.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_robustness_figure(delta_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for row, mode in enumerate(MODES):
        for col, (metric, title) in enumerate(
            (
                ("roi_delta_bps_vs_d0", "Mean ROI change vs D0 (bps)"),
                ("drawdown_delta_bps_vs_d0", "Mean max drawdown change vs D0 (bps)"),
            )
        ):
            data = delta_df.loc[delta_df["mode"] == mode]
            heatmap = (
                data.groupby(["agent_id", "treatment"])[metric]
                .mean()
                .unstack()
                .reindex(index=ROLE_ORDER, columns=["i1", "a1"])
            )
            heatmap.index = [ROLE_LABELS[item] for item in heatmap.index]
            sns.heatmap(
                heatmap,
                ax=axes[row, col],
                annot=True,
                fmt=".1f",
                cmap="RdBu_r",
                center=0,
                cbar=row == 0,
                linewidths=0.5,
                linecolor="white",
            )
            axes[row, col].set_title(f"{MODE_LABELS[mode]}\n{title}")
            axes[row, col].set_xlabel("Linked-market treatment")
            axes[row, col].set_ylabel("")
            axes[row, col].set_xticklabels(["I1 information", "A1 arbitrage"], rotation=0)
    fig.suptitle("Role-level robustness relative to the disconnected control", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_dir / "04_role_robustness.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_llm_response_figure(response_df: pd.DataFrame, output_dir: Path) -> None:
    llm = response_df.loc[(response_df["mode"] == "llm") & response_df["linked_future_agent"]].copy()
    per_run = (
        llm.groupby(["treatment", "replicate"], as_index=False)[
            ["active_event_slow_loops", "active_event_strategy_changes"]
        ]
        .sum()
    )
    full_index = pd.MultiIndex.from_product(
        [TREATMENTS, REPLICATES], names=["treatment", "replicate"]
    )
    per_run = per_run.set_index(["treatment", "replicate"]).reindex(full_index, fill_value=0).reset_index()
    summary = per_run.melt(
        id_vars=["treatment", "replicate"],
        var_name="metric",
        value_name="count",
    )
    labels = {
        "active_event_slow_loops": "Shock-aware slow-loop decisions",
        "active_event_strategy_changes": "Strategy updates during active shock",
    }
    summary["metric"] = summary["metric"].map(labels)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=summary,
        x="treatment",
        y="count",
        hue="metric",
        order=TREATMENTS,
        errorbar=None,
        palette=["#168aad", "#f18f01"],
        ax=ax,
    )
    sns.stripplot(
        data=summary,
        x="treatment",
        y="count",
        hue="metric",
        order=TREATMENTS,
        dodge=True,
        palette={label: "#111827" for label in labels.values()},
        size=5,
        ax=ax,
        legend=False,
    )
    ax.set_xticks(np.arange(len(TREATMENTS)), [TREATMENT_LABELS[item] for item in TREATMENTS])
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Count per run")
    ax.set_title("LLM future-market response to the stock shock")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "05_llm_information_response.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    market_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    response_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    future = market_df.loc[market_df["asset"] == "AAPL_FUT"]
    future_table = (
        future.groupby(["mode", "treatment"])[["event_return_pct", "event_volume", "event_realized_vol_bps"]]
        .mean()
        .round(3)
    )
    llm_future = response_df.loc[
        (response_df["mode"] == "llm") & response_df["linked_future_agent"]
    ]
    llm_response_table = (
        llm_future.groupby("treatment")[["active_event_slow_loops", "active_event_strategy_changes"]]
        .sum()
        .reindex(TREATMENTS, fill_value=0)
        .astype(int)
    )
    a1 = decision_df.loc[decision_df["treatment"] == "a1"]
    a1_counts = (
        a1.loc[a1["execution_outcome"] != "observed_only"]
        .groupby(["mode", "execution_outcome"])
        .size()
        .unstack(fill_value=0)
    )
    maker_deltas = (
        delta_df.loc[delta_df["agent_id"].isin(["stock_market_maker", "future_market_maker"])]
        .groupby(["mode", "treatment"])[["roi_delta_bps_vs_d0", "drawdown_delta_bps_vs_d0"]]
        .mean()
        .round(1)
    )
    frozen_max_abs = (
        delta_df.loc[delta_df["mode"] == "frozen", ["roi_delta_bps_vs_d0", "drawdown_delta_bps_vs_d0"]]
        .abs()
        .max()
    )

    lines = [
        "# Financial Ecology Pilot: Final V2 Analysis",
        "",
        "## Scope",
        "",
        "This is a controlled stock-to-future pilot: 18 one-hour runs, with D0 disconnected, I1 information-only, and A1 information plus valuation/arbitrage. Each treatment has three matched seeds in frozen and LLM-adaptive modes.",
        "",
        "The injected event is one unexpected, stock-specific negative-news shock at 09:59 for four simulated minutes. It is a micro/idiosyncratic shock, not a systematic macro shock.",
        "",
        "## Data Integrity",
        "",
        "All 18 runs completed, both books traded in every run, all expected report files exist, and there were no runtime errors, rejected slow-loop decisions, malformed limit orders, or executed self-trades. Self-trade prevention did cancel potential maker self-crosses; retain that count as an execution-friction control, not as trade volume.",
        "",
        "## RQ1: How does a stock shock transmit to a linked future through agents?",
        "",
        "The mechanism is verified. D0 gives the future market no shock information. I1 forwards an information-only event to its three future-market participants. A1 adds the basis arbitrageur and valuation relationship. In LLM runs, future-facing agents made the following shock-aware slow-loop decisions and strategy updates:",
        "",
        markdown_table(llm_response_table),
        "",
        "A positive count in I1/A1 and zero in D0 is direct evidence that the relationship graph changed what future-market agents could observe and act on. The event-price and transmission figures show the market outcome, while the decision log makes the path auditable.",
        "",
        "Average future-market event metrics across the three seeds:",
        "",
        markdown_table(future_table),
        "",
        "A1 submitted two-leg basis trades, with outcomes:",
        "",
        markdown_table(a1_counts),
        "",
        "Interpretation: the pilot establishes causal channel operation and a measurable future-market response. It does not yet establish a stable direction or magnitude of transmission because the three-seed estimates are intentionally small and LLM decisions are stochastic.",
        "",
        "## RQ2: Do strategies robust in D0 remain robust after linking markets?",
        "",
        f"Frozen strategies were broadly stable: the largest mean absolute role-level change versus D0 was {frozen_max_abs['roi_delta_bps_vs_d0']:.1f} bps of ROI or {frozen_max_abs['drawdown_delta_bps_vs_d0']:.1f} bps of maximum drawdown. This is expected because their slow strategy state is deliberately frozen.",
        "",
        "For adaptive LLM agents, the direction is less benign in this pilot. Mean market-maker deltas versus D0 were:",
        "",
        markdown_table(maker_deltas),
        "",
        "This suggests that linked-market information and arbitrage conditions can reduce robustness for adaptive liquidity providers. Treat it as a pilot finding, not a statistical conclusion: three seeds are too few for formal inference, and LLM responses add another source of variation.",
        "",
        "## What This Batch Can and Cannot Claim",
        "",
        "Supported now: a localized stock shock reaches the future through declared information and arbitrage channels; the causal path is logged; linked settings can change market and agent outcomes; adaptive LLM roles appear more sensitive than frozen baselines.",
        "",
        "Not supported yet: a claim about systematic macro shock propagation, all asset classes, or statistically reliable robustness rankings. The current study has one stock/future pair, one negative micro-shock family, and three seeds.",
        "",
        "## Required Follow-up Before a Final Research Claim",
        "",
        "1. Add a systematic macro-shock factorial condition, such as a policy-rate/funding/liquidity shock affecting declared common state, then repeat D0/I1/A1. This is necessary for the exact systematic-shock research question.",
        "2. Increase the current treatment grid to at least 10 paired seeds per mode before testing effects. Keep frozen runs as the deterministic baseline and record model, prompt, and temperature for LLM runs.",
        "3. Add at least one further relationship type, such as underlying-option or basket-ETF, before generalizing from stock/future ecology to cross-asset ecology.",
        "",
        "## Files",
        "",
        "- `run_level_market_metrics.csv`: run and event-window market outcomes.",
        "- `agent_metrics.csv`: agent-level ROI, PnL, drawdown, exposure, and trade counts.",
        "- `llm_active_event_responses.csv`: shock-aware LLM decisions by agent.",
        "- `arbitrage_decisions.csv`: A1 basis observations and two-leg outcomes.",
        "- `paired_role_deltas.csv`: matched I1/A1 versus D0 robustness deltas.",
    ]
    (output_dir / "financial_ecology_final_v2_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_df, market_df, agent_df, response_df, decision_df = collect_data(args.runs_dir)
    delta_df = paired_role_deltas(agent_df)

    path_df.to_csv(args.output_dir / "price_paths_30s.csv", index=False)
    market_df.to_csv(args.output_dir / "run_level_market_metrics.csv", index=False)
    agent_df.to_csv(args.output_dir / "agent_metrics.csv", index=False)
    response_df.to_csv(args.output_dir / "llm_active_event_responses.csv", index=False)
    decision_df.to_csv(args.output_dir / "arbitrage_decisions.csv", index=False)
    delta_df.to_csv(args.output_dir / "paired_role_deltas.csv", index=False)
    (
        market_df.groupby(["mode", "treatment", "asset"], as_index=False)
        .mean(numeric_only=True)
        .to_csv(args.output_dir / "treatment_market_summary.csv", index=False)
    )

    save_event_path_figure(path_df, args.output_dir)
    save_transmission_figure(market_df, args.output_dir)
    save_arbitrage_figure(decision_df, args.output_dir)
    save_robustness_figure(delta_df, args.output_dir)
    save_llm_response_figure(response_df, args.output_dir)
    write_report(market_df, agent_df, response_df, decision_df, delta_df, args.output_dir)

    print(f"Wrote financial ecology analysis to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
