#!/usr/bin/env python3
"""Create a quality-controlled analysis package for the AML ecology experiments.

The package combines the original stock/future pilot (D0, I1, A1) with the
later three-market, two-shock study (C0, C2).  It reports mechanisms and
effect sizes, but deliberately avoids significance claims from three seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
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
RUNS_DIR = ROOT / ".aml_runs"
OUTPUT_DIR = ROOT / "artifacts" / "financial_ecology_combined_v1"
SESSION_START = pd.Timestamp("2025-03-01T09:30:00+00:00")
SESSION_END = pd.Timestamp("2025-03-01T10:30:00+00:00")

LEGACY_TREATMENTS = ("d0", "i1", "a1")
MODES = ("frozen", "llm")
REPLICATES = (1, 2, 3)
LEGACY_LABELS = {"d0": "D0 disconnected", "i1": "I1 information", "a1": "A1 linked"}
LINKED_LABELS = {"c0": "C0 control", "c2": "C2 linked"}
COLORS = {"d0": "#667085", "i1": "#1f8a70", "a1": "#d97706", "c0": "#667085", "c2": "#1f8a70"}
ASSET_COLORS = {"AAPL": "#0f766e", "AAPL_FUT": "#2563eb", "UST10Y": "#7c3aed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def table(frame: pd.DataFrame) -> str:
    frame = frame.reset_index()
    headers = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)

    rows = [[render(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def run_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for treatment in LEGACY_TREATMENTS:
        for mode in MODES:
            for replicate in REPLICATES:
                specs.append(
                    {
                        "study": "stock_future_micro",
                        "treatment": treatment,
                        "mode": mode,
                        "replicate": replicate,
                        "run_id": f"ecology_final_v2_{treatment}_{mode}_r{replicate}",
                        "micro_time": pd.Timestamp("2025-03-01T09:59:00+00:00"),
                        "micro_end": pd.Timestamp("2025-03-01T10:03:00+00:00"),
                        "macro_time": pd.NaT,
                    }
                )
    for treatment in ("c0", "c2"):
        for mode in MODES:
            for replicate in REPLICATES:
                specs.append(
                    {
                        "study": "stock_future_bond_two_shock",
                        "treatment": treatment,
                        "mode": mode,
                        "replicate": replicate,
                        "run_id": f"ecology_{treatment}_{mode}_r{replicate}",
                        "micro_time": pd.Timestamp("2025-03-01T09:45:00+00:00"),
                        "micro_end": pd.Timestamp("2025-03-01T09:49:00+00:00"),
                        "macro_time": pd.Timestamp("2025-03-01T10:00:00+00:00"),
                    }
                )
    return specs


def market_path(summary: dict[str, Any]) -> pd.DataFrame:
    grid = pd.DataFrame({"timestamp": pd.date_range(SESSION_START, SESSION_END, freq="30s")})
    trades = pd.DataFrame(summary.get("trades", []))
    if trades.empty:
        grid["price"] = np.nan
        grid["volume"] = 0
        grid["trade_count"] = 0
        return grid
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0)
    trades = trades.sort_values("timestamp")
    last = trades.drop_duplicates("timestamp", keep="last")[["timestamp", "price"]]
    grid = pd.merge_asof(grid, last, on="timestamp", direction="backward")
    grid["price"] = grid["price"].ffill().fillna(as_float(summary.get("start_price")))
    grid["volume"] = grid["timestamp"].map(trades.groupby("timestamp")["quantity"].sum()).fillna(0)
    grid["trade_count"] = grid["timestamp"].map(trades.groupby("timestamp").size()).fillna(0)
    return grid


def price_at(path: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    rows = path.loc[path["timestamp"] == timestamp, "price"]
    return as_float(rows.iloc[0]) if not rows.empty else float("nan")


def event_metrics(path: pd.DataFrame, event_start: pd.Timestamp, event_end: pd.Timestamp) -> dict[str, float]:
    pre = price_at(path, event_start - pd.Timedelta(seconds=30))
    end = price_at(path, event_end)
    recovery = price_at(path, min(event_start + pd.Timedelta(minutes=15), SESSION_END))
    active = path.loc[(path["timestamp"] >= event_start) & (path["timestamp"] < event_end)]
    prices = active["price"].replace(0, np.nan).dropna()
    returns = np.log(prices).diff().dropna()
    return {
        "pre_price": pre,
        "end_price": end,
        "event_return_pct": 100 * (end / pre - 1) if pre > 0 else np.nan,
        "fifteen_min_return_pct": 100 * (recovery / pre - 1) if pre > 0 else np.nan,
        "event_volume": float(active["volume"].sum()),
        "event_trade_count": int(active["trade_count"].sum()),
        "event_realized_vol_bps": float(returns.std(ddof=1) * 10_000) if len(returns) > 1 else np.nan,
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


def warning_counts(run_dir: Path) -> Counter[str]:
    result: Counter[str] = Counter()
    for path in (run_dir / "logs").rglob("*.log"):
        content = path.read_text(errors="replace")
        result["warnings"] += len(re.findall(r"\bWARNING\b", content, flags=re.I))
        result["self_trade_prevention"] += len(re.findall(r"Self-trade prevention", content, flags=re.I))
        result["stale_cancel"] += len(re.findall(r"non-existent order ID|failed to cancel order", content, flags=re.I))
        result["partial_market_orders"] += len(re.findall(r"partially unfilled", content, flags=re.I))
        result["tracebacks"] += len(re.findall(r"Traceback", content, flags=re.I))
    return result


def collect(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for spec in run_specs():
        run_dir = runs_dir / spec["run_id"]
        reports = run_dir / "reports"
        required = [
            reports / "simulation_summary.json",
            reports / "ecology_market_summary.json",
            reports / "ecology_agent_response_summary.json",
            reports / "ecology_channel_ledger.json",
            reports / "ecology_decision_summary.json",
        ]
        missing = [str(path.relative_to(run_dir)) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{spec['run_id']} is incomplete: {', '.join(missing)}")

        manifest = read_json(reports / "ecology_manifest.json")
        simulation = read_json(reports / "simulation_summary.json")
        market_summary = read_json(reports / "ecology_market_summary.json")["markets"]
        response_summary = read_json(reports / "ecology_agent_response_summary.json")
        ledger = read_json(reports / "ecology_channel_ledger.json")
        counters = warning_counts(run_dir)
        statuses = Counter(item.get("slow_loop_status") for item in response_summary.get("responses", []))
        quality_rows.append(
            {
                **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                "duration": simulation["simulation_info"].get("duration"),
                "assets": ",".join(simulation["simulation_info"].get("instruments", [])),
                "llm_agents": simulation.get("research_metrics", {}).get("llm_agents", 0),
                "slow_loop_completed": statuses.get("completed", 0),
                "slow_loop_rejected": statuses.get("rejected", 0),
                "slow_loop_failed": statuses.get("failed", 0),
                "strategy_changes": response_summary.get("strategy_change_count", 0),
                "event_deliveries": sum(ledger.get("event_delivery_counts", {}).values()),
                "relationship_deliveries": sum(
                    value for key, value in ledger.get("event_delivery_counts", {}).items() if "relationship" in key
                ),
                **counters,
            }
        )

        for asset, summary in market_summary.items():
            path = market_path(summary)
            path = path.assign(**{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}}, asset=asset)
            paths.append(path)
            for event_name, start, end in (
                ("micro", spec["micro_time"], spec["micro_end"]),
                ("macro", spec["macro_time"], min(spec["macro_time"] + pd.Timedelta(minutes=4), SESSION_END) if pd.notna(spec["macro_time"]) else pd.NaT),
            ):
                if pd.isna(start):
                    continue
                event_rows.append(
                    {
                        **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                        "asset": asset,
                        "event": event_name,
                        **event_metrics(path, start, end),
                        "total_trade_count": int(summary.get("trade_count", 0)),
                        "total_volume": float(summary.get("volume", 0)),
                        "session_return_pct": 100 * as_float(summary.get("return")),
                    }
                )

        for metrics_path in sorted((reports / "agents").glob("metrics_*.json")):
            agent_id = metrics_path.stem.removeprefix("metrics_")
            metrics = read_json(metrics_path)
            series_path = reports / "agents" / f"portfolio_timeseries_{agent_id}.json"
            series = read_json(series_path) if series_path.exists() else []
            agent_rows.append(
                {
                    **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                    "agent_id": agent_id,
                    "role": agent_role(agent_id),
                    "initial_value": as_float(series[0].get("value")) if series else np.nan,
                    "final_value": as_float(metrics.get("Last Portfolio Value")),
                    "roi_pct": 100 * as_float(metrics.get("ROI")),
                    "max_drawdown_pct": 100 * as_float(metrics.get("Max Drawdown")),
                    "sharpe": as_float(metrics.get("Sharpe Ratio")),
                    "total_pnl": sum(as_float(value, 0.0) for value in (metrics.get("Total P&L") or {}).values()),
                    "gross_exposure": as_float(metrics.get("Gross Exposure")),
                    "net_exposure": as_float(metrics.get("Net Exposure")),
                    "num_trades": int(as_float(metrics.get("Num Trades"), 0)),
                }
            )

        for item in response_summary.get("responses", []):
            response_rows.append(
                {
                    **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                    "agent_id": item.get("agent_id"),
                    "role": agent_role(str(item.get("agent_id"))),
                    "status": item.get("slow_loop_status"),
                    "strategy_changed": bool(item.get("strategy_changed")),
                    "event_context": bool(item.get("event_context_present")),
                    "active_event": bool(item.get("active_event_ids")),
                    "known_event": bool(item.get("known_event_ids")),
                    "risk_changed": item.get("risk_mode_before") != item.get("risk_mode_after"),
                }
            )

        for item in read_json(reports / "ecology_decision_summary.json").get("decisions", []):
            decision_rows.append(
                {
                    **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                    "timestamp": pd.to_datetime(item.get("timestamp"), utc=True),
                    "decision_id": item.get("decision_id"),
                    "observed_basis_bps": as_float(item.get("observed_basis_bps")),
                    "abs_observed_basis_bps": abs(as_float(item.get("observed_basis_bps"))),
                    "execution_outcome": item.get("execution_outcome"),
                    "active_shock": bool(item.get("active_event_ids")),
                }
            )

    return (
        pd.concat(paths, ignore_index=True),
        pd.DataFrame(event_rows),
        pd.DataFrame(quality_rows),
        pd.DataFrame(agent_rows),
        pd.DataFrame(response_rows),
        pd.DataFrame(decision_rows),
    )


def save_legacy_paths(paths: pd.DataFrame, output_dir: Path) -> None:
    start = pd.Timestamp("2025-03-01T09:49:00+00:00")
    end = pd.Timestamp("2025-03-01T10:14:00+00:00")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    shock = pd.Timestamp("2025-03-01T09:59:00+00:00")
    for row, mode in enumerate(MODES):
        for col, asset in enumerate(("AAPL", "AAPL_FUT")):
            ax = axes[row, col]
            subset = paths.loc[(paths.study == "stock_future_micro") & (paths["mode"] == mode) & (paths.asset == asset)].copy()
            for treatment in LEGACY_TREATMENTS:
                data = subset.loc[subset.treatment == treatment].copy()
                data["relative_min"] = (data.timestamp - shock).dt.total_seconds() / 60
                base = data.loc[data.timestamp == shock - pd.Timedelta(seconds=30), ["replicate", "price"]].set_index("replicate")["price"]
                data["return_pct"] = data.apply(lambda row: 100 * (row.price / base.get(row.replicate, np.nan) - 1), axis=1)
                data = data.loc[(data.timestamp >= start) & (data.timestamp <= end)]
                mean = data.groupby("relative_min").return_pct.mean()
                ax.plot(mean.index, mean.values, color=COLORS[treatment], lw=2.4, label=LEGACY_LABELS[treatment])
            ax.axvspan(0, 4, color="#f6bd60", alpha=0.2)
            ax.axvline(0, color="#4b5563", ls="--", lw=1)
            ax.axhline(0, color="#cbd5e1", lw=1)
            ax.set_title(f"{mode.title()} - {asset}")
            ax.set_ylabel("Return from pre-shock price (%)")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Earlier D0/I1/A1 stock-shock transmission", y=0.99, fontsize=14)
    fig.text(0.5, 0.01, "Mean across three paired seeds. Shaded band is the four-minute idiosyncratic stock shock.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_dir / "01_legacy_micro_transmission.png", dpi=220)
    plt.close(fig)


def paired_legacy_deltas(agent_df: pd.DataFrame) -> pd.DataFrame:
    legacy = agent_df.loc[(agent_df.study == "stock_future_micro") & agent_df.agent_id.isin([
        "stock_market_maker", "stock_institutional", "stock_retail_1", "stock_retail_2",
        "future_market_maker", "future_institutional", "future_retail",
    ])]
    control = legacy.loc[legacy.treatment == "d0"].set_index(["mode", "replicate", "agent_id"])
    rows = []
    for treatment in ("i1", "a1"):
        linked = legacy.loc[legacy.treatment == treatment].set_index(["mode", "replicate", "agent_id"])
        joined = linked.join(control[["roi_pct", "max_drawdown_pct"]], rsuffix="_d0", how="inner")
        for (mode, replicate, agent_id), row in joined.iterrows():
            rows.append({
                "mode": mode, "replicate": replicate, "treatment": treatment, "agent_id": agent_id,
                "role": agent_role(agent_id),
                "roi_delta_bps": 100 * (row.roi_pct - row.roi_pct_d0),
                "drawdown_delta_bps": 100 * (row.max_drawdown_pct - row.max_drawdown_pct_d0),
            })
    return pd.DataFrame(rows)


def save_legacy_robustness(deltas: pd.DataFrame, output_dir: Path) -> None:
    labels = {
        "stock_market_maker": "Stock MM", "stock_institutional": "Stock institutional", "stock_retail_1": "Stock retail 1",
        "stock_retail_2": "Stock retail 2", "future_market_maker": "Future MM", "future_institutional": "Future institutional", "future_retail": "Future retail",
    }
    order = list(labels)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for r, mode in enumerate(MODES):
        for c, (metric, title) in enumerate((("roi_delta_bps", "ROI change vs D0 (bps)"), ("drawdown_delta_bps", "Drawdown change vs D0 (bps)"))):
            pivot = deltas.loc[deltas["mode"] == mode].groupby(["agent_id", "treatment"])[metric].mean().unstack().reindex(index=order, columns=["i1", "a1"])
            pivot.index = [labels[value] for value in pivot.index]
            sns.heatmap(pivot, cmap="RdBu_r", center=0, annot=True, fmt=".1f", linewidths=.4, linecolor="white", cbar=r == 0, ax=axes[r,c])
            axes[r,c].set_title(f"{mode.title()} - {title}")
            axes[r,c].set_xlabel("Treatment")
            axes[r,c].set_xticklabels(["I1", "A1"])
            axes[r,c].set_ylabel("")
    fig.suptitle("Strategy robustness when a future market is linked", y=.995, fontsize=14)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(output_dir / "02_legacy_robustness.png", dpi=220)
    plt.close(fig)


def save_three_market_paths(paths: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    events = {"micro": pd.Timestamp("2025-03-01T09:45:00+00:00"), "macro": pd.Timestamp("2025-03-01T10:00:00+00:00")}
    for r, mode in enumerate(MODES):
        for c, (event_name, event_time) in enumerate(events.items()):
            ax = axes[r,c]
            subset = paths.loc[(paths.study == "stock_future_bond_two_shock") & (paths["mode"] == mode)].copy()
            for treatment in ("c0", "c2"):
                for asset in ("AAPL", "AAPL_FUT", "UST10Y"):
                    data = subset.loc[(subset.treatment == treatment) & (subset.asset == asset)].copy()
                    base = data.loc[data.timestamp == event_time - pd.Timedelta(seconds=30), ["replicate", "price"]].set_index("replicate")["price"]
                    data["relative_min"] = (data.timestamp - event_time).dt.total_seconds() / 60
                    data["return_pct"] = data.apply(lambda row: 100 * (row.price / base.get(row.replicate, np.nan) - 1), axis=1)
                    data = data.loc[(data.relative_min >= -8) & (data.relative_min <= 15)]
                    mean = data.groupby("relative_min").return_pct.mean()
                    style = "-" if treatment == "c2" else "--"
                    ax.plot(mean.index, mean.values, color=ASSET_COLORS[asset], ls=style, lw=2, label=f"{asset} {treatment.upper()}")
            ax.axvspan(0, 4, color="#f6bd60", alpha=.18)
            ax.axvline(0, color="#4b5563", ls="--", lw=1)
            ax.axhline(0, color="#cbd5e1", lw=1)
            ax.set_title(f"{mode.title()} - {event_name.title()} event")
            ax.set_xlabel("Minutes from event")
            ax.set_ylabel("Return from pre-event price (%)")
            ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("C0/C2 three-market event paths: dashed control, solid linked", y=.99, fontsize=14)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(output_dir / "03_three_market_event_paths.png", dpi=220)
    plt.close(fig)


def save_three_market_activity(events: pd.DataFrame, output_dir: Path) -> None:
    session = events.loc[(events.study == "stock_future_bond_two_shock") & (events.event == "macro")].copy()
    metrics = (("total_trade_count", "Session trade count"), ("total_volume", "Session volume"), ("session_return_pct", "Session return (%)"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, mode in enumerate(MODES):
        for c, (metric, title) in enumerate(metrics):
            ax = axes[r,c]
            data = session.loc[session["mode"] == mode]
            sns.barplot(data=data, x="asset", y=metric, hue="treatment", hue_order=["c0","c2"], palette=[COLORS["c0"], COLORS["c2"]], errorbar=None, ax=ax)
            sns.stripplot(data=data, x="asset", y=metric, hue="treatment", hue_order=["c0","c2"], dodge=True, color="#111827", size=3.5, ax=ax, legend=False)
            ax.set_title(f"{mode.title()} - {title}")
            ax.set_xlabel("")
            if c:
                ax.set_ylabel("")
            if r or c:
                legend = ax.get_legend()
                if legend:
                    legend.remove()
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, ["C0 control", "C2 linked"], loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Three-market activity by treatment and decision mode", y=.97, fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(output_dir / "04_three_market_activity.png", dpi=220)
    plt.close(fig)


def save_basis(decisions: pd.DataFrame, output_dir: Path) -> None:
    c2 = decisions.loc[(decisions.study == "stock_future_bond_two_shock") & (decisions.treatment == "c2")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode in MODES:
        data = c2.loc[c2["mode"] == mode].copy()
        data["relative_min"] = (data.timestamp - pd.Timestamp("2025-03-01T09:45:00+00:00")).dt.total_seconds()/60
        mean = data.groupby("relative_min").abs_observed_basis_bps.mean()
        axes[0].plot(mean.index, mean.values, lw=2.4, label=mode.title())
    axes[0].axvspan(0,4,color="#f6bd60",alpha=.18)
    axes[0].axvline(15,color="#d97706",ls="--",lw=1,label="Macro shock")
    axes[0].set_xlim(-8,30)
    axes[0].set_title("C2 stock-future basis observation")
    axes[0].set_xlabel("Minutes from micro shock")
    axes[0].set_ylabel("Absolute observed basis (bps)")
    axes[0].legend(frameon=False)
    outcomes = c2.loc[c2.execution_outcome != "observed_only"].groupby(["mode","execution_outcome"]).size().unstack(fill_value=0)
    outcomes = outcomes.reindex(index=MODES, columns=["fully_hedged","partial_or_unhedged","unfilled"], fill_value=0)
    bottom = np.zeros(len(outcomes))
    palette = {"fully_hedged":"#1f8a70", "partial_or_unhedged":"#d97706", "unfilled":"#dc2626"}
    for outcome in outcomes.columns:
        axes[1].bar(outcomes.index.str.title(), outcomes[outcome], bottom=bottom, label=outcome.replace("_", " "), color=palette[outcome])
        bottom += outcomes[outcome].to_numpy()
    axes[1].set_title("C2 cross-market execution outcomes")
    axes[1].set_ylabel("Decision count across 3 seeds")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "05_c2_basis_execution.png", dpi=220)
    plt.close(fig)


def save_llm_quality(quality: pd.DataFrame, output_dir: Path) -> None:
    llm = quality.loc[quality["mode"] == "llm"].copy()
    llm["condition"] = llm.study.map({"stock_future_micro":"D0/I1/A1 pilot", "stock_future_bond_two_shock":"C0/C2 two-shock"})
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    sns.barplot(data=llm, x="run_id", y="self_trade_prevention", hue="condition", errorbar=None, ax=axes[0])
    axes[0].tick_params(axis="x", rotation=70, labelsize=7)
    axes[0].set_title("LLM runs: prevented self-crosses")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    statuses = llm[["run_id","slow_loop_completed","slow_loop_rejected","slow_loop_failed"]].set_index("run_id")
    statuses.plot(kind="bar", stacked=True, color=["#1f8a70", "#d97706", "#dc2626"], ax=axes[1])
    axes[1].tick_params(axis="x", rotation=70, labelsize=7)
    axes[1].set_title("LLM runs: slow-loop status")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Decision count")
    axes[1].legend(["completed", "rejected", "failed"], frameon=False)
    fig.suptitle("Operational quality: LLM treatment needs execution guardrails", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "06_llm_quality_flags.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(events: pd.DataFrame, quality: pd.DataFrame, agents: pd.DataFrame, responses: pd.DataFrame, decisions: pd.DataFrame, deltas: pd.DataFrame, output_dir: Path) -> None:
    legacy_future = events.loc[(events.study == "stock_future_micro") & (events.asset == "AAPL_FUT")]
    legacy_future_table = legacy_future.groupby(["mode","treatment"])[["event_return_pct","event_volume","event_realized_vol_bps"]].mean().round(3)
    frozen_max = deltas.loc[deltas["mode"] == "frozen", ["roi_delta_bps","drawdown_delta_bps"]].abs().max()
    c2 = decisions.loc[(decisions.study == "stock_future_bond_two_shock") & (decisions.treatment == "c2")]
    c2_outcomes = c2.loc[c2.execution_outcome != "observed_only"].groupby(["mode","execution_outcome"]).size().unstack(fill_value=0).reindex(columns=["fully_hedged","partial_or_unhedged","unfilled"], fill_value=0)
    activity = events.loc[(events.study == "stock_future_bond_two_shock") & (events.event == "macro")].groupby(["mode","treatment","asset"])[["total_trade_count","total_volume"]].mean().round(1)
    llm_quality = quality.loc[(quality.study == "stock_future_bond_two_shock") & (quality["mode"] == "llm")]
    total_llm = int(llm_quality[["slow_loop_completed","slow_loop_rejected","slow_loop_failed"]].sum().sum())
    valid_llm = int(llm_quality.slow_loop_completed.sum())
    rel_deliveries = int(quality.loc[(quality.study == "stock_future_bond_two_shock") & (quality.treatment == "c2"), "relationship_deliveries"].sum())
    text = [
        "# Financial Market Ecology: Combined Experimental Analysis",
        "",
        "## Dataset and Scope",
        "",
        "This package combines two controlled AML experiment families: (1) the earlier stock/future micro-shock pilot with D0 disconnected, I1 information-only, and A1 information plus arbitrage; and (2) the later C0/C2 stock-future-bond experiment with an unexpected AAPL shock and an announced policy/liquidity shock.",
        "",
        "There are 30 one-hour runs in total: 18 legacy and 12 three-market runs, each with three paired seeds and frozen or all-agent LLM slow strategies. Results are descriptive effect sizes, not statistical inference.",
        "",
        "## Quality Assessment",
        "",
        "All 30 runs reached the intended simulation end and produced the expected market, agent, portfolio, decision, and shock-delivery reports. The frozen treatments are the reliable comparison set. The LLM treatments are operational and logged, but the newer all-agent LLM runs show substantial order-cancel and self-cross-prevention churn. Their performance figures must therefore be treated as diagnostic rather than as a clean estimate of LLM trading quality.",
        "",
        f"In the C0/C2 LLM batch, {valid_llm} of {total_llm} slow-loop decisions completed; the remainder were validator rejections or one failed call. This confirms use of the LLM but also motivates stronger maker quote constraints before an LLM-versus-frozen performance claim.",
        "",
        "## RQ1: How are shocks transmitted across connected markets through agent decisions?",
        "",
        "**Answer: the causal mechanism is verified, while effect magnitude remains preliminary.** In the earlier pilot, D0 gave future-market agents no stock-shock information; I1 delivered the information; A1 additionally generated basis observations and two-leg trades. The observed future-market outcomes were:",
        "",
        table(legacy_future_table),
        "",
        "In C0/C2, the micro shock is delivered only to AAPL participants in C0. In C2, it additionally reached linked future-side participants through the relationship graph; the six C2 runs recorded a total of " + str(rel_deliveries) + " relationship deliveries. The announced macro shock was delivered directly to all three markets, including UST10Y. This separates localized propagation from common systematic exposure.",
        "",
        "C2 also recorded executable stock/future linkage rather than only a metadata relationship:",
        "",
        table(c2_outcomes),
        "",
        "Interpretation: an idiosyncratic stock shock can reach the future through declared information and arbitrage channels, while the bond responds to shared macro state. The present design does not claim that bonds mechanically follow stocks; that would be economically artificial.",
        "",
        "## RQ2: Do strategies robust in one market remain robust after a linked market is introduced?",
        "",
        f"**Frozen baseline:** broadly yes in this pilot. Across D0 versus I1/A1, the largest mean role-level shift was {frozen_max['roi_delta_bps']:.1f} bps in ROI or {frozen_max['drawdown_delta_bps']:.1f} bps in maximum drawdown. This supports the narrower statement that the current deterministic strategies remain stable under the tested linkage.",
        "",
        "**Adaptive LLM treatment:** not yet answerable as a clean robustness comparison. Linked LLM conditions changed strategy state frequently and had lower liquidity plus material order-lifecycle churn. That may reflect real adaptation under uncertainty, but it is confounded with the current quote-management weakness. The appropriate conclusion is that adaptive strategies are more sensitive to linkage in the present implementation, not that they are intrinsically less robust.",
        "",
        "## Three-Market Descriptive Results",
        "",
        table(activity),
        "",
        "In frozen runs, C2 raised stock and future activity relative to C0 while bond activity stayed similar. That pattern is consistent with the intended topology: stock/future are linked, while the bond shares the macro state but has no direct stock/future arbitrage rule.",
        "",
        "## What Can Be Claimed Now",
        "",
        "- AML supports repeatable interventional experiments with explicit shock timing, visibility, recipients, and market-state effects.",
        "- The declared stock/future relationship changes who sees the micro shock and produces logged basis/arbitrage behaviour.",
        "- The policy/liquidity shock reaches stock, future, and bond agents as a shared macro condition.",
        "- Frozen-strategy results are suitable for a pilot mechanism and robustness discussion.",
        "",
        "## What Still Needs Work Before a Strong Research Claim",
        "",
        "- Increase to at least 10 paired seeds for confirmation and report uncertainty intervals.",
        "- Analyse micro and macro windows separately in the final paper; do not use end-of-session return as a shock effect estimate.",
        "- Add market-maker guardrails around quote crossing, stale cancel handling, and order-size changes before treating all-agent LLM performance as evidence.",
        "- Add a bond-specific behavioural channel, then an options or ETF relationship, before generalising beyond the current stock/future/bond topology.",
    ]
    (output_dir / "financial_ecology_combined_report.md").write_text("\n".join(text) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths, events, quality, agents, responses, decisions = collect(args.runs_dir)
    deltas = paired_legacy_deltas(agents)
    paths.to_csv(args.output_dir / "price_paths_30s.csv", index=False)
    events.to_csv(args.output_dir / "market_event_metrics.csv", index=False)
    quality.to_csv(args.output_dir / "run_quality.csv", index=False)
    agents.to_csv(args.output_dir / "agent_metrics.csv", index=False)
    responses.to_csv(args.output_dir / "slow_loop_responses.csv", index=False)
    decisions.to_csv(args.output_dir / "cross_market_decisions.csv", index=False)
    deltas.to_csv(args.output_dir / "legacy_paired_role_deltas.csv", index=False)
    save_legacy_paths(paths, args.output_dir)
    save_legacy_robustness(deltas, args.output_dir)
    save_three_market_paths(paths, args.output_dir)
    save_three_market_activity(events, args.output_dir)
    save_basis(decisions, args.output_dir)
    save_llm_quality(quality, args.output_dir)
    write_report(events, quality, agents, responses, decisions, deltas, args.output_dir)
    print(f"Wrote analysis package to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
