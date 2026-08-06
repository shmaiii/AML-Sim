"""Mean recovery timelines across completed experiment replications."""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from aml_sim.evaluation.recovery_plot import (
    ROLE_LABELS,
    SHOCK_COLORS,
    _balance_position_metrics,
    _first_balance_baseline_positions,
    _ordered_agent_ids,
    _recovery_baseline,
    _shock_windows,
    _state_trajectories,
    _timestamp,
    _trajectory,
)


Series = list[tuple[datetime, float]]


def _load_runs(
    run_directories: list[str | Path],
) -> list[tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]]:
    runs = []
    for run_directory in run_directories:
        run_dir = Path(run_directory).resolve()
        report_path = run_dir / "reports" / "evaluation" / "role_recovery.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        runs.append((run_dir, report, _state_trajectories(run_dir)))
    if not runs:
        raise ValueError("No completed recovery reports were provided")
    return runs


def _mean_band(series: list[Series]) -> tuple[list[datetime], list[float], list[float], list[float]]:
    values_by_time: dict[datetime, list[float]] = {}
    for points in series:
        for timestamp, value in points:
            values_by_time.setdefault(timestamp, []).append(value)
    times = sorted(values_by_time)
    values = [values_by_time[timestamp] for timestamp in times]
    return (
        times,
        [mean(items) for items in values],
        [min(items) for items in values],
        [max(items) for items in values],
    )


def _plot_mean_series(
    axis: Any,
    series: list[Series],
    *,
    label: str,
    color: str,
) -> None:
    times, averages, lower, upper = _mean_band(series)
    if not times:
        return
    axis.fill_between(times, lower, upper, color=color, alpha=0.13, linewidth=0)
    axis.plot(times, averages, color=color, linewidth=2.0, label=label)


def _shade_shocks(axis: Any, windows: list[tuple[str, datetime, datetime]], index: int) -> None:
    for shock_index, (shock_id, onset, expiry) in enumerate(windows):
        label = shock_id.replace("role_recovery_", "").replace("_", " ").title()
        axis.axvspan(
            onset,
            expiry,
            color=SHOCK_COLORS[shock_index % len(SHOCK_COLORS)],
            alpha=0.14,
            linewidth=0,
            label=label if index == 0 else None,
        )


def _finish_figure(
    fig: Any,
    axes: list[Any],
    *,
    title: str,
    output_dir: Path,
    output_stem: str,
    legend_columns: int,
) -> tuple[Path, Path]:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.12),
        ncol=legend_columns,
        frameon=False,
    )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axes[-1].set_xlabel("Simulation time")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def _role_axes(agent_ids: list[str], *, height: float, sharey: bool = False) -> tuple[Any, list[Any]]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(agent_ids),
        1,
        figsize=(13, height * len(agent_ids)),
        sharex=True,
        sharey=sharey,
        constrained_layout=True,
    )
    return fig, list(axes) if len(agent_ids) > 1 else [axes]


def _style_role_axis(axis: Any, agent_id: str, *, ylabel: str) -> None:
    axis.set_title(
        ROLE_LABELS.get(agent_id, agent_id.replace("_", " ").title()),
        loc="left",
        fontsize=9.5,
        fontweight="bold",
    )
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_mean_financial_recovery(
    runs: list[tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]],
    output_dir: Path,
) -> tuple[Path, Path]:
    first_report = runs[0][1]
    agent_ids = _ordered_agent_ids(first_report.get("agents", {}))
    fig, axes = _role_axes(agent_ids, height=2.35, sharey=True)
    windows = _shock_windows(first_report)
    for index, (axis, agent_id) in enumerate(zip(axes, agent_ids)):
        series: list[Series] = []
        tolerances: list[float] = []
        for _, report, state in runs:
            agent = report.get("agents", {}).get(agent_id)
            if not agent:
                continue
            baseline = _recovery_baseline(
                agent, "financial_recovery", "baseline_portfolio_value"
            )
            if baseline in (None, 0):
                continue
            series.append(
                [
                    (_timestamp(row["timestamp"]), float(row["portfolio_value"]) / baseline)
                    for row in state.get(agent_id, [])
                    if row.get("portfolio_value") is not None
                ]
            )
            tolerances.append(float(report.get("config", {}).get("financial_tolerance", 0.05)))
        _shade_shocks(axis, windows, index)
        _plot_mean_series(axis, series, label="Mean across replications", color="#2563eb")
        tolerance = mean(tolerances) if tolerances else 0.05
        axis.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
        axis.axhspan(
            1.0 - tolerance,
            1.0 + tolerance,
            color="#22c55e",
            alpha=0.09,
            label="Recovery band" if index == 0 else None,
        )
        _style_role_axis(axis, agent_id, ylabel="Baseline ratio")
    return _finish_figure(
        fig,
        axes,
        title=(
            f"Mean financial recovery across {len(runs)} replications\n"
            "Line is the timestamp mean; shaded line band is the replication range"
        ),
        output_dir=output_dir,
        output_stem="mean_financial_recovery_timeline",
        legend_columns=5,
    )


def plot_mean_balance_sheet_recovery(
    runs: list[tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]],
    output_dir: Path,
) -> tuple[Path, Path]:
    first_report = runs[0][1]
    agent_ids = _ordered_agent_ids(first_report.get("agents", {}))
    fig, axes = _role_axes(agent_ids, height=2.65)
    windows = _shock_windows(first_report)
    for index, (axis, agent_id) in enumerate(zip(axes, agent_ids)):
        exposure_runs: list[Series] = []
        utilization_runs: list[Series] = []
        distance_runs: list[Series] = []
        for _, report, state in runs:
            agent = report.get("agents", {}).get(agent_id)
            if not agent:
                continue
            config = report.get("config", {})
            exposure_tolerance = float(config.get("balance_exposure_tolerance", 0.10))
            target_tolerance = float(config.get("balance_target_tolerance", 0.10))
            utilization_limit = float(config.get("sustainable_limit_utilization", 0.80))
            baseline_exposure = _recovery_baseline(
                agent, "balance_sheet_recovery", "baseline_gross_exposure"
            )
            baseline_positions = _first_balance_baseline_positions(agent)
            exposure: Series = []
            utilization: Series = []
            distance: Series = []
            for row in state.get(agent_id, []):
                timestamp = _timestamp(row["timestamp"])
                current_exposure = row.get("gross_exposure")
                if baseline_exposure not in (None, 0) and current_exposure is not None:
                    ratio = float(current_exposure) / baseline_exposure
                    exposure.append((timestamp, abs(ratio - 1.0) / exposure_tolerance))
                position_utilization, position_distance = _balance_position_metrics(
                    row, baseline_positions
                )
                if position_utilization is not None:
                    utilization.append((timestamp, position_utilization / utilization_limit))
                if position_distance is not None:
                    distance.append((timestamp, position_distance / target_tolerance))
            exposure_runs.append(exposure)
            utilization_runs.append(utilization)
            distance_runs.append(distance)

        _shade_shocks(axis, windows, index)
        _plot_mean_series(axis, exposure_runs, label="Mean exposure deviation", color="#2563eb")
        _plot_mean_series(axis, utilization_runs, label="Mean limit utilization", color="#f59e0b")
        _plot_mean_series(axis, distance_runs, label="Mean position distance", color="#7c3aed")
        axis.axhspan(
            0,
            1.0,
            color="#22c55e",
            alpha=0.09,
            label="Within recovery tolerance" if index == 0 else None,
        )
        axis.axhline(
            1.0,
            color="#dc2626",
            linewidth=1.1,
            linestyle="--",
            label="Recovery boundary" if index == 0 else None,
        )
        axis.set_ylim(bottom=0)
        _style_role_axis(axis, agent_id, ylabel="Threshold-normalized\npressure")
        if not any(utilization_runs):
            axis.text(
                0.995,
                0.94,
                "Limit utilization not recorded",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color="#9a6700",
                fontsize=8.5,
                style="italic",
            )
    return _finish_figure(
        fig,
        axes,
        title=(
            f"Mean balance-sheet recovery across {len(runs)} replications\n"
            "Lines are timestamp means; shaded line bands are replication ranges"
        ),
        output_dir=output_dir,
        output_stem="mean_balance_sheet_recovery_timeline",
        legend_columns=4,
    )


def plot_mean_behavioural_recovery(
    runs: list[tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]],
    output_dir: Path,
) -> tuple[Path, Path]:
    first_report = runs[0][1]
    first_agents = first_report.get("agents", {})
    agent_ids = _ordered_agent_ids(first_agents)
    fig, axes = _role_axes(agent_ids, height=2.7, sharey=True)
    windows = _shock_windows(first_report)
    for index, (axis, agent_id) in enumerate(zip(axes, agent_ids)):
        series = [
            _trajectory(report["agents"][agent_id])
            for _, report, _ in runs
            if agent_id in report.get("agents", {})
        ]
        _shade_shocks(axis, windows, index)
        _plot_mean_series(axis, series, label="Mean behavioural score", color="#2563eb")
        axis.axhline(
            0.8,
            color="#dc2626",
            linestyle="--",
            linewidth=1.1,
            label="Recovery threshold" if index == 0 else None,
        )
        axis.axhspan(0.8, 1.0, color="#22c55e", alpha=0.045)
        first_shock = next(iter(first_agents[agent_id].get("shocks", {}).values()), {})
        component_names = list(
            first_shock.get("behavioural_recovery", {}).get("baseline", {})
        )
        component_label = ", ".join(name.replace("mean_", "") for name in component_names)
        axis.set_title(
            f"{ROLE_LABELS.get(agent_id, agent_id.replace('_', ' ').title())}\n"
            f"Components: {textwrap.fill(component_label, width=115)}",
            loc="left",
            fontsize=9.5,
            fontweight="bold",
        )
        axis.set_ylim(0, 1.03)
        axis.set_yticks((0, 0.4, 0.8, 1.0))
        axis.set_ylabel("Score")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    return _finish_figure(
        fig,
        axes,
        title=(
            f"Mean behavioural recovery across {len(runs)} replications\n"
            "Line is the timestamp mean; shaded line band is the replication range"
        ),
        output_dir=output_dir,
        output_stem="mean_behavioural_recovery_timeline",
        legend_columns=5,
    )


def plot_experiment_mean_recovery(
    run_directories: list[str | Path],
    output_directory: str | Path,
) -> list[Path]:
    """Generate three timestamp-aligned mean plots for an experiment."""

    runs = _load_runs(run_directories)
    output_dir = Path(output_directory).resolve()
    outputs: list[Path] = []
    for plotter in (
        plot_mean_financial_recovery,
        plot_mean_balance_sheet_recovery,
        plot_mean_behavioural_recovery,
    ):
        outputs.extend(plotter(runs, output_dir))
    return outputs
