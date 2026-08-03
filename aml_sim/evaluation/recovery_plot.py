"""Plot role-specific financial, balance-sheet, and behavioural recovery."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any


ROLE_ORDER = (
    "market_maker",
    "retail_trader",
    "institutional_trader",
    "informed_trader",
    "liquidity_taker",
)

ROLE_LABELS = {
    "market_maker": "Market maker",
    "retail_trader": "Retail trader",
    "institutional_trader": "Institutional trader",
    "informed_trader": "Informed trader",
    "liquidity_taker": "Liquidity taker",
}

SHOCK_COLORS = ("#ef4444", "#8b5cf6", "#22c55e")


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _trajectory(agent: dict[str, Any]) -> list[tuple[datetime, float]]:
    """Combine episode trajectories, preferring the later episode at boundaries."""

    points: dict[datetime, float] = {}
    for shock in agent.get("shocks", {}).values():
        rows = shock.get("behavioural_recovery", {}).get("score_trajectory", [])
        for row in rows:
            timestamp = row.get("timestamp")
            score = row.get("score")
            if timestamp is not None and score is not None:
                points[_timestamp(timestamp)] = float(score)
    return sorted(points.items())


def _shock_windows(report: dict[str, Any]) -> list[tuple[str, datetime, datetime]]:
    windows: dict[str, tuple[datetime, datetime]] = {}
    for agent in report.get("agents", {}).values():
        for shock_id, shock in agent.get("shocks", {}).items():
            window = shock.get("window", {})
            if not window.get("onset") or not window.get("expiry"):
                continue
            onset = _timestamp(window["onset"])
            expiry = _timestamp(window["expiry"])
            previous = windows.get(shock_id)
            windows[shock_id] = (
                min(previous[0], onset) if previous else onset,
                max(previous[1], expiry) if previous else expiry,
            )
    return [
        (shock_id, onset, expiry)
        for shock_id, (onset, expiry) in sorted(windows.items(), key=lambda item: item[1][0])
    ]


def _rejected_update_times(run_dir: Path, agent_id: str) -> list[datetime]:
    log_path = run_dir / "logs" / "agents" / f"agent_{agent_id}.log"
    if not log_path.exists():
        return []

    current_tick: datetime | None = None
    rejected: list[datetime] = []
    tick_pattern = re.compile(r"Received TIME_TICK message: (\S+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        tick_match = tick_pattern.search(line)
        if tick_match:
            current_tick = _timestamp(tick_match.group(1))
        if "Rejected strategy proposal" in line and current_tick is not None:
            rejected.append(current_tick)
    return rejected


def _score_at_or_before(
    trajectory: list[tuple[datetime, float]],
    timestamp: datetime,
) -> float | None:
    eligible = [score for observed_at, score in trajectory if observed_at <= timestamp]
    return eligible[-1] if eligible else None


def _state_trajectories(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load chronologically ordered agent-state observations from a saved run."""

    trajectories: dict[str, list[dict[str, Any]]] = {}
    reports_dir = run_dir / "reports" / "agents"
    for path in sorted(reports_dir.glob("trader_actions_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        actions = payload if isinstance(payload, list) else payload.get("actions", [])
        rows = [row for row in actions if row.get("event_type") == "agent_state_tick"]
        if not rows:
            continue
        agent_id = str(rows[0].get("agent_id") or path.stem.removeprefix("trader_actions_"))
        trajectories[agent_id] = sorted(rows, key=lambda row: _timestamp(row["timestamp"]))
    return trajectories


def _ordered_agent_ids(agents: dict[str, Any]) -> list[str]:
    agent_ids = [agent_id for agent_id in ROLE_ORDER if agent_id in agents]
    agent_ids.extend(sorted(set(agents) - set(agent_ids)))
    return agent_ids


def _recovery_baseline(agent: dict[str, Any], section: str, field: str) -> float | None:
    for shock in agent.get("shocks", {}).values():
        value = shock.get(section, {}).get(field)
        if value is not None:
            return float(value)
    return None


def _save_state_plot(
    run_directory: str | Path,
    *,
    section: str,
    value_field: str,
    baseline_field: str,
    tolerance_config_field: str,
    output_stem: str,
    title: str,
    ylabel: str,
    constrained_markers: bool = False,
) -> tuple[Path, Path]:
    """Plot a state variable normalized to each role's pre-shock baseline."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    run_dir = Path(run_directory).resolve()
    report_path = run_dir / "reports" / "evaluation" / "role_recovery.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    agents = report.get("agents", {})
    agent_ids = _ordered_agent_ids(agents)
    if not agent_ids:
        raise ValueError(f"No agents found in {report_path}")
    state = _state_trajectories(run_dir)
    tolerance = float(report.get("config", {}).get(tolerance_config_field, 0.0))

    fig, axes = plt.subplots(
        len(agent_ids), 1, figsize=(13, 2.35 * len(agent_ids)),
        sharex=True, constrained_layout=True,
    )
    if len(agent_ids) == 1:
        axes = [axes]
    shock_windows = _shock_windows(report)
    for index, (axis, agent_id) in enumerate(zip(axes, agent_ids)):
        baseline = _recovery_baseline(agents[agent_id], section, baseline_field)
        rows = state.get(agent_id, [])
        points = [
            (_timestamp(row["timestamp"]), float(row[value_field]) / baseline)
            for row in rows
            if baseline not in (None, 0) and row.get(value_field) is not None
        ]
        for shock_index, (shock_id, onset, expiry) in enumerate(shock_windows):
            color = SHOCK_COLORS[shock_index % len(SHOCK_COLORS)]
            label = shock_id.replace("role_recovery_", "").replace("_", " ").title()
            axis.axvspan(onset, expiry, color=color, alpha=0.14, linewidth=0,
                        label=label if index == 0 else None)
        times = [timestamp for timestamp, _ in points]
        values = [value for _, value in points]
        axis.plot(times, values, color="#2563eb", linewidth=1.8,
                  label="Observed value" if index == 0 else None)
        axis.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
        axis.axhspan(1.0 - tolerance, 1.0 + tolerance, color="#22c55e", alpha=0.09,
                    label="Recovery band" if index == 0 else None)
        if constrained_markers:
            constrained = [
                (_timestamp(row["timestamp"]), float(row[value_field]) / baseline)
                for row in rows
                if baseline not in (None, 0) and row.get(value_field) is not None
                and any(
                    bool(asset.get(flag))
                    for asset in row.get("fast_loop_state", {}).values()
                    for flag in ("buy_constrained", "sell_constrained", "position_constrained")
                )
            ]
            if constrained:
                axis.scatter([t for t, _ in constrained], [v for _, v in constrained],
                             marker="x", s=28, color="#dc2626", zorder=4,
                             label="Position constrained" if index == 0 else None)
        axis.set_title(ROLE_LABELS.get(agent_id, agent_id.replace("_", " ").title()),
                       loc="left", fontsize=9.5, fontweight="bold")
        axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel(ylabel)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=5, frameon=False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axes[-1].set_xlabel("Simulation time")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_dir = run_dir / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    svg_path = output_dir / f"{output_stem}.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def plot_financial_recovery(run_directory: str | Path) -> tuple[Path, Path]:
    return _save_state_plot(
        run_directory, section="financial_recovery", value_field="portfolio_value",
        baseline_field="baseline_portfolio_value", tolerance_config_field="financial_tolerance",
        output_stem="financial_recovery_timeline",
        title="Financial recovery by trading role\nPortfolio value relative to pre-shock baseline",
        ylabel="Baseline ratio",
    )


def plot_balance_sheet_recovery(run_directory: str | Path) -> tuple[Path, Path]:
    """Plot threshold-normalized balance-sheet pressure for each role."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    run_dir = Path(run_directory).resolve()
    report_path = run_dir / "reports" / "evaluation" / "role_recovery.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    agents = report.get("agents", {})
    agent_ids = _ordered_agent_ids(agents)
    if not agent_ids:
        raise ValueError(f"No agents found in {report_path}")

    state = _state_trajectories(run_dir)
    config = report.get("config", {})
    exposure_tolerance = float(config.get("balance_exposure_tolerance", 0.10))
    target_tolerance = float(config.get("balance_target_tolerance", 0.10))
    utilization_limit = float(config.get("sustainable_limit_utilization", 0.80))
    shock_windows = _shock_windows(report)

    fig, axes = plt.subplots(
        len(agent_ids), 1, figsize=(13, 2.65 * len(agent_ids)),
        sharex=True, constrained_layout=True,
    )
    if len(agent_ids) == 1:
        axes = [axes]

    for role_index, (axis, agent_id) in enumerate(zip(axes, agent_ids)):
        rows = state.get(agent_id, [])
        baseline_exposure = _recovery_baseline(
            agents[agent_id], "balance_sheet_recovery", "baseline_gross_exposure"
        )
        baseline_positions = _first_balance_baseline_positions(agents[agent_id])

        exposure_pressure: list[tuple[datetime, float]] = []
        utilization_pressure: list[tuple[datetime, float]] = []
        distance_pressure: list[tuple[datetime, float]] = []
        constrained_times: list[datetime] = []
        for row in rows:
            timestamp = _timestamp(row["timestamp"])
            exposure = _optional_float(row.get("gross_exposure"))
            if baseline_exposure not in (None, 0) and exposure is not None:
                exposure_ratio = exposure / baseline_exposure
                exposure_pressure.append(
                    (timestamp, abs(exposure_ratio - 1.0) / exposure_tolerance)
                )

            utilization, distance = _balance_position_metrics(row, baseline_positions)
            if utilization is not None:
                utilization_pressure.append((timestamp, utilization / utilization_limit))
            if distance is not None:
                distance_pressure.append((timestamp, distance / target_tolerance))
            if _row_is_constrained(row):
                constrained_times.append(timestamp)

        for shock_index, (shock_id, onset, expiry) in enumerate(shock_windows):
            color = SHOCK_COLORS[shock_index % len(SHOCK_COLORS)]
            label = shock_id.replace("role_recovery_", "").replace("_", " ").title()
            axis.axvspan(
                onset, expiry, color=color, alpha=0.14, linewidth=0,
                label=label if role_index == 0 else None,
            )

        _plot_named_points(
            axis, exposure_pressure, label="Exposure deviation", color="#2563eb"
        )
        _plot_named_points(
            axis, utilization_pressure, label="Limit utilization", color="#f59e0b"
        )
        _plot_named_points(
            axis, distance_pressure, label="Position distance", color="#7c3aed"
        )

        axis.axhspan(
            0, 1.0, color="#22c55e", alpha=0.09,
            label="All-condition acceptable zone" if role_index == 0 else None,
        )
        axis.axhline(
            1.0,
            color="#dc2626",
            linewidth=1.1,
            linestyle="--",
            label="Recovery boundary" if role_index == 0 else None,
        )
        axis.set_ylim(bottom=0)
        axis.set_ylabel("Threshold-\nnormalized\npressure")
        axis.set_title(
            ROLE_LABELS.get(agent_id, agent_id.replace("_", " ").title()),
            loc="left",
            fontsize=10.5,
            fontweight="bold",
        )
        axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if not utilization_pressure:
            axis.text(
                0.995, 0.94, "Limit utilization not recorded",
                transform=axis.transAxes, ha="right", va="top",
                color="#9a6700", fontsize=8.5, style="italic",
            )

        if constrained_times:
            marker_points = []
            for timestamp in constrained_times:
                available_values = [
                    value
                    for value in (
                        _value_at_or_before(exposure_pressure, timestamp),
                        _value_at_or_before(utilization_pressure, timestamp),
                        _value_at_or_before(distance_pressure, timestamp),
                    )
                    if value is not None
                ]
                if available_values:
                    marker_points.append((timestamp, max(available_values)))
            marker_points = [(timestamp, value) for timestamp, value in marker_points if value is not None]
            if marker_points:
                axis.scatter(
                    [timestamp for timestamp, _ in marker_points],
                    [value for _, value in marker_points],
                    marker="x",
                    s=27,
                    color="#dc2626",
                    zorder=4,
                    label="Constraint flag" if role_index == 0 else None,
                )

    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.14), ncol=4, frameon=False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axes[-1].set_xlabel("Simulation time")
    fig.suptitle(
        "Balance-sheet recovery by trading role\n"
        "Each line is divided by its recovery threshold; values at or below 1.0 are acceptable",
        fontsize=14,
        fontweight="bold",
    )

    output_dir = run_dir / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "balance_sheet_recovery_timeline.png"
    svg_path = output_dir / "balance_sheet_recovery_timeline.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def _first_balance_baseline_positions(agent: dict[str, Any]) -> dict[str, float]:
    for shock in agent.get("shocks", {}).values():
        positions = shock.get("balance_sheet_recovery", {}).get("baseline_net_positions")
        if isinstance(positions, dict):
            return {
                str(instrument): float(value)
                for instrument, value in positions.items()
                if _optional_float(value) is not None
            }
    return {}


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _row_positions(row: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for instrument, details in row.get("positions", {}).items():
        if not isinstance(details, dict):
            continue
        value = _optional_float(details.get("net"))
        if value is not None:
            result[str(instrument)] = value
    return result


def _relative_distance(current: float, baseline: float) -> float:
    return abs(current - baseline) / abs(baseline) if baseline else abs(current - baseline)


def _balance_position_metrics(
    row: dict[str, Any],
    baseline_positions: dict[str, float],
) -> tuple[float | None, float | None]:
    """Return worst utilization and position distance using evaluator logic."""

    state = row.get("fast_loop_state", {})
    state = state if isinstance(state, dict) else {}
    positions = _row_positions(row)
    utilizations: list[float] = []
    distances: list[float] = []
    for instrument, position in positions.items():
        details = state.get(instrument, {})
        details = details if isinstance(details, dict) else {}
        utilization = _optional_float(details.get("position_limit_utilization"))
        effective_limit = _optional_float(details.get("effective_position_limit"))
        target_distance = _optional_float(details.get("target_distance"))
        if utilization is not None:
            utilizations.append(utilization)
        if target_distance is not None and effective_limit:
            distances.append(target_distance / abs(effective_limit))
        elif effective_limit:
            baseline = baseline_positions.get(instrument, position)
            distances.append(abs(position - baseline) / abs(effective_limit))
        else:
            baseline = baseline_positions.get(instrument, position)
            distances.append(_relative_distance(position, baseline))
    return (
        max(utilizations) if utilizations else None,
        max(distances) if distances else None,
    )


def _row_is_constrained(row: dict[str, Any]) -> bool:
    state = row.get("fast_loop_state", {})
    if not isinstance(state, dict):
        return False
    return any(
        bool(
            details.get("buy_constrained")
            or details.get("sell_constrained")
            or details.get("position_constrained")
        )
        for details in state.values()
        if isinstance(details, dict)
    )


def _plot_points(axis: Any, points: list[tuple[datetime, float]]) -> None:
    if points:
        axis.plot(
            [timestamp for timestamp, _ in points],
            [value for _, value in points],
            color="#2563eb",
            linewidth=1.8,
        )


def _plot_named_points(
    axis: Any,
    points: list[tuple[datetime, float]],
    *,
    label: str,
    color: str,
    linewidth: float = 1.8,
) -> None:
    """Plot one named recovery-pressure series when observations exist."""

    if points:
        axis.plot(
            [timestamp for timestamp, _ in points],
            [value for _, value in points],
            label=label,
            color=color,
            linewidth=linewidth,
        )


def _value_at_or_before(
    points: list[tuple[datetime, float]], timestamp: datetime
) -> float | None:
    values = [value for observed_at, value in points if observed_at <= timestamp]
    return values[-1] if values else None


def _mark_unavailable(axis: Any, message: str) -> None:
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="#6b7280",
        fontsize=9,
        style="italic",
    )


def plot_behavioural_recovery(run_directory: str | Path) -> tuple[Path, Path]:
    """Save PNG and SVG behavioural-recovery small multiples for one run."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    run_dir = Path(run_directory).resolve()
    report_path = run_dir / "reports" / "evaluation" / "role_recovery.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    agents = report.get("agents", {})
    agent_ids = _ordered_agent_ids(agents)
    if not agent_ids:
        raise ValueError(f"No agents found in {report_path}")

    fig, axes = plt.subplots(
        len(agent_ids),
        1,
        figsize=(13, 2.7 * len(agent_ids)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(agent_ids) == 1:
        axes = [axes]

    shock_windows = _shock_windows(report)
    for index, (axis, agent_id) in enumerate(zip(axes, agent_ids)):
        trajectory = _trajectory(agents[agent_id])
        times = [timestamp for timestamp, _ in trajectory]
        scores = [score for _, score in trajectory]

        for shock_index, (shock_id, onset, expiry) in enumerate(shock_windows):
            color = SHOCK_COLORS[shock_index % len(SHOCK_COLORS)]
            label = shock_id.replace("role_recovery_", "").replace("_", " ").title()
            axis.axvspan(
                onset,
                expiry,
                color=color,
                alpha=0.14,
                linewidth=0,
                label=label if index == 0 else None,
            )

        axis.plot(times, scores, color="#2563eb", linewidth=2.0)
        axis.axhline(0.8, color="#dc2626", linestyle="--", linewidth=1.1)
        axis.fill_between(times, 0.8, 1.0, color="#22c55e", alpha=0.045)

        rejected_times = _rejected_update_times(run_dir, agent_id)
        rejected_points = [
            (timestamp, _score_at_or_before(trajectory, timestamp))
            for timestamp in rejected_times
        ]
        rejected_points = [(timestamp, score) for timestamp, score in rejected_points if score is not None]
        if rejected_points:
            axis.scatter(
                [timestamp for timestamp, _ in rejected_points],
                [score for _, score in rejected_points],
                marker="x",
                s=48,
                linewidths=1.8,
                color="#111827",
                zorder=5,
                label="Rejected LLM update" if index == 0 else None,
            )

        first_shock = next(iter(agents[agent_id].get("shocks", {}).values()), {})
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
        axis.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel("Score")

    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.12), ncol=4, frameon=False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axes[-1].set_xlabel("Simulation time")
    fig.suptitle(
        "Behavioural recovery by trading role\n"
        "Scores are normalized to each role's own pre-shock baseline; 0.80 is the recovery threshold",
        fontsize=14,
        fontweight="bold",
    )

    output_dir = run_dir / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "behavioural_recovery_timeline.png"
    svg_path = output_dir / "behavioural_recovery_timeline.svg"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    args = parser.parse_args()
    for plotter in (
        plot_financial_recovery,
        plot_balance_sheet_recovery,
        plot_behavioural_recovery,
    ):
        for output in plotter(args.run_directory):
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
