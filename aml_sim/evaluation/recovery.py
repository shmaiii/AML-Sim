"""Post-shock recovery evaluation from completed AML-Sim run artifacts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RecoveryConfig:
    """Thresholds used by the three recovery evaluations."""

    baseline_seconds: int = 300
    financial_tolerance: float = 0.05
    balance_target_tolerance: float = 0.10
    balance_exposure_tolerance: float = 0.10
    sustainable_limit_utilization: float = 0.80
    behavioural_relative_tolerance: float = 0.10
    behavioural_share_tolerance: float = 0.10
    behavioural_recovery_score: float = 0.80
    sustained_observations: int = 3
    behavioural_window_ticks: int = 3
    horizons_seconds: tuple[int, ...] = (30, 60, 120)


class RecoveryEvaluator:
    """Calculate financial, balance-sheet, and behavioural recovery."""

    def __init__(self, config: RecoveryConfig | None = None) -> None:
        self.config = config or RecoveryConfig()

    def evaluate_run(self, run_dir: str | Path) -> dict[str, Any]:
        """Read one completed run and evaluate every agent/shock pair."""

        run_path = Path(run_dir)
        action_files = sorted(
            (run_path / "reports" / "agents").glob(
                "trader_actions_*.json"
            )
        )
        agents: dict[str, Any] = {}
        for action_file in action_files:
            actions = _read_action_file(action_file)
            state_ticks = sorted(
                (
                    action
                    for action in actions
                    if action.get("event_type") == "agent_state_tick"
                ),
                key=_timestamp_sort_key,
            )
            if not state_ticks:
                continue
            agent_id = str(
                state_ticks[0].get("agent_id")
                or action_file.stem.removeprefix("trader_actions_")
            )
            event_metadata = _event_metadata(actions)
            episodes = _shock_episodes(state_ticks)
            global_baseline = (
                _baseline_rows(
                    state_ticks,
                    onset=_parse_timestamp(episodes[0]["onset"]),
                    seconds=self.config.baseline_seconds,
                )
                if episodes
                else []
            )
            shocks: dict[str, Any] = {}
            for index, episode in enumerate(episodes):
                shock_ids = list(episode["shock_ids"])
                episode_key = _episode_key(shock_ids)
                next_onset = (
                    episodes[index + 1]["onset"]
                    if index + 1 < len(episodes)
                    else None
                )
                shocks[episode_key] = self.evaluate_agent_shock(
                    agent_id=agent_id,
                    state_ticks=state_ticks,
                    shock_id=episode_key,
                    shock_ids=shock_ids,
                    shock_window=episode,
                    shock_metadata={
                        shock_id: event_metadata.get(shock_id, {})
                        for shock_id in shock_ids
                    },
                    baseline_rows=global_baseline,
                    next_episode_onset=next_onset,
                )
            agents[agent_id] = {
                "agent_id": agent_id,
                "role": state_ticks[0].get("role"),
                "shock_count": sum(
                    len(episode["shock_ids"])
                    for episode in episodes
                ),
                "shock_episode_count": len(shocks),
                "shocks": shocks,
            }

        return {
            "run_directory": str(run_path.resolve()),
            "config": {
                key: value
                for key, value in vars(self.config).items()
            },
            "agents": agents,
        }

    def save_report(
        self,
        run_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Evaluate a run and save the research report without rerunning it."""

        run_path = Path(run_dir)
        destination = (
            Path(output_path)
            if output_path is not None
            else run_path
            / "reports"
            / "evaluation"
            / "role_recovery.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.evaluate_run(run_path), indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    def evaluate_agent_shock(
        self,
        *,
        agent_id: str,
        state_ticks: Sequence[Mapping[str, Any]],
        shock_id: str,
        shock_window: Mapping[str, Any],
        shock_metadata: Mapping[str, Any],
        shock_ids: Sequence[str] | None = None,
        baseline_rows: Sequence[Mapping[str, Any]] | None = None,
        next_episode_onset: Any = None,
    ) -> dict[str, Any]:
        """Evaluate one agent relative to one observed shock episode."""

        onset = _parse_timestamp(shock_window["onset"])
        expiry = _parse_timestamp(shock_window["expiry"])
        baseline = (
            list(baseline_rows)
            if baseline_rows is not None
            else _baseline_rows(
                state_ticks,
                onset=onset,
                seconds=self.config.baseline_seconds,
            )
        )
        evaluation_end = (
            _parse_timestamp(next_episode_onset)
            if next_episode_onset is not None
            else None
        )
        post_shock = [
            row
            for row in state_ticks
            if _row_time(row) >= onset
            and (
                evaluation_end is None
                or _row_time(row) < evaluation_end
            )
        ]
        interrupted = evaluation_end is not None
        episode_shock_ids = list(shock_ids or [shock_id])
        episode_metadata = dict(shock_metadata)
        return {
            "agent_id": agent_id,
            "role": post_shock[0].get("role") if post_shock else None,
            "shock_id": shock_id,
            "shock_ids": episode_shock_ids,
            "shock": (
                dict(episode_metadata.get(episode_shock_ids[0], {}))
                if len(episode_shock_ids) == 1
                else {}
            ),
            "shock_metadata": episode_metadata,
            "window": {
                **dict(shock_window),
                "next_episode_onset": (
                    evaluation_end.isoformat()
                    if evaluation_end is not None
                    else None
                ),
                "evaluation_end": (
                    evaluation_end.isoformat()
                    if evaluation_end is not None
                    else (
                        _row_time(post_shock[-1]).isoformat()
                        if post_shock
                        else None
                    )
                ),
                "baseline_reference": "before_first_observed_shock",
                "baseline_start": (
                    _row_time(baseline[0]).isoformat()
                    if baseline
                    else None
                ),
                "baseline_observations": len(baseline),
            },
            "financial_recovery": self._financial_recovery(
                baseline,
                post_shock,
                onset=onset,
                expiry=expiry,
                interrupted=interrupted,
            ),
            "balance_sheet_recovery": self._balance_sheet_recovery(
                baseline,
                post_shock,
                onset=onset,
                expiry=expiry,
                interrupted=interrupted,
            ),
            "behavioural_recovery": self._behavioural_recovery(
                baseline,
                post_shock,
                onset=onset,
                expiry=expiry,
                interrupted=interrupted,
            ),
        }

    def _financial_recovery(
        self,
        baseline_rows: Sequence[Mapping[str, Any]],
        post_rows: Sequence[Mapping[str, Any]],
        *,
        onset: datetime,
        expiry: datetime,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        values = _numbers(
            row.get("portfolio_value") for row in baseline_rows
        )
        if not values:
            return _unavailable("no pre-shock portfolio observations")
        baseline = mean(values)
        tolerance = abs(baseline) * self.config.financial_tolerance
        condition = [
            (
                row,
                abs(_number(row.get("portfolio_value"), baseline) - baseline)
                <= tolerance,
            )
            for row in post_rows
        ]
        recovery = _recovery_timing(
            condition,
            onset=onset,
            expiry=expiry,
            sustained=self.config.sustained_observations,
            interrupted=interrupted,
        )
        post_values = _numbers(
            row.get("portfolio_value") for row in post_rows
        )
        minimum = min(post_values) if post_values else None
        maximum_drawdown = (
            max(0.0, (baseline - minimum) / abs(baseline))
            if minimum is not None and baseline != 0
            else None
        )
        return {
            "available": True,
            "baseline_portfolio_value": baseline,
            "recovery_band": {
                "lower": baseline - tolerance,
                "upper": baseline + tolerance,
                "tolerance_fraction": self.config.financial_tolerance,
            },
            "maximum_drawdown_from_baseline": maximum_drawdown,
            "end_portfolio_value": (
                _optional_number(post_rows[-1].get("portfolio_value"))
                if post_rows
                else None
            ),
            "end_distance_from_baseline": (
                _relative_distance(
                    post_rows[-1].get("portfolio_value"),
                    baseline,
                )
                if post_rows
                else None
            ),
            "horizons": _horizon_values(
                post_rows,
                expiry,
                self.config.horizons_seconds,
                ("portfolio_value", "total_pnl"),
            ),
            **recovery,
        }

    def _balance_sheet_recovery(
        self,
        baseline_rows: Sequence[Mapping[str, Any]],
        post_rows: Sequence[Mapping[str, Any]],
        *,
        onset: datetime,
        expiry: datetime,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        if not baseline_rows:
            return _unavailable("no pre-shock balance-sheet observations")
        baseline_exposure_values = _numbers(
            row.get("gross_exposure") for row in baseline_rows
        )
        baseline_exposure = (
            mean(baseline_exposure_values)
            if baseline_exposure_values
            else None
        )
        baseline_cash_values = _numbers(
            row.get("cash") for row in baseline_rows
        )
        baseline_cash = (
            mean(baseline_cash_values) if baseline_cash_values else None
        )
        baseline_positions = _mean_positions(baseline_rows)
        status = [
            (
                row,
                _balance_condition(
                    row,
                    baseline_exposure=baseline_exposure,
                    baseline_positions=baseline_positions,
                    config=self.config,
                ),
            )
            for row in post_rows
        ]
        recovery = _recovery_timing(
            status,
            onset=onset,
            expiry=expiry,
            sustained=self.config.sustained_observations,
            interrupted=interrupted,
        )
        constrained_seconds = _duration_matching(
            post_rows,
            _is_constrained,
        )
        utilizations = [
            value
            for row in post_rows
            for value in _role_state_numbers(
                row,
                "position_limit_utilization",
            )
        ]
        return {
            "available": True,
            "baseline_gross_exposure": baseline_exposure,
            "baseline_cash": baseline_cash,
            "baseline_net_positions": baseline_positions,
            "maximum_position_limit_utilization": (
                max(utilizations) if utilizations else None
            ),
            "time_constrained_seconds": constrained_seconds,
            "end_gross_exposure": (
                _optional_number(post_rows[-1].get("gross_exposure"))
                if post_rows
                else None
            ),
            "end_net_positions": (
                _net_positions(post_rows[-1]) if post_rows else {}
            ),
            "end_role_state": (
                _select_fast_loop_state(
                    post_rows[-1],
                    (
                        "effective_position_limit",
                        "position_limit_utilization",
                        "target_distance",
                        "buy_constrained",
                        "sell_constrained",
                        "position_constrained",
                    ),
                )
                if post_rows
                else {}
            ),
            "end_cash_utilization": (
                _cash_utilization(
                    post_rows[-1].get("cash"),
                    baseline_cash,
                )
                if post_rows
                else None
            ),
            "horizons": _balance_horizons(
                post_rows,
                expiry,
                self.config.horizons_seconds,
                baseline_cash,
            ),
            **recovery,
        }

    def _behavioural_recovery(
        self,
        baseline_rows: Sequence[Mapping[str, Any]],
        post_rows: Sequence[Mapping[str, Any]],
        *,
        onset: datetime,
        expiry: datetime,
        interrupted: bool = False,
    ) -> dict[str, Any]:
        baseline_fast = [
            row for row in baseline_rows if row.get("fast_loop_executed")
        ]
        post_fast = [
            row for row in post_rows if row.get("fast_loop_executed")
        ]
        if not baseline_fast:
            return _unavailable("no pre-shock fast-loop observations")
        role = str(baseline_fast[0].get("role") or "unknown")
        profile = _behaviour_profile(baseline_fast, role)
        windows: list[tuple[Mapping[str, Any], bool]] = []
        scored_windows: list[dict[str, Any]] = []
        width = max(1, self.config.behavioural_window_ticks)
        for index, row in enumerate(post_fast):
            recent = post_fast[max(0, index - width + 1) : index + 1]
            current = _behaviour_profile(recent, role)
            score = _behaviour_score(
                current,
                profile,
                role,
                self.config,
            )
            scored_windows.append(
                {
                    "timestamp": _row_time(row).isoformat(),
                    **score,
                }
            )
            windows.append(
                (
                    row,
                    score["score"]
                    >= self.config.behavioural_recovery_score,
                )
            )
        recovery = _recovery_timing(
            windows,
            onset=onset,
            expiry=expiry,
            sustained=self.config.sustained_observations,
            interrupted=interrupted,
        )
        end_profile = (
            _behaviour_profile(post_fast[-width:], role) if post_fast else {}
        )
        end_score = (
            _behaviour_score(
                end_profile,
                profile,
                role,
                self.config,
            )
            if end_profile
            else _empty_behaviour_score(role)
        )
        return {
            "available": True,
            "role_profile": role,
            "scoring_method": "equal_weight_mean_of_available_components",
            "recovery_score_threshold": (
                self.config.behavioural_recovery_score
            ),
            "baseline": profile,
            "end": end_profile,
            "end_recovery_score": end_score["score"],
            "end_recovery_classification": end_score["classification"],
            "end_components": end_score["components"],
            "remaining_impairments": end_score["remaining_impairments"],
            "fast_loop_observations": len(post_fast),
            "score_trajectory": scored_windows,
            "horizons": _behaviour_horizons(
                post_fast,
                expiry,
                self.config.horizons_seconds,
            ),
            **recovery,
        }


def _recovery_timing(
    rows: Sequence[tuple[Mapping[str, Any], bool]],
    *,
    onset: datetime,
    expiry: datetime,
    sustained: int,
    interrupted: bool = False,
) -> dict[str, Any]:
    """Find the first sustained return after episode expiry."""

    breached = False
    run = 0
    for index, (row, inside) in enumerate(rows):
        timestamp = _row_time(row)
        if not breached:
            if not inside:
                breached = True
                run = 0
            if not breached or timestamp < expiry:
                continue
        if timestamp < expiry:
            continue
        run = run + 1 if inside else 0
        if run >= max(1, sustained):
            first_index = index - run + 1
            recovered_at = _row_time(rows[first_index][0])
            return {
                "disrupted": True,
                "recovered": True,
                "recovery_status": "recovered",
                "recovery_time_from_onset_seconds": (
                    recovered_at - onset
                ).total_seconds(),
                "recovery_time_from_expiry_seconds": (
                    recovered_at - expiry
                ).total_seconds(),
                "recovery_timestamp": recovered_at.isoformat(),
            }
    if not breached:
        return {
            "disrupted": False,
            "recovered": True,
            "recovery_status": "not_disrupted",
            "recovery_time_from_onset_seconds": 0.0,
            "recovery_time_from_expiry_seconds": None,
            "recovery_timestamp": onset.isoformat(),
        }
    return {
        "disrupted": True,
        "recovered": False,
        "recovery_status": (
            "interrupted_by_next_shock"
            if interrupted
            else "not_recovered_by_run_end"
        ),
        "recovery_time_from_onset_seconds": None,
        "recovery_time_from_expiry_seconds": None,
        "recovery_timestamp": None,
    }


def _balance_condition(
    row: Mapping[str, Any],
    *,
    baseline_exposure: float | None,
    baseline_positions: Mapping[str, float],
    config: RecoveryConfig,
) -> bool:
    exposure_ok = (
        True
        if baseline_exposure is None
        else _relative_distance(
            row.get("gross_exposure"),
            baseline_exposure,
        )
        <= config.balance_exposure_tolerance
    )
    state = row.get("fast_loop_state")
    state = state if isinstance(state, Mapping) else {}
    positions = _net_positions(row)
    instrument_checks: list[bool] = []
    for instrument, position in positions.items():
        details = state.get(instrument)
        details = details if isinstance(details, Mapping) else {}
        utilization = _optional_number(
            details.get("position_limit_utilization")
        )
        utilization_ok = (
            utilization is None
            or utilization <= config.sustainable_limit_utilization
        )
        effective_limit = _optional_number(
            details.get("effective_position_limit")
        )
        target_distance = _optional_number(details.get("target_distance"))
        if target_distance is not None and effective_limit:
            displacement = target_distance / abs(effective_limit)
        elif effective_limit:
            displacement = abs(
                position - baseline_positions.get(instrument, position)
            ) / abs(effective_limit)
        else:
            baseline_position = baseline_positions.get(instrument, position)
            displacement = _relative_distance(position, baseline_position)
        instrument_checks.append(
            utilization_ok
            and displacement <= config.balance_target_tolerance
        )
    return exposure_ok and all(instrument_checks or [True])


_COMMON_BEHAVIOUR_COMPONENTS = (
    "risk_mode",
    "participation_rate",
    "mean_order_size",
    "buy_share",
    "market_order_share",
)

_ROLE_BEHAVIOUR_COMPONENTS = {
    "market_maker": (
        "risk_mode",
        "participation_rate",
        "mean_quote_size",
        "mean_spread",
        "mean_active_quote_count",
    ),
    "retail": (
        *_COMMON_BEHAVIOUR_COMPONENTS,
        "mean_effective_participation_probability",
        "mean_effective_buy_probability",
        "mean_effective_order_size_cap",
    ),
    "institutional": (
        *_COMMON_BEHAVIOUR_COMPONENTS,
        "mean_effective_child_order_size",
        "mean_effective_entry_threshold",
        "mean_effective_exit_threshold",
    ),
    "informed": (
        *_COMMON_BEHAVIOUR_COMPONENTS,
        "mean_effective_signal_threshold",
        "signal_action_consistency",
    ),
    "liquidity_taker": (
        *_COMMON_BEHAVIOUR_COMPONENTS,
        "mean_effective_participation_probability",
        "mean_effective_buy_probability",
        "mean_effective_order_size_cap",
        "mean_aggression",
    ),
}

_SHARE_COMPONENTS = {
    "participation_rate",
    "buy_share",
    "market_order_share",
    "mean_effective_participation_probability",
    "mean_effective_buy_probability",
    "signal_action_consistency",
}


def _behaviour_profile(
    rows: Sequence[Mapping[str, Any]],
    role: str,
) -> dict[str, Any]:
    modes = [
        str(row.get("risk_mode"))
        for row in rows
        if row.get("risk_mode")
    ]
    quantities = [
        float(quantity)
        for row in rows
        for quantity in row.get("order_quantities", [])
        if _optional_number(quantity) is not None
    ]
    sides = [
        str(side).upper()
        for row in rows
        for side in row.get("order_sides", [])
    ]
    order_types = [
        str(order_type).upper()
        for row in rows
        for order_type in row.get("order_types", [])
    ]
    quote_sizes = [
        value
        for row in rows
        for value in _role_state_numbers(row, "effective_quote_size")
    ]
    spreads = [
        value
        for row in rows
        for value in _role_state_numbers(row, "effective_spread")
    ]
    profile = {
        "risk_mode": Counter(modes).most_common(1)[0][0] if modes else None,
        "participation_rate": (
            sum(bool(row.get("fast_loop_participated")) for row in rows)
            / len(rows)
            if rows
            else None
        ),
        "mean_order_size": mean(quantities) if quantities else None,
        "buy_share": (
            sum(side == "BUY" for side in sides) / len(sides)
            if sides
            else None
        ),
        "market_order_share": (
            sum(order_type == "MARKET" for order_type in order_types)
            / len(order_types)
            if order_types
            else None
        ),
        "mean_quote_size": mean(quote_sizes) if quote_sizes else None,
        "mean_spread": mean(spreads) if spreads else None,
        "mean_active_quote_count": _mean_role_state(
            rows,
            "active_quote_order_count",
        ),
        "mean_effective_participation_probability": _mean_role_state(
            rows,
            "effective_participation_probability",
        ),
        "mean_effective_buy_probability": _mean_role_state(
            rows,
            "effective_buy_probability",
        ),
        "mean_effective_order_size_cap": _mean_role_state(
            rows,
            "effective_order_size_cap",
        ),
        "mean_effective_child_order_size": _mean_role_state(
            rows,
            "effective_child_order_size",
        ),
        "mean_effective_entry_threshold": _mean_role_state(
            rows,
            "effective_entry_threshold",
        ),
        "mean_effective_exit_threshold": _mean_role_state(
            rows,
            "effective_exit_threshold",
        ),
        "mean_effective_signal_threshold": _mean_role_state(
            rows,
            "effective_signal_threshold",
        ),
        "mean_aggression": _mean_role_state(rows, "aggression"),
        "signal_action_consistency": _signal_action_consistency(rows),
    }
    components = _ROLE_BEHAVIOUR_COMPONENTS.get(
        role,
        _COMMON_BEHAVIOUR_COMPONENTS,
    )
    return {
        key: profile.get(key)
        for key in components
    }


def _behaviour_score(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    role: str,
    config: RecoveryConfig,
) -> dict[str, Any]:
    component_results: dict[str, Any] = {}
    for key in _ROLE_BEHAVIOUR_COMPONENTS.get(
        role,
        _COMMON_BEHAVIOUR_COMPONENTS,
    ):
        baseline_value = baseline.get(key)
        current_value = current.get(key)
        if baseline_value is None:
            continue
        if key == "risk_mode":
            matches = current_value == baseline_value
            distance = 0.0 if matches else 1.0
            component_score = 1.0 if matches else 0.0
            tolerance = 0.0
        else:
            current_number = _optional_number(current_value)
            baseline_number = _optional_number(baseline_value)
            absolute = key in _SHARE_COMPONENTS
            tolerance = (
                config.behavioural_share_tolerance
                if absolute
                else config.behavioural_relative_tolerance
            )
            if current_number is None or baseline_number is None:
                distance = None
                component_score = 0.0
                matches = False
            else:
                distance = (
                    abs(current_number - baseline_number)
                    if absolute
                    else _relative_distance(
                        current_number,
                        baseline_number,
                    )
                )
                component_score = 1.0 / (1.0 + distance)
                matches = distance <= tolerance
        component_results[key] = {
            "baseline": baseline_value,
            "current": current_value,
            "distance": distance,
            "score": component_score,
            "tolerance": tolerance,
            "within_tolerance": matches,
            "weight": 1.0,
        }

    scores = [
        component["score"]
        for component in component_results.values()
    ]
    aggregate = mean(scores) if scores else 0.0
    return {
        "role": role,
        "score": aggregate,
        "classification": _recovery_classification(aggregate),
        "components": component_results,
        "remaining_impairments": [
            key
            for key, component in component_results.items()
            if not component["within_tolerance"]
        ],
    }


def _empty_behaviour_score(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "score": 0.0,
        "classification": "not_recovered",
        "components": {},
        "remaining_impairments": [],
    }


def _recovery_classification(score: float) -> str:
    if score >= 0.90:
        return "fully_recovered"
    if score >= 0.75:
        return "mostly_recovered"
    if score >= 0.50:
        return "partially_recovered"
    return "not_recovered"


def _mean_role_state(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> float | None:
    values = [
        value
        for row in rows
        for value in _role_state_numbers(row, key)
    ]
    return mean(values) if values else None


def _signal_action_consistency(
    rows: Sequence[Mapping[str, Any]],
) -> float | None:
    qualifying = 0
    consistent = 0
    for row in rows:
        state = row.get("fast_loop_state")
        if not isinstance(state, Mapping):
            continue
        expected_sides: list[str] = []
        for details in state.values():
            if not isinstance(details, Mapping):
                continue
            signal = _optional_number(details.get("signal_strength"))
            threshold = _optional_number(
                details.get("effective_signal_threshold")
            )
            if (
                signal is None
                or threshold is None
                or abs(signal) < threshold
            ):
                continue
            expected_sides.append("BUY" if signal > 0 else "SELL")
        if not expected_sides:
            continue
        qualifying += len(expected_sides)
        submitted_sides = {
            str(side).upper()
            for side in row.get("order_sides", [])
        }
        consistent += sum(
            expected_side in submitted_sides
            for expected_side in expected_sides
        )
    return consistent / qualifying if qualifying else None


def _optional_metric_matches(
    current: Any,
    baseline: Any,
    tolerance: float,
    *,
    absolute: bool = False,
) -> bool:
    baseline_number = _optional_number(baseline)
    current_number = _optional_number(current)
    if baseline_number is None:
        return True
    if current_number is None:
        return False
    if absolute:
        return abs(current_number - baseline_number) <= tolerance
    return _relative_distance(current_number, baseline_number) <= tolerance


def _shock_window(
    rows: Sequence[Mapping[str, Any]],
    shock_id: str,
) -> dict[str, Any] | None:
    active_indexes = [
        index
        for index, row in enumerate(rows)
        if shock_id in _shock_ids(row)
    ]
    if not active_indexes:
        return None
    first = active_indexes[0]
    last = active_indexes[-1]
    onset = _row_time(rows[first])
    if last + 1 < len(rows):
        expiry = _row_time(rows[last + 1])
        inferred_end = False
    else:
        expiry = _row_time(rows[last])
        inferred_end = True
    return {
        "onset": onset.isoformat(),
        "expiry": expiry.isoformat(),
        "expiry_is_run_end": inferred_end,
    }


def _shock_episodes(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge overlapping observed shock intervals into ordered episodes."""

    intervals: list[dict[str, Any]] = []
    for shock_id in _observed_shock_ids(rows):
        window = _shock_window(rows, shock_id)
        if window is None:
            continue
        intervals.append(
            {
                "shock_ids": [shock_id],
                "onset": _parse_timestamp(window["onset"]),
                "expiry": _parse_timestamp(window["expiry"]),
                "expiry_is_run_end": window["expiry_is_run_end"],
            }
        )
    intervals.sort(key=lambda interval: interval["onset"])

    episodes: list[dict[str, Any]] = []
    for interval in intervals:
        if (
            episodes
            and interval["onset"] <= episodes[-1]["expiry"]
        ):
            episode = episodes[-1]
            episode["shock_ids"].extend(interval["shock_ids"])
            if interval["expiry"] > episode["expiry"]:
                episode["expiry"] = interval["expiry"]
            episode["expiry_is_run_end"] = (
                episode["expiry_is_run_end"]
                or interval["expiry_is_run_end"]
            )
            continue
        episodes.append(dict(interval))

    return [
        {
            "shock_ids": episode["shock_ids"],
            "onset": episode["onset"].isoformat(),
            "expiry": episode["expiry"].isoformat(),
            "expiry_is_run_end": episode["expiry_is_run_end"],
        }
        for episode in episodes
    ]


def _episode_key(shock_ids: Sequence[str]) -> str:
    if len(shock_ids) == 1:
        return shock_ids[0]
    return "episode__" + "__".join(shock_ids)


def _baseline_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    onset: datetime,
    seconds: int,
) -> list[Mapping[str, Any]]:
    start = onset - timedelta(seconds=max(0, seconds))
    selected = [
        row
        for row in rows
        if start <= _row_time(row) < onset
    ]
    if selected:
        return selected
    previous = [row for row in rows if _row_time(row) < onset]
    return previous[-1:]


def _event_metadata(
    actions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for action in actions:
        if action.get("event_type") != "event_observed":
            continue
        shock_id = action.get("shock_id")
        if not shock_id or str(action.get("phase", "active")).lower() not in {
            "active",
            "shock",
        }:
            continue
        result[str(shock_id)] = {
            key: action.get(key)
            for key in (
                "shock_type",
                "shock_class",
                "scope",
                "severity",
                "direction",
                "affected_instruments",
                "affected_asset_classes",
            )
        }
    return result


def _observed_shock_ids(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    return sorted(
        {
            shock_id
            for row in rows
            for shock_id in _shock_ids(row)
        }
    )


def _shock_ids(row: Mapping[str, Any]) -> list[str]:
    value = row.get("active_shock_ids")
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _mean_positions(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    observations: dict[str, list[float]] = {}
    for row in rows:
        for instrument, value in _net_positions(row).items():
            observations.setdefault(instrument, []).append(value)
    return {
        instrument: mean(values)
        for instrument, values in observations.items()
        if values
    }


def _net_positions(row: Mapping[str, Any]) -> dict[str, float]:
    positions = row.get("positions")
    if not isinstance(positions, Mapping):
        return {}
    result: dict[str, float] = {}
    for instrument, details in positions.items():
        if not isinstance(details, Mapping):
            continue
        value = _optional_number(details.get("net"))
        if value is not None:
            result[str(instrument)] = value
    return result


def _role_state_numbers(
    row: Mapping[str, Any],
    key: str,
) -> list[float]:
    state = row.get("fast_loop_state")
    if not isinstance(state, Mapping):
        return []
    return [
        value
        for details in state.values()
        if isinstance(details, Mapping)
        and (value := _optional_number(details.get(key))) is not None
    ]


def _is_constrained(row: Mapping[str, Any]) -> bool:
    state = row.get("fast_loop_state")
    if not isinstance(state, Mapping):
        return False
    return any(
        bool(
            details.get("buy_constrained")
            or details.get("sell_constrained")
            or details.get("position_constrained")
        )
        for details in state.values()
        if isinstance(details, Mapping)
    )


def _duration_matching(
    rows: Sequence[Mapping[str, Any]],
    predicate: Any,
) -> float:
    total = 0.0
    for index, row in enumerate(rows[:-1]):
        if predicate(row):
            total += (
                _row_time(rows[index + 1]) - _row_time(row)
            ).total_seconds()
    return total


def _horizon_values(
    rows: Sequence[Mapping[str, Any]],
    reference: datetime,
    horizons: Iterable[int],
    fields: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seconds in horizons:
        row = _first_at_or_after(
            rows,
            reference + timedelta(seconds=seconds),
        )
        result[f"+{seconds}s"] = (
            {
                "observed_at": _row_time(row).isoformat(),
                **{field: row.get(field) for field in fields},
            }
            if row is not None
            else None
        )
    return result


def _balance_horizons(
    rows: Sequence[Mapping[str, Any]],
    reference: datetime,
    horizons: Iterable[int],
    baseline_cash: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seconds in horizons:
        row = _first_at_or_after(
            rows,
            reference + timedelta(seconds=seconds),
        )
        result[f"+{seconds}s"] = (
            {
                "observed_at": _row_time(row).isoformat(),
                "gross_exposure": row.get("gross_exposure"),
                "net_positions": _net_positions(row),
                "cash_utilization": _cash_utilization(
                    row.get("cash"),
                    baseline_cash,
                ),
                "constrained": _is_constrained(row),
                "role_state": _select_fast_loop_state(
                    row,
                    (
                        "effective_position_limit",
                        "position_limit_utilization",
                        "target_distance",
                        "buy_constrained",
                        "sell_constrained",
                        "position_constrained",
                    ),
                ),
            }
            if row is not None
            else None
        )
    return result


def _behaviour_horizons(
    rows: Sequence[Mapping[str, Any]],
    reference: datetime,
    horizons: Iterable[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seconds in horizons:
        row = _first_at_or_after(
            rows,
            reference + timedelta(seconds=seconds),
        )
        result[f"+{seconds}s"] = (
            {
                "observed_at": _row_time(row).isoformat(),
                "risk_mode": row.get("risk_mode"),
                "participated": row.get("fast_loop_participated"),
                "order_sides": row.get("order_sides"),
                "order_types": row.get("order_types"),
                "order_quantities": row.get("order_quantities"),
                "role_state": _select_fast_loop_state(
                    row,
                    (
                        "effective_participation_probability",
                        "effective_order_size_cap",
                        "effective_quote_size",
                        "effective_spread",
                        "effective_child_order_size",
                    ),
                ),
            }
            if row is not None
            else None
        )
    return result


def _first_at_or_after(
    rows: Sequence[Mapping[str, Any]],
    target: datetime,
) -> Mapping[str, Any] | None:
    return next(
        (row for row in rows if _row_time(row) >= target),
        None,
    )


def _select_fast_loop_state(
    row: Mapping[str, Any],
    keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    state = row.get("fast_loop_state")
    if not isinstance(state, Mapping):
        return {}
    selected: dict[str, dict[str, Any]] = {}
    for instrument, details in state.items():
        if not isinstance(details, Mapping):
            continue
        selected[str(instrument)] = {
            key: details.get(key)
            for key in keys
            if key in details
        }
    return selected


def _cash_utilization(
    current: Any,
    baseline: float | None,
) -> float | None:
    current_number = _optional_number(current)
    if current_number is None or baseline in {None, 0.0}:
        return None
    return 1.0 - (current_number / baseline)


def _relative_distance(value: Any, baseline: Any) -> float:
    value_number = _number(value, 0.0)
    baseline_number = _number(baseline, 0.0)
    denominator = abs(baseline_number)
    if denominator == 0:
        return abs(value_number - baseline_number)
    return abs(value_number - baseline_number) / denominator


def _read_action_file(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [
        dict(item)
        for item in payload
        if isinstance(item, Mapping)
    ]


def _row_time(row: Mapping[str, Any]) -> datetime:
    return _parse_timestamp(row.get("timestamp"))


def _timestamp_sort_key(row: Mapping[str, Any]) -> datetime:
    return _row_time(row)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value or "")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


def _numbers(values: Iterable[Any]) -> list[float]:
    return [
        number
        for value in values
        if (number := _optional_number(value)) is not None
    ]


def _optional_number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _number(value: Any, default: float) -> float:
    parsed = _optional_number(value)
    return parsed if parsed is not None else default


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "disrupted": None,
        "recovered": None,
    }
