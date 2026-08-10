"""Validate and summarize locked validation runs for the D0-D4 experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "experiments" / "diversity_matrix.yaml"
DEFAULT_RUNS_DIR = ROOT / ".aml_runs"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "diversity_analysis" / "validation"
LEVELS = ("D0", "D1", "D2", "D3", "D4")
METRIC_FIELDS = (
    "mean_signed_flow_herding_index",
    "mean_absolute_within_role_signed_flow_correlation",
    "mean_absolute_cross_role_signed_flow_correlation",
    "mean_relative_spread_bps",
    "mean_top5_total_depth",
    "trade_active_tick_rate",
)
REQUIRED_REPORTS = (
    "research_metrics.json",
    "agent_decisions_detailed.csv",
    "interval_outcomes.csv",
    "order_book_microstructure.csv",
    "llm_strategy_updates.json",
)
SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
ERROR_PATTERN = re.compile(
    r"(?:\bERROR\b|Traceback \(most recent call last\)|Simulation encountered)",
    re.IGNORECASE,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _has_invalid_result_timestamp(item: dict[str, str]) -> bool:
    """Allow unavailable timestamps only when the outcome is explicitly missing."""
    status = str(item.get("outcome_status", "")).lower()
    interval_end = _parse_timestamp(item.get("interval_end"))
    result_time = _parse_timestamp(item.get("result_available_timestamp"))
    if interval_end is None:
        return True
    if status == "missing" and result_time is None:
        return False
    return result_time is None or result_time < interval_end


def _parse_interval_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    units = {
        "seconds": 1.0,
        "second": 1.0,
        "secs": 1.0,
        "sec": 1.0,
        "s": 1.0,
        "minutes": 60.0,
        "minute": 60.0,
        "mins": 60.0,
        "min": 60.0,
        "m": 60.0,
        "hours": 3600.0,
        "hour": 3600.0,
        "h": 3600.0,
    }
    for suffix in sorted(units, key=len, reverse=True):
        if text.endswith(suffix):
            return float(text[: -len(suffix)].strip()) * units[suffix]
    raise ValueError(f"Unsupported interval value: {value!r}")


def _expected_microstructure_rows(scenario: dict[str, Any]) -> int:
    simulation = scenario["stocksim_config"]["simulation"]
    start = _parse_timestamp(str(simulation["start_time"]))
    end = _parse_timestamp(str(simulation["end_time"]))
    if start is None or end is None or end <= start:
        raise ValueError("Scenario has an invalid simulation window")
    interval = _parse_interval_seconds(simulation["tick_interval"])
    return math.ceil((end - start).total_seconds() / interval)


def _expected_api_calls(scenario: dict[str, Any]) -> int:
    simulation = scenario["stocksim_config"]["simulation"]
    start = _parse_timestamp(str(simulation["start_time"]))
    end = _parse_timestamp(str(simulation["end_time"]))
    if start is None or end is None or end <= start:
        raise ValueError("Scenario has an invalid simulation window")
    duration = (end - start).total_seconds()
    llm_defaults = scenario.get("aml_config", {}).get("llm", {}) or {}
    calls = 0
    for details in scenario["stocksim_config"].get("agents", {}).values():
        parameters = details.get("parameters", {}) or {}
        strategist = parameters.get("slow_strategist")
        if not isinstance(strategist, dict):
            continue
        effective = {**llm_defaults, **strategist}
        if str(effective.get("type", "static")).lower() not in {
            "openai",
            "openai_json",
        }:
            continue
        interval = _parse_interval_seconds(parameters.get("slow_loop_interval", "1h"))
        calls += int(details.get("count", 1)) * max(1, math.ceil(duration / interval))
    return calls


def _nested_metric(metrics: dict[str, Any], field: str) -> float | None:
    section = "synchrony" if field in METRIC_FIELDS[:3] else "liquidity"
    value = metrics.get(section, {}).get(field)
    return float(value) if isinstance(value, (int, float)) else None


def _scan_text_files(run_dir: Path) -> tuple[int, int]:
    error_lines = 0
    secret_matches = 0
    allowed_suffixes = {".csv", ".json", ".jsonl", ".log", ".txt", ".yaml", ".yml"}
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        secret_matches += len(SECRET_PATTERN.findall(text))
        if path.suffix.lower() == ".log":
            error_lines += sum(bool(ERROR_PATTERN.search(line)) for line in text.splitlines())
    return error_lines, secret_matches


def _read_api_records(run_dir: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    parse_errors = 0
    for path in sorted((run_dir / "decision_context").rglob("llm_responses.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                if isinstance(value, dict):
                    records.append(value)
                else:
                    parse_errors += 1
    return records, parse_errors


def inspect_run(run_dir: Path, level: str, seed: int) -> dict[str, Any]:
    """Return primary metrics and structural quality checks for one run."""
    reports_dir = run_dir / "reports"
    missing = [name for name in REQUIRED_REPORTS if not (reports_dir / name).exists()]
    row: dict[str, Any] = {
        "run_id": run_dir.name,
        "diversity_level": level,
        "seed": seed,
        "run_exists": run_dir.exists(),
        "missing_required_file_count": len(missing),
        "missing_required_files": ";".join(missing),
    }
    if not run_dir.exists() or missing:
        row.update({field: None for field in METRIC_FIELDS})
        row["quality_pass"] = False
        return row

    scenario = _read_yaml(run_dir / "scenario.yaml")
    metrics = _read_json(reports_dir / "research_metrics.json")
    decisions = _read_csv(reports_dir / "agent_decisions_detailed.csv")
    outcomes = _read_csv(reports_dir / "interval_outcomes.csv")
    microstructure = _read_csv(reports_dir / "order_book_microstructure.csv")
    updates = _read_json(reports_dir / "llm_strategy_updates.json")
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid metrics object in {run_dir}")
    if not isinstance(updates, list):
        raise ValueError(f"Invalid LLM update audit in {run_dir}")

    decision_ids = [item.get("decision_id", "") for item in decisions]
    decision_id_counts = Counter(item for item in decision_ids if item)
    outcome_ids = [item.get("decision_id", "") for item in outcomes]
    outcome_id_set = {item for item in outcome_ids if item}
    decision_id_set = set(decision_id_counts)

    lookahead_count = 0
    invalid_decision_timestamp_count = 0
    action_mismatch_count = 0
    invalid_decision_split_count = 0
    for item in decisions:
        decision_time = _parse_timestamp(item.get("decision_timestamp"))
        cutoff_time = _parse_timestamp(item.get("data_cutoff_timestamp"))
        if decision_time is None or cutoff_time is None:
            invalid_decision_timestamp_count += 1
        elif cutoff_time > decision_time:
            lookahead_count += 1
        if str(item.get("action", "")).upper() != str(
            item.get("submitted_action", "")
        ).upper():
            action_mismatch_count += 1
        if item.get("split") != "validation":
            invalid_decision_split_count += 1

    allowed_statuses = {"completed", "inactive", "missing"}
    invalid_outcome_status_count = 0
    invalid_result_timestamp_count = 0
    invalid_outcome_split_count = 0
    outcome_status_counts = Counter()
    for item in outcomes:
        status = str(item.get("outcome_status", "")).lower()
        outcome_status_counts[status] += 1
        if status not in allowed_statuses:
            invalid_outcome_status_count += 1
        if _has_invalid_result_timestamp(item):
            invalid_result_timestamp_count += 1
        if item.get("split") != "validation":
            invalid_outcome_split_count += 1

    api_records, api_parse_errors = _read_api_records(run_dir)
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    missing_response_id_count = 0
    for record in api_records:
        if not record.get("response_id"):
            missing_response_id_count += 1
        usage = record.get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)

    invalid_update_audit_count = sum(
        not isinstance(item, dict)
        or not all(
            isinstance(item.get(field), dict)
            for field in ("proposed_updates", "applied_updates", "rejected_updates")
        )
        for item in updates
    )
    update_statuses = Counter(str(item.get("status", "")) for item in updates if isinstance(item, dict))
    error_lines, secret_matches = _scan_text_files(run_dir)

    expected_api_calls = _expected_api_calls(scenario)
    expected_micro_rows = _expected_microstructure_rows(scenario)
    quality_errors = (
        len(missing)
        + sum(not item for item in decision_ids)
        + sum(count - 1 for count in decision_id_counts.values() if count > 1)
        + lookahead_count
        + invalid_decision_timestamp_count
        + action_mismatch_count
        + invalid_decision_split_count
        + len(outcome_id_set - decision_id_set)
        + len(decision_id_set - outcome_id_set)
        + invalid_outcome_status_count
        + invalid_result_timestamp_count
        + invalid_outcome_split_count
        + api_parse_errors
        + missing_response_id_count
        + invalid_update_audit_count
        + error_lines
        + secret_matches
        + abs(len(api_records) - expected_api_calls)
        + abs(len(updates) - expected_api_calls)
        + abs(len(microstructure) - expected_micro_rows)
    )

    row.update({field: _nested_metric(metrics, field) for field in METRIC_FIELDS})
    row.update(
        {
            "expected_api_call_count": expected_api_calls,
            "api_call_count": len(api_records),
            "api_response_parse_error_count": api_parse_errors,
            "missing_response_id_count": missing_response_id_count,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "decision_row_count": len(decisions),
            "missing_decision_id_count": sum(not item for item in decision_ids),
            "duplicate_decision_id_count": sum(
                count - 1 for count in decision_id_counts.values() if count > 1
            ),
            "lookahead_count": lookahead_count,
            "invalid_decision_timestamp_count": invalid_decision_timestamp_count,
            "action_mismatch_count": action_mismatch_count,
            "invalid_decision_split_count": invalid_decision_split_count,
            "outcome_row_count": len(outcomes),
            "completed_outcome_count": outcome_status_counts["completed"],
            "inactive_outcome_count": outcome_status_counts["inactive"],
            "missing_outcome_count": outcome_status_counts["missing"],
            "orphan_outcome_decision_id_count": len(outcome_id_set - decision_id_set),
            "decision_without_outcome_count": len(decision_id_set - outcome_id_set),
            "invalid_outcome_status_count": invalid_outcome_status_count,
            "invalid_result_timestamp_count": invalid_result_timestamp_count,
            "invalid_outcome_split_count": invalid_outcome_split_count,
            "expected_microstructure_row_count": expected_micro_rows,
            "microstructure_row_count": len(microstructure),
            "valid_book_observation_count": metrics.get("liquidity", {}).get(
                "valid_book_observation_count"
            ),
            "llm_update_audit_count": len(updates),
            "applied_update_count": update_statuses["applied"],
            "partially_applied_update_count": update_statuses["partially_applied"],
            "rejected_update_count": update_statuses["rejected"],
            "invalid_update_audit_count": invalid_update_audit_count,
            "log_error_line_count": error_lines,
            "secret_pattern_match_count": secret_matches,
            "primary_metric_null_count": sum(
                row.get(field) is None for field in METRIC_FIELDS
            ),
            "quality_error_count": quality_errors,
            "quality_pass": quality_errors == 0
            and all(row.get(field) is not None for field in METRIC_FIELDS),
        }
    )
    return row


def _safe_relative(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def build_level_summary(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    requested_seeds: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate requested validation runs and apply the frozen selection rule."""
    locked_seeds = [int(seed) for seed in protocol["validation_seeds"]]
    complete_validation = set(requested_seeds) == set(locked_seeds)
    summary: list[dict[str, Any]] = []
    for level in LEVELS:
        level_rows = [row for row in rows if row["diversity_level"] == level]
        item: dict[str, Any] = {
            "diversity_level": level,
            "run_count": len(level_rows),
            "quality_pass_run_count": sum(bool(row.get("quality_pass")) for row in level_rows),
        }
        for field in METRIC_FIELDS:
            values = [
                float(row[field])
                for row in level_rows
                if isinstance(row.get(field), (int, float))
            ]
            item[field] = mean(values) if values else None
        summary.append(item)

    baseline = summary[0]
    thresholds = protocol["validation_thresholds"]
    all_quality_pass = bool(rows) and all(bool(row.get("quality_pass")) for row in rows)
    selected_level: str | None = None
    for item in summary:
        if item["diversity_level"] == "D0":
            item.update(
                {
                    "herding_reduction_relative_to_d0": 0.0,
                    "within_role_correlation_reduction": 0.0,
                    "cross_role_correlation_reduction": 0.0,
                    "relative_spread_increase_vs_d0": 0.0,
                    "relative_depth_decrease_vs_d0": 0.0,
                    "trade_active_tick_rate_decrease_vs_d0": 0.0,
                    "threshold_status": "baseline",
                    "eligible": False,
                }
            )
            continue

        item["herding_reduction_relative_to_d0"] = _safe_relative(
            baseline[METRIC_FIELDS[0]] - item[METRIC_FIELDS[0]]
            if baseline[METRIC_FIELDS[0]] is not None and item[METRIC_FIELDS[0]] is not None
            else None,
            baseline[METRIC_FIELDS[0]],
        )
        item["within_role_correlation_reduction"] = (
            baseline[METRIC_FIELDS[1]] - item[METRIC_FIELDS[1]]
            if baseline[METRIC_FIELDS[1]] is not None and item[METRIC_FIELDS[1]] is not None
            else None
        )
        item["cross_role_correlation_reduction"] = (
            baseline[METRIC_FIELDS[2]] - item[METRIC_FIELDS[2]]
            if baseline[METRIC_FIELDS[2]] is not None and item[METRIC_FIELDS[2]] is not None
            else None
        )
        item["relative_spread_increase_vs_d0"] = _safe_relative(
            item[METRIC_FIELDS[3]] - baseline[METRIC_FIELDS[3]]
            if baseline[METRIC_FIELDS[3]] is not None and item[METRIC_FIELDS[3]] is not None
            else None,
            baseline[METRIC_FIELDS[3]],
        )
        item["relative_depth_decrease_vs_d0"] = _safe_relative(
            baseline[METRIC_FIELDS[4]] - item[METRIC_FIELDS[4]]
            if baseline[METRIC_FIELDS[4]] is not None and item[METRIC_FIELDS[4]] is not None
            else None,
            baseline[METRIC_FIELDS[4]],
        )
        item["trade_active_tick_rate_decrease_vs_d0"] = (
            baseline[METRIC_FIELDS[5]] - item[METRIC_FIELDS[5]]
            if baseline[METRIC_FIELDS[5]] is not None and item[METRIC_FIELDS[5]] is not None
            else None
        )

        comparisons = (
            item["herding_reduction_relative_to_d0"],
            item["within_role_correlation_reduction"],
            item["cross_role_correlation_reduction"],
            item["relative_spread_increase_vs_d0"],
            item["relative_depth_decrease_vs_d0"],
            item["trade_active_tick_rate_decrease_vs_d0"],
        )
        if not complete_validation:
            item["threshold_status"] = "exploratory_only"
            item["eligible"] = None
        elif not all_quality_pass or any(value is None for value in comparisons):
            item["threshold_status"] = "insufficient_quality_or_data"
            item["eligible"] = None
        else:
            eligible = (
                comparisons[0] >= thresholds["minimum_herding_reduction_relative_to_d0"]
                and comparisons[1] >= thresholds["minimum_absolute_correlation_reduction"]
                and comparisons[2] >= thresholds["minimum_absolute_correlation_reduction"]
                and comparisons[3] <= thresholds["maximum_relative_spread_increase"]
                and comparisons[4] <= thresholds["maximum_relative_depth_decrease"]
                and comparisons[5] <= thresholds["maximum_trade_active_tick_rate_decrease"]
            )
            item["threshold_status"] = "pass" if eligible else "fail"
            item["eligible"] = eligible
            if eligible and selected_level is None:
                selected_level = item["diversity_level"]

    analysis_status = (
        "validation_complete"
        if complete_validation and all_quality_pass
        else "validation_quality_failure"
        if complete_validation
        else "exploratory_only"
    )
    metadata = {
        "phase": "validation",
        "analysis_status": analysis_status,
        "requested_seeds": requested_seeds,
        "locked_validation_seed_count": len(locked_seeds),
        "all_requested_runs_pass_quality": all_quality_pass,
        "selection_permitted": complete_validation and all_quality_pass,
        "selected_level": selected_level if complete_validation and all_quality_pass else None,
    }
    return summary, metadata


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in values:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def analyze(
    protocol_path: Path,
    runs_dir: Path,
    output_dir: Path,
    levels: list[str],
    seeds: list[int],
    allow_partial: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    protocol = _read_yaml(protocol_path)
    locked = {int(seed) for seed in protocol["validation_seeds"]}
    unlocked = sorted(set(seeds) - locked)
    if unlocked:
        raise ValueError(f"Seeds are not locked validation seeds: {unlocked}")
    rows: list[dict[str, Any]] = []
    missing_runs: list[str] = []
    for level in levels:
        for seed in seeds:
            run_id = f"diversity_{level.lower()}_seed_{seed}"
            run_dir = runs_dir / run_id
            if not run_dir.exists():
                missing_runs.append(run_id)
                continue
            rows.append(inspect_run(run_dir, level, seed))
    if missing_runs and not allow_partial:
        raise FileNotFoundError("Missing requested runs: " + ", ".join(missing_runs))

    level_summary, metadata = build_level_summary(rows, protocol, seeds)
    metadata.update(
        {
            "levels": levels,
            "run_count": len(rows),
            "missing_runs": missing_runs,
            "total_api_calls": sum(int(row.get("api_call_count") or 0) for row in rows),
            "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
            "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
            "selection_rule": protocol.get("selection_rule"),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "run_quality.csv", rows)
    _write_csv(output_dir / "level_summary.csv", level_summary)
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "levels": level_summary}, handle, indent=2)
    return rows, level_summary, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze locked validation D0-D4 outputs without opening OOS results."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write exploratory output when some requested run directories are absent.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = _read_yaml(args.protocol.resolve())
    seeds = args.seeds or [int(seed) for seed in protocol["validation_seeds"]]
    rows, level_summary, metadata = analyze(
        args.protocol.resolve(),
        args.runs_dir.resolve(),
        args.output_dir.resolve(),
        list(args.levels),
        seeds,
        args.allow_partial,
    )
    print(f"Analysis status: {metadata['analysis_status']}")
    print(f"Runs analyzed: {len(rows)}; API calls: {metadata['total_api_calls']}")
    print(f"Selected level: {metadata['selected_level'] or 'none'}")
    for item in level_summary:
        print(
            f"{item['diversity_level']}: runs={item['run_count']} "
            f"quality={item['quality_pass_run_count']} "
            f"status={item['threshold_status']}"
        )
    print(f"Outputs: {args.output_dir.resolve()}")
    if metadata["missing_runs"] or any(not row.get("quality_pass") for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
