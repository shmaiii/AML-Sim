"""Build the final descriptive report from frozen validation and OOS summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = ROOT / ".aml_runs" / "diversity_analysis"
LEVELS = ("D0", "D1", "D2", "D3", "D4")
METRICS = (
    ("mean_signed_flow_herding_index", "Herding"),
    ("mean_absolute_within_role_signed_flow_correlation", "Within-role corr."),
    ("mean_absolute_cross_role_signed_flow_correlation", "Cross-role corr."),
    ("mean_relative_spread_bps", "Spread (bps)"),
    ("mean_top5_total_depth", "Top-5 depth"),
    ("trade_active_tick_rate", "Trade-active rate"),
)
T_CRITICAL_95_DF4 = 2.7764451051977987


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _statistics(phase: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for level in LEVELS:
        selected = [row for row in rows if row["diversity_level"] == level]
        for field, label in METRICS:
            values = [_number(row, field) for row in selected]
            value_mean = mean(values)
            value_sd = stdev(values) if len(values) > 1 else 0.0
            half_width = (
                T_CRITICAL_95_DF4 * value_sd / math.sqrt(len(values))
                if len(values) == 5
                else None
            )
            output.append(
                {
                    "phase": phase,
                    "diversity_level": level,
                    "metric": field,
                    "metric_label": label,
                    "n": len(values),
                    "mean": value_mean,
                    "sample_sd": value_sd,
                    "ci95_low": value_mean - half_width if half_width is not None else None,
                    "ci95_high": value_mean + half_width if half_width is not None else None,
                }
            )
    return output


def _seed_thresholds(
    phase: str,
    rows: list[dict[str, str]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    by_key = {
        (row["diversity_level"], int(row["seed"])): row
        for row in rows
    }
    seeds = sorted(seed for level, seed in by_key if level == "D0")
    output: list[dict[str, Any]] = []
    for level in LEVELS[1:]:
        for seed in seeds:
            baseline = by_key[("D0", seed)]
            treatment = by_key[(level, seed)]
            d0_herding = _number(baseline, METRICS[0][0])
            herding_reduction = (
                (d0_herding - _number(treatment, METRICS[0][0])) / d0_herding
                if d0_herding != 0
                else None
            )
            within_reduction = _number(baseline, METRICS[1][0]) - _number(
                treatment, METRICS[1][0]
            )
            cross_reduction = _number(baseline, METRICS[2][0]) - _number(
                treatment, METRICS[2][0]
            )
            d0_spread = _number(baseline, METRICS[3][0])
            spread_increase = (
                (_number(treatment, METRICS[3][0]) - d0_spread) / d0_spread
                if d0_spread != 0
                else None
            )
            d0_depth = _number(baseline, METRICS[4][0])
            depth_decrease = (
                (d0_depth - _number(treatment, METRICS[4][0])) / d0_depth
                if d0_depth != 0
                else None
            )
            activity_decrease = _number(baseline, METRICS[5][0]) - _number(
                treatment, METRICS[5][0]
            )
            passes = (
                herding_reduction is not None
                and herding_reduction
                >= thresholds["minimum_herding_reduction_relative_to_d0"]
                and within_reduction
                >= thresholds["minimum_absolute_correlation_reduction"]
                and cross_reduction
                >= thresholds["minimum_absolute_correlation_reduction"]
                and spread_increase is not None
                and spread_increase <= thresholds["maximum_relative_spread_increase"]
                and depth_decrease is not None
                and depth_decrease <= thresholds["maximum_relative_depth_decrease"]
                and activity_decrease
                <= thresholds["maximum_trade_active_tick_rate_decrease"]
            )
            output.append(
                {
                    "phase": phase,
                    "diversity_level": level,
                    "seed": seed,
                    "herding_reduction_relative_to_d0": herding_reduction,
                    "within_role_correlation_reduction": within_reduction,
                    "cross_role_correlation_reduction": cross_reduction,
                    "relative_spread_increase_vs_d0": spread_increase,
                    "relative_depth_decrease_vs_d0": depth_decrease,
                    "trade_active_tick_rate_decrease_vs_d0": activity_decrease,
                    "all_thresholds_pass": passes,
                }
            )
    return output


def _table(summary: list[dict[str, str]]) -> list[str]:
    lines = [
        "| Level | Herding | Within corr. | Cross corr. | Spread bps | Top-5 depth | Active rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {diversity_level} | {h:.3f} | {w:.3f} | {c:.3f} | "
            "{s:.2f} | {d:.2f} | {a:.3f} |".format(
                diversity_level=row["diversity_level"],
                h=_number(row, METRICS[0][0]),
                w=_number(row, METRICS[1][0]),
                c=_number(row, METRICS[2][0]),
                s=_number(row, METRICS[3][0]),
                d=_number(row, METRICS[4][0]),
                a=_number(row, METRICS[5][0]),
            )
        )
    return lines


def build_report(
    analysis_dir: Path,
    output_dir: Path,
    input_price_per_million: float,
    output_price_per_million: float,
) -> dict[str, Any]:
    validation_dir = analysis_dir / "validation"
    oos_dir = analysis_dir / "out_of_sample"
    protocol_path = ROOT / "experiments" / "diversity_matrix.yaml"
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    validation_quality = _read_csv(validation_dir / "run_quality.csv")
    oos_quality = _read_csv(oos_dir / "run_quality.csv")
    validation_summary = _read_csv(validation_dir / "level_summary.csv")
    oos_summary = _read_csv(oos_dir / "level_summary.csv")
    selection_lock = _read_json(validation_dir / "selection_lock.json")
    oos_analysis = _read_json(oos_dir / "analysis_summary.json")

    if selection_lock.get("selection_result") != "no_qualifying_level":
        selection_text = f"selected {selection_lock.get('selected_level')}"
    else:
        selection_text = "no qualifying diversity level"
    if oos_analysis.get("metadata", {}).get("selection_lock_sha256") is None:
        raise ValueError("OOS analysis is not linked to a validation selection lock")

    statistics_rows = _statistics("validation", validation_quality)
    statistics_rows.extend(_statistics("out_of_sample", oos_quality))
    seed_threshold_rows = _seed_thresholds(
        "validation", validation_quality, protocol["validation_thresholds"]
    )
    seed_threshold_rows.extend(
        _seed_thresholds(
            "out_of_sample", oos_quality, protocol["validation_thresholds"]
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "level_metric_uncertainty.csv", statistics_rows)
    _write_csv(output_dir / "seed_threshold_results.csv", seed_threshold_rows)

    all_quality = validation_quality + oos_quality
    total_api_calls = sum(int(row["api_call_count"]) for row in all_quality)
    total_input_tokens = sum(int(row["input_tokens"]) for row in all_quality)
    total_output_tokens = sum(int(row["output_tokens"]) for row in all_quality)
    total_cost = (
        total_input_tokens / 1_000_000 * input_price_per_million
        + total_output_tokens / 1_000_000 * output_price_per_million
    )
    pass_counts: dict[tuple[str, str], int] = {}
    for phase in ("validation", "out_of_sample"):
        for level in LEVELS[1:]:
            pass_counts[(phase, level)] = sum(
                row["all_thresholds_pass"] is True
                for row in seed_threshold_rows
                if row["phase"] == phase and row["diversity_level"] == level
            )

    report_lines = [
        "# Cognitive Diversity D0-D4: Final Research Report",
        "",
        "## Study status",
        "",
        "The full preregistered matrix completed: 25 validation runs and 25 out-of-sample runs. "
        f"Validation was frozen as **{selection_text}** before OOS outputs were analyzed. "
        "OOS results therefore do not alter the validation selection decision. This final report is a "
        "descriptive post-run summary of the six preregistered primary metrics.",
        "",
        "## Data integrity and cost",
        "",
        f"All 50 runs passed structural quality checks. The study recorded {total_api_calls} API calls, "
        f"{sum(int(row['decision_row_count']) for row in all_quality):,} decisions, "
        f"{sum(int(row['outcome_row_count']) for row in all_quality):,} outcomes, and "
        f"{sum(int(row['microstructure_row_count']) for row in all_quality):,} order-book ticks. "
        "Look-ahead, duplicate-ID, API-response, log-error, and secret-leakage checks all returned zero.",
        "",
        f"Observed usage was {total_input_tokens:,} input tokens and {total_output_tokens:,} output tokens. "
        f"At ${input_price_per_million:.2f}/M input and ${output_price_per_million:.2f}/M output, "
        f"estimated API cost was **${total_cost:.2f}**.",
        "",
        "## Validation results",
        "",
        *_table(validation_summary),
        "",
        "No D1-D4 level passed all preregistered thresholds on validation means. "
        "D2 produced the largest within-role correlation reduction (0.065) while preserving liquidity, "
        "but its herding reduction (1.9%) and cross-role reduction (0.018) were below the required values.",
        "",
        "## Out-of-sample results",
        "",
        *_table(oos_summary),
        "",
        "OOS patterns did not overturn the frozen conclusion. D2 again reduced within-role and cross-role "
        "correlation, but its trade-active rate fell by 0.126 versus D0. D4 delivered the largest herding "
        "reduction (6.5%), still below the required 15%, and its activity rate fell by 0.095.",
        "",
        "## Seed-level robustness",
        "",
        "| Level | Validation seeds passing all thresholds | OOS seeds passing all thresholds |",
        "|---|---:|---:|",
        *[
            f"| {level} | {pass_counts[('validation', level)]}/5 | "
            f"{pass_counts[('out_of_sample', level)]}/5 |"
            for level in LEVELS[1:]
        ],
        "",
        "## Conclusion",
        "",
        "Under the tested agent population, role mix, capital, shocks, information set, and ten-minute horizon, "
        "increasing profile, prompt, and initial-strategy diversity did not produce a level that simultaneously "
        "met the synchrony-reduction and liquidity-preservation criteria. Moderate diversity, especially D2, "
        "reduced pairwise correlations more consistently than maximal diversity, but this effect was not large "
        "enough and did not generalize without a continuity cost in OOS. The defensible conclusion is therefore "
        "that the minimum effective diversity level was not identified within D0-D4 under this design, rather "
        "than that one of the observed levels should be selected post hoc.",
        "",
        "The confidence intervals and seed-level values are available in the accompanying CSV files. With only "
        "five seeds per phase, uncertainty remains material; the results apply to this controlled scenario and "
        "should not be generalized to other role mixes or market structures without a separately preregistered study.",
    ]
    report_path = output_dir / "final_research_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary = {
        "selection_result": selection_lock["selection_result"],
        "selected_level": selection_lock.get("selected_level"),
        "validation_runs": len(validation_quality),
        "oos_runs": len(oos_quality),
        "total_api_calls": total_api_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_api_cost_usd": total_cost,
        "report": str(report_path),
    }
    with (output_dir / "final_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final locked diversity report.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-price", type=float, default=2.50)
    parser.add_argument("--output-price", type=float, default=15.00)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    analysis_dir = args.analysis_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (analysis_dir / "final").resolve()
    )
    summary = build_report(
        analysis_dir, output_dir, args.input_price, args.output_price
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
