"""Run seeded recovery replications sequentially and generate per-run plots.

Example:
    python scripts/run_recovery_experiment.py \
        scenarios/aml_role_recovery_baseline.yaml \
        --experiment-id role-recovery-pilot \
        --replications 3 \
        --seed-base 2026080400
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNS_DIR = ROOT / ".aml_runs"
EXPERIMENTS_DIR = ROOT / ".aml_experiments"
SEEDED_PARAMETER = "random_seed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path, help="Source recovery scenario YAML.")
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Stable directory/run prefix used to resume an experiment.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=3,
        help="Number of paired replications to run (default: 3).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        required=True,
        help="Base integer used to derive deterministic per-agent seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write seeded scenarios and the manifest without launching AML-Sim.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later replications if one simulation or plot step fails.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario must contain a YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def _scenario_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_scenario(
    source: dict[str, Any],
    *,
    experiment_id: str,
    replication: int,
    seed_base: int,
) -> tuple[dict[str, Any], dict[str, int]]:
    scenario = copy.deepcopy(source)
    scenario["name"] = f"{source.get('name', experiment_id)}_r{replication:03d}"
    agents = scenario.get("stocksim_config", {}).get("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("Scenario stocksim_config.agents must be a mapping")

    seeds: dict[str, int] = {}
    seeded_agents = [
        (agent_id, config)
        for agent_id, config in sorted(agents.items())
        if isinstance(config, dict)
        and isinstance(config.get("parameters"), dict)
        and SEEDED_PARAMETER in config["parameters"]
    ]
    for offset, (agent_id, config) in enumerate(seeded_agents, start=1):
        seed = seed_base + ((replication - 1) * 100) + offset
        config["parameters"][SEEDED_PARAMETER] = seed
        seeds[str(agent_id)] = seed
    return scenario, seeds


def _plot_outputs(run_dir: Path) -> list[Path]:
    evaluation_dir = run_dir / "reports" / "evaluation"
    stems = (
        "financial_recovery_timeline",
        "balance_sheet_recovery_timeline",
        "behavioural_recovery_timeline",
    )
    return [evaluation_dir / f"{stem}.{suffix}" for stem in stems for suffix in ("png", "svg")]


def _postprocess_run(run_dir: Path) -> list[str]:
    from aml_sim.evaluation.recovery import RecoveryEvaluator
    from aml_sim.evaluation.recovery_plot import (
        plot_balance_sheet_recovery,
        plot_behavioural_recovery,
        plot_financial_recovery,
    )

    report_path = RecoveryEvaluator().save_report(run_dir)
    outputs = [report_path]
    for plotter in (
        plot_financial_recovery,
        plot_balance_sheet_recovery,
        plot_behavioural_recovery,
    ):
        outputs.extend(plotter(run_dir))
    return [str(path.resolve()) for path in outputs]


def _completed(run_dir: Path) -> bool:
    report = run_dir / "reports" / "evaluation" / "role_recovery.json"
    return report.exists() and all(path.exists() for path in _plot_outputs(run_dir))


def _can_postprocess(run_dir: Path) -> bool:
    return any((run_dir / "reports" / "agents").glob("trader_actions_*.json"))


def _behavioural_trajectory_metrics(section: dict[str, Any]) -> dict[str, float | None]:
    trajectory = section.get("score_trajectory", [])
    points = [
        (
            datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")),
            float(row["score"]),
        )
        for row in trajectory
        if row.get("timestamp") is not None and row.get("score") is not None
    ]
    if not points:
        return {
            "behavioural_mean_score": None,
            "behavioural_min_score": None,
            "behavioural_area_below_threshold_seconds": None,
        }
    threshold = float(section.get("recovery_score_threshold", 0.8))
    area = 0.0
    for (left_time, left_score), (right_time, right_score) in zip(points, points[1:]):
        elapsed = max(0.0, (right_time - left_time).total_seconds())
        left_deficit = max(0.0, threshold - left_score)
        right_deficit = max(0.0, threshold - right_score)
        area += ((left_deficit + right_deficit) / 2.0) * elapsed
    scores = [score for _, score in points]
    return {
        "behavioural_mean_score": mean(scores),
        "behavioural_min_score": min(scores),
        "behavioural_area_below_threshold_seconds": area,
    }


def _recovery_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        if run.get("status") != "completed":
            continue
        report_path = (
            Path(run["run_directory"])
            / "reports"
            / "evaluation"
            / "role_recovery.json"
        )
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for agent_id, agent in report.get("agents", {}).items():
            for shock_id, shock in agent.get("shocks", {}).items():
                for recovery_type in (
                    "financial_recovery",
                    "balance_sheet_recovery",
                    "behavioural_recovery",
                ):
                    section = shock.get(recovery_type, {})
                    row: dict[str, Any] = {
                        "replication": run["replication"],
                        "run_id": run["run_id"],
                        "agent_id": agent_id,
                        "role": agent.get("role"),
                        "shock_id": shock_id,
                        "recovery_type": recovery_type.removesuffix("_recovery"),
                        "available": bool(section.get("available", False)),
                        "disrupted": bool(section.get("disrupted", False)),
                        "recovered": bool(section.get("recovered", False)),
                        "recovery_status": section.get("recovery_status"),
                        "recovery_time_from_onset_seconds": section.get(
                            "recovery_time_from_onset_seconds"
                        ),
                        "recovery_time_from_expiry_seconds": section.get(
                            "recovery_time_from_expiry_seconds"
                        ),
                    }
                    if recovery_type == "financial_recovery":
                        row.update(
                            {
                                "financial_max_drawdown_from_baseline": section.get(
                                    "maximum_drawdown_from_baseline"
                                ),
                                "financial_end_distance_from_baseline": section.get(
                                    "end_distance_from_baseline"
                                ),
                            }
                        )
                    elif recovery_type == "balance_sheet_recovery":
                        baseline = section.get("baseline_gross_exposure")
                        end = section.get("end_gross_exposure")
                        exposure_deviation = (
                            abs((float(end) / float(baseline)) - 1.0)
                            if baseline not in (None, 0) and end is not None
                            else None
                        )
                        row.update(
                            {
                                "balance_end_exposure_deviation": exposure_deviation,
                                "balance_max_position_limit_utilization": section.get(
                                    "maximum_position_limit_utilization"
                                ),
                                "balance_time_constrained_seconds": section.get(
                                    "time_constrained_seconds"
                                ),
                            }
                        )
                    else:
                        row["behavioural_end_score"] = section.get("end_recovery_score")
                        row.update(_behavioural_trajectory_metrics(section))
                    rows.append(row)
    return rows


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["agent_id"]), str(row["shock_id"]), str(row["recovery_type"]))
        groups.setdefault(key, []).append(row)

    numeric_fields = (
        "recovery_time_from_onset_seconds",
        "recovery_time_from_expiry_seconds",
        "financial_max_drawdown_from_baseline",
        "financial_end_distance_from_baseline",
        "balance_end_exposure_deviation",
        "balance_max_position_limit_utilization",
        "balance_time_constrained_seconds",
        "behavioural_end_score",
        "behavioural_mean_score",
        "behavioural_min_score",
        "behavioural_area_below_threshold_seconds",
    )
    summaries: list[dict[str, Any]] = []
    for (agent_id, shock_id, recovery_type), group in sorted(groups.items()):
        available = [row for row in group if row["available"]]
        disrupted = [row for row in available if row["disrupted"]]
        recovered_disruptions = [row for row in disrupted if row["recovered"]]
        summary: dict[str, Any] = {
            "agent_id": agent_id,
            "role": group[0].get("role"),
            "shock_id": shock_id,
            "recovery_type": recovery_type,
            "replication_count": len(group),
            "available_count": len(available),
            "disrupted_count": len(disrupted),
            "disruption_rate": len(disrupted) / len(available) if available else None,
            "recovered_disruption_count": len(recovered_disruptions),
            "recovery_rate_given_disruption": (
                len(recovered_disruptions) / len(disrupted) if disrupted else None
            ),
        }
        for field_name in numeric_fields:
            values = [
                float(row[field_name])
                for row in group
                if row.get(field_name) is not None
            ]
            if not values:
                continue
            for statistic, value in _numeric_summary(values).items():
                summary[f"{field_name}_{statistic}"] = value
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_aggregate_outputs(
    experiment_dir: Path,
    manifest: dict[str, Any],
) -> list[str]:
    rows = _recovery_rows(manifest)
    if not rows:
        return []
    summaries = _aggregate_rows(rows)
    evaluation_dir = experiment_dir / "evaluation"
    replication_csv = evaluation_dir / "recovery_replications.csv"
    summary_csv = evaluation_dir / "recovery_summary.csv"
    summary_json = evaluation_dir / "recovery_summary.json"
    _write_csv(replication_csv, rows)
    _write_csv(summary_csv, summaries)
    _write_json(
        summary_json,
        {
            "generated_at": _utc_now(),
            "experiment_id": manifest["experiment_id"],
            "replication_rows": len(rows),
            "groups": summaries,
        },
    )
    from aml_sim.evaluation.recovery_experiment_plot import (
        plot_experiment_mean_recovery,
    )

    completed_run_directories = [
        run["run_directory"]
        for run in manifest.get("runs", [])
        if run.get("status") == "completed"
    ]
    plot_paths = plot_experiment_mean_recovery(
        completed_run_directories,
        evaluation_dir,
    )
    return [
        str(path.resolve())
        for path in (replication_csv, summary_csv, summary_json, *plot_paths)
    ]


def _initial_manifest(
    *,
    experiment_id: str,
    scenario_path: Path,
    scenario_hash: str,
    seed_base: int,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "source_scenario": str(scenario_path),
        "source_scenario_sha256": scenario_hash,
        "seed_base": seed_base,
        "seed_formula": "seed_base + (replication - 1) * 100 + seeded_agent_offset",
        "execution": "sequential",
        "note": "Simulator seeds do not make remote LLM responses deterministic.",
        "runs": [],
    }


def _load_or_create_manifest(
    path: Path,
    *,
    experiment_id: str,
    scenario_path: Path,
    scenario_hash: str,
    seed_base: int,
) -> dict[str, Any]:
    if not path.exists():
        return _initial_manifest(
            experiment_id=experiment_id,
            scenario_path=scenario_path,
            scenario_hash=scenario_hash,
            seed_base=seed_base,
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": experiment_id,
        "source_scenario": str(scenario_path),
        "source_scenario_sha256": scenario_hash,
        "seed_base": seed_base,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise ValueError(
            "Existing experiment manifest does not match: " + ", ".join(mismatches)
        )
    return manifest


def _run_record(manifest: dict[str, Any], replication: int) -> dict[str, Any]:
    for record in manifest["runs"]:
        if record.get("replication") == replication:
            return record
    record: dict[str, Any] = {"replication": replication}
    manifest["runs"].append(record)
    manifest["runs"].sort(key=lambda item: item["replication"])
    return record


def main() -> int:
    args = _parse_args()
    if args.replications < 1:
        raise ValueError("--replications must be at least 1")

    scenario_path = args.scenario.expanduser().resolve()
    source = _load_yaml(scenario_path)
    scenario_hash = _scenario_digest(scenario_path)
    experiment_dir = EXPERIMENTS_DIR / args.experiment_id
    generated_scenarios_dir = experiment_dir / "scenarios"
    manifest_path = experiment_dir / "manifest.json"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(experiment_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(experiment_dir / ".cache"))

    manifest = _load_or_create_manifest(
        manifest_path,
        experiment_id=args.experiment_id,
        scenario_path=scenario_path,
        scenario_hash=scenario_hash,
        seed_base=args.seed_base,
    )

    for replication in range(1, args.replications + 1):
        run_id = f"{args.experiment_id}-r{replication:03d}"
        run_dir = RUNS_DIR / run_id
        generated_path = generated_scenarios_dir / f"replication-{replication:03d}.yaml"
        seeded_scenario, seeds = _seed_scenario(
            source,
            experiment_id=args.experiment_id,
            replication=replication,
            seed_base=args.seed_base,
        )
        _write_yaml(generated_path, seeded_scenario)

        record = _run_record(manifest, replication)
        record.update(
            {
                "run_id": run_id,
                "run_directory": str(run_dir.resolve()),
                "generated_scenario": str(generated_path.resolve()),
                "seeds": seeds,
            }
        )

        if _completed(run_dir):
            record.update({"status": "completed", "resumed": True})
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
            print(f"[{replication}/{args.replications}] already complete: {run_id}")
            continue

        if args.dry_run:
            record["status"] = "planned"
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
            print(f"[{replication}/{args.replications}] planned: {generated_path}")
            continue

        record.update({"status": "running", "started_at": _utc_now()})
        manifest["updated_at"] = _utc_now()
        _write_json(manifest_path, manifest)
        print(f"[{replication}/{args.replications}] running: {run_id}")

        try:
            if not run_dir.exists():
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "aml_runner.py"),
                        str(generated_path),
                        "--run-id",
                        run_id,
                        "--reports",
                    ],
                    cwd=ROOT,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"AML runner exited with code {result.returncode}")
            elif not _can_postprocess(run_dir):
                raise RuntimeError(
                    f"Run directory exists but has no agent reports: {run_dir}. "
                    "Use a new experiment id or inspect the incomplete run manually."
                )

            artifacts = _postprocess_run(run_dir)
            record.update(
                {
                    "status": "completed",
                    "completed_at": _utc_now(),
                    "artifacts": artifacts,
                }
            )
            print(f"[{replication}/{args.replications}] completed: {run_id}")
        except KeyboardInterrupt:
            record.update({"status": "interrupted", "finished_at": _utc_now()})
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
            raise
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "error": str(exc),
                }
            )
            print(f"[{replication}/{args.replications}] failed: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                manifest["updated_at"] = _utc_now()
                _write_json(manifest_path, manifest)
                return 1
        finally:
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)

    aggregate_artifacts = _write_aggregate_outputs(experiment_dir, manifest)
    if aggregate_artifacts:
        manifest["aggregate_artifacts"] = aggregate_artifacts
        manifest["updated_at"] = _utc_now()
        _write_json(manifest_path, manifest)

    completed = sum(record.get("status") == "completed" for record in manifest["runs"])
    print(f"Experiment manifest: {manifest_path.resolve()}")
    for artifact in aggregate_artifacts:
        print(f"Aggregate artifact: {artifact}")
    print(f"Completed replications: {completed}/{args.replications}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
