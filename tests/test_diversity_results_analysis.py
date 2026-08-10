from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "analyze_diversity_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_diversity_results", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def metric_row(level: str, seed: int, values: tuple[float, ...]) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": f"diversity_{level.lower()}_seed_{seed}",
        "diversity_level": level,
        "seed": seed,
        "quality_pass": True,
    }
    row.update(dict(zip(MODULE.METRIC_FIELDS, values)))
    return row


class DiversityResultsAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "validation_seeds": [11, 22],
            "validation_thresholds": {
                "minimum_herding_reduction_relative_to_d0": 0.15,
                "minimum_absolute_correlation_reduction": 0.10,
                "maximum_relative_spread_increase": 0.10,
                "maximum_relative_depth_decrease": 0.10,
                "maximum_trade_active_tick_rate_decrease": 0.05,
            },
        }
        self.d0 = (0.8, 0.6, 0.5, 20.0, 1000.0, 0.50)
        self.d1 = (0.6, 0.45, 0.35, 21.0, 950.0, 0.47)

    def test_partial_validation_is_exploratory_and_selects_nothing(self) -> None:
        rows = [
            metric_row(level, 11, self.d0 if level == "D0" else self.d1)
            for level in MODULE.LEVELS
        ]

        summary, metadata = MODULE.build_level_summary(rows, self.protocol, [11])

        self.assertEqual("exploratory_only", metadata["analysis_status"])
        self.assertFalse(metadata["selection_permitted"])
        self.assertIsNone(metadata["selected_level"])
        self.assertTrue(
            all(
                item["threshold_status"] == "exploratory_only"
                for item in summary
                if item["diversity_level"] != "D0"
            )
        )

    def test_comparisons_follow_frozen_formulas(self) -> None:
        rows = []
        for seed in self.protocol["validation_seeds"]:
            rows.extend(
                metric_row(level, seed, self.d0 if level == "D0" else self.d1)
                for level in MODULE.LEVELS
            )

        summary, metadata = MODULE.build_level_summary(
            rows, self.protocol, self.protocol["validation_seeds"]
        )
        d1 = summary[1]

        self.assertAlmostEqual(0.25, d1["herding_reduction_relative_to_d0"])
        self.assertAlmostEqual(0.15, d1["within_role_correlation_reduction"])
        self.assertAlmostEqual(0.15, d1["cross_role_correlation_reduction"])
        self.assertAlmostEqual(0.05, d1["relative_spread_increase_vs_d0"])
        self.assertAlmostEqual(0.05, d1["relative_depth_decrease_vs_d0"])
        self.assertAlmostEqual(0.03, d1["trade_active_tick_rate_decrease_vs_d0"])
        self.assertTrue(d1["eligible"])
        self.assertEqual("D1", metadata["selected_level"])

    def test_unlocked_seed_is_rejected_before_any_output_is_read(self) -> None:
        with self.assertRaisesRegex(ValueError, "not locked validation seeds"):
            MODULE.analyze(
                ROOT / "experiments" / "diversity_matrix.yaml",
                ROOT / ".aml_runs",
                ROOT / ".aml_runs" / "unused-test-output",
                ["D0"],
                [999999],
                True,
            )


if __name__ == "__main__":
    unittest.main()
