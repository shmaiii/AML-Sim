from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_diversity_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_diversity_matrix", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DiversityMatrixRunnerTests(unittest.TestCase):
    def test_research_scenario_has_one_call_per_openai_agent(self) -> None:
        scenario_path = ROOT / "scenarios" / "diversity" / "d0_seed_104729.yaml"
        with scenario_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        planned, worst_case = MODULE.estimate_api_calls(raw)

        self.assertEqual(6, planned)
        self.assertEqual(6, worst_case)

    def test_interval_parser_supports_research_units(self) -> None:
        self.assertEqual(30.0, MODULE._parse_interval_seconds("30s"))
        self.assertEqual(600.0, MODULE._parse_interval_seconds("10m"))
        self.assertEqual(3600.0, MODULE._parse_interval_seconds("1h"))

    def test_locked_oos_seeds_do_not_overlap_validation(self) -> None:
        protocol_path = ROOT / "experiments" / "diversity_matrix.yaml"
        with protocol_path.open("r", encoding="utf-8") as handle:
            protocol = yaml.safe_load(handle)

        self.assertFalse(
            set(protocol["validation_seeds"])
            & set(protocol["out_of_sample_seeds"])
        )


if __name__ == "__main__":
    unittest.main()
