from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from aml_sim.agents.models.profile import (
    InformedProfile,
    InstitutionalProfile,
    LiquidityTakerProfile,
    MarketMakerProfile,
    RetailProfile,
    coerce_profile,
)
from aml_sim.agents.profile_effects import (
    informed_profile_effects,
    institutional_profile_effects,
    liquidity_taker_profile_effects,
    market_maker_profile_effects,
    retail_profile_effects,
)
from aml_sim.experiments.diversity_matrix import build_condition
from aml_sim.launcher import _make_seed
from aml_sim.scenario import load_scenario


class ProfileAndConfigTests(unittest.TestCase):
    def test_profile_formula_is_explicit_and_bounded(self) -> None:
        self.assertEqual(
            {
                "participation_multiplier": 1.0,
                "herding_multiplier": 1.0,
                "panic_multiplier": 1.0,
                "order_size_multiplier": 1.0,
            },
            retail_profile_effects(RetailProfile()),
        )
        self.assertGreater(
            market_maker_profile_effects({"inventory_discipline": 1.0})[
                "inventory_skew_multiplier"
            ],
            market_maker_profile_effects({"inventory_discipline": 0.0})[
                "inventory_skew_multiplier"
            ],
        )

    def test_default_profiles_preserve_latest_main_behavior(self) -> None:
        effect_sets = [
            market_maker_profile_effects(MarketMakerProfile()),
            retail_profile_effects(RetailProfile()),
            institutional_profile_effects(InstitutionalProfile(), urgency=0.5),
            informed_profile_effects(InformedProfile()),
            liquidity_taker_profile_effects(LiquidityTakerProfile()),
        ]
        for effects in effect_sets:
            with self.subTest(effects=effects):
                self.assertTrue(all(value == 1.0 for value in effects.values()))

    def test_urgency_changes_institutional_child_size(self) -> None:
        profile = {"execution_patience": 0.5, "market_impact_aversion": 0.5}
        low = institutional_profile_effects(profile, urgency=0.0)["child_size_multiplier"]
        high = institutional_profile_effects(profile, urgency=1.0)["child_size_multiplier"]
        self.assertLess(low, high)

    def test_unknown_or_invalid_profile_fields_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown RetailProfile field"):
            coerce_profile({"herding_tendancy": 0.5}, RetailProfile)
        with self.assertRaisesRegex(ValueError, "must be in"):
            coerce_profile({"herding_tendency": 1.1}, RetailProfile)

    def test_scenario_rejects_unknown_yaml_fields(self) -> None:
        payload = {
            "name": "bad",
            "unknown_root": True,
            "stocksim_config": {
                "exchange_mode": "orderbook",
                "instruments": ["AAPL"],
                "exchanges": {"AAPL": {}},
                "agents": {},
                "simulation": {},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown_root"):
                load_scenario(path)

    def test_seed_is_cross_process_stable(self) -> None:
        self.assertEqual(_make_seed("104729", "retail_1"), _make_seed("104729", "retail_1"))
        self.assertNotEqual(_make_seed("104729", "retail_1"), _make_seed("104729", "retail_2"))

    def test_d0_d4_matrix_changes_only_declared_dimensions(self) -> None:
        base_path = Path("scenarios/aml_diversity_api_10min.yaml")
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        d0 = build_condition(base, "D0", 104729)
        d4 = build_condition(base, "D4", 104729)
        d0_agents = d0["stocksim_config"]["agents"]
        d4_agents = d4["stocksim_config"]["agents"]
        self.assertEqual(4, sum(name.startswith("retail_") for name in d0_agents))
        d0_herding = {
            d0_agents[f"retail_{index}"]["parameters"]["herding_tendency"]
            for index in range(1, 5)
        }
        d4_herding = {
            d4_agents[f"retail_{index}"]["parameters"]["herding_tendency"]
            for index in range(1, 5)
        }
        self.assertEqual(1, len(d0_herding))
        self.assertGreater(len(d4_herding), 1)
