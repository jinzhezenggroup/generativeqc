"""Behavior tests of independent composite admission using an archived sample."""

from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from model_contract import coefficients, equal_coefficient
from replay import qualification

HERE = Path(__file__).parent


class CompositeModelContract(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = json.loads((HERE / "inputs-v2.json").read_bytes())
        plan = json.loads((HERE / "prepared-plan-v2.json").read_bytes())
        self.call = next(
            c
            for c in plan["calls"]
            if c["fixture"] == "archive-water"
            and c["hf_reference"] == "exact"
            and c["geometry_id"] == 0
            and c["policy"] == "standard"
        )
        self.sample = json.loads((HERE / "test-model-fixture.json").read_bytes())[
            "sample"
        ]
        self.sample.update(call=self.call, converged=True)

    def test_archived_correct_result_passes_independent_gates(self) -> None:
        self.assertEqual(
            qualification(self.call, self.sample, self.inputs)["kind"],
            "independent_full_vector",
        )

    def test_returned_coefficients_cannot_control_reference(self) -> None:
        corrupted = copy.deepcopy(self.sample)
        corrupted.update(energy=0.0, forces=[[0.0] * 3 for _ in range(3)])
        for i, c in enumerate(corrupted["components"]):
            c.update(
                coefficient=0.0,
                energy=self.inputs["water_cases"][0]["oracle"]["component_energies"][i],
                forces=self.inputs["water_cases"][0]["oracle"]["component_forces"][i],
            )
        with self.assertRaises(ValueError):
            qualification(self.call, corrupted, self.inputs)

    def test_reordered_components_preserve_semantics(self) -> None:
        self.sample["components"].reverse()
        self.assertEqual(
            qualification(self.call, self.sample, self.inputs)["kind"],
            "independent_full_vector",
        )

    def test_duplicate_role_refuses(self) -> None:
        self.sample["components"][-1]["roles"] = self.sample["components"][0]["roles"]
        with self.assertRaises(ValueError):
            qualification(self.call, self.sample, self.inputs)

    def test_changed_method_or_basis_refuses(self) -> None:
        for field, value in (("method", "rhf"), ("basis_identity", "0" * 64)):
            sample = copy.deepcopy(self.sample)
            sample["components"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualification(self.call, sample, self.inputs)

    def test_changed_extrapolation_or_l1_refuses(self) -> None:
        for field in ("hf_weights", "mp2_weights", "coefficient_l1"):
            sample = copy.deepcopy(self.sample)
            if field == "coefficient_l1":
                sample["convergence"][field] = 0.0
            else:
                sample[field] = [0.0, 0.0]
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualification(self.call, sample, self.inputs)

    def test_positive_and_negative_coefficient_rounding_boundary(self) -> None:
        for expected in (coefficients()[0][2], coefficients()[0][4]):
            for direction in (-math.inf, math.inf):
                value = expected
                for _ in range(4):
                    value = math.nextafter(value, direction)
                self.assertTrue(equal_coefficient(value, expected))
                self.assertFalse(
                    equal_coefficient(math.nextafter(value, direction), expected)
                )

    def test_ambiguous_or_nonfinite_coefficients_refuse(self) -> None:
        expected = coefficients()[0][2]
        self.assertFalse(equal_coefficient(0.0, expected))
        for value in (True, None, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                equal_coefficient(value, expected)


if __name__ == "__main__":
    unittest.main()
