"""Host-only failure injection for retained CBS reference admission."""

import copy
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "admission", Path(__file__).with_name("admission.py")
)
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)


class ReferenceAdmission(unittest.TestCase):
    def oracle(self) -> dict:
        rows = [
            {"atom": i, "axis": j, "energies_minus_plus": [[1.0] * 5, [1.0] * 5]}
            for i in range(3)
            for j in range(3)
        ]
        return {
            "component_energies": [1.0] * 5,
            "component_forces": [[[0.0] * 3 for _ in range(3)] for _ in range(5)],
            "step_bohr": 3e-5,
            "displaced_energies": rows,
        }

    def test_full_displaced_vector(self) -> None:
        self.assertEqual(
            admission.reconstruct(self.oracle()), self.oracle()["component_forces"]
        )

    def test_missing_coordinate_fails(self) -> None:
        o = self.oracle()
        o["displaced_energies"].pop()
        with self.assertRaises(ValueError):
            admission.reconstruct(o)

    def test_duplicate_coordinate_fails(self) -> None:
        o = self.oracle()
        o["displaced_energies"][-1] = copy.deepcopy(o["displaced_energies"][0])
        with self.assertRaises(ValueError):
            admission.reconstruct(o)

    def test_nonfinite_energy_fails(self) -> None:
        o = self.oracle()
        o["displaced_energies"][0]["energies_minus_plus"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            admission.reconstruct(o)

    def test_stored_vector_must_match_raw(self) -> None:
        o = self.oracle()
        o["component_forces"][0][0][0] = 1e-12
        with self.assertRaises(ValueError):
            admission.reconstruct(o)

    def test_shape_cannot_truncate(self) -> None:
        o = self.oracle()
        o["displaced_energies"][0]["energies_minus_plus"][0].pop()
        with self.assertRaises(ValueError):
            admission.reconstruct(o)

    def test_strict_archive_gate(self) -> None:
        with self.assertRaises(ValueError):
            admission.compare(
                1.0,
                [[0.0, 0.0, 0.0]],
                0.0,
                [[0.0, 0.0, 0.0]],
                {"energy_abs": 1.0, "force_max_abs": 1.0, "force_rms": 1.0},
            )

    def test_conventional_inclusive_original_gate(self) -> None:
        errors = admission.compare(
            1.0,
            [[0.0, 0.0, 0.0]],
            0.0,
            [[0.0, 0.0, 0.0]],
            {"energy_abs": 1.0, "force_max_abs": 1.0, "force_rms": 1.0},
            strict=False,
        )
        self.assertEqual(errors["energy_abs"], 1.0)

    def test_no_nonfinite_or_missing_forces(self) -> None:
        for forces in (None, [[float("nan"), 0.0, 0.0]], [[0.0, 0.0]]):
            with self.assertRaises(ValueError):
                admission.compare(
                    0.0,
                    forces,
                    0.0,
                    [[0.0, 0.0, 0.0]],
                    {"energy_abs": 1.0, "force_max_abs": 1.0, "force_rms": 1.0},
                )


if __name__ == "__main__":
    unittest.main()
