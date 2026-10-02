"""Check CLI basis/units/spin routing against pinned independent HF references."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(sys.argv.pop(1)).resolve()
REFERENCES = Path(__file__).resolve().parents[1] / "reference_data/validation"
BOHR_PER_ANGSTROM = 1.8897261254578281


class GaussianCliTests(unittest.TestCase):
    def test_pinned_hf_energy_and_force_references(self) -> None:
        # These repository fixtures retain independent PySCF inputs and provenance.
        for name in ("h2", "h2o", "hf-plus-uhf"):
            fixture = json.loads((REFERENCES / f"{name}.json").read_text())
            inputs = fixture["inputs"]
            for units in ("bohr", "angstrom"):
                with (
                    self.subTest(name=name, units=units),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    scale = 1.0 if units == "bohr" else 1.0 / BOHR_PER_ANGSTROM
                    path = Path(tmp) / "molecule.xyz"
                    lines = [str(len(inputs["atomic_numbers"])), "pinned reference"]
                    lines.extend(
                        f"{number} "
                        + " ".join(f"{value * scale:.17g}" for value in position)
                        for number, position in zip(
                            inputs["atomic_numbers"], inputs["coordinates"], strict=True
                        )
                    )
                    path.write_text("\n".join(lines) + "\n")
                    result = subprocess.run(
                        [
                            str(CLI),
                            "run",
                            str(path),
                            "--method",
                            inputs["method"],
                            "--basis",
                            inputs["basis_name"].upper().replace("-", "_"),
                            "--units",
                            units,
                            "--charge",
                            str(inputs["charge"]),
                            "--multiplicity",
                            str(inputs["multiplicity"]),
                            "--forces",
                            "--json",
                        ],
                        env={
                            **os.environ,
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                        },
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    actual = json.loads(result.stdout)
                    self.assertEqual(actual["basis"], inputs["basis_name"])
                    self.assertEqual(actual["backend"], "cpu_reference")
                    self.assertAlmostEqual(
                        actual["energy_hartree"], fixture["data"]["energy"], delta=1e-8
                    )
                    for actual_atom, expected_atom in zip(
                        actual["forces_hartree_per_bohr"],
                        fixture["data"]["forces"],
                        strict=True,
                    ):
                        for actual_value, expected_value in zip(
                            actual_atom, expected_atom, strict=True
                        ):
                            self.assertAlmostEqual(
                                actual_value, expected_value, delta=1e-7
                            )

    def test_invalid_gaussian_flags_fail_before_input(self) -> None:
        for flags, message in (
            (["--method", "gfn2-xtb", "--basis", "sto-3g"], "intrinsic basis"),
            (
                ["--method", "gfn2-xtb", "--representation", "spherical"],
                "intrinsic basis",
            ),
            (["--method", "rhf", "--representation", "bad"], "representation"),
            (["--method", "uhf", "--multiplicity", "0"], "positive"),
            (["--method", "rhf", "--charge", "2147483648"], "int32"),
            (["--method", "rhf", "--units", "bad"], "units"),
        ):
            with self.subTest(flags=flags):
                result = subprocess.run(
                    [str(CLI), "run", "nonexistent.xyz", *flags, "--json"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
