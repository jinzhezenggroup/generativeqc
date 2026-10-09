"""Exercise Python-free DFT-DF CLI execution and explicit failure boundaries."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

CLI = Path(sys.argv.pop(1)).resolve()
MOLECULE = Path(sys.argv.pop(1)).resolve()


class NativeDftDfCliTests(unittest.TestCase):
    def call_named(self, method: str, *flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(CLI),
                "run",
                str(MOLECULE),
                "--method",
                method,
                "--basis",
                "sto-3g",
                "--units",
                "bohr",
                *flags,
                "--json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def call(self, *flags: str) -> subprocess.CompletedProcess[str]:
        return self.call_named("pbe-rks", *flags)

    def test_explicit_pbe0_rks_energy_and_fail_closed_force(self) -> None:
        # The reference is the independently qualified native C++ H2/STO-3G
        # PBE0 energy in test_cpp_batch.cpp at +/-0.7 Bohr, not PBE energy.
        result = self.call_named("PBE0-RKS", "--backend", "cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["method"], "pbe0-rks")
        self.assertEqual(data["backend"], "cpu_reference")
        self.assertAlmostEqual(data["energy_hartree"], -1.1543107969377155, delta=1e-6)
        self.assertNotIn("forces_hartree_per_bohr", data)

        explicit_grid = self.call_named(
            "pbe0-rks",
            "--backend",
            "cpu",
            "--pbe0-radial-points",
            "64",
            "--pbe0-polar-points",
            "12",
            "--pbe0-azimuth-points",
            "24",
        )
        self.assertEqual(explicit_grid.returncode, 0, explicit_grid.stderr)
        self.assertAlmostEqual(
            json.loads(explicit_grid.stdout)["energy_hartree"],
            data["energy_hartree"],
            delta=1e-9,
        )
        for method, extra, message in (
            ("pbe0-rks", ("--forces",), "DFT forces are not exposed"),
            (
                "pbe-rks",
                ("--pbe0-radial-points", "64"),
                "only valid with --method pbe0-rks",
            ),
            ("pbe0-uks", (), "native run method must be"),
        ):
            with self.subTest(method=method, extra=extra):
                rejected = self.call_named(method, *extra)
                self.assertEqual(rejected.returncode, 2, rejected.stderr)
                self.assertEqual(rejected.stdout, "")
                self.assertIn(message, rejected.stderr)

    def test_cpu_df_pbe_energy_and_metadata(self) -> None:
        flags = (
            "--backend",
            "cpu",
            "--density-fitting",
            "cpu",
            "--auxiliary-basis",
            "def2-svp",
        )
        first = self.call(*flags)
        self.assertEqual(first.returncode, 0, first.stderr)
        data = json.loads(first.stdout)
        self.assertEqual(data["method"], "pbe-rks")
        self.assertEqual(data["backend"], "cpu_reference")
        self.assertEqual(data["density_fitting"], "cpu")
        self.assertEqual(data["auxiliary_basis"], "def2-svp")
        self.assertTrue(math.isfinite(data["energy_hartree"]))
        self.assertGreater(data["iterations"], 0)
        self.assertNotIn("forces_hartree_per_bohr", data)

        again = self.call(*flags)
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertAlmostEqual(
            json.loads(again.stdout)["energy_hartree"],
            data["energy_hartree"],
            delta=1.0e-9,
        )

    def test_cpu_auto_df_same_orbital_default(self) -> None:
        result = self.call("--backend", "cpu", "--density-fitting", "auto")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["density_fitting"], "auto")
        self.assertEqual(data["auxiliary_basis"], "same-as-orbital")
        self.assertTrue(math.isfinite(data["energy_hartree"]))

    def test_rejected_requests_are_not_silently_downgraded(self) -> None:
        for flags, reason, expected_code in (
            (
                ("--density-fitting", "cuda", "--backend", "cpu"),
                "invalid argument (status 1)",
                1,
            ),
            (("--auxiliary-basis", "def2-svp"), "requires density fitting", 2),
            (("--density-fitting", "cpu", "--forces"), "DFT forces are not exposed", 2),
        ):
            with self.subTest(flags=flags):
                result = self.call(*flags)
                self.assertEqual(result.returncode, expected_code, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(reason, result.stderr)


if __name__ == "__main__":
    unittest.main()
