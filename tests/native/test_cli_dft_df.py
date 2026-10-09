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
    def call(self, *flags: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(CLI),
                "run",
                str(MOLECULE),
                "--method",
                "pbe-rks",
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
                "DFT density-fitting backend must match",
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
