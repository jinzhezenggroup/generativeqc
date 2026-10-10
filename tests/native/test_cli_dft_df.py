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

    def test_generated_method_catalog_backend_and_correction_bounds(self) -> None:
        listing = subprocess.run(
            [str(CLI), "methods", "--compositions", "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(listing.returncode, 0, listing.stderr)
        data = {entry["name"]: entry for entry in json.loads(listing.stdout)}
        for name in ("pbe0-rks", "pbe0-uks", "b3lyp-rks", "b3lyp-uks"):
            with self.subTest(method=name):
                self.assertTrue(data[name]["cpu"])
                self.assertTrue(data[name]["cuda"])
                self.assertFalse(data[name]["reason"])
        self.assertFalse(data["m06-2x-rks"]["cpu"])
        self.assertTrue(data["m06-2x-rks"]["cuda"])
        self.assertFalse(data["wb97m-v-rks"]["cpu"])
        self.assertFalse(data["wb97m-v-rks"]["cuda"])
        self.assertIn("nonlocal", data["wb97m-v-rks"]["reason"])
        self.assertIn("correction", data["r2scan-3c-rks"]["reason"])

    def test_generated_pbe0_and_b3lyp_energy(self) -> None:
        pbe0 = self.call_named("PBE0-RKS", "--backend", "cpu")
        self.assertEqual(pbe0.returncode, 0, pbe0.stderr)
        data = json.loads(pbe0.stdout)
        self.assertEqual(data["method"], "pbe0-rks")
        self.assertEqual(data["backend"], "cpu_reference")
        # Independent C++ native SDK H2/STO-3G reference at +/-0.7 Bohr.
        self.assertAlmostEqual(data["energy_hartree"], -1.1543107969377155, delta=1e-6)
        self.assertEqual(len(data["method_ir_identity"]), 64)
        self.assertNotIn("forces_hartree_per_bohr", data)

        explicit_grid = self.call_named(
            "pbe0-rks",
            "--backend",
            "cpu",
            "--grid-radial-points",
            "64",
            "--grid-polar-points",
            "12",
            "--grid-azimuth-points",
            "24",
        )
        self.assertEqual(explicit_grid.returncode, 0, explicit_grid.stderr)
        self.assertAlmostEqual(
            json.loads(explicit_grid.stdout)["energy_hartree"],
            data["energy_hartree"],
            delta=1e-9,
        )

        b3lyp = self.call_named("B3LYP-RKS", "--backend", "cpu")
        self.assertEqual(b3lyp.returncode, 0, b3lyp.stderr)
        b3lyp_data = json.loads(b3lyp.stdout)
        self.assertEqual(b3lyp_data["method"], "b3lyp-rks")
        self.assertTrue(math.isfinite(b3lyp_data["energy_hartree"]))
        self.assertNotEqual(
            b3lyp_data["method_ir_identity"], data["method_ir_identity"]
        )

    def test_generated_methods_reject_unsupported_graphs_and_forces(self) -> None:
        for method, flags, reason in (
            ("pbe0-rks", ("--forces",), "DFT forces are not exposed"),
            ("pbe-rks", ("--grid-radial-points", "64"), "require a generated"),
            ("b3lyp-rks", ("--grid-polar-points", "0"), "must be positive"),
            ("m06-2x-rks", (), "no qualified native lowerer"),
            ("wb97m-v-rks", (), "nonlocal"),
            ("r2scan-3c-rks", (), "correction"),
            ("r2scan0-rks", (), "no qualified native semilocal"),
            ("invented-ks-rks", (), "unknown native or compiler"),
        ):
            with self.subTest(method=method, flags=flags):
                rejected = self.call_named(method, *flags)
                self.assertEqual(rejected.returncode, 2, rejected.stderr)
                self.assertEqual(rejected.stdout, "")
                self.assertIn(reason, rejected.stderr)

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

    def test_cpu_df_pbe_analytic_forces(self) -> None:
        result = self.call("--backend", "cpu", "--density-fitting", "cpu", "--forces")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(math.isfinite(data["energy_hartree"]))
        self.assertEqual(data["density_fitting"], "cpu")
        force = data["forces_hartree_per_bohr"]
        self.assertEqual(len(force), 2)
        for vector in force:
            self.assertEqual(len(vector), 3)
            self.assertTrue(all(math.isfinite(component) for component in vector))
        for axis in range(3):
            self.assertAlmostEqual(force[0][axis] + force[1][axis], 0.0, delta=2e-5)

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
            (("--density-fitting", "none", "--forces"), "DFT forces are not exposed", 2),
        ):
            with self.subTest(flags=flags):
                result = self.call(*flags)
                self.assertEqual(result.returncode, expected_code, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(reason, result.stderr)


if __name__ == "__main__":
    unittest.main()
