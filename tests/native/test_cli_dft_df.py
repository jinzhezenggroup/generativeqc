"""Exercise Python-free DFT-DF CLI execution and explicit failure boundaries."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(sys.argv.pop(1)).resolve()
MOLECULE = Path(sys.argv.pop(1)).resolve()
DIRECT_AOT_ENABLED = bool(int(sys.argv.pop(1)))


class NativeDftDfCliTests(unittest.TestCase):
    SMALL_GRID = (
        "--grid-radial-points",
        "8",
        "--grid-polar-points",
        "4",
        "--grid-azimuth-points",
        "8",
    )

    def test_complete_wb97mv_energy_against_independent_pyscf(self) -> None:
        # PySCF 2.14 / Libxc 7; independently reconstructed 8x4x8 product rules,
        # original Becke without radius adjustment, identical XC/VV10 grids,
        # no density pruning, STO-3G H2 at +/-0.7 Bohr. This checks graph
        # completeness, not convergence of the deliberately small quadrature.
        for method, charge, energy in (
            ("wb97m-v-rks", 0, -1.150312466420175),
            ("wb97m-v-uks", 1, -0.528250800668613),
        ):
            with self.subTest(method=method):
                result = self.call_named(
                    method,
                    *self.SMALL_GRID,
                    "--charge",
                    str(charge),
                    "--multiplicity",
                    str(charge + 1),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertAlmostEqual(
                    json.loads(result.stdout)["energy_hartree"], energy, delta=2e-8
                )
        rejected = self.call_named(
            "wb97m-v-rks", *self.SMALL_GRID, "--nonlocal-memory-bytes", "0"
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(rejected.stdout, "")

    def test_complete_dispersion_forces_match_total_energy_differences(self) -> None:
        if not DIRECT_AOT_ENABLED:
            self.skipTest("native direct force derivatives are not linked")
        flags = (
            *self.SMALL_GRID,
            "--energy-tolerance",
            "1e-12",
            "--density-tolerance",
            "1e-10",
        )
        for method in (
            "pbe-d3(bj)-rks",
            "pbe-d3(bj)-atm-rks",
            "pbe0-d3(0)-uks",
            "pbe0-d4(bj-eeq-atm)-rks",
            "b3lyp-d4(bj-eeq-atm)-rks",
        ):
            controls = flags + (
                ("--charge", "1", "--multiplicity", "2")
                if method.endswith("-uks")
                else ()
            )
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                result = self.call_named(method, *controls, "--forces")
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                force = data["forces_hartree_per_bohr"][1][2]
                for step in (3e-4, 1e-4):
                    energies = []
                    for sign in (-1, 1):
                        path = Path(directory) / "moved.xyz"
                        path.write_text(
                            f"2\nH2\nH 0 0 -0.7\nH 0 0 {0.7 + sign * step:.14f}\n"
                        )
                        moved = self.call_xyz(path, method, *controls)
                        self.assertEqual(moved.returncode, 0, moved.stderr)
                        energies.append(json.loads(moved.stdout)["energy_hartree"])
                    self.assertAlmostEqual(
                        force, -(energies[1] - energies[0]) / (2 * step), delta=2e-5
                    )
                refused = self.call_named(
                    method, *controls, "--dispersion-memory-bytes", "1"
                )
                self.assertNotEqual(refused.returncode, 0)
                self.assertEqual(refused.stdout, "")

    def test_bound_r2scan3c_native_basis_and_complete_energy(self) -> None:
        # Independent PySCF/Libxc + dftd4 4.3 + audited gCP reference;
        # exact spherical def2-mTZVPP H2, product grid 8 x 4 x 8.
        for spin, energy in (
            ("rks", -1.1741390564619811),
            ("uks", -0.5709191317293616),
        ):
            flags = ("--charge", "1", "--multiplicity", "2") if spin == "uks" else ()
            command = [
                str(CLI),
                "run",
                str(MOLECULE),
                "--method",
                f"r2scan-3c-{spin}",
                "--backend",
                "cpu",
                "--units",
                "bohr",
                *self.SMALL_GRID,
                "--energy-tolerance",
                "1e-12",
                "--density-tolerance",
                "1e-10",
                *flags,
                "--json",
            ]

            def invoke(
                extra: tuple[str, ...], command: list[str] = command
            ) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [*command, *extra],
                    text=True,
                    capture_output=True,
                    env={
                        **os.environ,
                        "PATH": "/no-python-runtime",
                        "GENERATIVEQC_PROFILE": "off",
                    },
                    timeout=60,
                    check=False,
                )

            for basis in ((), ("--basis", "def2-mTZVPP")):
                result = invoke(basis)
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(result.stdout)
                self.assertAlmostEqual(data["energy_hartree"], energy, delta=2e-8)
                self.assertEqual(data["basis"], "def2-mTZVPP")
                self.assertEqual(data["representation"], "spherical")
                self.assertEqual(
                    data["basis_identity"],
                    "af55e4a3e21efd9d813231d5c5b043429ebc6ca3878d2cb08802b97920719996",
                )
            for rejected in (
                ("--basis", "sto-3g"),
                ("--representation", "cartesian"),
                ("--dispersion-memory-bytes", "1"),
                ("--forces",),
            ):
                result = invoke(rejected)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")

    def call_named(self, method: str, *flags: str) -> subprocess.CompletedProcess[str]:
        return self.call_xyz(MOLECULE, method, *flags)

    def call_xyz(
        self, xyz: Path, method: str, *flags: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(CLI),
                "run",
                str(xyz),
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
            env={
                **os.environ,
                "PATH": "/no-python-runtime",
                "GENERATIVEQC_PROFILE": "off",
            },
            timeout=60,
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
        self.assertTrue(data["wb97m-v-rks"]["cpu"])
        self.assertTrue(data["wb97m-v-rks"]["cuda"])
        for name in ("pbe-d3(bj)-rks", "pbe0-d3(0)-uks", "b3lyp-d4(bj-eeq-atm)-rks"):
            self.assertTrue(data[name]["cpu"])
        self.assertTrue(data["r2scan-3c-rks"]["cpu"])
        self.assertFalse(data["r2scan-3c-rks"]["cuda"])
        self.assertIn("gCP", data["r2scan-3c-rks"]["reason"])

    def test_generated_pbe0_and_b3lyp_energy(self) -> None:
        pbe0 = self.call_named("PBE0-RKS", "--backend", "cpu")
        self.assertEqual(pbe0.returncode, 0, pbe0.stderr)
        data = json.loads(pbe0.stdout)
        self.assertEqual(data["method"], "pbe0-rks")
        self.assertEqual(data["backend"], "cpu_reference")
        # Independent PySCF 2.14 / Libxc 7, H2/STO-3G at +/-0.7 Bohr:
        # level 3 Treutler-Ahlrichs/NWChem, Treutler radii adjustment,
        # original Becke, no density pruning, conv_tol=1e-14.
        self.assertAlmostEqual(data["energy_hartree"], -1.15431095433684, delta=1e-6)
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
        # Preserve the original SDK oracle for the explicit legacy v1 grid.
        self.assertAlmostEqual(
            json.loads(explicit_grid.stdout)["energy_hartree"],
            -1.1543107969377155,
            delta=1e-9,
        )
        explicit_level = self.call_named(
            "pbe0-rks", "--backend", "cpu", "--grid-level", "3"
        )
        self.assertEqual(explicit_level.returncode, 0, explicit_level.stderr)
        self.assertAlmostEqual(
            json.loads(explicit_level.stdout)["energy_hartree"],
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
            ("pbe-rks", ("--grid-radial-points", "64"), "require a generated"),
            ("b3lyp-rks", ("--grid-polar-points", "0"), "must be positive"),
            ("m06-2x-rks", (), "no qualified native lowerer"),
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

    def test_scf_control_overrides_and_rejection(self) -> None:
        flags = ("--backend", "cpu", "--density-fitting", "cpu")
        baseline = self.call(*flags)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        controlled = self.call(
            *flags,
            "--max-iterations",
            "120",
            "--energy-tolerance",
            "1e-12",
            "--density-tolerance",
            "1e-10",
            "--screening-tolerance",
            "1e-14",
        )
        self.assertEqual(controlled.returncode, 0, controlled.stderr)
        self.assertAlmostEqual(
            json.loads(controlled.stdout)["energy_hartree"],
            json.loads(baseline.stdout)["energy_hartree"],
            delta=1e-9,
        )
        for flag, value, reason in (
            ("--max-iterations", "0", "must be positive"),
            ("--max-iterations", "2147483648", "must fit int32"),
            ("--energy-tolerance", "nan", "positive finite number"),
            ("--energy-tolerance", "inf", "positive finite number"),
            ("--density-tolerance", "0", "positive finite number"),
            ("--screening-tolerance", "-1e-12", "positive finite number"),
            ("--screening-tolerance", "1e-14junk", "positive finite number"),
        ):
            with self.subTest(flag=flag, value=value):
                result = self.call(*flags, flag, value)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(reason, result.stderr)

    def test_cpu_pbe0_analytic_force_reconverged_finite_difference(self) -> None:
        for fitting in ("cpu", "none"):
            if fitting == "none" and not DIRECT_AOT_ENABLED:
                continue
            with self.subTest(fitting=fitting):
                self.assert_pbe0_force_reconverged_finite_difference(fitting)

    def test_cpu_uks_analytic_force_reconverged_finite_difference(self) -> None:
        for method in ("pbe-uks", "pbe0-uks"):
            for fitting in ("cpu", "none"):
                if fitting == "none" and not DIRECT_AOT_ENABLED:
                    continue
                with self.subTest(method=method, fitting=fitting):
                    self.assert_pbe0_force_reconverged_finite_difference(
                        fitting, method
                    )

    def test_cpu_b3lyp_analytic_force_reconverged_finite_difference(self) -> None:
        for fitting in ("cpu", "none"):
            if fitting == "none" and not DIRECT_AOT_ENABLED:
                continue
            with self.subTest(fitting=fitting):
                self.assert_pbe0_force_reconverged_finite_difference(
                    fitting, "b3lyp-rks"
                )

    def assert_pbe0_force_reconverged_finite_difference(
        self, fitting: str, method: str = "pbe0-rks"
    ) -> None:
        # This is a complete molecular E+F gate, not the fixed-density XC
        # derivative test. Both displaced geometries must reconverge natively.
        flags = (
            "--backend",
            "cpu",
            "--density-fitting",
            fitting,
            "--max-iterations",
            "200",
            "--energy-tolerance",
            "1e-12",
            "--density-tolerance",
            "1e-10",
        )
        if method.endswith("-uks"):
            flags += ("--charge", "1", "--multiplicity", "2")
        result = self.call_named(method, *flags, "--forces")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(math.isfinite(data["energy_hartree"]))
        self.assertEqual(data["method"], method)
        if method != "pbe-uks":
            self.assertEqual(len(data["method_ir_identity"]), 64)
        force = data["forces_hartree_per_bohr"]
        self.assertEqual(len(force), 2)
        self.assertTrue(
            all(
                len(vector) == 3 and all(math.isfinite(x) for x in vector)
                for vector in force
            )
        )
        for axis in range(3):
            self.assertAlmostEqual(force[0][axis] + force[1][axis], 0.0, delta=2e-5)

        step = 5e-4  # Bohr, matching --units bohr
        original = MOLECULE.read_text(encoding="utf-8").splitlines()
        self.assertEqual(int(original[0]), 2)
        displaced_energies = []
        with tempfile.TemporaryDirectory(
            prefix=f"generativeqc-{method}-force-"
        ) as directory:
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                lines = list(original)
                atom = lines[3].split()
                self.assertEqual(len(atom), 4)
                atom[3] = f"{float(atom[3]) + sign * step:.12f}"
                lines[3] = " ".join(atom)
                geometry = Path(directory) / f"{label}.xyz"
                geometry.write_text("\n".join(lines) + "\n", encoding="utf-8")
                displaced = self.call_xyz(geometry, method, *flags)
                self.assertEqual(displaced.returncode, 0, displaced.stderr)
                displaced_energies.append(
                    json.loads(displaced.stdout)["energy_hartree"]
                )
        fd_force = -(displaced_energies[0] - displaced_energies[1]) / (2 * step)
        self.assertTrue(math.isfinite(fd_force))
        self.assertAlmostEqual(force[1][2], fd_force, delta=2e-3)

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
        ):
            with self.subTest(flags=flags):
                result = self.call(*flags)
                self.assertEqual(result.returncode, expected_code, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(reason, result.stderr)


if __name__ == "__main__":
    unittest.main()
