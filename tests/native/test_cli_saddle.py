"""Python-free native molecular task admission, results and failure receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CLI = Path(sys.argv.pop(1)).resolve()
HOST = os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN", "1") == "1"


class SaddleCli(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="gqc-native-saddle-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.xyz = self.root / "Input With Spaces.xyz"
        self.xyz.write_text("2\nBohr\nH 0 0 0\nH 0 0 1.5\n")
        self.masses = self.root / "Masses With Spaces.json"
        self.masses.write_text("[1.00782503223, 1.00782503223]")

    def call(
        self,
        *arguments: str,
        masses: bool = True,
        timeout: int = 60,
        journal: Path | None = None,
    ) -> subprocess.CompletedProcess:
        command = [str(CLI), "saddle", str(self.xyz), "--units", "bohr", "--json"]
        if masses:
            command += ["--masses", str(self.masses)]
        return subprocess.run(
            [*command, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=dict(
                os.environ,
                PATH="/nonexistent",
                PYTHONHOME="/nonexistent",
                PYTHONPATH="/nonexistent",
                CXX="/nonexistent",
                CC="/nonexistent",
                **({"GENERATIVEQC_DF_PROGRESS_TRACE": str(journal)} if journal else {}),
            ),
        )

    def scf_journal(self, path: Path, spin: str) -> list[dict]:
        """Check completed scopes without prescribing internal evaluation counts."""
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        starts = {row["id"]: row for row in rows if row["event"] == "BEGIN"}
        ends = {row["id"]: row for row in rows if row["event"] == "END"}
        self.assertTrue(starts)
        self.assertEqual(starts.keys(), ends.keys())
        for identity, start in starts.items():
            self.assertEqual(ends[identity]["name"], start["name"])
            self.assertGreaterEqual(ends[identity]["time_ns"], start["time_ns"])
        scopes = [row for row in starts.values() if row["name"] == f"{spin}_scf"]
        self.assertTrue(scopes)
        self.assertTrue(
            any(row["name"] == f"{spin}_physical_evaluation" for row in starts.values())
        )
        observations = []
        for scope in scopes:
            values = {
                row["key"]: row["value"]
                for row in rows
                if row["id"] == scope["id"] and row["event"] == "VALUE"
            }
            self.assertGreater(values["basis_dimension"], 0)
            self.assertGreater(values["grid_points"], 0)
            self.assertGreaterEqual(
                values["terminal_fock_builds"], values["terminal_iterations"]
            )
            self.assertEqual(ends[scope["id"]]["status"], "returned")
            observations.append(values)
        return observations

    def rejected(self, result: subprocess.CompletedProcess, message: str) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)

    def test_help(self) -> None:
        result = subprocess.run(
            [str(CLI), "saddle", "--help"], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in (
            "--masses",
            "--max-attempts",
            "unscaled",
            "no Python/JIT or preoptimization",
        ):
            self.assertIn(text, result.stdout)

    def test_explicit_masses_and_attempt_bound(self) -> None:
        self.rejected(self.call(masses=False), "requires --masses")
        for value in ("0", "101", "-1", "1.5", "18446744073709551616"):
            with self.subTest(value=value):
                self.rejected(
                    self.call("--max-attempts", value), "positive bounded integer"
                )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_invalid_masses(self) -> None:
        for value in ("[true, 1]", "[1]", "[1, -1]", "[1, 0]", '[1, "2"]'):
            with self.subTest(value=value):
                self.masses.write_text(value)
                self.rejected(self.call(), "one positive finite dalton mass")

    def test_no_implicit_backend_or_method_switch(self) -> None:
        for arguments in (
            ("--backend", "cuda"),
            ("--method", "wb97m-v-rks"),
            ("--method", "rhf"),
        ):
            with self.subTest(arguments=arguments):
                self.rejected(
                    self.call(*arguments), "requires a linked native CPU B3LYP"
                )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_unqualified_ecp_is_refused_before_any_molecular_state(self) -> None:
        """An admission-only potential must never reach an all-electron frame."""
        potential = [
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [0],
                "r_exponents": [2],
                "gaussian_exponents": ["1"],
                "coefficients": [["0.3"]],
            }
        ]
        record = {
            "schema": "generativeqc.basis",
            "schema_version": 1,
            "name": "synthetic-He-ECP-saddle-refusal",
            "representation": "spherical",
            "exponent_units": "bohr^-2",
            "normalization": "normalized-primitives; native-normalized-contractions",
            "ordering": "CCA-cartesian; libcint-real-spherical",
            "provenance": {
                "source": "synthetic admission control, not numerical qualification",
                "version": "1",
                "license": "CC0",
                "checksum": "c" * 64,
            },
            "elements": [
                {
                    "atomic_number": 2,
                    "nuclear_charge": 2,
                    "ecp_core_electrons": 1,
                    "ecp_data": json.dumps(
                        potential, sort_keys=True, separators=(",", ":")
                    ),
                    "shells": [
                        {
                            "angular_momentum": 0,
                            "source_group": 0,
                            "exponents": ["1E0"],
                            "coefficients": [["1E0"]],
                        }
                    ],
                }
            ],
        }
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record["checksum"] = hashlib.sha256(payload.encode()).hexdigest()
        basis = self.root / "Unqualified ECP With Spaces.json"
        basis.write_text(json.dumps(record))
        self.xyz.write_text("2\nBohr\nHe 0 0 0\nHe 0 0 1.5\n")
        self.masses.write_text("[4.00260325413, 4.00260325413]")
        for method, charge, multiplicity in (
            ("b3lyp-rks", "0", "1"),
            ("b3lyp-d3(bj)-uks", "1", "2"),
        ):
            with self.subTest(method=method):
                self.rejected(
                    self.call(
                        "--basis",
                        str(basis),
                        "--method",
                        method,
                        "--charge",
                        charge,
                        "--multiplicity",
                        multiplicity,
                        "--max-iterations",
                        "1",
                        "--progress-json",
                    ),
                    "ECP requires separate real-element qualification",
                )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_direct_fp64_and_resource_admission(self) -> None:
        self.rejected(self.call("--precision", "fp32"), "--precision must be")
        self.rejected(self.call("--precision", "auto"), "requires direct FP64")
        self.rejected(self.call("--output-budget-bytes", "1"), "result panels exceed")
        self.rejected(
            self.call("--saddle-budget-bytes", "1"), "argument/resource refusal"
        )
        self.rejected(self.call("--max-coordinates", "3"), "exceeds --max-coordinates")
        self.rejected(self.call("--device-budget-bytes", "1"), "does not use CUDA")

    def test_no_preoptimization_or_worker_controls(self) -> None:
        for option in ("--stages", "--preopt-basis", "--compiler-python", "--cache"):
            with self.subTest(option=option):
                self.rejected(self.call(option, "/nonexistent"), "option")
        self.rejected(
            self.call("--initial-radius", "0.01", "--minimum-radius", "0.1"),
            "invalid saddle trust",
        )
        self.rejected(
            self.call("--minimum-occupied-overlap", "1.01"), "overlap controls"
        )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_nonnegative_target_retains_native_state_without_frequencies(self) -> None:
        journal = self.root / "rks-scf.jsonl"
        result = self.call("--progress-json", journal=journal)
        self.assertEqual(result.returncode, 1, result.stderr)
        panel = json.loads(result.stdout)
        self.assertFalse(panel["converged"])
        receipt = panel["receipt"]
        self.assertEqual(receipt["status"], "target_not_negative")
        self.assertTrue(receipt["has_valid_state"])
        self.assertFalse(receipt["stationary_verified"])
        self.assertEqual(receipt["attempts"], 0)
        self.assertEqual(panel["coordinates_bohr"], [[0, 0, 0], [0, 0, 1.5]])
        self.assertEqual(len(panel["raw_hessian_hartree_per_bohr2"]), 6)
        self.assertIsNone(panel["harmonics"])
        completed = next(
            event
            for event in panel["evaluation_records"]
            if event["stage"] == "energy_gradient" and event["status"] == "complete"
        )
        self.assertEqual(len(completed["occupied_counts"]), 1)
        self.assertEqual(len(completed["minimum_occupied_overlaps_by_spin"]), 1)
        self.assertEqual(
            completed["occupied_overlap_coordinate_frame"],
            "mass-centroid-translation-registered",
        )
        self.assertEqual(completed["root_registration_translation_bohr"], [0, 0, 0])
        self.assertGreater(panel["work"]["scf_iterations"], 0)
        self.assertGreaterEqual(
            panel["work"]["scf_fock_builds"], panel["work"]["scf_iterations"]
        )
        observations = self.scf_journal(journal, "rks")
        self.assertTrue(all(row["terminal_converged"] for row in observations))
        self.assertEqual(
            sum(row["terminal_iterations"] for row in observations),
            panel["work"]["scf_iterations"],
        )
        self.assertEqual(
            sum(row["terminal_fock_builds"] for row in observations),
            panel["work"]["scf_fock_builds"],
        )
        diagnostics = panel["diagnostics"]
        self.assertEqual(diagnostics["numerical_runtime"], "native-cpp-cpu")
        self.assertFalse(diagnostics["runtime_compilation"])
        self.assertEqual(diagnostics["preoptimization_stages"], 0)
        events = [json.loads(line) for line in result.stderr.splitlines()]
        for stage in ("system_prepare", "batch_prepare", "scf_forces"):
            started = next(
                event
                for event in events
                if event["stage"] == stage and event["status"] == "start"
            )
            completed = next(
                event
                for event in events
                if event["stage"] == stage and event["status"] == "complete"
            )
            self.assertGreaterEqual(
                completed["elapsed_seconds"], started["elapsed_seconds"]
            )
            self.assertGreaterEqual(completed["phase_seconds"], 0)
        self.assertTrue(any(event["stage"] == "optimizer" for event in events))
        self.assertTrue(
            any(
                event["stage"] == "hessian" and event["status"] == "complete"
                for event in events
            )
        )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_derivative_refusal_is_not_a_successful_state(self) -> None:
        result = self.call("--integral-budget-bytes", "1")
        self.assertEqual(result.returncode, 1, result.stderr)
        panel = json.loads(result.stdout)
        self.assertEqual(panel["receipt"]["status"], "hessian_failed")
        self.assertFalse(panel["receipt"]["has_valid_state"])
        self.assertFalse(panel["converged"])
        self.assertIsNone(panel["raw_hessian_hartree_per_bohr2"])
        self.assertEqual(panel["coordinates_bohr"], [[0, 0, 0], [0, 0, 1.5]])
        self.assertTrue(panel["derivative_failure"])
        self.assertEqual(panel["evaluation_records"][-1]["status"], "failed")

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_root_and_unconverged_scf_refusals_retain_no_fake_hessian(self) -> None:
        for arguments in (("--root-budget-bytes", "1"), ("--max-iterations", "1")):
            with self.subTest(arguments=arguments):
                journal = self.root / f"failure-{arguments[0]}.jsonl"
                result = self.call(*arguments, journal=journal)
                self.assertEqual(result.returncode, 1, result.stderr)
                panel = json.loads(result.stdout)
                self.assertEqual(panel["receipt"]["status"], "evaluation_failed")
                self.assertFalse(panel["receipt"]["has_valid_state"])
                self.assertFalse(panel["converged"])
                self.assertIsNone(panel["raw_hessian_hartree_per_bohr2"])
                self.assertTrue(panel["derivative_failure"])
                if arguments[0] == "--max-iterations":
                    observations = self.scf_journal(journal, "rks")
                    self.assertFalse(
                        any(row["terminal_converged"] for row in observations)
                    )
                    self.assertTrue(
                        all(row["terminal_iterations"] <= 1 for row in observations)
                    )

    @unittest.skipUnless(HOST, "native CPU Hessian closure is disabled")
    def test_uks_native_corrected_total(self) -> None:
        for method in ("b3lyp-uks", "b3lyp-d3(bj)-uks"):
            with self.subTest(method=method):
                journal = self.root / f"{method}.jsonl"
                result = self.call("--method", method, journal=journal)
                self.assertEqual(result.returncode, 1, result.stderr)
                panel = json.loads(result.stdout)
                self.assertEqual(panel["receipt"]["status"], "target_not_negative")
                self.assertEqual(panel["method"], method)
                self.assertEqual(panel["multiplicity"], 1)
                observations = self.scf_journal(journal, "uks")
                self.assertTrue(all(row["terminal_converged"] for row in observations))
                self.assertEqual(
                    sum(row["terminal_iterations"] for row in observations),
                    panel["work"]["scf_iterations"],
                )
                self.assertEqual(
                    sum(row["terminal_fock_builds"] for row in observations),
                    panel["work"]["scf_fock_builds"],
                )
                event = next(
                    event
                    for event in panel["evaluation_records"]
                    if event["stage"] == "energy_gradient"
                    and event["status"] == "complete"
                )
                self.assertEqual(event["occupied_counts"], [1, 1])
                self.assertEqual(event["minimum_occupied_overlaps_by_spin"], [1, 1])
                if "d3" in method:
                    self.assertGreater(panel["work"]["d3_pair_visits"], 0)
                else:
                    self.assertEqual(panel["work"]["d3_pair_visits"], 0)

    @unittest.skipUnless(
        HOST and os.environ.get("GENERATIVEQC_TEST_SADDLE_ENDPOINT") == "1",
        "opt-in complete NH3 endpoint, not a fast CLI contract test",
    )
    def test_complete_nh3_endpoint_and_independent_mass_weighting(self) -> None:
        import numpy as np
        from generativeqc.vibrations import analyze_harmonic_modes

        specification = json.loads(
            (
                Path(__file__).resolve().parents[1] / "data/native_saddle/manifest.json"
            ).read_text()
        )
        radius = specification["cases"]["saddle"]["hydrogen_radius_bohr"]
        height = specification["cases"]["saddle"]["nitrogen_z_bohr"]
        off_axis = radius * math.sqrt(3) / 2
        self.xyz.write_text(
            f"4\nBohr\nN 0 0 {height}\nH {radius} 0 0\n"
            f"H {-radius / 2} {off_axis} 0\nH {-radius / 2} {-off_axis} 0\n"
        )
        self.masses.write_text(json.dumps(specification["masses_dalton"]))
        result = self.call(
            "--representation",
            "spherical",
            "--response-rtol",
            "1e-11",
            "--raw-symmetry-tolerance",
            str(specification["controls"]["raw_symmetry_tolerance"]),
            timeout=900,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        panel = json.loads(result.stdout)
        self.assertTrue(panel["converged"])
        receipt = panel["receipt"]
        self.assertTrue(receipt["stationary_verified"])
        self.assertEqual(receipt["negative_curvatures"], 1)
        self.assertLessEqual(receipt["attempts"], 100)
        self.assertLessEqual(receipt["accepted_steps"], receipt["attempts"])
        for field in (
            "gradient_rms",
            "gradient_max",
            "energy_change",
            "displacement_rms_angstrom",
            "displacement_max_angstrom",
        ):
            self.assertLessEqual(receipt[field], specification["controls"][field])
        matrix = np.asarray(panel["raw_hessian_hartree_per_bohr2"])
        modes = analyze_harmonic_modes(
            matrix,
            coordinates=panel["coordinates_bohr"],
            masses=panel["masses_dalton"],
            hessian_units="hartree/bohr^2",
            coordinate_units="bohr",
            mass_units="dalton",
            symmetrize=True,
            raw_symmetry_tolerance=specification["controls"]["raw_symmetry_tolerance"],
        )
        self.assertEqual(modes.negative_mode_count, 1)
        self.assertFalse(panel["harmonics"]["scaled"])
        np.testing.assert_allclose(
            panel["harmonics"]["frequencies_cm1"],
            modes.frequencies_cm1,
            rtol=0,
            atol=1e-7,
        )
        self.assertEqual(panel["harmonics"]["rigid_mode_count"], modes.rigid_mode_count)
        for name in ("scf_iterations", "scf_fock_builds"):
            completed = (
                event
                for event in panel["evaluation_records"]
                if event["stage"] == "energy_gradient" and event["status"] == "complete"
            )
            self.assertEqual(
                panel["work"][name], sum(event[name] for event in completed)
            )


if __name__ == "__main__":
    unittest.main()
