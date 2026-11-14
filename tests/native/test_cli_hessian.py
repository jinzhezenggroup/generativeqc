"""Host-only native CLI admission: never start a worker or initialize a GPU.

Independent molecular and real CUDA publication/lifetime gates live in the
Slurm-only stationary probe and compiled CLI tests, not fake worker replies.
"""

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


class HessianCli(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="gqc-native-hessian-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.xyz = self.root / "Input With Spaces.xyz"
        self.xyz.write_text("2\nBohr\nH 0.03 -0.02 0.01\nH 0.15 0.13 1.5\n")
        self.direction = self.root / "Direction With Spaces.json"
        self.direction.write_text(json.dumps([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]]))

    def call(
        self, *arguments: str, operation: str = "hvp"
    ) -> subprocess.CompletedProcess:
        command = [str(CLI), operation, str(self.xyz), "--units", "bohr", "--json"]
        if operation == "hvp":
            command += ["--direction", str(self.direction)]
        return subprocess.run(
            [*command, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            env=dict(
                os.environ,
                PATH="/nonexistent",
                PYTHONHOME="/nonexistent",
                PYTHONPATH="/nonexistent",
            ),
        )

    def rejected(self, result: subprocess.CompletedProcess, message: str) -> None:
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(message, result.stderr)

    def test_help_documents_linked_native_execution(self) -> None:
        result = subprocess.run(
            [str(CLI), "hessian", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no Python/JIT", result.stdout)
        self.assertIn("--device-budget-bytes", result.stdout)
        self.assertNotIn("--compiler-python", result.stdout)

    def test_no_cpu_or_implicit_python_fallback(self) -> None:
        self.rejected(self.call(), "requires --backend cuda")

    def test_obsolete_worker_controls_cannot_execute_any_interpreter(self) -> None:
        worker = self.root / "Fake Worker"
        marker = self.root / "was-run"
        worker.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
        worker.chmod(0o700)
        for option in (
            "--compiler-python",
            "--compiler-library",
            "--cache",
            "--timeout-seconds",
        ):
            with self.subTest(option=option):
                self.rejected(
                    self.call(option, str(worker)),
                    "no Python worker or runtime compilation",
                )
                self.assertFalse(marker.exists())

    def test_invalid_response_and_resource_controls(self) -> None:
        for option, value in (
            ("--response-rtol", "1"),
            ("--response-atol", "nan"),
            ("--response-restart", "0"),
            ("--response-max-iterations", "-1"),
            ("--device-budget-bytes", "0"),
            ("--max-nonlocal-pairs", "18446744073709551616"),
        ):
            with self.subTest(option=option):
                self.rejected(self.call(option, value), option)

    def test_full_output_budget_fails_before_device_initialization(self) -> None:
        self.rejected(
            self.call(
                "--backend", "cuda", "--output-budget-bytes", "1", operation="hessian"
            ),
            "--output-budget-bytes",
        )

    def test_direction_validation_fails_before_device_initialization(self) -> None:
        for value in ([[1, 2, 3]], [[1, 2], [3, 4]], [["nan", 0, 0], [0, 0, 0]]):
            self.direction.write_text(json.dumps(value))
            self.rejected(self.call("--backend", "cuda"), "finite natoms-by-3")

    def test_only_hvp_accepts_direction(self) -> None:
        self.rejected(
            self.call("--direction", str(self.direction), operation="hessian"),
            "only hvp",
        )

    def test_wrong_method_precision_and_provider_never_touch_cuda(self) -> None:
        self.rejected(
            self.call("--method", "pbe-rks"), "linked CPU MethodIR or CUDA WB97M-V"
        )
        self.rejected(self.call("--backend", "cuda", "--precision", "mixed"), "fp64")
        self.rejected(
            self.call("--backend", "cuda", "--density-fitting", "cpu"),
            "CPU density-fitting",
        )

    def test_cpu_full_matrix_hvp_and_total_composition_without_python(self) -> None:
        """Run the actual binary with no interpreter path, not a worker fixture."""
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
        self.xyz.write_text("3\nBohr\nH 0 .1 -1.6\nH .3 -.2 0\nH -.1 .4 1.6\n")
        self.direction.write_text(
            json.dumps([[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]])
        )
        for suffix, multiplicity, charge in (("rks", "1", "1"), ("uks", "2", "0")):
            with self.subTest(spin=suffix):
                controls = (
                    "--backend",
                    "cpu",
                    "--representation",
                    "spherical",
                    "--charge",
                    charge,
                    "--multiplicity",
                    multiplicity,
                    "--grid-radial-points",
                    "8",
                    "--grid-polar-points",
                    "4",
                    "--grid-azimuth-points",
                    "8",
                )
                method = f"b3lyp-d3(bj)-{suffix}"
                matrix_call = self.call(
                    "--method", method, *controls, operation="hessian"
                )
                self.assertEqual(matrix_call.returncode, 0, matrix_call.stderr)
                matrix = json.loads(matrix_call.stdout)
                action_call = self.call("--method", method, *controls)
                self.assertEqual(action_call.returncode, 0, action_call.stderr)
                action = json.loads(action_call.stdout)
                vector = [
                    component
                    for atom in json.loads(self.direction.read_text())
                    for component in atom
                ]
                flattened = [
                    component for atom in action["value"] for component in atom
                ]
                for row, actual in zip(matrix["matrix"], flattened, strict=True):
                    expected = sum(
                        first * second
                        for first, second in zip(row, vector, strict=True)
                    )
                    self.assertTrue(math.isfinite(actual))
                    self.assertAlmostEqual(actual, expected, delta=2e-8)
                self.assertEqual(matrix["backend"], "cpu")
                diagnostic = matrix["diagnostics"]
                self.assertFalse(diagnostic["runtime_compilation"])
                self.assertFalse(diagnostic["posthoc_symmetrization"])
                self.assertLess(diagnostic["raw_symmetry_error"], 1e-7)
                self.assertGreater(diagnostic["d3_pair_visits"], 0)
                self.assertGreater(diagnostic["first_one_electron_primitives"], 0)
                self.assertGreater(diagnostic["curvature_one_electron_primitives"], 0)
                self.assertGreater(diagnostic["source_xc_point_visits"], 0)
                self.assertGreater(diagnostic["curvature_xc_point_visits"], 0)
                self.assertGreater(diagnostic["response_xc_point_visits"], 0)
                self.assertEqual(diagnostic["ecp_kernel_blocks"], 0)
                self.assertEqual(diagnostic["max_eri_primitives"], 100_000_000)
                self.assertEqual(
                    diagnostic["d3_pair_visits"],
                    9 * action["diagnostics"]["d3_pair_visits"],
                )
                bare_call = self.call(
                    "--method", f"b3lyp-{suffix}", *controls, operation="hessian"
                )
                self.assertEqual(bare_call.returncode, 0, bare_call.stderr)
                bare = json.loads(bare_call.stdout)
                self.assertEqual(bare["diagnostics"]["d3_pair_visits"], 0)
                self.assertAlmostEqual(
                    matrix["energy_hartree"] - bare["energy_hartree"],
                    diagnostic["d3_energy"],
                    delta=2e-12,
                )
                self.assertLess(diagnostic["d3_energy"], 0)

    def test_cpu_work_refusal_publishes_no_prefix_or_electronic_only_result(
        self,
    ) -> None:
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
        result = self.call(
            "--backend",
            "cpu",
            "--method",
            "b3lyp-d3(bj)-rks",
            "--grid-radial-points",
            "8",
            "--grid-polar-points",
            "4",
            "--grid-azimuth-points",
            "8",
            "--max-d3-pairs",
            "1",
        )
        self.rejected(result, "D3 pair-work cap")

    def test_cpu_ecp_reaches_bounded_native_owner_without_a_python_worker(self) -> None:
        """Synthetic admission/resource control, not real-element qualification."""
        if os.environ.get("GENERATIVEQC_TEST_HOST_HESSIAN") == "0":
            self.skipTest("this build explicitly omits the CPU Hessian AOT closure")
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
            "name": "synthetic-He-ECP-admission",
            "representation": "spherical",
            "exponent_units": "bohr^-2",
            "normalization": "normalized-primitives; native-normalized-contractions",
            "ordering": "CCA-cartesian; libcint-real-spherical",
            "provenance": {
                "source": "synthetic admission control",
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
        basis = self.root / "ECP With Spaces.json"
        basis.write_text(json.dumps(record))
        self.xyz.write_text("1\nBohr\nHe 0 0 0\n")
        self.direction.write_text("[[0.02,-0.03,0.1]]")
        for endpoint in ("hessian", "hvp"):
            with self.subTest(endpoint=endpoint):
                self.rejected(
                    self.call(
                        "--backend",
                        "cpu",
                        "--method",
                        "b3lyp-uks",
                        "--multiplicity",
                        "2",
                        "--basis",
                        str(basis),
                        "--grid-radial-points",
                        "8",
                        "--grid-polar-points",
                        "4",
                        "--grid-azimuth-points",
                        "8",
                        "--numeric-budget-bytes",
                        "1",
                        operation=endpoint,
                    ),
                    "CPU Hessian aggregate numerical reservation exceeded",
                )


if __name__ == "__main__":
    unittest.main()
