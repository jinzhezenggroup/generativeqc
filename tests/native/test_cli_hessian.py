"""Host-only native CLI admission: never start a worker or initialize a GPU.

Independent molecular and real CUDA publication/lifetime gates live in the
Slurm-only stationary probe and compiled CLI tests, not fake worker replies.
"""

from __future__ import annotations

import json
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
        self.rejected(self.call("--method", "pbe-rks"), "only WB97M-V")
        self.rejected(self.call("--backend", "cuda", "--precision", "mixed"), "fp64")
        self.rejected(
            self.call("--backend", "cuda", "--density-fitting", "cpu"),
            "CPU density-fitting",
        )


if __name__ == "__main__":
    unittest.main()
