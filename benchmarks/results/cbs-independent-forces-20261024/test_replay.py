"""Bounded host checks for geometry integrity and observable derivative behavior."""

import copy
import importlib.util
import json
import lzma
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "force_replay", Path(__file__).with_name("replay.py")
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def test_timeout_retained_and_strict_replay_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "C7-h1-plus"
            task.mkdir()
            outcome = task / "outcome.json"
            raw = b'{"exit_code":124,"time_limit_seconds":21600}\n'
            outcome.write_bytes(raw)
            command = [
                sys.executable,
                str(Path(__file__).with_name("replay.py")),
                "--raw",
                str(root),
                "--output",
                str(root / "comparison.json"),
                "--require-pass",
            ]
            process = subprocess.run(command, capture_output=True, check=False)
            self.assertEqual(process.returncode, 1, process.stderr)
            result = json.loads((root / "comparison.json").read_bytes())
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(
                [r["status"] for r in result["molecules"]], ["INCOMPLETE", "FAIL"]
            )
            self.assertEqual(
                result["molecules"][1]["tasks"]["C7-h1-plus"]["outcome"]["exit_code"],
                124,
            )
            self.assertEqual(outcome.read_bytes(), raw)

    def test_exact_inputs(self) -> None:
        inputs = json.loads(
            lzma.decompress(Path(__file__).with_name("inputs.json.xz").read_bytes())
        )
        replay.validate_inputs(inputs)
        for field in ("direction", "atoms_bohr"):
            altered = copy.deepcopy(inputs)
            task = altered["tasks"]["C4-h0-minus"]
            if field == "direction":
                task[field][0][0] += 1e-10
            else:
                task[field][0][1][0] += 1e-10
            with self.assertRaises(AssertionError):
                replay.validate_inputs(altered)

    def test_missing_task_rejected(self) -> None:
        inputs = json.loads(
            lzma.decompress(Path(__file__).with_name("inputs.json.xz").read_bytes())
        )
        del inputs["tasks"]["C7-h1-plus"]
        with self.assertRaises((AssertionError, KeyError)):
            replay.validate_inputs(inputs)

    def test_force_sign_from_independent_quadratic(self) -> None:
        center = np.array([[0.5, -0.2, 0.8], [-0.3, 0.4, 0.1]])
        direction = np.random.default_rng(709).normal(size=center.shape)
        direction /= np.linalg.norm(direction)
        # E=sum(x**2), hence analytic F=-2x. Both prescribed steps must agree.
        for step in (1e-4, 3e-5):
            minus = {"composite": float(np.sum((center - step * direction) ** 2))}
            plus = {"composite": float(np.sum((center + step * direction) ** 2))}
            error = replay.projection_errors(
                {"composite": -2 * center}, direction, minus, plus, step
            )
            self.assertLess(error["composite"]["error_eh_per_bohr"], 1e-10)
            wrong = replay.projection_errors(
                {"composite": 2 * center}, direction, minus, plus, step
            )
            self.assertGreater(
                wrong["composite"]["error_eh_per_bohr"], replay.FORCE_GATE
            )

    def test_nonfinite_refused(self) -> None:
        with self.assertRaises(AssertionError):
            replay.projection_errors(
                {"a": np.zeros((1, 3))},
                [[1, 0, 0]],
                {"a": 0},
                {"a": float("nan")},
                1e-4,
            )


if __name__ == "__main__":
    unittest.main()
