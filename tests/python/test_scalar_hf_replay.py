"""Run all 34 reviewed replay mocks without importing scientific libraries.

The helper runs under isolated, site-free Python, even when an enclosing pytest
session has already imported NumPy or GenerativeQC. It replaces the calculator,
native library, resource controls, and subprocesses with injected stdlib mocks.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "benchmarks/results/cpu-bounded-scalar-hf-20261003"
PUBLIC_INPUT = ROOT / "benchmarks/results/cpu-larger-systems-20261003"
REPLAY_SHA = "7053d9284a92d16d51871180552683d5531731a76b8d3545213f11988979334d"
INPUT_SHA = "d94b0c325258e97609ad45c23d58a7932140048e0fae8146345d343ba3360a73"
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


class TestScalarHFReplay(unittest.TestCase):
    def test_all_34_reviewed_mock_cases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scalar-hf-replay-") as tmp:
            root = Path(tmp)
            for decoder, flag, output in (
                (CAPSULE / "decode.py", "--output-dir", root / "scalar"),
                (PUBLIC_INPUT / "decode.py", "--output", root / "public"),
            ):
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-B", str(decoder), flag, str(output)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            replay = root / "scalar/fresh_endpoint.py"
            case = root / "public/frozen/inputs/n-pentane.json"
            # Pin the reviewed historical script/input, never current source bytes.
            self.assertEqual(
                hashlib.sha256(replay.read_bytes()).hexdigest(), REPLAY_SHA
            )
            self.assertEqual(hashlib.sha256(case.read_bytes()).hexdigest(), INPUT_SHA)
            helper = root / "test_replay.py"
            shutil.copyfile(ROOT / "tests/helpers/cpu_scalar_replay_mocks.py", helper)
            env = dict(os.environ)
            env.update({key: "1" for key in THREADS})
            env.update(REPLAY_SCRIPT=str(replay), REPLAY_INPUT=str(case))
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", str(helper)],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("Ran 34 tests", output)
            self.assertIn("\nOK", output)


if __name__ == "__main__":
    unittest.main()
