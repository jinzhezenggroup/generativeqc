"""Evidence replay must not silently disable its scientific assertions."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/xc-spatial-bisection-20261023/reproduce.py"
)


class TestReplayOptimization(unittest.TestCase):
    def test_optimized_replay_is_refused(self) -> None:
        for flags, value in ((["-O"], None), (["-OO"], None), ([], "1"), ([], "2")):
            with self.subTest(flags=flags, value=value):
                env = os.environ.copy()
                env.pop("PYTHONOPTIMIZE", None)
                if value is not None:
                    env["PYTHONOPTIMIZE"] = value
                result = subprocess.run(
                    [sys.executable, *flags, str(SCRIPT)],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Evidence validation requires assertions", result.stderr)
                self.assertEqual(result.stdout, "")
