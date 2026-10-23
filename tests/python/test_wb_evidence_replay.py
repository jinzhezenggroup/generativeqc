"""Evidence replay must fail closed when Python removes its acceptance checks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/wb-range-reuse-20261023/reproduce.py"
)


@pytest.mark.parametrize(
    "flag,optimize", [("-O", ""), ("-OO", ""), ("", "1"), ("", "2")]
)
def test_optimized_evidence_replay_is_rejected(flag, optimize):
    env = {**os.environ, "PYTHONOPTIMIZE": optimize}
    result = subprocess.run(
        [sys.executable, *([flag] if flag else []), str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "requires assertions" in result.stderr
    assert "native_calls" not in result.stdout
