"""Historical qualification must not silently disable its acceptance gates."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECHECK = ROOT / "benchmarks/results/public-schwarz-cold-20261023/recheck.py"


@pytest.mark.parametrize(
    "flags,optimize", [(["-O"], ""), (["-OO"], ""), ([], "1"), ([], "2")]
)
def test_requalification_rejects_disabled_assertions(
    flags: list[str], optimize: str, tmp_path: Path
) -> None:
    """Reject before reading receipts, including environment-selected optimization."""
    environment = dict(
        os.environ,
        PYTHONOPTIMIZE=optimize,
        PYTHONPATH=os.pathsep.join(
            (str(ROOT), str(ROOT / "python"), os.environ.get("PYTHONPATH", ""))
        ),
    )
    result = subprocess.run(
        [sys.executable, *flags, str(RECHECK), "--raw-directory", str(tmp_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "Requalification requires Python assertions" in result.stderr
    assert "FileNotFoundError" not in result.stderr
    assert "complete E/F calls passed" not in result.stdout


def test_requalification_normal_help_remains_available() -> None:
    """Normal Python keeps the documented reader interface."""
    result = subprocess.run(
        [sys.executable, str(RECHECK), "--help"],
        cwd=ROOT,
        env=dict(
            os.environ,
            PYTHONOPTIMIZE="",
            PYTHONPATH=os.pathsep.join(
                (str(ROOT), str(ROOT / "python"), os.environ.get("PYTHONPATH", ""))
            ),
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--raw-directory" in result.stdout
