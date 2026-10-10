"""Cold-endpoint CLI validation must precede scientific imports and work."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli_root(tmp_path: Path) -> Path:
    """Run the real entry points with only their stdlib output guard available."""
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    (benchmarks / "__init__.py").touch()
    for name in ("pbe0_df_cold.py", "_retention.py"):
        shutil.copyfile(ROOT / "benchmarks" / name, benchmarks / name)
    return tmp_path


def run_cli(
    root: Path, entry_point: str, output: Path
) -> subprocess.CompletedProcess[str]:
    """Bound the CLI probe and leave real GPU execution unavailable."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("SLURM_JOB_ID", None)
    command = (
        ["-m", "benchmarks.pbe0_df_cold"]
        if entry_point == "module"
        else [str(root / "benchmarks" / "pbe0_df_cold.py")]
    )
    return subprocess.run(
        [
            sys.executable,
            *command,
            "native",
            "--properties",
            "energy",
            "--output",
            str(output),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize("entry_point", ("module", "script"))
@pytest.mark.parametrize("spelling", ("absolute", "relative", "parent", "symlink"))
def test_rejects_retained_output_before_runtime(
    cli_root: Path, entry_point: str, spelling: str
) -> None:
    retained = cli_root / "benchmarks" / "results"
    retained.mkdir()
    reviewed = retained / "reviewed.json"
    original = b"reviewed evidence\n"
    reviewed.write_bytes(original)
    output = reviewed
    if spelling == "relative":
        output = reviewed.relative_to(cli_root)
    elif spelling == "parent":
        output = Path(".artifacts/../benchmarks/results/reviewed.json")
    elif spelling == "symlink":
        alias = cli_root / "alias"
        try:
            alias.symlink_to(retained, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symlink creation unavailable: {error}")
        output = alias / "reviewed.json"
    result = run_cli(cli_root, entry_point, output)
    assert result.returncode == 2, result.stderr
    assert "argument --output: invalid raw_output_path value" in result.stderr
    assert "Traceback" not in result.stderr
    assert reviewed.read_bytes() == original
    assert not (cli_root / ".artifacts").exists()


@pytest.mark.parametrize("entry_point", ("module", "script"))
def test_accepts_scratch_output_before_requiring_slurm(
    cli_root: Path, entry_point: str
) -> None:
    output = cli_root / ".artifacts" / "benchmarks" / "cold.json"
    result = run_cli(cli_root, entry_point, output)
    assert result.returncode == 1, result.stderr
    assert "real GPU measurements require Slurm-assigned visibility" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not output.parent.exists()
