"""Exercise the host/container cache ownership boundary without privilege."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_archive_preparation_is_scoped_and_preserves_private_modes(
    tmp_path: Path, kind: str
) -> None:
    runner = tmp_path / "runner temp"
    runner.mkdir()
    cache = runner / "stationary-aot-source-cache"
    outside = tmp_path / "unrelated"
    outside.mkdir(mode=0o700)
    if kind == "directory":
        cache.mkdir(mode=0o700)
        (cache / "outside-link").symlink_to(outside, target_is_directory=True)
    elif kind == "symlink":
        cache.symlink_to(outside, target_is_directory=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.json"
    sudo = bin_dir / "sudo"
    sudo.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CALLS']).write_text(json.dumps(sys.argv[1:]))\n"
        "sys.exit(subprocess.call(sys.argv[1:]))\n"
    )
    sudo.chmod(0o755)
    subprocess.run(
        ["bash", str(ROOT / ".github/scripts/prepare-stationary-source-cache.sh")],
        env={
            **os.environ,
            "RUNNER_TEMP": str(runner),
            "CALLS": str(calls),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        check=True,
    )
    assert outside.stat().st_mode & 0o777 == 0o700
    if kind == "directory":
        arguments = json.loads(calls.read_text())
        assert arguments[-1] == str(cache)
        assert "--no-dereference" in arguments and "-P" in arguments
        assert cache.stat().st_mode & 0o777 == 0o700
    else:
        assert not calls.exists()
