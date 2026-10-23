"""Execute installed-dependency cache identity without requiring Homebrew."""

from __future__ import annotations

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/cumetal-cuda.yml"


def fingerprint(tmp_path: Path, inventory: str, exit_code: int = 0) -> tuple[int, str]:
    step = (
        WORKFLOW.read_text()
        .split("      - name: Fingerprint installed CuMetal dependencies\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    brew = tmp_path / "brew"
    brew.write_text('#!/bin/sh\nprintf "%s\\n" "$INVENTORY"\nexit "$BREW_EXIT"\n')
    brew.chmod(0o755)
    output = tmp_path / "output"
    output.write_text("")
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "INVENTORY": inventory,
            "BREW_EXIT": str(exit_code),
            "GITHUB_OUTPUT": str(output),
        },
        check=False,
    )
    return result.returncode, output.read_text()


def test_package_update_invalidates_cache(tmp_path: Path) -> None:
    old = fingerprint(tmp_path, "llvm 21.1.8\nz3 5.1.0")[1]
    new = fingerprint(tmp_path, "llvm 21.1.8\nz3 5.3.1")[1]
    assert old != new
    assert re.fullmatch(r"fingerprint=[0-9a-f]{64}\n", new)
    assert new == fingerprint(tmp_path, "z3 5.3.1\nllvm 21.1.8")[1]


@pytest.mark.parametrize("inventory,exit_code", [("", 0), ("z3 5.3.1", 1)])
def test_missing_inventory_fails_closed(
    tmp_path: Path, inventory: str, exit_code: int
) -> None:
    code, output = fingerprint(tmp_path, inventory, exit_code)
    assert code != 0
    assert not output


def test_immutable_toolchain_has_no_cross_dependency_fallback() -> None:
    step = WORKFLOW.read_text().split("      - name: Restore CuMetal toolchain\n", 1)[1]
    step = step.split("      - name:", 1)[0]
    assert "${{ steps.cumetal_dependencies.outputs.fingerprint }}" in step
    assert "restore-keys:" not in step
