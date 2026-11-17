"""Retain consumer-owned CLI emission while admitting canonical f/g families."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("derivatives", [False, True])
def test_existing_owner_emission_remains_available(
    tmp_path: Path, derivatives: bool
) -> None:
    output = tmp_path / "owner.cuh"
    argv = [
        sys.executable,
        str(ROOT / "tools/generate_one_electron_kernels.py"),
        "--owner",
        "cosx",
        "--max-angular-momentum",
        "1",
        "--output",
        str(output),
    ]
    if derivatives:
        argv.append("--derivatives")
    subprocess.run(argv, check=True, capture_output=True, timeout=60)
    assert output.stat().st_size > 0


@pytest.mark.parametrize(
    "extra",
    [
        ["--owner", "cosx", "--max-angular-momentum", "4", "--family-output"],
        ["--max-angular-momentum", "3", "--family-output"],
        ["--max-angular-momentum", "4", "--family-metadata-output"],
    ],
)
def test_invalid_family_combinations_fail_before_emission(
    tmp_path: Path, extra: list[str]
) -> None:
    output, family = tmp_path / "value.cuh", tmp_path / "family.cuh"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_one_electron_kernels.py"),
            "--output",
            str(output),
            *extra,
            str(family),
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2
    assert not output.exists()
    assert not family.exists()
