"""Compiled finite owner retention/close contracts with no device execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def owner_probe(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("provider-owner-retention")
    return required_native_cxx.build_executable(
        [ROOT / "tests/native/test_provider_initialization_owner.cpp"],
        directory / "owner",
        compile_args=[
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=0",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ],
    )


def test_partial_report_is_not_whole_owner_close(owner_probe: Path) -> None:
    result = subprocess.run(
        [str(owner_probe), "partial-report"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "mode",
    [
        "inventory",
        "owner",
        "foreign-owner",
        "failed-close",
        "missing-operation",
        "duplicate-schema",
        "uncertain-close-retry",
        "finalization-throws",
    ],
)
def test_concrete_owner_retention(owner_probe: Path, mode: str) -> None:
    result = subprocess.run(
        [str(owner_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
