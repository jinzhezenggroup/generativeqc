"""Execute provider admission and one-way observation without a CUDA device."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("standard", ["c++17", "c++20"])
def test_prepared_admission_and_observations(
    tmp_path: Path, required_native_cxx: NativeCxx, standard: str
) -> None:
    """Context refusal and diagnostic mutation cannot promote an execution route."""
    executable = required_native_cxx.build_executable(
        [ROOT / "tests/native/test_prepared_contraction_admission.cpp"],
        tmp_path / "prepared-admission",
        compile_args=(
            f"-std={standard}",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic-errors",
            "-I" + str(ROOT / "src"),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=15)
