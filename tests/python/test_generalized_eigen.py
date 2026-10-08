"""Independent CPU numerical and borrowed-binding gates for shared eigen phases."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("probe", ("test_generalized_eigen", "test_cpu_target_eigen"))
def test_cpu_generalized_phases(
    probe: str, tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    binary = required_native_cxx.build_executable(
        [
            ROOT / f"tests/native/{probe}.cpp",
            ROOT / "src/tensor/cpu_linalg.cpp",
            ROOT / "src/tensor/cpu/lp64_provider.cpp",
            ROOT / "src/scf/solver/cpu_target_eigen.cpp",
            ROOT / "src/scf/solver/eigen_frame.cpp",
            ROOT / "src/scf/reference/linalg.cpp",
        ],
        tmp_path / probe,
        compile_args=["-std=c++20", "-O2", "-I" + str(ROOT / "src")],
        link_args=["-ldl"],
    )
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=30)
