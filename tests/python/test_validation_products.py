"""Independent physical gates and optional BLAS thread-ownership contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("provider,local", [(0, 0), (1, 0), (1, 1)])
@pytest.mark.parametrize("owner", ["eigen_frame", "validation_proposal"])
def test_physical_validation_products(
    tmp_path: Path,
    required_native_cxx: NativeCxx,
    provider: int,
    local: int,
    owner: str,
) -> None:
    """Real validation owners retain physical rejection and scalar fallbacks."""
    sources = [
        ROOT / f"tests/native/test_{owner}.cpp",
        ROOT / "tests/native/fixtures/cpu_linked_provider/validation_gemm.cpp",
        ROOT / "src/scf/reference/linalg.cpp",
        ROOT / "src/scf/solver/eigen_frame.cpp",
        ROOT / "src/scf/solver/proposal_control.cpp",
        ROOT / "src/tensor/cpu_linalg.cpp",
    ]
    binary = required_native_cxx.build_executable(
        sources,
        tmp_path / "validation",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-ffunction-sections",
            "-fdata-sections",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "tests/native/fixtures/cpu_linked_provider"),
            f"-DGENERATIVEQC_HAS_OPENBLAS={provider}",
            f"-DGENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS={local}",
            "-DGENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS=1",
        ],
        link_args=["-Wl,--gc-sections"],
    )
    for setting in (None, "0", "1", "true", "2"):
        env = dict(os.environ)
        env.pop("GENERATIVEQC_SCF_VALIDATION_GEMM", None)
        env.pop("GENERATIVEQC_TEST_VALIDATION_GEMM_CORRUPT", None)
        if setting is not None:
            env["GENERATIVEQC_SCF_VALIDATION_GEMM"] = setting
        subprocess.run(
            [str(binary)], env=env, check=True, capture_output=True, timeout=30
        )
    if provider and local and owner == "eigen_frame":
        env["GENERATIVEQC_SCF_VALIDATION_GEMM"] = "1"
        env["GENERATIVEQC_TEST_VALIDATION_GEMM_CORRUPT"] = "1"
        failure = subprocess.run(
            [str(binary)], check=False, env=env, capture_output=True, timeout=30
        )
        assert failure.returncode == 1
        assert b"physical eigen residual or metric orthogonality" in failure.stderr
