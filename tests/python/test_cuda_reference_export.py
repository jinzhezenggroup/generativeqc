"""Execute the production retention guard and exceptional export lifetime on CPU.

Only CUDA transport and unreachable numerical validation are substituted; the
production export header and its typed CUDA error boundary are compiled intact.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_reference_export_host_contract(tmp_path: Path, native_cxx: NativeCxx) -> None:
    (tmp_path / "cuda_runtime_api.h").write_text(
        """#pragma once
#include <cstddef>
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess, cudaErrorMemoryAllocation, cudaErrorInvalidValue };
enum cudaMemcpyKind { cudaMemcpyDeviceToHost };
inline const char* cudaGetErrorString(cudaError_t status) {
  return status == cudaErrorMemoryAllocation ? "physical CUDA allocation failure"
                                           : "CUDA transport failure";
}
""",
        encoding="utf-8",
    )
    validation = tmp_path / "validation.cpp"
    validation.write_text(
        """#include "scf/mean_field.hpp"
#include <stdexcept>
namespace generativeqc::scf {
void validate_physical_reference(PhysicalReference&) {
  throw std::logic_error("fault-injected transport must not reach numerical validation");
}
}
""",
        encoding="utf-8",
    )
    try:
        executable = native_cxx.build_executable(
            [ROOT / "tests/native/test_cuda_reference_export.cpp", validation],
            tmp_path / "reference_export",
            compile_args=[
                "-std=c++20",
                "-O2",
                f"-I{tmp_path}",
                f"-I{ROOT / 'src'}",
                f"-I{ROOT / 'include'}",
            ],
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stdout + error.stderr, pytrace=False)
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
