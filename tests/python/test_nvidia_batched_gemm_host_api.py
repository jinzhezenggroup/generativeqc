"""Qualify the independent-pair wheel ABI without an NVIDIA provider.

Signature: https://docs.nvidia.com/cuda/archive/12.9.1/cublas/index.html#cublas-t-gemmbatched
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools.link_cuda_implib import link_with_auto_implib, provider_for_symbol

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def includes(tmp_path: Path) -> list[str]:
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t=void*; enum cudaDataType { CUDA_R_32F=0 };\n"
    )
    (tmp_path / "library_types.h").write_text(
        "#pragma once\nenum libraryPropertyType { MAJOR_VERSION=0 };\n"
    )
    return ["-I", str(tmp_path), "-I", str(ROOT / "src/runtime/nvidia_host_api")]


def test_batched_gemm_is_in_automatic_cublas_provider_family() -> None:
    provider = provider_for_symbol("cublasDgemmBatched")
    assert provider is not None
    assert provider.name == "cublas"
    assert provider.load_name == "libcublas.so.12"


@pytest.mark.parametrize("standard", ["c++17", "c++20"])
def test_batched_gemm_declaration(
    tmp_path: Path, includes: list[str], native_cxx: NativeCxx, standard: str
) -> None:
    source = tmp_path / "signature.cpp"
    source.write_text(r"""
#include <type_traits>
#include "cublas_v2.h"
// CUDA 12.9 cuBLAS, section 2.7.3: cublas<t>gemmBatched.
using BatchedGemm = cublasStatus_t (*)(cublasHandle_t, cublasOperation_t,
 cublasOperation_t, int, int, int, const double*, const double* const*, int,
 const double* const*, int, const double*, double* const*, int, int);
static_assert(std::is_same_v<decltype(&cublasDgemmBatched), BatchedGemm>);
""")
    native_cxx.compile_object(
        source, tmp_path / "signature.o", args=(f"-std={standard}", *includes)
    )


def test_independent_pair_call_lazy_import_forwards_abi_without_provider_dependency(
    tmp_path: Path, includes: list[str], native_cxx: NativeCxx
) -> None:
    targets = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    machine = platform.machine().lower()
    cc, readelf = (shutil.which(name) for name in ("cc", "readelf"))
    if platform.system() != "Linux" or machine not in targets or not cc or not readelf:
        pytest.skip("supported Linux ELF target, C compiler and inspector required")

    cached_cc = tmp_path / "cached-cc"
    cached_cc.write_text(
        f'#!/bin/sh\nexec {shlex.quote(native_cxx.cache)} {shlex.quote(cc)} "$@"\n'
    )
    cached_cc.chmod(0o755)
    # Compile the actual invocation and pointer-table type from the CUDA owner.
    # Kernel syntax is outside this host-ABI probe; it does not qualify GPU work.
    owner = (ROOT / "src/tensor/cuda_contraction.cuh").read_text()
    table_start = owner.index("struct IndependentPairPointers {")
    table = owner[table_start : owner.index("};", table_start) + 2]
    call_start = owner.index("generativeqc_tensor::blas_check(cublasDgemmBatched(")
    call = owner[call_start : owner.index("));", call_start) + 3]
    consumer = tmp_path / "consumer.cpp"
    consumer.write_text(
        r"""
#include <cstddef>
#include "cublas_v2.h"
static cublasStatus_t observed_status;
namespace generativeqc_tensor {
// Observe status forwarding only; the production checker is not under test.
void blas_check(cublasStatus_t status) { observed_status = status; }
}
"""
        + table
        + r"""
struct Request {
  std::size_t m=8, n=7, k=9;
  std::size_t leading_dimension(int operand) const { return 11 + operand; }
};
struct Context {
  cublasHandle_t value;
  cublasHandle_t handle() const { return value; }
};
extern "C" int probe(int left_transpose, int right_transpose) {
  double left[2] = {13, 17}, right[2] = {19, 23}, output[2] = {29, 31};
  IndependentPairPointers storage{{left, left+1}, {right, right+1},
                                  {output, output+1}};
  auto* table = &storage;
  Context context{reinterpret_cast<cublasHandle_t>(left)};
  auto* context_ = &context;
  const Request request;
  const auto transpose_left = left_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto transpose_right = right_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  const double alpha=2, beta=-3;
"""
        + call
        + r"""
  return observed_status == 73 && output[0] == 101 + right_transpose &&
         output[1] == 103 + left_transpose && left[0] == 13 && left[1] == 17 &&
         right[0] == 19 && right[1] == 23 ? 0 : 1;
}
"""
    )
    obj = tmp_path / "consumer.o"
    native_cxx.compile_object(consumer, obj, args=("-std=c++20", "-fPIC", *includes))
    library = tmp_path / "consumer.so"
    assert (
        link_with_auto_implib(
            command=[
                native_cxx.compiler,
                "-shared",
                str(obj),
                "-Wl,-z,defs",
                "-ldl",
                "-o",
                str(library),
            ],
            cc=str(cached_cc),
            implib_root=ROOT / "cmake/3rdparty/implib",
            work_dir=tmp_path / "implib",
            target=targets[machine],
        )
        == 0
    )
    dynamic = subprocess.run(
        [readelf, "-d", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert not any(
        name in dynamic for name in ("libcublas", "libcudart", "libcusolver")
    )

    # Independent declarations validate pointer constness, integer widths, the
    # unversioned C symbol, all scalar/table arguments and a non-success return.
    provider_source = tmp_path / "provider.cpp"
    provider_source.write_text(r"""
#include <cstdint>
extern "C" std::uint32_t cublasDgemmBatched(void* h, int ta, int tb,
 int m, int n, int k, const double* alpha, const double* const* a, int lda,
 const double* const* b, int ldb, const double* beta, double* const* c,
 int ldc, int count) {
  if ((ta != 0 && ta != 1) || (tb != 0 && tb != 1) || m != 7 || n != 8 || k != 9 ||
      lda != 12 || ldb != 11 || ldc != 13 || count != 2 ||
      !alpha || *alpha != 2 || !beta || *beta != -3 || !a || !b || !c ||
      h != b[0] || a[1] != a[0]+1 || b[1] != b[0]+1 || c[1] != c[0]+1 ||
      *a[0] != 19 || *a[1] != 23 || *b[0] != 13 || *b[1] != 17 ||
      *c[0] != 29 || *c[1] != 31) return 7;
  *c[0] = 101 + ta;
  *c[1] = 103 + tb;
  return 73;
}
""")
    available = tmp_path / "available-provider.so"
    native_cxx.build_shared([provider_source], available, compile_args=("-std=c++17",))
    provider = tmp_path / "libcublas.so.12"
    script = """
import ctypes
import pathlib
import sys
library, available, provider = map(pathlib.Path, sys.argv[1:])
assert not provider.exists()
loaded = ctypes.CDLL(str(library))
loaded.probe.argtypes = [ctypes.c_int, ctypes.c_int]
loaded.probe.restype = ctypes.c_int
available.rename(provider)
for _ in range(2):
    for left in (0, 1):
        for right in (0, 1):
            assert loaded.probe(left, right) == 0
"""
    subprocess.run(
        [sys.executable, "-c", script, str(library), str(available), str(provider)],
        env={**os.environ, "LD_LIBRARY_PATH": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
