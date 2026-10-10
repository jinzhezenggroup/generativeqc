"""Qualify the square-solve wheel ABI without an installed NVIDIA provider.

Signature: https://docs.nvidia.com/cuda/archive/12.9.0/cublas/index.html#cublas-t-trsmbatched
Enum values: NVIDIA nvmath-python's SideMode and DiagType reference pages.
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
    return [
        "-I",
        str(tmp_path),
        "-I",
        str(ROOT / "src/runtime/nvidia_host_api"),
        "-I",
        str(ROOT / "src"),
    ]


def test_trsm_is_in_automatic_cublas_provider_family() -> None:
    provider = provider_for_symbol("cublasDtrsmBatched")
    assert provider is not None
    assert provider.name == "cublas"
    assert provider.load_name == "libcublas.so.12"


def test_trsm_declaration_and_square_consumer(
    tmp_path: Path, includes: list[str], native_cxx: NativeCxx
) -> None:
    source = tmp_path / "signature.cpp"
    source.write_text(r"""
#include <type_traits>
#include "tensor/cuda_square_linalg.hpp"
// CUDA 12.9 cuBLAS, section 2.7.12: cublas<t>trsmBatched.
using Trsm = cublasStatus_t (*)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
 cublasOperation_t, cublasDiagType_t, int, int, const double*,
 const double* const*, int, double* const*, int, int);
static_assert(std::is_same_v<decltype(&cublasDtrsmBatched), Trsm>);
static_assert(CUBLAS_SIDE_LEFT == 0 && CUBLAS_SIDE_RIGHT == 1);
static_assert(CUBLAS_DIAG_NON_UNIT == 0 && CUBLAS_DIAG_UNIT == 1);
static_assert(sizeof(cublasSideMode_t) == sizeof(int));
static_assert(sizeof(cublasDiagType_t) == sizeof(int));
""")
    native_cxx.compile_object(
        source, tmp_path / "signature.o", args=("-std=c++17", *includes)
    )
    # Compile the exact translation unit that failed in the manylinux wheel.
    native_cxx.compile_object(
        ROOT / "src/solver/cuda/generalized_eigen.cpp",
        tmp_path / "generalized_eigen.o",
        args=("-std=c++20", *includes),
    )


def test_square_trsm_lazy_import_forwards_abi_without_provider_dependency(
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

    # The linker launcher takes one C compiler executable. Keep its generated
    # C/assembly compilations behind the same verified cache as the C++ probes.
    cached_cc = tmp_path / "cached-cc"
    cached_cc.write_text(
        f'#!/bin/sh\nexec {shlex.quote(native_cxx.cache)} {shlex.quote(cc)} "$@"\n'
    )
    cached_cc.chmod(0o755)
    consumer = tmp_path / "consumer.cpp"
    consumer.write_text(r"""
#include "tensor/cuda_square_linalg.hpp"
extern "C" int probe(int right, int transpose) {
  double a[2] = {13, 17}, b[2] = {19, 23};
  double* factors[2] = {a, a+1};
  double* matrices[2] = {b, b+1};
  auto handle = reinterpret_cast<cublasHandle_t>(a);
  auto status = generativeqc::tensor::cuda::square_lower_solve(
      handle, right, transpose, 7, 2, factors, matrices);
  // A non-success status and output sentinels must survive the lazy trampoline.
  return status == 73 && b[0] == 101 + right && b[1] == 103 + transpose ? 0 : 1;
}
""")
    obj = tmp_path / "consumer.o"
    native_cxx.compile_object(consumer, obj, args=("-std=c++17", "-fPIC", *includes))
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

    # Independent mock declarations avoid validating the shim against itself.
    # Numeric enum values, scalar arguments, pointer-table entries, status and
    # writes through both table entries all cross the real generated trampoline.
    provider_source = tmp_path / "provider.cpp"
    provider_source.write_text(r"""
#include <cstdint>
extern "C" std::uint32_t cublasDtrsmBatched(void* h, int side, int uplo, int trans,
 int diag, int m, int n, const double* alpha, const double* const* a, int lda,
 double* const* b, int ldb, int count) {
  if ((side != 0 && side != 1) || uplo != 0 || (trans != 0 && trans != 1) || diag != 0 ||
      m != 7 || n != 7 || lda != 7 || ldb != 7 || count != 2 || !alpha || *alpha != 1 ||
      !a || !b || h != a[0] || a[1] != a[0]+1 || b[1] != b[0]+1 ||
      *a[0] != 13 || *a[1] != 17 || *b[0] != 19 || *b[1] != 23) return 7;
  *b[0] = 101 + side;
  *b[1] = 103 + trans;
  return 73;
}
""")
    available = tmp_path / "available-provider.so"
    native_cxx.build_shared([provider_source], available, compile_args=("-std=c++17",))
    provider = tmp_path / "libcublas.so.12"
    # Keep abort-prone lazy resolution isolated. Load before the provider exists,
    # then exercise both sides/transpositions twice to cover resolved replay.
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
    for right in (0, 1):
        for transpose in (0, 1):
            assert loaded.probe(right, transpose) == 0
"""
    subprocess.run(
        [sys.executable, "-c", script, str(library), str(available), str(provider)],
        env={**os.environ, "LD_LIBRARY_PATH": str(tmp_path)},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
