"""Qualify the native potential's rank-2k wheel ABI without a CUDA provider."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _symbols() -> list[str]:
    cmake = (ROOT / "cmake/GenerativeQCCudaImplib.cmake").read_text()
    return cmake.split("set(GENERATIVEQC_CUBLAS_SYMBOLS", 1)[1].split(")", 1)[0].split()


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=30
    )


@pytest.fixture
def toolchain(tmp_path: Path) -> tuple[str, str, str, list[str]]:
    cc, cxx, cache = (shutil.which(name) for name in ("cc", "c++", "ccache"))
    if cc is None or cxx is None or cache is None:
        pytest.skip("host C/C++ compilers and ccache required")
    _run(cache, "--version")
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaStream_t=void*; enum cudaDataType { CUDA_R_32F=0 };\n"
    )
    (tmp_path / "library_types.h").write_text(
        "#pragma once\nenum libraryPropertyType { MAJOR_VERSION=0 };\n"
    )
    includes = ["-I", str(tmp_path), "-I", str(ROOT / "src/runtime/nvidia_host_api")]
    return cc, cxx, cache, includes


def test_rank2k_is_in_curated_wheel_imports() -> None:
    assert "cublasDsyr2k_v2" in _symbols()


def test_rank2k_declaration_and_alias(
    tmp_path: Path, toolchain: tuple[str, str, str, list[str]]
) -> None:
    _, cxx, cache, includes = toolchain
    source = tmp_path / "signature.cpp"
    source.write_text(r"""
#include <type_traits>
#include "cublas_v2.h"
using Rank2k = cublasStatus_t (*)(cublasHandle_t,cublasFillMode_t,cublasOperation_t,
 int,int,const double*,const double*,int,const double*,int,const double*,double*,int);
static_assert(std::is_same_v<decltype(&cublasDsyr2k), Rank2k>);
static_assert(std::is_same_v<decltype(&cublasDsyr2k_v2), Rank2k>);
""")
    _run(
        cache,
        cxx,
        "-std=c++17",
        *includes,
        "-c",
        str(source),
        "-o",
        str(tmp_path / "abi.o"),
    )


def test_rank2k_lazy_import_forwards_all_arguments_without_provider_dependency(
    tmp_path: Path, toolchain: tuple[str, str, str, list[str]]
) -> None:
    from tools.generate_cuda_implib import generate

    targets = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    machine = platform.machine().lower()
    readelf = shutil.which("readelf")
    if platform.system() != "Linux" or machine not in targets or readelf is None:
        pytest.skip("supported Linux ELF target and inspector required")
    cc, cxx, cache, includes = toolchain
    provider = tmp_path / "mock-provider.so"
    generate(
        "libcublas.so",
        _symbols(),
        str(provider),
        targets[machine],
        ROOT / "cmake/3rdparty/implib",
        tmp_path,
    )
    objects = []
    for suffix in ("init.c", "tramp.S"):
        source, obj = tmp_path / f"libcublas.so.{suffix}", tmp_path / f"{suffix}.o"
        _run(cache, cc, "-fPIC", "-c", str(source), "-o", str(obj))
        objects.append(str(obj))
    consumer = tmp_path / "consumer.cpp"
    consumer.write_text(r"""
#include "cublas_v2.h"
extern "C" int probe() {
  double alpha=2, beta=3, a=4, b=5, c=6;
  auto h=reinterpret_cast<cublasHandle_t>(&a);
  auto status=cublasDsyr2k(h,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,7,9,
                          &alpha,&a,11,&b,12,&beta,&c,13);
  if(status!=CUBLAS_STATUS_SUCCESS || c!=98) return 1;
  status=cublasDsyr2k(h,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,7,9,
                     &alpha,&a,11,&b,12,&beta,&c,13);
  return status==CUBLAS_STATUS_SUCCESS && c==374 ? 0 : 2;
}
""")
    consumer_object, library = tmp_path / "consumer.o", tmp_path / "consumer.so"
    _run(
        cache,
        cxx,
        "-std=c++17",
        "-fPIC",
        *includes,
        "-c",
        str(consumer),
        "-o",
        str(consumer_object),
    )
    _run(
        cxx,
        "-shared",
        str(consumer_object),
        *objects,
        "-Wl,-z,defs",
        "-ldl",
        "-o",
        str(library),
    )
    dynamic = _run(readelf, "-d", str(library)).stdout
    assert not any(
        name in dynamic for name in ("libcublas", "libcudart", "libcusolver")
    )
    fake = tmp_path / "provider.cpp"
    fake.write_text(r"""
#include "cublas_v2.h"
extern "C" cublasStatus_t cublasDsyr2k_v2(cublasHandle_t h,cublasFillMode_t uplo,
  cublasOperation_t trans,int n,int k,const double* alpha,const double* a,int lda,
  const double* b,int ldb,const double* beta,double* c,int ldc) {
  if(h!=reinterpret_cast<cublasHandle_t>(const_cast<double*>(a)) ||
     uplo!=CUBLAS_FILL_MODE_LOWER || trans!=CUBLAS_OP_N || n!=7 || k!=9 ||
     lda!=11 || ldb!=12 || ldc!=13 || *alpha!=2 || *beta!=3 || *a!=4 || *b!=5 || !c)
    return CUBLAS_STATUS_INVALID_VALUE;
  *c=*alpha * (*a * *b + *b * *a) + *beta * *c;
  return CUBLAS_STATUS_SUCCESS;
}
""")
    provider_object, available = (
        tmp_path / "provider.o",
        tmp_path / "available-provider.so",
    )
    _run(
        cache,
        cxx,
        "-std=c++17",
        "-fPIC",
        *includes,
        "-c",
        str(fake),
        "-o",
        str(provider_object),
    )
    _run(cxx, "-shared", str(provider_object), "-o", str(available))
    _run(
        sys.executable,
        "-c",
        """
import ctypes
import pathlib
import sys
library, available, provider = map(pathlib.Path, sys.argv[1:])
assert not provider.exists()
loaded = ctypes.CDLL(str(library))
loaded.probe.argtypes = []
loaded.probe.restype = ctypes.c_int
available.rename(provider)
assert loaded.probe() == 0
assert loaded.probe() == 0
""",
        str(library),
        str(available),
        str(provider),
    )
