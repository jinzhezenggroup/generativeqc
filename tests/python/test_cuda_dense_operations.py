"""Compile the actual prepared dense provider with independent host vendors."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cpp_source_support import cpp_function_definition, cpp_record_definition

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

HEADER = r"""
#pragma once
#include <cstdint>
struct Handle { void* stream{}; int pointer_mode{71}; };
using cublasHandle_t = Handle*;
using cublasStatus_t = int;
using cudaStream_t = void*;
enum cublasOperation_t { CUBLAS_OP_N, CUBLAS_OP_T };
enum cublasFillMode_t { CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER };
constexpr int CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_INVALID_VALUE=7;
constexpr int CUBLAS_POINTER_MODE_HOST=0;
int cublasCreate(cublasHandle_t*);
int cublasDestroy(cublasHandle_t);
int cublasSetStream(cublasHandle_t,cudaStream_t);
int cublasSetPointerMode(cublasHandle_t,int);
int cublasDcopy(cublasHandle_t,int,const double*,int,double*,int);
int cublasDscal(cublasHandle_t,int,const double*,double*,int);
int cublasDaxpy(cublasHandle_t,int,const double*,const double*,int,double*,int);
int cublasDdot(cublasHandle_t,int,const double*,int,const double*,int,double*);
int cublasDnrm2(cublasHandle_t,int,const double*,int,double*);
int cublasDgemm(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,
               const double*,const double*,int,const double*,int,const double*,double*,int);
int cublasDgemmStridedBatched(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,
               const double*,const double*,int,long long,const double*,int,long long,
               const double*,double*,int,long long,int);
int cublasDgemv(cublasHandle_t,cublasOperation_t,int,int,const double*,const double*,int,
               const double*,int,const double*,double*,int);
#ifdef GENERATIVEQC_TEST_HAS_GEAM
int cublasDgeam(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,const double*,
               const double*,int,const double*,const double*,int,double*,int);
#endif
"""


@pytest.mark.parametrize("standard", ["c++17", "c++20"])
@pytest.mark.parametrize("has_geam", [False, True])
def test_actual_dense_provider_contract(
    tmp_path: Path, required_native_cxx: NativeCxx, standard: str, has_geam: bool
) -> None:
    """Check ABI/status forwarding, partial preparation and optional-GEAM numerics."""
    (tmp_path / "cublas_v2.h").write_text(HEADER)
    source = (ROOT / "src/tensor/cuda_runtime.cuh").read_text()
    first = cpp_record_definition(source, "ColumnGemmView")
    last = cpp_function_definition(source, "submit_column_geam", include_template=True)
    start, end = source.index(first), source.index(last) + len(last)
    (tmp_path / "tensor").mkdir()
    (tmp_path / "tensor/cuda_runtime.cuh").write_text(
        "#include <type_traits>\n#include <utility>\n"
        + "namespace generativeqc_tensor {\n"
        + source[start:end]
        + "\n}\n"
    )
    executable = required_native_cxx.build_executable(
        [
            ROOT / "src/tensor/cuda_dense_operations.cu",
            ROOT / "tests/native/test_cuda_dense_operations.cpp",
        ],
        tmp_path / "dense-provider",
        compile_args=(
            "-x",
            "c++",
            f"-std={standard}",
            "-O1",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic-errors",
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
            *(("-DGENERATIVEQC_TEST_HAS_GEAM",) if has_geam else ()),
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=15)
