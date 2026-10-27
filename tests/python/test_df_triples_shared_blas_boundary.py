"""Execute the triples FP64 submission at its shared Tensor provider boundary.

The HOST provider captures the real wrapper/adapter and evaluates its physical
column-major views. CUDA numerical and complete E+F qualification stay separate.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/cc/df_triples_cuda.cu"
VENDOR_INVENTORY = ROOT / "manifests/maintenance/vendor_boundaries.json"


def _definition(source: str, marker: str) -> str:
    """Extract the current production definition, never a replacement body."""
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        depth += (source[position] == "{") - (source[position] == "}")
        if depth == 0:
            return source[start : position + 1]
    raise AssertionError(f"unterminated production definition: {marker}")


@pytest.mark.parametrize("mutation", (None, "transpose", "stride", "order", "status"))
def test_triples_column_gemm_executes_forwarding_and_error_contract(
    tmp_path: Path, required_native_cxx: NativeCxx, mutation: str | None
) -> None:
    runtime = (ROOT / "src/tensor/cuda_runtime.cuh").read_text()
    wrapper = _definition(
        SOURCE.read_text(), "inline void execute_strided_fp64_product("
    )
    if mutation is not None:
        before, after = {
            "transpose": (".transpose_left = ta != 'N'", ".transpose_left = ta == 'N'"),
            "stride": (
                ".left_ld = static_cast<int>(lda)",
                ".left_ld = static_cast<int>(ldb)",
            ),
            "order": (".rows = static_cast<int>(m)", ".rows = static_cast<int>(n)"),
            "status": (
                "generativeqc_tensor::blas_check(generativeqc_tensor::submit_column_gemm(request));",
                "(void)generativeqc_tensor::submit_column_gemm(request);",
            ),
        }[mutation]
        assert before in wrapper, (
            "mutation must reach the current production definition"
        )
        wrapper = wrapper.replace(before, after, 1)
    # The SDK stub supplies types only. Production cuda_error.hpp owns exceptions.
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\nusing cudaError_t=int;\n"
        "constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2;\n"
        'inline const char* cudaGetErrorString(int) { return "stub"; }\n'
    )
    source = tmp_path / "column_gemm.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
using cublasHandle_t=void*;
using cublasStatus_t=int;
constexpr int CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_ALLOC_FAILED=3;
constexpr int CUBLAS_STATUS_EXECUTION_FAILED=13, CUBLAS_OP_N=0, CUBLAS_OP_T=1;
struct Capture {
  cublasHandle_t handle{};
  int ta{},tb{},m{},n{},k{},lda{},ldb{},ldc{},calls{};
  const double *alpha{},*a{},*b{},*beta{};
  double *c{},alpha_value{},beta_value{};
} captured;
int provider_status=CUBLAS_STATUS_SUCCESS;
void require(bool value) { if (!value) std::exit(17); }
cublasStatus_t cublasDgemm(cublasHandle_t h,int ta,int tb,int m,int n,int k,
                          const double* alpha,const double* a,int lda,
                          const double* b,int ldb,const double* beta,double* c,int ldc) {
  captured={h,ta,tb,m,n,k,lda,ldb,ldc,captured.calls+1,alpha,a,b,beta,c,*alpha,*beta};
  if(provider_status!=CUBLAS_STATUS_SUCCESS) return provider_status;
  for(int col=0;col<n;++col) for(int row=0;row<m;++row) {
    double sum=0;
    for(int inner=0;inner<k;++inner) {
      const double av=ta==CUBLAS_OP_N?a[row+inner*lda]:a[inner+row*lda];
      const double bv=tb==CUBLAS_OP_N?b[inner+col*ldb]:b[col+inner*ldb];
      sum+=av*bv;
    }
    c[row+col*ldc]=*alpha*sum+*beta*c[row+col*ldc];
  }
  return CUBLAS_STATUS_SUCCESS;
}
"""
        + f'#include "{ROOT / "src/tensor/cuda_error.hpp"}"\n'
        + "namespace generativeqc_tensor {\n"
        + "struct Context { cublasHandle_t handle{}; void* stream{}; };\n"
        + _definition(runtime, "inline void blas_check(")
        + "\n"
        + _definition(runtime, "struct ColumnGemmView {")
        + ";\n"
        + _definition(runtime, "inline cublasStatus_t submit_column_gemm(")
        + "\n}\n"
        + wrapper
        + r"""
int main() {
  int handle_token=0,stream_token=0;
  generativeqc_tensor::Context context{&handle_token,&stream_token};
  constexpr size_t m=2,n=3,k=4,ldc=7;
  // Unequal rectangular shapes and padding expose transposition/stride mistakes.
  for(char ta : {'N','T'}) for(char tb : {'N','T'}) {
    const size_t lda=(ta=='N'?m:k)+2,ldb=(tb=='N'?k:n)+3;
    std::vector<double> a(128,-37),b(128,-41),c(128,97),expected=c;
    std::array<double,m*k> logical_a{};
    std::array<double,k*n> logical_b{};
    for(size_t row=0;row<m;++row) for(size_t inner=0;inner<k;++inner) {
      const double v=double(3*row+inner+1)/4;
      logical_a[row*k+inner]=v;
      a[ta=='N'?row+inner*lda:inner+row*lda]=v;
    }
    for(size_t inner=0;inner<k;++inner) for(size_t col=0;col<n;++col) {
      const double v=double(2*inner+col+2)/8;
      logical_b[inner*n+col]=v;
      b[tb=='N'?inner+col*ldb:col+inner*ldb]=v;
    }
    for(size_t col=0;col<n;++col) for(size_t row=0;row<m;++row) {
      c[row+col*ldc]=expected[row+col*ldc]=double(row+col+1);
      double sum=0;
      for(size_t inner=0;inner<k;++inner)
        sum+=logical_a[row*k+inner]*logical_b[inner*n+col];
      expected[row+col*ldc]=1.25*sum-.75*expected[row+col*ldc];
    }
    const auto original_a=a,original_b=b;
    captured={}; provider_status=CUBLAS_STATUS_SUCCESS;
    execute_strided_fp64_product(context,ta,tb,m,n,k,1.25,a.data(),lda,b.data(),ldb,-.75,c.data(),ldc);
    require(captured.calls==1 && captured.handle==context.handle);
    require(captured.ta==(ta=='N'?CUBLAS_OP_N:CUBLAS_OP_T));
    require(captured.tb==(tb=='N'?CUBLAS_OP_N:CUBLAS_OP_T));
    require(captured.m==m && captured.n==n && captured.k==k);
    require(captured.lda==lda && captured.ldb==ldb && captured.ldc==ldc);
    require(captured.a==a.data() && captured.b==b.data() && captured.c==c.data());
    // Wrapper scalars are by value; observe borrowed host pointers while live,
    // not their dangling addresses or the caller's distinct scalar objects.
    require(captured.alpha && captured.beta && captured.alpha_value==1.25 && captured.beta_value==-.75);
    require(context.handle==&handle_token && context.stream==&stream_token);
    require(a==original_a && b==original_b);
    for(size_t i=0;i<c.size();++i) require(std::abs(c[i]-expected[i])<1e-12);
    for(int failure : {CUBLAS_STATUS_ALLOC_FAILED,CUBLAS_STATUS_EXECUTION_FAILED}) {
      provider_status=failure; captured={}; const auto before=c;
      bool allocation=false,runtime=false;
      try {
        execute_strided_fp64_product(context,ta,tb,m,n,k,1.25,a.data(),lda,b.data(),ldb,-.75,c.data(),ldc);
      } catch(const generativeqc_tensor::DeviceAllocationError&) { allocation=true; }
        catch(const generativeqc_tensor::DeviceRuntimeError&) { runtime=true; }
      require(allocation==(failure==CUBLAS_STATUS_ALLOC_FAILED));
      require(runtime==(failure==CUBLAS_STATUS_EXECUTION_FAILED));
      require(captured.calls==1 && c==before && a==original_a && b==original_b);
      require(context.handle==&handle_token && context.stream==&stream_token);
      double alpha=1.25,beta=-.75;
      generativeqc_tensor::ColumnGemmView view{context.handle,ta!='N',tb!='N',m,n,k,
          &alpha,a.data(),int(lda),b.data(),int(ldb),&beta,c.data(),ldc};
      captured={};
      require(generativeqc_tensor::submit_column_gemm(view)==failure);
      require(captured.calls==1 && captured.alpha==&alpha && captured.beta==&beta && c==before);
    }
  }
}
"""
    )
    executable = tmp_path / "column_gemm"
    required_native_cxx.build_executable(
        [source], executable, compile_args=["-std=c++20", "-I", str(tmp_path)]
    )
    if mutation is None:
        subprocess.run([str(executable)], check=True, timeout=10)
    else:
        result = subprocess.run([str(executable)], check=False, timeout=10)
        assert result.returncode == 17, (
            "the executable semantic check must reject the mutant"
        )


def test_triples_no_longer_owns_a_direct_vendor_submission() -> None:
    inventory = json.loads(VENDOR_INVENTORY.read_text())
    assert "src/cc/df_triples_cuda.cu" not in inventory["files"]
    assert "cublasDgemm(" not in SOURCE.read_text()
