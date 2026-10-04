"""Compile the real CUDA owner's construction path with injected API failures."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner, write_df_cpu_headers

ROOT = Path(__file__).resolve().parents[2]


def test_cuda_owner_unwinds_every_setup_failure(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/cc/cuda_solver.cu").read_text()
    helpers = source[
        source.index("std::size_t checked_mul(") : source.index(
            "__global__ void damped_advance"
        )
    ]
    # Extract the live production members/constructor/destructor, not a copied
    # model. Kernel methods are irrelevant to constructor unwind and excluded.
    owner = source[
        source.index("struct Layout {") : source.index("  void virtual_corrections()")
    ]
    owner += source[
        source.index("  void cleanup() noexcept {") : source.index(
            "  template <class Output>"
        )
    ]
    # Host-compile the constructor's callback body without CUDA launch syntax.
    # The callback is never executed by this ownership test; keep all provider
    # construction, configuration and cleanup calls in the extracted code.
    owner, replaced = re.subn(
        r"audit_df_matrix<<<.*?>>>", "audit_df_matrix", owner, flags=re.DOTALL
    )
    assert replaced == 1
    support = (ROOT / "src/cc/cuda_solver_support.cuh").read_text()
    state = support[
        support.index("struct CudaState {") : support.index(
            "struct DeviceIterationOutputs"
        )
    ]
    write_df_cpu_headers(tmp_path)
    cpp = tmp_path / "owner.cpp"
    cpp.write_text(PREFIX + state + GENERATED + helpers + owner + "};\n" + MAIN)
    exe = tmp_path / "owner"
    compile_owner(compiler, tmp_path, [cpp], exe)
    result = subprocess.run(
        [str(exe)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include "cc/solver.hpp"
#include "cc/df_plan.hpp"
#include "generated_rccsd_cpu.hpp"
#include "runtime/allocation_measurement.hpp"
#include <functional>
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr int cudaStreamNonBlocking = 1, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToDevice = 2;
int calls = 0, fail_at = 0, streams = 0, events = 0, allocations = 0, handles = 0, device = 7;
int step() { return ++calls == fail_at ? 999 : 0; }
using cublasHandle_t = void*;
constexpr int cudaErrorMemoryAllocation=2, CUBLAS_STATUS_ALLOC_FAILED=3;
constexpr int CUBLAS_POINTER_MODE_HOST=0, CUBLAS_PEDANTIC_MATH=0, CUBLAS_OP_N=0, CUBLAS_OP_T=1;
int cublasCreate(cublasHandle_t* p) {
  if (const int error=step()) return error;
  *p=new int(1); ++handles; return 0;
}
int cublasDestroy(cublasHandle_t p) { delete static_cast<int*>(p); --handles; return 0; }
int cublasSetStream(cublasHandle_t, cudaStream_t) { return step(); }
int cublasSetPointerMode(cublasHandle_t, int) { return step(); }
int cublasSetMathMode(cublasHandle_t, int) { return step(); }
int cublasSetWorkspace(cublasHandle_t, void*, std::size_t) { return step(); }
int cudaMemGetInfo(std::size_t* free, std::size_t* total) {
  *free=*total=1ULL<<30; return step();
}
int cudaGetLastError() { return 0; }
int cublasDgemm(cublasHandle_t,int,int,int,int,int,const double*,const double*,int,
                 const double*,int,const double*,double*,int) {
  throw std::logic_error("ownership test must not execute numerical callback");
}
void audit_df_matrix(const double*,std::size_t,int*) {
  throw std::logic_error("ownership test must not execute numerical callback");
}
void blas_check(int code) { if (code) throw std::runtime_error("injected CUDA failure"); }
int cudaGetDevice(int* p) { *p = device; return 0; }
int cudaSetDevice(int d) { device = d; return 0; }
int cudaStreamCreateWithFlags(cudaStream_t* p, int) {
  if (const int error = step()) return error;
  *p = new int(1); ++streams; return 0;
}
int cudaEventCreate(cudaEvent_t* p) {
  if (const int error = step()) return error;
  *p = new int(1); ++events; return 0;
}
int cudaEventDestroy(cudaEvent_t p) { delete static_cast<int*>(p); --events; return 0; }
int cudaMalloc(void** p, std::size_t bytes) {
  if (const int error = step()) return error;
  *p = new unsigned char[bytes]; ++allocations; return 0;
}
int cudaMemcpyAsync(void* d, const void* s, std::size_t n, int, cudaStream_t) {
  if (const int error = step()) return error;
  std::memcpy(d, s, n); return 0;
}
"""

PREFIX += r"""
int cudaStreamSynchronize(cudaStream_t) { return step(); }
int cudaFree(void* p) { delete[] static_cast<unsigned char*>(p); --allocations; return 0; }
int cudaStreamDestroy(cudaStream_t p) { delete static_cast<int*>(p); --streams; return 0; }
void cuda_check(int code) { if (code) throw std::runtime_error("injected CUDA failure"); }
namespace generativeqc::cc {
constexpr std::size_t kDFBlasProviderAllowance=96ULL<<20;
namespace generated {
"""
GENERATED = r"""
namespace dfcore {
struct CudaState : generated::CudaState {
  const double *df_virtual_singles{}, *df_virtual_doubles{};
};
}
namespace df {
struct CudaState {
  std::size_t o{}, v{};
  cudaStream_t stream{};
  int* error{};
  double* response_arena{};
};

}
namespace dfhoist {
struct CudaState : dfcore::CudaState {
  double *prepare_arena{}, *auxiliary_arena{};
  std::function<void(char,char,std::size_t,std::size_t,std::size_t,double,
                     const double*,const double*,double*)> gemm;
};
}
}
std::size_t problem_host_bytes(const Problem&) { return 128; }
"""
MAIN = r"""
}  // namespace generativeqc::cc
int main() {
  generativeqc::cc::Problem p; p.nocc = p.nvir = 1;
  for (auto* v : {&p.foo, &p.fov, &p.fvv, &p.ovov, &p.ovvo, &p.oovv,
                 &p.ovvv, &p.ovoo, &p.oooo, &p.vvvv, &p.d1, &p.d2,
                 &p.initial_t1, &p.initial_t2}) v->push_back(1.0);
  // Compile the production owner once, then exercise disabled, one-slot and
  // ordinary DIIS. Event creation participates in the same failure sequence.
  bool saw_matrix=false;
  for (const unsigned naux : {0U, 2U}) {
  p.naux = naux; p.df_bov.assign(naux, 0.1); p.df_bvv.assign(naux, 0.1);
  for (const unsigned history : {0U, 1U, 6U}) {
    generativeqc::cc::SolverOptions options;
    options.diis_size = history;
    calls = 0; fail_at = 0;
    int constructor_calls = 0;
    { generativeqc::cc::Owner good(p, options, 0); constructor_calls = calls;
      saw_matrix = saw_matrix || good.plan.matrix_gemm;
      if (handles != (good.plan.matrix_gemm ? 1 : 0)) return 10;
      const auto detached = (good.n1 + good.n2) * sizeof(double);
      if (good.diagnostic.numeric_capacity_bytes < 128 + good.layout.total + detached) {
        std::cerr << "CUDA detached result storage was not reserved\n"; return 8;
      }
      if (events != (history ? 2 : 0)) return 9;
    }
    if (streams || events || allocations || handles || device != 7 || constructor_calls < 18) return 1;
    for (int failure = 1; failure <= constructor_calls; ++failure) {
      calls = 0; fail_at = failure;
      try { generativeqc::cc::Owner broken(p, options, 0); return 2; }
      catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "injected CUDA failure") return 3;
      }
      if (streams || events || allocations || handles || device != 7) {
        std::cerr << "leaked owners after setup operation " << failure << '\n';
        return 4;
      }
      calls = 0; fail_at = 0;
      { generativeqc::cc::Owner retry(p, options, 0); }
      if (streams || events || allocations || handles || device != 7) return 5;
    }
    options.max_bytes = 1; calls = 0;
    try { generativeqc::cc::Owner over_budget(p, options, 0); return 6; }
    catch (const std::length_error&) {}
    if (calls || streams || events || allocations || handles || device != 7) return 7;
    std::cout << "DIIS " << history << ": setup failures and retries checked: "
              << constructor_calls << '\n';
  }
  }
  if (!saw_matrix) return 11;
}
"""
