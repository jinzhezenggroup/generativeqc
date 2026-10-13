"""DF-MP2 host transfers survive delayed CUDA copies and every error edge.

Execute the production owner and its real stream/buffer RAII with a delayed
CUDA shim. ASan observes expired host destinations without dereferencing them;
this is a lifetime regression, not a replacement for GPU numerical qualification.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def definition(source: str, marker: str) -> str:
    begin = source.index(marker)
    opening = source.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


SHIM = r"""
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sanitizer/asan_interface.h>
using cudaError_t = int;
using cudaStream_t = void*;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaErrorUnknown=999;
constexpr int cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2, cudaStreamNonBlocking=1;
int fail_copy=0, copy_calls=0, fail_sync=0, sync_calls=0, error_kind=999;
int invalid=0, streams=0, buffers=0, current_device=3;
std::vector<std::function<void()>> pending;
const char* cudaGetErrorString(int) { return "injected CUDA failure"; }
int cudaGetDevice(int* value) { *value=current_device; return 0; }
int cudaSetDevice(int value) { current_device=value; return 0; }
int cudaStreamCreateWithFlags(cudaStream_t* stream,unsigned) {
  *stream=reinterpret_cast<void*>(1); ++streams; return 0;
}
int cudaStreamDestroy(cudaStream_t) { --streams; return 0; }
int cudaMemcpyAsync(void* target,const void* source,std::size_t count,int kind,cudaStream_t) {
  if (++copy_calls==fail_copy) return error_kind;
  pending.emplace_back([=] {
    const void* host=kind==cudaMemcpyHostToDevice ? source : target;
    if (__asan_region_is_poisoned(const_cast<void*>(host),count)) {
      ++invalid;
      std::fprintf(stderr,"expired host transfer: kind=%d bytes=%zu\n",kind,count);
    } else std::memcpy(target,source,count);
  });
  return 0;
}
int cudaMemsetAsync(void* target,int value,std::size_t count,cudaStream_t) {
  pending.emplace_back([=] { std::memset(target,value,count); }); return 0;
}
int cudaStreamSynchronize(cudaStream_t) {
  if (++sync_calls==fail_sync) return error_kind;
  for (auto& operation:pending) operation();
  pending.clear(); return 0;
}
namespace generativeqc::runtime {
template<class T> struct TensorView { T* data; std::size_t size; };
std::size_t size_mul(std::size_t a,std::size_t b,const char*) { return a*b; }
template<class T> int resource_cuda_malloc(T** data,std::size_t count) {
  *data=static_cast<T*>(std::malloc(count));
  if (!*data) return cudaErrorMemoryAllocation;
  ++buffers; return 0;
}
int resource_cuda_free(void* data) { std::free(data); --buffers; return 0; }
"""

GENERATED = r"""
namespace generated::df_source {
constexpr const char* mp2_energy_equation_hash="test-equations";
std::size_t mp2_response_arena_elements(std::size_t,std::size_t,std::size_t) { return 32; }
std::size_t mp2_energy_arena_elements(std::size_t,std::size_t,std::size_t) { return 32; }
struct Mp2CudaState {
  std::size_t o,v,q;
  const double *bov,*ei,*ej,*ea,*eb,*unit;
  double* arena;
  int* error;
  cudaStream_t stream;
};
struct Outputs {
  const double *opposite_spin,*same_spin,*bar_foo,*bar_fvv,*bar_bov,*bar_ei,*bar_ea,*bar_ej,*bar_eb;
};
Outputs mp2_energy_cuda(const Mp2CudaState& state) {
  pending.emplace_back([=] {
    std::fill_n(state.arena,32,1.0);
    state.arena[0]=-0.125; state.arena[1]=-0.025;
  });
  const auto p=state.arena;
  return {p,p+1,p+2,p+6,p+10,p+14,p+16,p+18,p+20};
}
Outputs mp2_response_cuda(const Mp2CudaState& state) { return mp2_energy_cuda(state); }
}
std::size_t checked_mul(std::size_t a,std::size_t b) { return a*b; }
std::size_t checked_add(std::size_t a,std::size_t b) { return a+b; }
std::size_t bytes(std::size_t values) { return values*sizeof(double); }
void admit(std::size_t peak,std::size_t budget) {
  if (peak>budget) throw std::length_error("budget");
}
"""

DRIVER = r"""
}
int main(int argc,char** argv) {
  if (argc!=5) return 99;
  const bool response=std::atoi(argv[1]);
  fail_copy=std::atoi(argv[2]); fail_sync=std::atoi(argv[3]); error_kind=std::atoi(argv[4]);
  double bov[4]{1,2,3,4}, eo[2]{-2,-1}, ev[2]{1,2};
  bool threw=false;
  try {
    const auto result=generativeqc::cc::evaluate_df_mp2_cuda(
        2,2,1,bov,eo,ev,17,response,4096,0,0,1e-10);
    if (result.opposite_spin!=-0.125 || result.same_spin!=-0.025 ||
        result.minimum_absolute_denominator!=4 || result.numeric_capacity_bytes>4096) return 5;
    if (response) {
      if (result.factors.source_identity!=17 || result.factors.bov.size()!=4 ||
          result.eps_o!=std::vector<double>(2,2) || result.eps_v!=std::vector<double>(2,2) ||
          result.foo!=std::vector<double>(4,1) || result.fvv!=std::vector<double>(4,1)) return 6;
    } else if (!result.factors.bov.empty() || !result.eps_o.empty()) return 7;
  } catch (const std::bad_alloc&) {
    if (error_kind!=cudaErrorMemoryAllocation) return 8;
    threw=true;
  } catch (const std::runtime_error& error) {
    if (error_kind!=cudaErrorUnknown || std::string(error.what())!="injected CUDA failure") return 9;
    threw=true;
  }
  // A copy failure is checked by the production owner. Cleanup synchronization
  // is deliberately noexcept; failures injected there must still retire owners.
  if (fail_copy && fail_copy<=copy_calls && !threw) return 10;
  if (!fail_copy && !fail_sync && threw) return 10;
  if (streams || buffers || !pending.empty() || current_device!=3 || !sync_calls) return 11;
  std::printf("%d %d\n",copy_calls,sync_calls);
  return invalid ? 1 : 0;
}
"""


@pytest.fixture(scope="module")
def lifetime_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires host C++ with AddressSanitizer and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (ROOT / "src/cc/df_source_cuda.cu").read_text()
    header = (ROOT / "src/cc/df_source.hpp").read_text()
    resources = (ROOT / "src/runtime/cuda_resources.cuh").read_text()
    unit = SHIM + definition(resources, "inline void cuda_resource_check(")
    for marker in (
        "class CudaDeviceScope",
        "class OwnedCudaStream",
        "template <class T>\nstruct BorrowedCudaBuffer",
        "template <class T>\nclass OwnedCudaBuffer",
    ):
        unit += "\n" + definition(resources, marker) + ";\n"
    unit += "}\nnamespace generativeqc::cc {\n" + GENERATED
    for marker in ("struct DFFactorResponseResult", "struct DFMP2Result"):
        unit += definition(header, marker) + ";\n"
    unit += definition(source, "DFMP2Result evaluate_df_mp2_cuda(") + DRIVER
    directory = tmp_path_factory.mktemp("df-mp2-host-lifetime")
    cpp, obj, executable = (
        directory / name for name in ("lifetime.cpp", "lifetime.o", "lifetime")
    )
    cpp.write_text(unit)
    compile_result = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O1",
            "-g",
            "-fno-elide-constructors",
            "-fsanitize=address",
            "-fsanitize-address-use-after-scope",
            "-fno-pie",
            "-c",
            str(cpp),
            "-o",
            str(obj),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    link_result = subprocess.run(
        [compiler, "-fsanitize=address", "-no-pie", str(obj), "-o", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert link_result.returncode == 0, link_result.stdout + link_result.stderr
    return executable


@pytest.mark.parametrize("response", (False, True))
@pytest.mark.parametrize("error", (999, 2))
@pytest.mark.parametrize("phase", ("copy", "sync"))
def test_df_mp2_transfers_outlive_failure_cleanup(
    lifetime_probe: Path, response: bool, error: int, phase: str
) -> None:
    def run(copy: int, sync: int) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [str(lifetime_probe), str(int(response)), str(copy), str(sync), str(error)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    # Discover the actual execution's edges instead of pinning transfer counts.
    # Future batching/elision is valid if every remaining edge preserves lifetime.
    copies, synchronizations = map(int, run(0, 0).stdout.split())
    for point in range(1, (copies if phase == "copy" else synchronizations) + 1):
        run(point if phase == "copy" else 0, point if phase == "sync" else 0)
