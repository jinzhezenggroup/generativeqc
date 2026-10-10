"""Host doubles check actual optional panel-provider live fallback cleanup."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.native_lowering import native_lowering_portfolio
from generativeqc_compiler.dft.xc_density_lowering import density_portfolio

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

CUDA = r"""
#pragma once
#include <cassert>
#include <cstddef>
#include <cstdlib>
using cudaStream_t = void*;
using cudaError_t = int;
using cudaStreamCaptureStatus = int;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2, cudaErrorInvalidValue=3;
constexpr int cudaErrorInvalidDevice=4, cudaStreamCaptureStatusNone=0;
inline int device=1, live_handles=0, sync_calls=0, destroy_calls=0;
inline int failure=0, failed_handle=0, retried_handle=0;
inline bool device_oom=false;
inline cudaStream_t stream=reinterpret_cast<void*>(7);
inline int cudaGetDevice(int* p){*p=device;return 0;}
inline int cudaSetDevice(int d){device=d;return 0;}
inline int cudaStreamIsCapturing(cudaStream_t s,int* p){assert(s==stream);*p=0;return 0;}
inline int cudaGetLastError(){return 0;}
inline const char* cudaGetErrorString(int){return "injected CUDA failure";}
inline int cudaMemGetInfo(std::size_t* f,std::size_t* t){*f=*t=1ULL<<30;return 0;}
inline int cudaRuntimeGetVersion(int* p){*p=1;return 0;}
inline int cudaStreamSynchronize(cudaStream_t s){
  assert(s==stream && device==1);++sync_calls;
  if(failure==1){assert(live_handles==1);failed_handle=1;failure=0;return 3;}
  if(failed_handle){assert(live_handles==1);retried_handle=1;}
  return 0;
}
inline int cudaMalloc(void** p,std::size_t n){
  assert(device==1);*p=nullptr;if(device_oom)return cudaErrorMemoryAllocation;
  *p=std::malloc(n);return *p?0:cudaErrorMemoryAllocation;
}
inline int cudaFree(void* p){assert(device==1);std::free(p);return 0;}
inline int cudaMallocAsync(void** p,std::size_t n,cudaStream_t){return cudaMalloc(p,n);}
inline int cudaFreeAsync(void* p,cudaStream_t){return cudaFree(p);}
"""

PREFIX = r"""
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include "runtime/allocation_measurement.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_cuda.cuh"
#include "tensor/cuda_error.hpp"
#include "tensor/cuda_panel_product.hpp"
using std::isfinite;
#define __global__
#define GENERATIVEQC_TEST_HOOKS
struct Dim {std::size_t x;};
Dim blockIdx{0},threadIdx{0},blockDim{1},gridDim{1};
int atomicCAS(int* p,int old,int value){int before=*p;if(before==old)*p=value;return before;}
using cublasHandle_t=void*;
constexpr int CUBLAS_STATUS_ALLOC_FAILED=2, CUBLAS_POINTER_MODE_HOST=0;
constexpr int CUBLAS_PEDANTIC_MATH=0, CUBLAS_OP_N=0;
bool measurement_locked(){
  // A distinct thread avoids trying to acquire a mutex already owned by self.
  bool acquired=false;
  std::thread observer([&]{
    auto& mutex=generativeqc::runtime::allocation_measurement_mutex;
    acquired=mutex.try_lock();if(acquired)mutex.unlock();
  });
  observer.join();return !acquired;
}
int cublasCreate(void** h){assert(device==1);*h=reinterpret_cast<void*>(11);++live_handles;return 0;}
int cublasDestroy(void* h){
  assert(device==1 && h==reinterpret_cast<void*>(11) && live_handles==1);
  assert(measurement_locked());++destroy_calls;
  if(failure==2){failed_handle=1;failure=0;return 3;}
  if(failed_handle)retried_handle=1;
  --live_handles;return 0;
}
int cublasSetStream(void*,cudaStream_t s){assert(s==stream);return 0;}
int cublasSetPointerMode(void*,int){return 0;}
int cublasSetMathMode(void*,int){return 0;}
int cublasSetWorkspace(void*,void* p,std::size_t n){assert(!p&&!n);return 0;}
int cublasGetVersion(void*,int* p){*p=1;return 0;}
int cublasDgemm(void*,int,int,int,int,int,const double*,const double*,int,
               const double*,int,const double*,double*,int){std::abort();}
namespace generativeqc_tensor {
void blas_check(int s){
  if(s==2)throw DeviceAllocationError("injected BLAS allocation failure");
  if(s)throw DeviceRuntimeError("injected BLAS runtime failure");
}
}
// The extracted context now drains optional storage under its device scope.
namespace generativeqc::runtime {
struct CudaDeviceScope {
  int previous;
  template<class Check> CudaDeviceScope(int selected,Check check) {
    check(cudaGetDevice(&previous)); check(cudaSetDevice(selected));
  }
  ~CudaDeviceScope() { (void)cudaSetDevice(previous); }
};
}
namespace generativeqc::tensor {
std::size_t contraction_product(std::size_t a,std::size_t b){
  if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::overflow_error("product");
  return a*b;
}
}
"""

DRIVER = r"""
int main(int argc,char** argv){
  assert(argc==4);
  using namespace generativeqc::tensor;
  using namespace generativeqc::runtime;
  const bool bounded=std::string(argv[1])=="bounded";
  const std::string route=argv[2], fault=argv[3];
  failure=fault=="sync"?1:fault=="destroy"?2:0;
  const int expected_failure=failure;
  panel_product_library_for_test=panel_product_bounded_library_for_test=route!="unqualified";
  constexpr std::size_t cache_bytes=2*2*sizeof(double);
  auto ledger=std::make_shared<DeviceResourceLedger>();
  ledger->device=1;ledger->limit=route=="ledger"?0:cache_bytes;
  active_device_resource_ledger=ledger;device_oom=route=="device";
  std::unique_ptr<CudaPanelProduct> owner;
  bool propagated=false;
  try{
    owner=std::make_unique<CudaPanelProduct>(probe_request,probe_candidates,
        probe_target,probe_compilation,2,1,1,stream,
        CudaContractionContext::kProviderAllowance+cache_bytes,0,bounded?5:4,true);
  }catch(const generativeqc_tensor::DeviceRuntimeError&){propagated=true;}
  if(expected_failure){
    if(!propagated){std::cerr<<"live cleanup failure admitted generated retry\n";return 1;}
    assert(!owner && failed_handle==1 && retried_handle==1);
    assert(sync_calls==2 && destroy_calls==(expected_failure==2?2:1));
  }else{
    assert(!propagated && owner);
    assert(owner->enabled()==(route=="admit"));
    assert(owner->diagnostic().matrix_bytes==(route=="admit"?cache_bytes:0));
    assert(owner->diagnostic().provider_allowance==
           (route=="admit"?CudaContractionContext::kProviderAllowance:0));
    if(route=="unqualified")assert(!destroy_calls && !sync_calls && !live_handles);
    else if(route!="admit")assert(destroy_calls==1 && sync_calls==1 && !live_handles);
  }
  owner.reset();
  assert(device==1 && live_handles==0 && ledger->live==0 && device_allocation_owners.empty());
  if(route=="ledger"||route=="device")assert(ledger->rejected==1);
  std::cout<<argv[1]<<" "<<route<<" "<<fault<<" passed\n";
}
"""


@pytest.fixture(scope="module")
def panel_cleanup_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    """Compile actual classes/ledger; remove only includes and CUDA launch syntax.

    CUDA and cuBLAS are explicit host doubles. This establishes ownership and
    error handling, not GPU arithmetic, capture, allocation behavior or timing.
    """
    folder = tmp_path_factory.mktemp("panel-provider-cleanup")
    (folder / "cuda_runtime_api.h").write_text(CUDA)
    context = (ROOT / "src/tensor/cuda_contraction.cuh").read_text()
    start = context.index("class CudaContractionContext {")
    context = context[start : context.index("\ntemplate <class T>", start)]
    panel = (ROOT / "src/tensor/cuda_panel_product.cuh").read_text()
    panel = re.sub(r"^#include.*\n|^#pragma once.*\n", "", panel, flags=re.MULTILINE)
    panel, launches = re.subn(r"<<<.*?>>>", "", panel, flags=re.DOTALL)
    assert launches == 1
    metadata = native_lowering_portfolio(*density_portfolio(16, "1" * 64), name="probe")
    unit, executable = folder / "probe.cpp", folder / "probe"
    unit.write_text(
        PREFIX
        + "namespace generativeqc::tensor {\n"
        + context
        + "}\n"
        + panel
        + metadata
        + DRIVER
    )
    return native_cxx.build_executable(
        [unit],
        executable,
        compile_args=(
            "-std=c++20",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pthread",
            "-I" + str(folder),
            "-I" + str(ROOT / "src"),
        ),
        link_args=("-pthread",),
    )


@pytest.mark.parametrize("domain", ["dense", "bounded"])
@pytest.mark.parametrize(
    ("route", "fault"),
    [
        ("ledger", "none"),
        ("ledger", "sync"),
        ("ledger", "destroy"),
        ("device", "none"),
        ("device", "sync"),
        ("device", "destroy"),
        ("admit", "none"),
        ("unqualified", "none"),
    ],
)
def test_panel_live_cleanup_propagates_before_fallback(
    panel_cleanup_probe: Path, domain: str, route: str, fault: str
) -> None:
    result = subprocess.run(
        [str(panel_cleanup_probe), domain, route, fault],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
