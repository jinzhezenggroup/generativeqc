"""Host-check the production grid factory's complete device preparation scope."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]


def _definition(path: str, marker: str) -> str:
    text = (ROOT / path).read_text()
    start = text.index(marker)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_grid_factory_prepares_projection_on_requested_device(
    tmp_path: Path, generated: bool, failure: bool
) -> None:
    """Real factory/context bodies with an explicit two-device runtime double.

    This checks device selection, cleanup and publication, not CUDA arithmetic.
    The double records handle/stream devices and permits device-independent stream
    queries, so a generated binding cannot accidentally inherit the caller device.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler required")
    prelude = r"""
#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "dft/ao_grid_work.hpp"
#include "tensor/metrics.hpp"
#include "runtime/allocation_measurement.hpp"
#include "runtime/bounded_workspace.hpp"
using cudaError_t=int;
struct Stream {int device;};
struct Handle {int device;};
struct Event {int device;};
using cudaStream_t=Stream*;
using cudaEvent_t=Event*;
using cublasHandle_t=Handle*;
using cublasStatus_t=int;
using cudaStreamCaptureStatus=int;
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2;
constexpr int cudaStreamCaptureStatusNone=0, cudaStreamNonBlocking=1;
constexpr int cudaMemcpyHostToDevice=1, CUBLAS_STATUS_ALLOC_FAILED=2;
constexpr int CUBLAS_POINTER_MODE_HOST=0,CUBLAS_DEFAULT_MATH=0,CUBLAS_PEDANTIC_MATH=1;
constexpr int GENERATIVEQC_STATUS_OUT_OF_MEMORY=2,GENERATIVEQC_STATUS_CUDA_ERROR=3;
int current_device=0, observed_provider=-1, live_streams=0, live_handles=0;
bool generated=false, fail_upload=false;
struct cudaDeviceProp {int major=12,minor=0;};
int cudaGetDevice(int* p){*p=current_device;return 0;}
int cudaSetDevice(int d){current_device=d;return 0;}
const char* cudaGetErrorString(int){return "runtime failure";}
int cudaGetDeviceProperties(cudaDeviceProp*,int){return 0;}
int cudaMemGetInfo(std::size_t* free,std::size_t* total){*free=*total=1ULL<<30;return 0;}
int cudaStreamCreateWithFlags(cudaStream_t* s,unsigned){*s=new Stream{current_device};++live_streams;return 0;}
int cudaStreamDestroy(cudaStream_t s){assert(s->device==current_device);delete s;--live_streams;return 0;}
int cudaStreamSynchronize(cudaStream_t){return 0;}
int cudaStreamIsCapturing(cudaStream_t,cudaStreamCaptureStatus* p){*p=0;return 0;}
int cudaMalloc(void** p,std::size_t n){*p=new unsigned char[n];return 0;}
int cudaFree(void* p){delete[] static_cast<unsigned char*>(p);return 0;}
int cudaMemcpyAsync(void* d,const void* s,std::size_t n,int,cudaStream_t){if(fail_upload)return 1;std::memcpy(d,s,n);return 0;}
int cudaMemsetAsync(void* d,int v,std::size_t n,cudaStream_t){std::memset(d,v,n);return 0;}
int cudaEventCreate(cudaEvent_t* p){*p=new Event{current_device};return 0;}
int cudaEventRecord(cudaEvent_t e,cudaStream_t s){return e->device!=s->device;}
int cudaEventSynchronize(cudaEvent_t){return 0;}
int cudaEventElapsedTime(float* p,cudaEvent_t,cudaEvent_t){*p=0;return 0;}
int cudaEventDestroy(cudaEvent_t p){assert(p->device==current_device);delete p;return 0;}
int cudaGetLastError(){return 0;}
int cudaRuntimeGetVersion(int* p){*p=12000;return 0;}
int cublasCreate(cublasHandle_t* p){*p=new Handle{current_device};++live_handles;return 0;}
int cublasDestroy(cublasHandle_t p){assert(p->device==current_device);delete p;--live_handles;return 0;}
int cublasSetStream(cublasHandle_t h,cudaStream_t s){return h->device==s->device?0:1;}
int cublasSetPointerMode(cublasHandle_t,int){return 0;}
int cublasSetMathMode(cublasHandle_t,int){return 0;}
int cublasSetWorkspace(cublasHandle_t,void*,std::size_t){return 0;}
int cublasGetVersion(cublasHandle_t,int* p){*p=12000;return 0;}
namespace generativeqc_tensor {
struct DeviceAllocationError:std::runtime_error{using std::runtime_error::runtime_error;};
struct DeviceRuntimeError:std::runtime_error{using std::runtime_error::runtime_error;};
void cuda_check(int s){if(s)throw DeviceRuntimeError("runtime failure");}
void blas_check(int s){if(s)throw DeviceRuntimeError("provider failure");}
void error_text(char* p,std::size_t n,const char* s){if(p&&n)std::snprintf(p,n,"%s",s);}
}
namespace generativeqc::runtime {
void cuda_resource_check(int s){generativeqc_tensor::cuda_check(s);}
template <class T> int resource_cuda_malloc(T** p,std::size_t n,bool* host_oom=nullptr){
  if(host_oom)*host_oom=false;
  return cudaMalloc(reinterpret_cast<void**>(p),n);
}
int resource_cuda_free(void* p){return cudaFree(p);}
"""
    source = prelude
    # Keep the actual CSR buffer ownership/destruction types in the extracted
    # GridPlan; only CUDA API allocation is replaced by the host double.
    for marker in (
        "class CudaDeviceScope",
        "template <class T>\nstruct BorrowedCudaBuffer",
        "template <class T>\nclass OwnedCudaBuffer",
    ):
        source += _definition("src/runtime/cuda_resources.cuh", marker) + ";\n"
    source += "}\nnamespace generativeqc_tensor {\n"
    source += _definition("src/tensor/cuda_runtime.cuh", "struct Context")
    source += ";}\nnamespace generativeqc::tensor {\n"
    source += _definition(
        "src/tensor/cuda_contraction.cuh", "class CudaContractionContext"
    )
    source += r""";
struct PreparedBoundedContraction {
  static constexpr std::size_t host_reservation=32U<<10;
  CudaContractionContext context;
  explicit PreparedBoundedContraction(cudaStream_t s){
    if(generated)context.prepare_generated(s);else context.prepare(s);
    observed_provider=context.device();
  }
  std::size_t retained_provider_bytes()const{return context.retained_bytes();}
};
}
namespace generativeqc::dft::generated {
std::unique_ptr<tensor::PreparedBoundedContraction> prepare_grid_panel(
    std::size_t,std::size_t,std::size_t,std::size_t,cudaStream_t s,std::size_t){
  return std::make_unique<tensor::PreparedBoundedContraction>(s);
}
}
using namespace generativeqc_tensor;
"""
    for marker in (
        "struct ResidentAoMap",
        "struct GridPlan",
        "size_t mul(",
        "size_t add(",
        "template <class F>\nint guarded(",
        "int grid_cuda_create_v3(",
    ):
        source += _definition("src/dft/cuda_grid.cu", marker) + ";\n"
    source += r"""
int main(int argc,char** argv){
  generated=std::atoi(argv[1]);fail_upload=std::atoi(argv[2]);
  const std::size_t dimensions[]{1,1,1};
  double basis[21]{};basis[3]=1;basis[7]=1;basis[8]=1;
  char error[256]{};void* output=reinterpret_cast<void*>(1);
  const int status=grid_cuda_create_v3(1,12,0,dimensions,basis,2,1,0,0,
      nullptr,0,15,&output,error,sizeof(error));
  if(observed_provider!=1){std::fprintf(stderr,"projection device=%d, expected=1; status=%d, error=%s\n",observed_provider,status,error);return 1;}
  assert(current_device==0);
  if(fail_upload){assert(status!=0&&output==nullptr);}
  else {assert(status==0&&output);delete static_cast<GridPlan*>(output);}
  assert(current_device==0&&live_handles==0&&live_streams==0);
}
"""
    cpp, executable = tmp_path / "device.cpp", tmp_path / "device"
    cpp.write_text(source)
    compile_owner(compiler, tmp_path, [cpp], executable)
    result = subprocess.run(
        [str(executable), str(int(generated)), str(int(failure))],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
