"""Run the real runtime observer against HOST DSOs with the selected CUDA ABI.

The driver/runtime/vendor objects here are small test fixtures. No real CUDA
library, context, numeric operation, SDK build or device is accessed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux ELF loader contract"
)

DRIVER = r"""
#include <cuda.h>
static unsigned long long current_id=41;
static CUresult query_error=CUDA_SUCCESS;
extern "C" void fixture_driver_state(unsigned long long id,int status) {
 current_id=id;query_error=static_cast<CUresult>(status);
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context) {
 *context=reinterpret_cast<CUcontext>(17);return CUDA_SUCCESS;
}
#ifndef OMIT_CONTEXT_ID
extern "C" CUresult CUDAAPI cuCtxGetId(CUcontext,unsigned long long* id) {
 *id=current_id;return query_error;
}
#endif
extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {*device=3;return CUDA_SUCCESS;}
extern "C" CUresult CUDAAPI cuDeviceGetUuid_v2(CUuuid* uuid,CUdevice) {
 *uuid={};uuid->bytes[7]=19;return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuDriverGetVersion(int* version) {*version=13020;return CUDA_SUCCESS;}
"""
RUNTIME = r"""
#include <cuda_runtime_api.h>
static cudaError_t query_error=cudaSuccess;
extern "C" void fixture_runtime_error(int status) {query_error=static_cast<cudaError_t>(status);}
extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* version) {
 *version=12090;return query_error;
}
extern "C" const char* CUDARTAPI cudaGetErrorString(cudaError_t) {return "HOST fixture CUDA error";}
"""
BLAS = r"""
#include <cublas_v2.h>
extern "C" cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t*) {
 return CUBLAS_STATUS_NOT_INITIALIZED;
}
"""
SOLVER = r"""
#include <cusolverDn.h>
extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t*) {
 return CUSOLVER_STATUS_NOT_INITIALIZED;
}
"""
PROBE = r"""
#include <dlfcn.h>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include "runtime/cuda_provider_identity.hpp"
#include "scf/cuda/provider_initialization.cuh"
using namespace generativeqc;
void require(bool ok,const char* text) {if(!ok)throw std::runtime_error(text);}
int main(int argc,char** argv) {
 try {
  require(argc==6,"HOST arguments");
  std::array<void*,4> libraries{};
  for(unsigned i=0;i<4;++i) {
   libraries[i]=dlopen(argv[i+1],RTLD_NOW|RTLD_LOCAL);
   require(libraries[i]!=nullptr,"HOST fixture library did not load");
  }
  const std::array<const void*,3> execution{
   dlsym(libraries[0],"cudaRuntimeGetVersion"),dlsym(libraries[1],"cublasCreate_v2"),
   dlsym(libraries[2],"cusolverDnCreate")};
  auto observed=runtime::observe_cuda_provider_identity(3,execution);
  if(std::string(argv[5])=="missing") {
   require(observed.status==runtime::ContextIdentityStatus::unavailable && !observed.identity &&
           observed.loaded_status==runtime::LoadedIdentityStatus::missing_symbol,
           "missing context-ID entrypoint was accepted or turned into a guessed context");
  } else {
   require(observed.status==runtime::ContextIdentityStatus::available && observed.identity &&
           observed.identity->context.context_id==41 && observed.identity->context.device==3 &&
           observed.identity->context.device_uuid[7]==19 &&
           observed.identity->context.driver_version==13020 && observed.identity->context.runtime_version==12090,
           "actual observer did not bind the HOST driver ABI/context identity");
   require(observed.identity->build_ids==std::array<std::optional<std::string>,4>{
            "01020304","11121314","21222324","31323334"},"direct loaded build identities differ");
   scf::ProviderInitializationIdentity expected{
    *observed.identity,scf::ProviderInitializationEigenFamily::xsyevd,1};
   scf::ProviderInitializationRequest request{
    scf::ProviderInitializationMode::explicit_qualified,128,expected};
   auto bound=scf::bind_reference_provider_initialization(
    request,{true,true},scf::ProviderInitializationEigenFamily::xsyevd,3,observed);
   require(bound.retained_capacity(16)==128,"native observation did not reach the actual admission adapter");
   auto state=reinterpret_cast<void(*)(unsigned long long,int)>(dlsym(libraries[3],"fixture_driver_state"));
   auto runtime_error=reinterpret_cast<void(*)(int)>(dlsym(libraries[0],"fixture_runtime_error"));
   require(state && runtime_error,"HOST controls missing");
   state(42,0);
   auto changed=runtime::observe_cuda_provider_identity(3,execution);
   bool refused=false;
   try {scf::bind_reference_provider_initialization(
    request,{true,true},scf::ProviderInitializationEigenFamily::xsyevd,3,changed);
   } catch(const std::invalid_argument&) {refused=true;}
   require(refused,"old context-ID request accepted a new current context");
   state(41,700);
   auto failed=runtime::observe_cuda_provider_identity(3,execution);
   require(failed.status==runtime::ContextIdentityStatus::query_error && failed.error_status==700 && !failed.identity,
           "actual driver error was relabelled unavailable/success");
   bool driver_preserved=false;
   try {scf::bind_reference_provider_initialization(
    request,{true,true},scf::ProviderInitializationEigenFamily::xsyevd,3,failed);
   } catch(const scf::ProviderIdentityDriverError& error) {driver_preserved=error.status()==700;}
   require(driver_preserved,"adapter lost the actual driver error");
   state(41,0);runtime_error(cudaErrorMemoryAllocation);
   failed=runtime::observe_cuda_provider_identity(3,execution);
   bool cuda_preserved=false;
   try {scf::bind_reference_provider_initialization(
    request,{true,true},scf::ProviderInitializationEigenFamily::xsyevd,3,failed);
   } catch(const generativeqc_tensor::DeviceAllocationError&) {cuda_preserved=true;}
   require(cuda_preserved,"adapter lost the actual CUDA allocation error type");
   runtime_error(0);
   auto wrong=execution;wrong[1]=execution[2];
   const auto mismatch=runtime::observe_cuda_provider_identity(3,wrong);
   require(mismatch.status==runtime::ContextIdentityStatus::unavailable && !mismatch.identity &&
           mismatch.loaded_status==runtime::LoadedIdentityStatus::wrong_symbol_owner,
           "execution trampoline/foreign DSO was treated as the selected library");
  }
  for(auto library:libraries)dlclose(library);
  std::cout<<"HOST actual runtime observer/admission binding and typed failures passed\n";
  return 0;
 } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
"""


def _cuda_include() -> Path:
    roots = [
        os.environ.get(name) for name in ("CUDAToolkit_ROOT", "CUDA_HOME", "CUDA_PATH")
    ]
    nvcc = shutil.which("nvcc")
    if nvcc:
        roots.append(str(Path(nvcc).resolve().parent.parent))
    roots.append("/usr/local/cuda")
    for root in roots:
        if root and (Path(root) / "include/cuda.h").is_file():
            return Path(root) / "include"
    pytest.skip("CUDA headers required for the HOST ABI fixture; no device is needed")


def test_actual_runtime_observer_host_abi(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    include = _cuda_include()
    flags = ["-std=c++20", "-O2", f"-I{include}", f"-I{ROOT / 'src'}"]
    libraries = []
    try:
        for index, (name, code, build_id) in enumerate(
            (
                ("libcudart.so.12", RUNTIME, "01020304"),
                ("libcublas.so.12", BLAS, "11121314"),
                ("libcusolver.so.11", SOLVER, "21222324"),
                ("libcuda.so.1", DRIVER, "31323334"),
            )
        ):
            source = tmp_path / f"fixture{index}.cpp"
            source.write_text(code)
            library = tmp_path / name
            required_native_cxx.build_shared(
                [source],
                library,
                compile_args=flags,
                link_args=[f"-Wl,-soname,{name}", f"-Wl,--build-id=0x{build_id}"],
            )
            libraries.append(library)
        missing = tmp_path / "missing.so"
        required_native_cxx.build_shared(
            [tmp_path / "fixture3.cpp"],
            missing,
            compile_args=[*flags, "-DOMIT_CONTEXT_ID"],
            link_args=["-Wl,-soname,libcuda.so.1", "-Wl,--build-id=0x31323334"],
        )
        probe = tmp_path / "probe.cpp"
        probe.write_text(PROBE)
        binary = required_native_cxx.build_executable(
            [probe, ROOT / "src/runtime/cuda_device_facts.cpp"],
            tmp_path / "observer",
            compile_args=[
                *flags,
                "-ffunction-sections",
                "-fdata-sections",
                "-DGENERATIVEQC_CUDA_PROVIDER_CUMETAL=0",
            ],
            link_args=[str(libraries[0]), "-Wl,--gc-sections", "-ldl"],
        )
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stdout + error.stderr, pytrace=False)
    for driver, mode in ((libraries[3], "complete"), (missing, "missing")):
        result = subprocess.run(
            [str(binary), *map(str, libraries[:3]), str(driver), mode],
            env={**os.environ, "LD_LIBRARY_PATH": str(tmp_path)},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
