"""Short-lived regions cannot acquire context-retained CUTLASS obligations."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.tensor.native_lowering import (
    emit_contraction_region_portfolio,
)
from test_joint_lowering import _request

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cutlass", [0, 1])
@pytest.mark.parametrize("hooks", [False, True])
def test_region_requires_context_lifetime_before_cutlass(
    tmp_path: Path, cutlass: int, hooks: bool
) -> None:
    """Exercise the real selector/constructor with a tiny provider API double.

    The double observes setup/descriptor calls. It does not model device math or
    claim resource qualification; admission must reject before reaching it.
    """
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("host C++ compiler and ccache required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    tensor = tmp_path / "tensor"
    tensor.mkdir()
    (tensor / "cuda_contraction.cuh").write_text(
        r"""
#pragma once
#include <vector>
#include <utility>
#include <string>
#include "tensor/native_contraction.hpp"
using cudaStream_t=void*;
using cudaStreamCaptureStatus=int;
constexpr int cudaStreamCaptureStatusNone=0;
inline int cudaGetDevice(int* d){*d=0;return 0;}
inline int cudaStreamIsCapturing(cudaStream_t,int* s){*s=0;return 0;}
namespace generativeqc_tensor { inline void cuda_check(int){} }
#define CUTLASS_VERSION 392
namespace generativeqc::tensor {
inline int setup_calls{}, table_calls{};
enum class ContractionAlgorithm { PedanticBlas, GeneratedOrdered,
  CutensorAffine, CublasLtMatmul, CutlassAot };
struct ContractionPreparationUnavailable:std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct ContractionProviderReservation {
  std::size_t workspace_bytes{},provider_bytes{},host_bytes{},cache_bytes{};
  static std::size_t checked_add(std::size_t a,std::size_t b) {
    return runtime::lowering_add(a,b);
  }
  std::size_t total_bytes(std::size_t plans) const {
    return contraction_product(plans,checked_add(checked_add(workspace_bytes,
      provider_bytes),checked_add(host_bytes,cache_bytes)));
  }
};
inline ContractionProviderReservation tensor_reservation,matmul_reservation;
inline auto qualified_cutensor_reservation(){return tensor_reservation;}
inline auto qualified_cublaslt_reservation(){return matmul_reservation;}
inline std::size_t cutensor_provider_version(){return 20800;}
inline std::size_t cublaslt_provider_version(){return 120901;}
struct CudaContractionContext {
  static constexpr std::size_t kProviderAllowance=96ULL<<20;
  bool prepare(cudaStream_t){++setup_calls;return true;}
  void prepare_generated(cudaStream_t){++setup_calls;}
  std::size_t retained_bytes()const{return 0;}
  int provider_version()const{return 12090;}
  cudaStream_t stream()const{return nullptr;}
  int device()const{return 0;}
};
// The bounded-domain sibling shares this header but is not executed by this
// lifetime-admission test; use the real descriptor and a no-op launch double.
template<class... A>void execute_matrix_contraction(A&&...){}
struct PreparedContractions {
  static std::size_t storage_bytes(std::size_t n){return 128+n*128;}
  template<class... A>void add(A&&...){++table_calls;}
  template<class... A>void execute(A&&...){}
  ContractionProviderReservation optional_resources()const{return {};}
  void release(){}
};
}
"""
    )
    request = _request()
    request = replace(request, precisions=(request.precisions[0],))
    portfolio = emit_contraction_region_portfolio(request, "e" * 64, name="region")
    source = tmp_path / "region.cpp"
    source.write_text(
        '#include "tensor/cuda_contraction_selection.cuh"\n#include <cassert>\n'
        + portfolio
        + r"""
int main(){
  using namespace generativeqc::tensor;
  constexpr auto cl=ContractionAlgorithm::CutlassAot;
  const auto host=PreparedContractionRegion::storage_bytes(8);
  auto select=[&](std::size_t bytes){return select_contraction_region(
    region_request,region_candidates,region_target,region_compilation,8,host,bytes);};
#ifdef GENERATIVEQC_TEST_HOOKS
  cutlass_region_qualification_for_test={{0,0,65536,1<<20},
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",392};
#endif
  assert(qualified_cutlass_region().reservation.total_bytes(1)==0);
  for(int attempt=0;attempt<3;++attempt){
    auto admitted=select(host+CudaContractionContext::kProviderAllowance);
    assert(admitted.algorithm==ContractionAlgorithm::PedanticBlas);
    assert(select(host).algorithm==ContractionAlgorithm::GeneratedOrdered);
    bool refused=false;
    try{(void)select(host-1);}catch(const std::length_error&){refused=true;}
    assert(refused);
    admitted.algorithm=cl;
    admitted.reservation={0,0,65536,1<<20};
    admitted.binding_bytes=host+admitted.reservation.total_bytes(8);
    std::size_t calls{},summands{};
    bool rejected=false;
    try{PreparedContractionRegion region(admitted,region_request,region_candidates,
      region_target,region_compilation,8,{2,3,1},nullptr,calls,summands,
      []{return std::vector<int>(8);});}
    catch(const std::invalid_argument&e){rejected=std::string_view(e.what())==
      "CUTLASS region requires a context-lifetime retention owner";}
    assert(rejected&&setup_calls==0&&table_calls==0);
  }
#ifdef GENERATIVEQC_TEST_HOOKS
  tensor_reservation={1,2,3,0};
  assert(select(1ULL<<30).algorithm==ContractionAlgorithm::CutensorAffine);
  tensor_reservation={};matmul_reservation={1,2,3,0};
  assert(select(1ULL<<30).algorithm==ContractionAlgorithm::CublasLtMatmul);
#endif
}
"""
    )
    obj, executable = tmp_path / "region.o", tmp_path / "region"
    flags = [f"-DGENERATIVEQC_HAS_CUTLASS={cutlass}"]
    if hooks:
        flags.append("-DGENERATIVEQC_TEST_HOOKS=1")
    for command in (
        [
            cache,
            compiler,
            "-std=c++20",
            "-O0",
            *flags,
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        [compiler, str(obj), "-o", str(executable)],
        [str(executable)],
    ):
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
