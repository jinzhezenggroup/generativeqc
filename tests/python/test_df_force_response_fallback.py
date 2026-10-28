"""Run the production response dispatcher with injected device/bridge failures.

The bridge is a boundary double; this qualifies dispatch, lease consumption and
bounded retry rather than CUDA arithmetic. Public response/token types are real.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]

SHIM = r"""
#pragma once
#include <algorithm>
#include <cassert>
#include <cstring>
#include <optional>
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "scf/density_fitting.hpp"
using cudaStream_t = void*;
constexpr int cudaSuccess=0, cudaMemcpyDeviceToHost=1, cudaMemcpyHostToDevice=2;
constexpr int cudaErrorMemoryAllocation=2, cudaErrorLaunchFailure=3;
int pending_error=0,error_clears=0;
int cudaPeekAtLastError() { return pending_error; }
int cudaGetLastError() { ++error_clears; const int result=pending_error; pending_error=0; return result; }
int cudaSetDevice(int) { return 0; }
int cudaMemcpyAsync(void* out,const void* in,std::size_t n,int,void*) {
  std::memcpy(out,in,n); return 0;
}
int cudaMemsetAsync(void* out,int value,std::size_t n,void*) {
  std::memset(out,value,n); return 0;
}
int cudaStreamSynchronize(void*) { return 0; }
namespace generativeqc::runtime::cuda_trace {
struct TraceShape { std::size_t systems,nbf,naux; bool source,streamed; std::size_t offset; };
struct TraceOperation { TraceOperation(const char*,void*,TraceShape) {} };
void trace_counter(const char*,std::uint64_t) {}
}
namespace generativeqc::scf {
namespace cuda_execution {
struct CudaDfMdOwner {};
// The optional producer stays absent in this strict occupied-response fixture.
int cuda_df_md_charge(CudaDfMdOwner*,const double*,double*) { assert(false); return 3; }
int cuda_df_md_three_center_gradient(CudaDfMdOwner*,const double*,const double*,double,double*) {
  assert(false); return 3;
}
}
CudaDfFinalStateToken current;
int mode=0,precision_authentication_calls=0;
struct CudaDensityFittingJkPlan {
  cuda_execution::CudaDfMdOwner* md_coulomb=nullptr;
  struct { std::size_t device_bytes=0; } md_coulomb_resources;
  std::size_t nbf=2,naux=8,batch_size=1,matrix_elements=4,tensor_elements_per_system=32;
  std::size_t row_tile=2,auxiliary_tile=8,stored_pair_count=3,stored_tensor_elements_per_system=24;
  std::size_t projection_capacity=16,panel_capacity=32,factor_basis_identity=1;
  std::size_t completed_occupied_projection_rank=1;
  int device_id=0;
  void* stream=reinterpret_cast<void*>(1);
  void* blas=reinterpret_cast<void*>(1);
  bool streamed=false,occupied_scf_reserved=true,resident_exchange_enabled=true,resident_raw_valid=false;
  bool resident_spectral_factors=true,resident_raw_coulomb=false;
  DfValueStorageOptions value_storage{DfPairStorage::SymmetricLowerSingle,1};
  CudaDensityFittingIntegralSource* integral_source=reinterpret_cast<CudaDensityFittingIntegralSource*>(1);
  double values[64]{};
  double *three_center=values,*auxiliary_tile_values=values,*inverse_square_roots=values;
  double *metric_eigenvectors=values,*metric_eigenvalues=values,*primary_density=values;
  double* cholesky_response_root=nullptr;
  double *exchange_intermediate=values,*exchange_contributions=values,*packed_raw=nullptr;
  const double* response_host_raw=nullptr;
  const core::Atom *response_orbital_atoms=nullptr,*response_auxiliary_atoms=nullptr;
  const core::Shell *response_orbital_shells=nullptr,*response_auxiliary_shells=nullptr;
  generativeqc_basis_representation response_orbital_representation{},response_auxiliary_representation{};
  double metric_relative_threshold=1e-10;
  std::vector<std::uint8_t> metric_full_rank{1},metric_response_valid{1};
  void* persistent_scf_state=nullptr;
  std::optional<CudaDfFinalStateToken> final_projection_token;
  void revoke_projection_leases() { final_projection_token.reset(); completed_occupied_projection_rank=0; }
};
namespace cuda_df {
struct PersistentScfState {
  bool unrestricted=false,occupied_exchange=true,final_frames_available=true;
  std::size_t alpha_factor_rank=1,beta_factor_rank=1;
  std::vector<std::size_t> factor_alpha_ranks{1},factor_beta_ranks{1},final_iterations{0};
  std::uint32_t generation=0;
  std::uint32_t *d_alpha_factor_generation=&generation,*d_beta_factor_generation=&generation;
  double density[4]{2,0,0,0},factor[2]{1,0};
  double *d_alpha_density=density,*d_beta_density=density,*d_density=density;
  double *d_alpha_factor=factor,*d_beta_factor=factor,*d_final_alpha_coefficients=factor;
};
bool qualified_resident_rhf_exchange(const CudaDensityFittingJkPlan&,std::size_t) { return false; }
bool qualified_value_rhf_exchange(const CudaDensityFittingJkPlan&,std::size_t rank) { return rank==1; }
generativeqc_status cuda_failure(int error,const char*,std::string&) {
  return error==cudaErrorMemoryAllocation ? GENERATIVEQC_STATUS_OUT_OF_MEMORY : GENERATIVEQC_STATUS_CUDA_ERROR;
}
generativeqc_status factor_density_for_exchange(CudaDensityFittingJkPlan& plan,PersistentScfState& state,
    const double* density,bool& accepted,std::size_t& rank,std::string&) {
  assert(mode>=19);
  assert(density==plan.primary_density && density[0]==0 && density[3]==2);
  assert(state.generation==0); // The old factor is revoked before scratch is overwritten.
  if(mode==21) { pending_error=cudaErrorMemoryAllocation; return GENERATIVEQC_STATUS_OUT_OF_MEMORY; }
  if(mode==23) { pending_error=cudaErrorLaunchFailure; return GENERATIVEQC_STATUS_OUT_OF_MEMORY; }
  if(mode==24) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
  accepted=mode!=20; rank=1;
  state.factor[0]=0; state.factor[1]=std::sqrt(2.0);
  return GENERATIVEQC_STATUS_SUCCESS;
}
}
generativeqc_status cuda_density_fitting_final_state_token(const CudaDensityFittingJkPlan*,
    std::size_t,CudaDfFinalStateToken& result,std::string&) { if(mode==14 || mode==15) pending_error=mode==14 ? cudaErrorMemoryAllocation : cudaErrorLaunchFailure;
  if(mode==11 || mode==14 || mode==15 || mode==17) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  result=current; return GENERATIVEQC_STATUS_SUCCESS; }
// These boundaries must remain unused when no arithmetic request was supplied.
// They deliberately cannot authenticate a candidate; this probe tests strict
// response dispatch and failure handling, not mixed-precision qualification.
std::uint64_t cuda_density_fitting_solve_epoch(const CudaDensityFittingJkPlan*) noexcept {
  ++precision_authentication_calls; return 0;
}
DensityFactorIdentity cuda_density_fitting_factor_identity(const CudaDensityFittingJkPlan*,
    std::size_t,std::uint64_t,std::uint64_t) noexcept {
  ++precision_authentication_calls; return {};
}
FockBuildSpec make_hf_fock_spec(FockSpin,FockApproximation) {
  ++precision_authentication_calls; return {};
}
bool cuda_density_fitting_integral_source_geometry_matches(const CudaDensityFittingIntegralSource*,
    std::size_t,const core::System&,const core::System&) noexcept { return true; }
DensityFittingMetricFactor factor_density_fitting_metric(const std::vector<double>&,std::size_t,double) { return {}; }
}
namespace generativeqc::molecule {
std::size_t ao_count(const core::System& system) noexcept { return system.atoms.size()==1 ? 2 : 8; }
}
"""

DRIVER = r"""
#include "src/scf/cuda/df_force_response.cpp"
namespace generativeqc::scf {
int occupied_calls=0,panel_calls=0;
CudaDensityFittingJkPlan* active=nullptr;
generativeqc_status execute_cuda_df_hf_gradient(int,void*,CudaDensityFittingIntegralSource*,
    std::size_t,const core::System&,const core::System&,std::span<const double>,
    const std::vector<double>&,const std::vector<double>&,
    std::span<const DensityFittingDensityResponse>,double,unsigned,std::size_t,std::size_t,
    std::vector<double>& output,std::string&,DfGradientResources* resources,const CudaDfMetricView*,void*,
    const CudaDfResponseBuffers*,const CudaDfPackedRawTensorView*,const CudaDfWhitenedTensorView* fitted,
    const CudaDfOccupiedResponseView* occupied,const CudaDfBorrowedResponseDensity*,bool mixed_shell,
    const CudaDfRawChargeView* raw_charge, const CudaDfResidentForceRequest* resident_force) {
  assert(!raw_charge && !resident_force);
  // A retry may drop the final projection, but its immutable forward values
  // must retain the producing plan's auxiliary coordinates.
  if(fitted) assert(fitted->spectral_basis==active->resident_spectral_factors);
  assert(!active->final_projection_token); // The lease is consumed before any response attempt.
  assert(!mixed_shell && precision_authentication_calls==0);
  assert(resources && !resources->precision_work_collected);
  assert((resources->precision_work==std::array<std::uint64_t,3>{}));
  if (occupied) {
    ++occupied_calls;
    if(mode>=19) {
      // A corrected determinant must project B anew, never reuse the old U.
      assert(!occupied->final_fitted_occupied_projection);
      assert(occupied->factors[0].density_scale==1.0);
      assert(occupied->factors[0].coefficients[0]==0);
    } else assert(occupied->final_fitted_occupied_projection==active->auxiliary_tile_values);
    if(mode==12 || mode==13) pending_error=mode==12 ? cudaErrorMemoryAllocation : cudaErrorLaunchFailure;
    if(mode==16) return GENERATIVEQC_STATUS_CUDA_ERROR;
    if(mode==2) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    if(mode!=0 && mode!=19) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } else {
    ++panel_calls;
    assert(pending_error==0);
    if(mode==3 || mode==17 || mode==18 || mode==25) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  }
  output={42}; return GENERATIVEQC_STATUS_SUCCESS;
}
}
int main(int argc,char** argv) {
  using namespace generativeqc;
  using namespace scf;
  assert(argc==2); mode=std::atoi(argv[1]);
  CudaDensityFittingJkPlan plan; active=&plan;
  cuda_df::PersistentScfState state; plan.persistent_scf_state=&state;
  current.identity.occupied={1}; current.identity.factor.density_generation=1;
  plan.final_projection_token=current;
  auto requested=current;
  auto density=std::vector<double>(state.density,state.density+4);
  core::System orbital,auxiliary; orbital.atoms.resize(1); auxiliary.atoms.resize(2);
  DensityFittingDensityResponse term{density,1.0,.25};
  CudaDfBorrowedFittedProjection borrowed{0,state.factor,plan.auxiliary_tile_values,2,8,1,plan.stream};
  const CudaDfBorrowedFittedProjection* lease=nullptr;
  if(mode==4) lease=&borrowed;
  if(mode==5) { setenv("GENERATIVEQC_DF_RESPONSE_SPACE","occupied",1); setenv("GENERATIVEQC_DF_OCCUPIED_RESPONSE_SOURCE","fitted",1); }
  if(mode==6) setenv("GENERATIVEQC_DF_FINAL_PROJECTION","off",1);
  if(mode==7) ++requested.identity.factor.density_generation;
  if(mode==8 || mode==18) density[0]=1.5;
  if(mode==9) plan.metric_full_rank[0]=0;
  if(mode==10) plan.final_projection_token.reset();
  if(mode>=19) {
    ++requested.identity.factor.density_generation;
    ++requested.identity.factor.orbital_generation;
    density[0]=0; density[3]=2;
  }
  if(mode==26) ++requested.identity.solve_epoch;
  if(mode==27) { requested.identity.factor.density_generation+=16;
                requested.identity.factor.orbital_generation+=16; }
  if(mode==28) ++requested.identity.factor.orbital_generation;
  if(mode==29) requested.identity.occupied[0]=0;
  if(mode==30) setenv("GENERATIVEQC_DF_FINAL_PROJECTION","off",1);
  std::vector<double> output{-1}; std::string detail;
  DfGradientResources resources;
  resources.precision_work_collected=true;
  resources.precision_work={7,8,9};
  auto status=execute_cuda_density_fitting_generated_force_response(&plan,0,orbital,auxiliary,{},
      {},{&term,1},0,4096,0,output,detail,&resources,&requested,lease,nullptr);
  const auto expected_status=(mode==2 || mode==24) ? GENERATIVEQC_STATUS_NUMERICAL_FAILURE
      : (mode==3 || mode==5 || mode==17 || mode==18 || mode==25) ? GENERATIVEQC_STATUS_OUT_OF_MEMORY
      : (mode==13 || mode==15 || mode==16 || mode==23) ? GENERATIVEQC_STATUS_CUDA_ERROR
      : GENERATIVEQC_STATUS_SUCCESS;
  assert(status==expected_status);
  assert(occupied_calls==int(mode<=5 || mode==12 || mode==13 || mode==16 || mode==19 || mode==22 || mode==25));
  assert(panel_calls==int(mode!=0 && mode!=2 && mode!=5 && mode!=13 && mode!=15 && mode!=16
                          && mode!=19 && mode!=23 && mode!=24));
  assert(error_clears==int(mode==12 || mode==14 || mode==21));
  if(mode==13 || mode==15 || mode==23) assert(pending_error==cudaErrorLaunchFailure);
  if(status==GENERATIVEQC_STATUS_SUCCESS) assert(output==std::vector<double>{42});
  if(status!=GENERATIVEQC_STATUS_SUCCESS) assert(output==std::vector<double>{-1});
  assert(!resources.precision_work_collected);
  assert((resources.precision_work==std::array<std::uint64_t,3>{}));
  assert(precision_authentication_calls==0);
  assert(!plan.final_projection_token);
  if(mode==0 || mode==19) { // Only a corrected density can reconstruct again; U stays consumed.
    status=execute_cuda_density_fitting_generated_force_response(&plan,0,orbital,auxiliary,{},
        {},{&term,1},0,4096,0,output,detail,&resources,&requested,nullptr,nullptr);
    assert(status==GENERATIVEQC_STATUS_SUCCESS);
    assert(occupied_calls==(mode==0 ? 1 : 2) && panel_calls==(mode==0 ? 1 : 0));
  }
}
"""


@pytest.fixture(scope="module")
def response_probe(
    tmp_path_factory: pytest.TempPathFactory, native_cxx: NativeCxx
) -> Path:
    directory = tmp_path_factory.mktemp("df-response-fallback")
    (directory / "shim.hpp").write_text(SHIM)
    for name in (
        "runtime/cuda_component_trace.hpp",
        "scf/cuda/df_plan_internal.hpp",
        "scf/cuda/df_runtime.hpp",
        "scf/cuda/df_scf_factor.hpp",
        "scf/cuda/df_scf_state.hpp",
    ):
        header = directory / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text('#include "shim.hpp"\n')
    source = directory / "probe.cpp"
    source.write_text(DRIVER)
    executable = directory / "probe"
    native_cxx.build_executable(
        [source],
        executable,
        compile_args=(
            "-std=c++20",
            "-I" + str(directory),
            "-I" + str(ROOT),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
    )
    return executable


@pytest.mark.parametrize(
    "mode",
    range(31),
    ids=(
        "success-and-repeat",
        "occupied-oom",
        "numerical-failure",
        "persistent-oom",
        "borrowed-dft-oom",
        "explicit-fitted-oom",
        "projection-off",
        "stale-token",
        "corrected-density",
        "truncated-metric",
        "consumed-lease",
        "provenance-oom",
        "device-allocation-error",
        "pending-device-error",
        "provenance-allocation-error",
        "provenance-pending-error",
        "compute-error",
        "provenance-and-panel-oom",
        "corrected-density-panel-oom",
        "corrected-auto-and-repeat",
        "reconstruction-rejected",
        "reconstruction-oom",
        "reconstructed-bridge-oom",
        "reconstruction-pending-error",
        "reconstruction-numerical-failure",
        "reconstructed-and-panel-oom",
        "corrected-stale-epoch",
        "corrected-generation-limit",
        "corrected-generation-mismatch",
        "corrected-empty-occupations",
        "corrected-projection-off",
    ),
)
def test_fitted_response_failure_and_lease_fallback(
    response_probe: Path, mode: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "GENERATIVEQC_DF_RESPONSE_SPACE",
        "GENERATIVEQC_DF_RESPONSE_STORAGE",
        "GENERATIVEQC_DF_OCCUPIED_RESPONSE_SOURCE",
        "GENERATIVEQC_DF_FINAL_PROJECTION",
    ):
        monkeypatch.delenv(name, raising=False)
    result = subprocess.run(
        [str(response_probe), str(mode)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
