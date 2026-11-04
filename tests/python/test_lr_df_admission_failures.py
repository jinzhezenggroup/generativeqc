"""Execute native optional-LR admission with typed failures and owned providers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from _cpp_source_support import cpp_function_definition

if TYPE_CHECKING:
    from conftest import NativeCxx


def test_optional_lr_admission_preserves_errors_caps_and_cleanup(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Only an initial optional refusal may choose exact, after candidate cleanup."""
    root = Path(__file__).resolve().parents[2]
    production = (root / "src/methods/dft_method.cpp").read_text()
    implementation = cpp_function_definition(production, "prepare_range_exchange")
    factory = cpp_function_definition(production, "make_plan")
    source = tmp_path / "admission.cpp"
    source.write_text(
        PRELUDE + implementation + "\n};\n" + FACTORY + factory + "\n};\n" + MAIN
    )
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "admission",
        compile_args=("-std=c++20", "-O2", f"-I{root / 'include'}"),
    )
    subprocess.run([str(executable)], check=True, timeout=10)


PRELUDE = r"""
#include "generativeqc/generativeqc.hpp"
#include <cassert>
#include <memory>
#include <optional>
#define GENERATIVEQC_HAS_CUDA 1
constexpr int cudaSuccess=0, cudaErrorMemoryAllocation=2;
int pending_error=0;
int cudaGetLastError() { const int result=pending_error; pending_error=0; return result; }
const char* cudaGetErrorString(int) { return "injected CUDA failure"; }
using MethodError=generativeqc::Error;
struct AutomaticLrFitRefused final : std::exception {};
enum class Failure { none, allocation, primary_allocation, unsupported, invalid, cuda, unresolved };
Failure failure=Failure::none;
int live_fitted=0;
namespace core { struct System {}; }
namespace scf {
enum class FockSpin { Restricted, Unrestricted };
enum class FockBackend { Cpu, Cuda };
enum class FockApproximation { Exact, DensityFitted };
struct Strategy {
  struct { struct { FockApproximation approximation; } coulomb; } spec;
};
Strategy make_rsh_correction_fock_spec(FockSpin,double,double,double,FockApproximation value) {
  return {{{value}}};
}
Strategy resolve_fock_build(Strategy value,FockBackend,double,double) { return value; }
class PreparedFockPlan {
  bool fitted_;
 public:
  PreparedFockPlan(const core::System&,const core::System*,Strategy strategy,int,
                   std::size_t budget,unsigned direct,unsigned fitted,int)
      : fitted_(strategy.spec.coulomb.approximation==FockApproximation::DensityFitted) {
    assert(budget<=64);
    if (!fitted_) { assert(live_fitted==0 && direct<=1 && !fitted); return; }
    assert(fitted==1 && !direct);
    if (failure==Failure::allocation) throw std::bad_alloc();
    if (failure==Failure::unsupported)
      throw generativeqc::Error(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,"unsupported");
    if (failure==Failure::invalid) throw std::invalid_argument("invalid control");
    if (failure==Failure::cuda)
      throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR,"solver failure");
    ++live_fitted;
  }
  ~PreparedFockPlan() { if (fitted_) --live_fitted; }
  bool density_fitting_response_valid() const { return failure!=Failure::unresolved; }
  Strategy strategy() const { return {{{FockApproximation::DensityFitted}}}; }
  const core::System* auxiliary_system() const { return nullptr; }
};
}
struct ExecutionPlan {
  bool density_fitted_range_exchange=true,automatic_density_fitted_range_exchange=true;
  double short_range_exchange=.15,long_range_exchange=1,range_omega=.3;
};
struct Options { double screening_tolerance=1e-14,density_fitting_relative_threshold=1e-10; };
bool ks_is_unrestricted(const ExecutionPlan&) { return false; }
std::size_t ks_fitted_provider_budget(const Options&,const ExecutionPlan&,bool) { return 64; }
std::size_t ks_provider_bytes(const core::System&,generativeqc_backend,unsigned) { return 32; }
int ks_fitted_projection_reservation(const core::System&,const ExecutionPlan&,scf::Strategy) {
  return 0;
}
struct Primary {
  scf::Strategy strategy() const { return {{{scf::FockApproximation::DensityFitted}}}; }
  const core::System* auxiliary_system() const { return nullptr; }
};
struct Owner {
  ExecutionPlan execution_plan_;
  Options options_;
  core::System system_;
  generativeqc_backend backend_=GENERATIVEQC_BACKEND_CPU_REFERENCE;
  Primary fock_;
  std::optional<scf::Strategy> range_strategy_;
  std::unique_ptr<scf::PreparedFockPlan> range_correction_;
"""

FACTORY = r"""
struct KsPreparedState : Owner {
  KsPreparedState(int,const core::System&,ExecutionPlan plan,Options,int,
                  generativeqc_backend backend,int,const core::System*,const int*) {
    execution_plan_=plan;
    backend_=backend;
    if (plan.density_fitted_range_exchange && failure==Failure::primary_allocation)
      throw std::bad_alloc();
    prepare_range_exchange(0);
  }
  generativeqc_status native_cuda_stationary_limit(std::size_t,std::string&) {
    return GENERATIVEQC_STATUS_SUCCESS;
  }
};
namespace runtime { struct DeviceCapacityScope { explicit DeviceCapacityScope(const int*) {} }; }
const core::System* ks_auxiliary_for_system(const core::System&,int) { return nullptr; }
struct Batch {
  std::unique_ptr<int> capacity_;
  ExecutionPlan execution_plan_;
  Options options_;
  int capabilities_=0,grid_spec_=0,device_=0,auxiliary_=0;
  generativeqc_backend backend_=GENERATIVEQC_BACKEND_CPU_REFERENCE;
"""

MAIN = r"""
int main() {
  for (const auto backend : {GENERATIVEQC_BACKEND_CPU_REFERENCE, GENERATIVEQC_BACKEND_CUDA}) {
    for (const auto mode : {Failure::none,Failure::unresolved,Failure::allocation,Failure::primary_allocation,
                            Failure::unsupported,Failure::invalid,Failure::cuda}) {
      for (const bool automatic : {false,true}) {
        failure=mode;
        Batch owner;
        owner.backend_=backend;
        owner.execution_plan_.automatic_density_fitted_range_exchange=automatic;
        bool threw=false;
        std::unique_ptr<KsPreparedState> plan;
        try { plan=owner.make_plan(core::System{}); }
        catch (const std::bad_alloc&) {
          assert(mode==Failure::allocation || mode==Failure::primary_allocation); threw=true;
        }
        catch (const std::invalid_argument&) { assert(mode==Failure::invalid); threw=true; }
        catch (const generativeqc::Error& error) {
          assert((mode==Failure::unsupported && error.status()==GENERATIVEQC_STATUS_NOT_IMPLEMENTED) ||
                 (mode==Failure::cuda && error.status()==GENERATIVEQC_STATUS_CUDA_ERROR));
          threw=true;
        }
        const bool refused=mode==Failure::allocation || mode==Failure::primary_allocation ||
                           mode==Failure::unsupported;
        assert(threw==(mode==Failure::invalid || mode==Failure::cuda || (!automatic && refused)));
        if (!threw) {
          const bool fitted=!(automatic && (refused || mode==Failure::unresolved));
          assert(plan->execution_plan_.density_fitted_range_exchange==fitted);
          assert(!plan->execution_plan_.automatic_density_fitted_range_exchange);
          assert(plan->range_correction_);
          assert((plan->range_strategy_->spec.coulomb.approximation==scf::FockApproximation::DensityFitted)
                 ==fitted);
        }
      }
      assert(live_fitted==0);
    }
  }
  failure=Failure::allocation;
  pending_error=7;
  Batch owner;
  owner.backend_=GENERATIVEQC_BACKEND_CUDA;
  try { owner.make_plan(core::System{}); return 1; }
  catch (const generativeqc::Error& error) {
    assert(error.status()==GENERATIVEQC_STATUS_CUDA_ERROR);
  }
  // Frozen exact admission survives a failed candidate on later geometry.
  failure=Failure::primary_allocation;
  const auto replay=owner.make_plan(core::System{}, {}, false);
  assert(!replay->execution_plan_.density_fitted_range_exchange);
}
"""
