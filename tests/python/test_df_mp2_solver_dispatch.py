"""DF-MP2 demand stays independent of the native CC warmup policy."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def test_mp2_demand_bypasses_cc_iteration(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute the production dispatch with observable MP2 and CC boundaries."""
    source = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    start = source.index(
        "    if (df_mp2_mode) {", source.index('"DF-MP2 energy/adjoint"')
    )
    end = source.index("    const double solver_seconds", start)
    dispatch = source[start:end]
    unit = tmp_path / "dispatch.cpp"
    unit.write_text(
        r"""
#include <cassert>
#include "methods/rccsd_method.hpp"
#include "posthf/capacity.hpp"
using namespace generativeqc;
int mp2_entries = 0, cc_entries = 0;
std::size_t expected_caller = 0;
bool expected_response = false;
namespace generativeqc::cc {
std::size_t problem_host_bytes(const Problem&) { return 72; }
DFMP2Result evaluate_df_mp2_cuda(std::size_t o, std::size_t v, std::size_t q,
    std::span<const double> bov, std::span<const double> eo, std::span<const double> ev,
    std::uint64_t identity, bool response, std::size_t budget, int device,
    std::size_t caller, double denominator) {
  ++mp2_entries;
  assert(o == 1 && v == 2 && q == 3 && bov.size() == 6);
  assert(eo.size() == 1 && ev.size() == 2 && identity == 123);
  assert(response == expected_response && budget == 4096 && device == 7);
  assert(caller == expected_caller && denominator == 1e-8);
  DFMP2Result result;
  result.opposite_spin = -0.25;
  result.same_spin = -0.125;
  result.numeric_capacity_bytes = 512;
  return result;
}
SolverResult solve_cuda(const Problem&, const SolverOptions&, int) {
  ++cc_entries;
  SolverResult result;
  result.status = SolveStatus::Converged;
  result.total_energy = -3;
  return result;
}
SolverResult solve_cpu(const Problem& p, const SolverOptions& o) {
  return solve_cuda(p, o, 0);
}
}
int main() {
  using methods::MethodError;
  for (bool mp2 : {false, true}) for (bool mixed : {false, true})
  for (bool response : {false, true}) for (bool cuda : {false, true}) {
    methods::detail::RccsdNativeState state;
    state.problem.nocc=1; state.problem.nvir=2; state.problem.naux=3;
    state.problem.df_bov.resize(6); state.problem.df_source_identity=123;
    state.problem.reference_retained_bytes=16;
    state.eps_o.resize(1); state.eps_v.resize(2);
    expected_caller=16+72+(state.eps_o.capacity()+state.eps_v.capacity())*sizeof(double);
    expected_response=response;
    hf::PhysicalReference reference_value; reference_value.energy=-2;
    auto* reference=&reference_value;
    core::System auxiliary;
    auto* correlation_auxiliary=&auxiliary;
    bool df_mp2_mode=mp2, retain_df_response=response;
    cc::SolverOptions solver_options;
    solver_options.max_bytes=4096; solver_options.df_mixed_precision=mixed;
    auto correlation_options=solver_options;
    correlation_options.denominator_threshold=1e-8;
    struct { int device_id() const { return 7; } } execution;
    scf::CudaRhfBucketPlan** cuda_reference_plan=nullptr;
    auto run_with_cuda_reference_budget=[](auto*, std::size_t budget, auto&& action) {
      action(budget);
    };
    auto retire_optional_source=[] { return false; };
    std::size_t exact_source_retained=16;
    mp2_entries=cc_entries=0;
    try {
"""
        + dispatch
        + r"""
      if (mp2) {
        assert(cuda && mp2_entries == 1 && cc_entries == 0 && state.df_mp2_mode);
        assert(state.solved.converged() && state.solved.total_energy == -2.375);
        assert(state.solved.correlation_energy == -0.375);
        assert(state.solved.diagnostic.numeric_capacity_bytes == 512);
      } else {
        assert(mp2_entries == 0 && cc_entries == 1 && !state.df_mp2_mode);
        assert(state.solved.total_energy == -3);
      }
    } catch (const MethodError& error) {
      assert(mp2 && !cuda && error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
      assert(mp2_entries == 0 && cc_entries == 0);
    }
  }
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "dispatch"
    native_cxx.build_executable(
        [unit],
        executable,
        compile_args=(
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
        compile_timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=10)


def test_recursive_mp2_companion_keeps_independent_controls(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Execute the actual recursive call and publication against its signature."""
    source = (ROOT / "src/methods/df_ccsdt_force.cu").read_text()
    start = source.index("static DFCCSDTResult run_df_ccsdt_native_attempt(")
    signature = source[start : source.index(") {", start) + 3]
    start = source.index("\n  if (demand.companions & kMp2Component) {") + 1
    end = source.index("  state.df_source = {};", start)
    companion = source[start:end]
    (tmp_path / "cuda_runtime_api.h").write_text(
        "#pragma once\ntypedef struct CUstream_st* cudaStream_t;\n", encoding="utf-8"
    )
    unit = tmp_path / "recursive-companion.cpp"
    unit.write_text(
        r"""
#include <cassert>
#include "methods/df_ccsdt_force.hpp"
#include "posthf/capacity.hpp"
#include "runtime/execution_context.hpp"
namespace generativeqc::methods::detail {
using posthf::checked_add;
std::size_t capacity(std::initializer_list<const std::vector<double>*> arrays) {
  std::size_t total=0;
  for (const auto* array : arrays) total=checked_add(total,array->capacity()*sizeof(double));
  return total;
}
RccsdNativeState* expected_state{};
double expected_residual{};
bool expected_scratch{};
std::size_t expected_reservation{};
"""
        + signature
        + r"""
  assert(forces && !with_triples && !request_triples_gap_cotangents);
  assert(ccsd_mixed_precision.has_value() && !*ccsd_mixed_precision);
  assert(!lambda_mixed_precision && !lambda_resident_krylov);
  assert(ccsd_mixed_switch_residual == expected_residual);
  assert(demand.mp2 && demand.companions == 0);
  assert(lambda_serial_scratch_reuse == expected_scratch);
  assert(replay_state == expected_state && retained_host_bytes == expected_reservation);
  assert(retained_df_source_bytes == 0 && reference_experiment == nullptr);
  // Model consumption of the borrowed state so the real caller must restore it.
  replay_state->reference.reset();
  replay_state->budget=1;
  DFCCSDTResult output;
  output.energy=-1.25;
  output.forces={0.5,-0.75,0.25};
  output.correlation.opposite_spin_energy=-0.125;
  output.numeric_capacity_bytes=512;
  return output;
}
int probe() {
  core::ContextState context;
  runtime::ExecutionContext execution(context);
  core::System system,auxiliary;
  generativeqc_method_descriptor descriptor{};
  for (double ccsd_mixed_switch_residual :
       {cc::SolverOptions{}.df_mixed_precision_switch_residual,1e-4})
  for (bool lambda_serial_scratch_reuse : {false,true})
  for (std::size_t old_peak : {std::size_t(256),std::size_t(1024)}) {
    RccsdNativeState state;
    state.reference=std::make_shared<hf::PhysicalReference>();
    const auto saved_reference=state.reference;
    state.budget=8192;
    DFCCSDTResult result;
    result.numeric_capacity_bytes=old_peak;
    const std::vector<double> bar_f(3),bar_c(5),correlation_gradient(7);
    hf::RHFFrameDFPreconditionerPreparation preconditioner;
    hf::RHFFrameResponseOptions frame_options;
    DFCorrelationDemand demand{false,kMp2Component};
    const std::size_t retained_host_bytes=17;
    bool df_auxiliary_reduction=true,df_matrix_gemm=true,lambda_matrix_gemm=true;
    std::size_t lambda_batch_limit=7,ccsd_batch_limit=11,lambda_true_residual_interval=30;
    bool derived_denominators=true,packed_diis=false,parallel_gap_reduction=true;
    bool lambda_core_reuse=true,lambda_audit_matrix=true,lambda_primal_matrix=true;
    expected_state=&state;
    expected_residual=ccsd_mixed_switch_residual;
    expected_scratch=lambda_serial_scratch_reuse;
    expected_reservation=retained_host_bytes+capacity({&bar_f,&bar_c,&correlation_gradient});
"""
        + companion
        + r"""
    assert(state.reference == saved_reference && state.budget == 8192);
    assert(result.method_result.mp2_component.has_value());
    const auto& published=*result.method_result.mp2_component;
    assert(published.energy == -1.25 && published.forces == std::vector<double>({0.5,-0.75,0.25}));
    assert(published.correlation->opposite_spin_energy == -0.125);
    assert(result.numeric_capacity_bytes == std::max(old_peak,std::size_t(512)));
  }
  return 0;
}
}
int main() { return generativeqc::methods::detail::probe(); }
""",
        encoding="utf-8",
    )
    executable = tmp_path / "recursive-companion"
    native_cxx.build_executable(
        [unit],
        executable,
        compile_args=(
            "-std=c++20",
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
        compile_timeout=60,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
