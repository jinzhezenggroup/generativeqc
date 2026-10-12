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
