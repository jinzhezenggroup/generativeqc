"""Execute the production component census without requiring a CUDA device.

This checks accounting and admission only. Numerical AUTO SCF/force acceptance
still requires the unmodified real-device gates.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _method(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_component_precision_census_preserves_complete_inventory(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    methods = "\n".join(
        _method(source, name)
        for name in (
            "  bool complete_precision_inventory_domain()",
            "  void record_precision_operator(",
            "  void record_fock_precision_work(",
            "  void record_precision_retry()",
        )
    )
    finish = _method(source, "  bool finish_legacy()")
    start = finish.index("    ++output.iterations;")
    end = finish.index("    output.precision.requested_mode", start)
    census = finish[start:end]
    publication = finish[
        finish.index("    try {", finish.rindex("    previous_energy =")) :
    ]
    harness = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "scf/types.hpp"
namespace scf = generativeqc::scf;
using Kind = scf::PrecisionOperatorKind;
using Mode = scf::PrecisionArithmeticMode;
using Dtype = scf::PrecisionDtype;
struct Owner {
  bool pending_mixed_coulomb{}, pending_mixed_density{}, mixed_precision_executed{};
  bool device_chunk_mode{}, has_exchange{true}, has_range_correction{true};
  bool stabilize_occupations{}, final_closure{}, is_active{}, is_failed{};
  bool final_state_ready{true}, occupied_fitted_factor_ready{};
  bool incremental_jk{}, pending_incremental_full{};
  double observed_max_delta{};
  void* nonlocal_correlation{};
  unsigned refinement_iterations{};
  std::uint64_t owner{17}, solve_epoch{3}, generation{5}, final_generation{5};
  scf::ScfOptions options;
  scf::ScfResult output;
  Owner() { options.xc_execution_schedule = scf::ScfOptions::XcExecutionSchedule::DeviceFused; }
"""
        + methods
        + r"""
  void complete(bool coulomb, bool density, std::uint64_t mixed_coulomb_recurrences) {
    pending_mixed_coulomb = coulomb;
    pending_mixed_density = density;
    mixed_precision_executed |= coulomb || density;
"""
        + census
        + r"""
  }
  bool publish() {
"""
        + publication
        + r"""
};
std::uint64_t count(const Owner& p, Kind kind, Mode mode = Mode::Strict) {
  std::uint64_t total = 0;
  for (const auto& op : p.output.precision_work.operators) {
    assert(op.storage == Dtype::Fp64 && op.accumulation == Dtype::Fp64);
    assert(op.reduction == Dtype::Fp64);
    assert(op.compute == (op.arithmetic_mode == Mode::Mixed ? Dtype::Fp32 : Dtype::Fp64));
    if (op.kind == kind && op.arithmetic_mode == mode) total += op.count;
  }
  return total;
}
void check_inventory(const Owner& p, bool coulomb, bool density, bool stabilized, bool closure) {
  assert(count(p, Kind::CoulombJ) == !coulomb);
  assert(count(p, Kind::CoulombJ, Mode::Mixed) == coulomb);
  assert(count(p, Kind::MatrixProduct) == (stabilized ? 9U : 7U));
  assert(count(p, Kind::MatrixProduct, Mode::Mixed) == density);
  assert(count(p, Kind::CoulombRecurrence, Mode::Mixed) == (coulomb ? 11U : 0U));
  assert(count(p, Kind::ExchangeK) == 2 && count(p, Kind::ExchangeK, Mode::Mixed) == 0);
  for (const auto kind : {Kind::Xc, Kind::FockAssembly, Kind::PhysicalResidual,
                          Kind::Eigensolver, Kind::DensityBuild, Kind::Diagnostics})
    assert(count(p, kind) == 1 && count(p, kind, Mode::Mixed) == 0);
  assert(count(p, Kind::Diis) == !closure);
  assert(count(p, Kind::OccupationStabilization) == stabilized);
  const auto expected_rows = 9U + !closure + stabilized + density + coulomb;
  assert(p.output.precision_work.operators.size() == expected_rows);
}
int main() {
  // All four independent component settings, including density-only, must
  // retain the complete strict inventory and exact exchange arithmetic.
  for (bool coulomb : {false, true}) for (bool density : {false, true})
    for (bool stabilized : {false, true}) for (bool closure : {false, true}) {
      Owner p;
      p.stabilize_occupations = stabilized;
      p.final_closure = closure;
      p.complete(coulomb, density, 11);
      check_inventory(p, coulomb, density, stabilized, closure);
      assert(p.output.precision_work.events.size() == 1);
      assert(p.output.precision_work.events[0].kind ==
             (coulomb || density ? scf::PrecisionWorkEventKind::MixedFock
                                 : scf::PrecisionWorkEventKind::StrictFock));
      assert(!p.output.precision_work.complete && !p.output.precision_work.operator_inventory_complete);
    }
  Owner p;
  p.complete(false, false, 0);
  p.complete(true, false, 11);
  p.complete(true, true, 13);
  p.complete(false, true, 99); // Density-only cannot inflate the J recurrence census.
  p.complete(false, false, 0);
  assert(p.output.fock_builds == 5 && p.output.iterations == 5);
  assert(p.output.precision.mixed_stage_fock_builds == 3);
  assert(p.output.precision.strict_stage_fock_builds == 2 && p.refinement_iterations == 1);
  assert(count(p, Kind::CoulombJ) == 3 && count(p, Kind::CoulombJ, Mode::Mixed) == 2);
  assert(count(p, Kind::MatrixProduct) == 35 && count(p, Kind::MatrixProduct, Mode::Mixed) == 2);
  assert(count(p, Kind::CoulombRecurrence, Mode::Mixed) == 24);
  assert(count(p, Kind::ExchangeK) == 10);
  const auto& work = p.output.precision_work;
  for (std::size_t i = 0; i != work.events.size(); ++i) {
    const auto& event = work.events[i];
    assert(event.sequence == i && event.iteration == i + 1);
    assert(event.owner_id == p.owner && event.solve_epoch == p.solve_epoch);
    assert(event.state_generation == p.generation);
    assert(event.phase == (i == 4 ? scf::PrecisionWorkPhase::Refinement
                                 : scf::PrecisionWorkPhase::Scf));
  }
  p.record_precision_retry();
  assert(work.events.back().sequence == 5 && work.events.back().kind == scf::PrecisionWorkEventKind::Retry);
  assert(work.events.back().phase == scf::PrecisionWorkPhase::Retry);
  assert(count(p, Kind::CoulombRecurrence, Mode::Mixed) == 24);
  p.publish(); // An unconverged result must never claim a complete inventory.
  assert(!work.complete && !work.operator_inventory_complete);
  p.output.converged = true;
  p.publish();
  assert(work.complete && work.operator_inventory_complete);
  assert(work.events.back().kind == scf::PrecisionWorkEventKind::FinalAudit);
  assert(work.events.back().phase == scf::PrecisionWorkPhase::Finalization);
  assert(work.events.back().sequence == 6 && work.events.back().iteration == 5);
  assert(work.events.back().owner_id == p.owner && work.events.back().solve_epoch == p.solve_epoch);
  assert(work.events.back().state_generation == p.final_generation);
  assert(p.output.precision.final_residual_audits == 1 && p.output.precision.operator_work_counters_valid == 1);
  assert(p.output.fock_builds == 5); // FinalAudit must not invent a Fock build.
  Owner incremental;
  incremental.incremental_jk = true;
  incremental.pending_incremental_full = true;
  incremental.complete(false, false, 0);
  incremental.pending_incremental_full = false;
  incremental.complete(false, false, 0);
  incremental.final_closure = true;
  incremental.complete(false, false, 0);
  incremental.output.converged = true;
  incremental.publish();
  assert(incremental.output.incremental_direct_jk.anchor_full_builds == 1);
  assert(incremental.output.incremental_direct_jk.delta_builds == 1);
  assert(incremental.output.incremental_direct_jk.post_scf_full_builds == 1);
  assert(incremental.output.fock_builds == 3);
  assert(incremental.output.precision.final_residual_audits == 1);
  assert(incremental.output.precision_work.events.size() == 4);
  assert(!incremental.output.precision_work.complete);
  assert(!incremental.output.precision_work.operator_inventory_complete);
  assert(!incremental.output.precision.operator_work_counters_valid);
  assert(!incremental.output.incremental_direct_jk.quartet_work_counters_valid);
  Owner screened;
  screened.complete(true, false, 0);
  assert(count(screened, Kind::CoulombJ, Mode::Mixed) == 1);
  assert(count(screened, Kind::CoulombRecurrence, Mode::Mixed) == 0);
  for (int excluded = 0; excluded < 3; ++excluded) {
    Owner partial;
    partial.device_chunk_mode = excluded == 0;
    partial.nonlocal_correlation = excluded == 1 ? &partial : nullptr;
    if (excluded == 2)
      partial.options.xc_execution_schedule = scf::ScfOptions::XcExecutionSchedule::HostUnfused;
    partial.complete(true, false, 7);
    assert(partial.output.precision_work.operators.size() == 1);
    assert(count(partial, Kind::CoulombRecurrence, Mode::Mixed) == 7);
    partial.output.converged = true;
    partial.publish();
    assert(!partial.output.precision_work.complete && !partial.output.precision_work.operator_inventory_complete);
    assert(partial.output.precision_work.events.size() == 1);
  }
  Owner overflow;
  overflow.record_precision_operator(Kind::CoulombRecurrence, Dtype::Fp32, Mode::Mixed,
                                      std::numeric_limits<std::uint64_t>::max());
  bool rejected = false;
  try { overflow.record_precision_operator(Kind::CoulombRecurrence, Dtype::Fp32, Mode::Mixed); }
  catch (const std::overflow_error&) { rejected = true; }
  assert(rejected && !overflow.output.precision_work.complete);
}
"""
    )
    cpp = tmp_path / "component_census.cpp"
    executable = tmp_path / "component_census"
    cpp.write_text(harness)
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(cpp),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(executable)], check=True, timeout=10)


def test_component_schedule_limits_lowering_to_qualified_density_families(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    definitions = "\n".join(
        (
            _method(source, "constexpr bool is_semilocal_family("),
            _method(source, "struct CudaKsPrecisionSchedule") + ";",
            _method(
                source, "CudaKsPrecisionSchedule resolve_cuda_ks_precision_schedule("
            ),
        )
    )
    harness = (
        r"""
#include <cassert>
#include <optional>
#include <stdexcept>
#include "generativeqc/generativeqc.h"
#include "dft/semilocal_family.hpp"
using namespace generativeqc::dft;
"""
        + definitions
        + r"""
int main() {
  for (const auto family : {SemilocalFamily::Lda, SemilocalFamily::Pbe,
                            SemilocalFamily::R2scan, SemilocalFamily::B3lyp}) {
    const auto code = semilocal_family_code(family);
    const auto strict = resolve_cuda_ks_precision_schedule(GENERATIVEQC_PRECISION_FP64, code, false, false);
    assert(!strict.any_mixed() && !strict.mixed_coulomb && !strict.mixed_density_contraction);
    assert(!resolve_cuda_ks_precision_schedule(std::nullopt, code, false, false).any_mixed());
    const auto automatic = resolve_cuda_ks_precision_schedule(GENERATIVEQC_PRECISION_AUTO, code, false, false);
    assert(automatic.any_mixed() && automatic.mixed_coulomb);
    assert(automatic.mixed_density_contraction == (family != SemilocalFamily::B3lyp));
    for (const bool fitted : {false, true}) {
      bool rejected = false;
      try { resolve_cuda_ks_precision_schedule(GENERATIVEQC_PRECISION_AUTO, code, fitted, !fitted); }
      catch (const std::invalid_argument&) { rejected = true; }
      assert(rejected); // Fitted Coulomb and nonlocal composition remain fail-closed.
    }
  }
  bool rejected = false;
  try { resolve_cuda_ks_precision_schedule(static_cast<generativeqc_precision_mode>(999), 0, false, false); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
"""
    )
    cpp = tmp_path / "component_schedule.cpp"
    executable = tmp_path / "component_schedule"
    cpp.write_text(harness)
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            str(cpp),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(executable)], check=True, timeout=10)
