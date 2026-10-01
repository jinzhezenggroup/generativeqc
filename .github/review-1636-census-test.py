"""Compile the real component precision census with the public provenance types."""

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


def test_component_precision_census_preserves_events_and_recurrences(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = (ROOT / "src/dft/cuda_ks.cpp").read_text()
    methods = "\n".join(
        _method(source, name)
        for name in (
            "  void record_fock_precision_work(",
            "  void record_precision_retry()",
        )
    )
    finish = _method(source, "  bool finish_legacy()")
    start = finish.index("    ++output.iterations;")
    end = finish.index("    output.precision.requested_mode", start)
    census = finish[start:end]
    harness = r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "scf/precision_work.hpp"
namespace scf = generativeqc::scf;
struct Owner {
  bool pending_mixed_coulomb{}, pending_mixed_density{}, mixed_precision_executed{};
  unsigned refinement_iterations{};
  std::uint64_t owner{17}, solve_epoch{3}, generation{5};
  struct {
    std::uint32_t iterations{};
    std::uint64_t fock_builds{};
    struct { std::uint64_t mixed_stage_fock_builds{}, strict_stage_fock_builds{}; } precision;
    scf::PrecisionWork precision_work;
  } output;
""" + methods + r"""
  void complete(bool coulomb, bool density, std::uint64_t mixed_coulomb_recurrences) {
    pending_mixed_coulomb = coulomb;
    pending_mixed_density = density;
    mixed_precision_executed |= coulomb || density;
""" + census + r"""
  }
};
int main() {
  Owner p;
  p.complete(false, false, 0);
  p.complete(true, false, 11);
  p.complete(true, true, 13);
  p.complete(false, true, 99); // Density-only work must never enter the J census.
  p.complete(false, false, 0);
  assert(p.output.fock_builds == 5 && p.output.iterations == 5);
  assert(p.output.precision.mixed_stage_fock_builds == 3);
  assert(p.output.precision.strict_stage_fock_builds == 2);
  assert(p.refinement_iterations == 1);
  const auto& work = p.output.precision_work;
  assert(work.events.size() == 5 && work.operators.size() == 1);
  assert(work.operators[0].count == 24);
  assert(work.operators[0].kind == scf::PrecisionOperatorKind::CoulombRecurrence);
  assert(work.operators[0].compute == scf::PrecisionDtype::Fp32);
  assert(work.operators[0].storage == scf::PrecisionDtype::Fp64);
  assert(work.operators[0].accumulation == scf::PrecisionDtype::Fp64);
  assert(work.operators[0].reduction == scf::PrecisionDtype::Fp64);
  assert(!work.complete && !work.operator_inventory_complete);
  for (std::size_t i = 0; i != work.events.size(); ++i) {
    const auto& event = work.events[i];
    assert(event.sequence == i && event.iteration == i + 1);
    assert(event.owner_id == p.owner && event.solve_epoch == p.solve_epoch);
    assert(event.state_generation == p.generation);
    const bool mixed = i > 0 && i < 4;
    assert(event.kind == (mixed ? scf::PrecisionWorkEventKind::MixedFock
                               : scf::PrecisionWorkEventKind::StrictFock));
    assert(event.phase == (i == 4 ? scf::PrecisionWorkPhase::Refinement
                                 : scf::PrecisionWorkPhase::Scf));
  }
  p.record_precision_retry();
  assert(work.events.back().sequence == 5);
  assert(work.events.back().kind == scf::PrecisionWorkEventKind::Retry);
  assert(work.events.back().phase == scf::PrecisionWorkPhase::Retry);
  assert(work.operators[0].count == 24);
  Owner screened;
  screened.complete(true, false, 0);
  assert(screened.output.precision.mixed_stage_fock_builds == 1);
  assert(screened.output.precision_work.operators.empty());
  p.pending_mixed_coulomb = true;
  p.output.precision_work.operators[0].count = std::numeric_limits<std::uint64_t>::max();
  bool overflow = false;
  try { p.record_fock_precision_work(1); }
  catch (const std::overflow_error&) { overflow = true; }
  assert(overflow);
}
"""
    cpp = tmp_path / "component_census.cpp"
    executable = tmp_path / "component_census"
    cpp.write_text(harness)
    subprocess.run(
        [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "src"),
         "-I", str(ROOT / "include"), str(cpp), "-o", str(executable)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=10)
