"""The source-aware CC force admission must bind the actual nuclear system."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _function(source, signature):
    start = source.index(signature)
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_force_source_identity_precedes_admission_and_reads(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/cc/rccsdt_force.cpp").read_text()
    helper = (
        _function(source, "bool force_source_matches_system(")
        if "bool force_source_matches_system(" in source
        else ""
    )
    wrappers = "\n".join(
        _function(source, f"RccsdtForcePlan plan_{name}_force_cpu(")
        for name in ("rccsd", "rccsdt")
    )
    code = (
        r"""
#include <iostream>
#include <stdexcept>
#include "hf/reference.hpp"
#include "integrals/electron_interaction_source.hpp"
namespace generativeqc::cc {
struct Problem {};
struct SolverResult {};
struct RccsdtForcePlan {};
unsigned planner_calls = 0;
RccsdtForcePlan plan_relaxed_rccsd_force_cpu(
    const core::System&, const hf::PhysicalReference&, const Problem&,
    const SolverResult&, std::size_t, bool, std::size_t) {
  ++planner_calls;
  return {};
}
"""
        + helper
        + "\n"
        + wrappers
        + r"""
}
class Source final : public generativeqc::integrals::ElectronInteractionSource {
 public:
  explicit Source(const generativeqc::core::System& system) : system_(system) {}
  const generativeqc::core::System& orbital() const override { return system_; }
  std::size_t nbf() const override { return 2; }
  std::size_t naux() const override { return 0; }
  std::size_t retained_numeric_bytes() const override { return 0; }
  bool supports(Operator op) const noexcept override { return op == Operator::eri; }
  void read(Operator, const std::array<std::size_t,4>&,
            const std::array<std::size_t,4>&, double*, std::size_t) const override {
    ++reads;
  }
  mutable unsigned reads = 0;
 private:
  const generativeqc::core::System& system_;
};
int main() {
  using namespace generativeqc;
  core::System system;
  system.atoms = {{1, {0., 0., 0.}, 0}, {1, {0., 0., 1.4}, 0}};
  system.shells = {{0, 0, {{1., 1.}}}, {1, 0, {{1., 1.}}}};
  system.electron_count = 2;
  hf::PhysicalReference reference;
  reference.nbf = 2;
  for (unsigned mutation = 0; mutation < 16; ++mutation) {
    auto other = system;
    switch (mutation) {
      case 1: other.atoms[1].position[2] += 0.1; break;
      case 2: ++other.atoms[0].atomic_number; break;
      case 3: ++other.atoms[0].ecp_core; break;
      case 4: ++other.shells[0].angular_momentum; break;
      case 5: ++other.shells[0].atom_index; break;
      case 6: other.shells[0].primitives[0].exponent += 0.1; break;
      case 7: other.shells[0].primitives[0].coefficient += 0.1; break;
      case 8: other.shells[0].primitives.push_back({0.2, 0.1}); break;
      case 9: other.shells.pop_back(); break;
      case 10: other.atoms.pop_back(); break;
      case 11: other.basis_representation = GENERATIVEQC_BASIS_SPHERICAL; break;
      case 12: ++other.charge; break;
      case 13: ++other.multiplicity; break;
      case 14: ++other.electron_count; break;
      case 15: other.ecp_terms.push_back({0, -1, 0, 1., 1.}); break;
    }
    Source source(other);
    for (const bool triples : {false, true}) {
      cc::planner_calls = 0;
      bool rejected = false;
      try {
        if (triples)
          (void)cc::plan_rccsdt_force_cpu(system, source, reference, {}, {}, 1024);
        else
          (void)cc::plan_rccsd_force_cpu(system, source, reference, {}, {}, 1024);
      } catch (const std::invalid_argument&) { rejected = true; }
      if (rejected != (mutation != 0) || cc::planner_calls != (mutation == 0) || source.reads) {
        std::cerr << "force source admitted wrong system: " << mutation << '\n';
        return 1;
      }
    }
  }
  std::cout << "32 source identity admission cases passed\n";
}
"""
    )
    path = tmp_path / "source.cpp"
    path.write_text(code)
    exe = tmp_path / "source"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            str(path),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
