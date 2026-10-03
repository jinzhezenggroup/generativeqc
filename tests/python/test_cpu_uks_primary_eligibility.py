"""Exercise the actual CPU UKS eligibility predicate without a native library.

The production maximum norm and method-owned callable are extracted unchanged.
The generic controller's ordering, budget, CURRENT retention, move counts and
scalar-gate short circuit are covered by test_self_consistent.cpp.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_cpu_uks_primary_eligibility(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    cache = shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires ccache and a host C++ compiler")
    source = (ROOT / "src/dft/uks.cpp").read_text()
    start = source.index(
        "      [&](const ::generativeqc::solver::SelfConsistentProgress&,"
    )
    end = source.index("\n      });", start)
    predicate = source[start:end] + "\n      }"
    algebra = (ROOT / "src/scf/reference/mean_field.cpp").read_text()
    start = algebra.index("double residual_max_abs(")
    norm = algebra[start : algebra.index("\n\n", start)]
    program = (
        r"""
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "solver/self_consistent.hpp"
using Matrix = std::vector<double>;
enum class FockBackend { Cpu, Cuda };
struct UksLoopEvaluation {
  Matrix alpha_residual, beta_residual;
  double energy{}, state_rms{}, residual_rms{};
};
"""
        + norm
        + r"""
int main() {
  struct { FockBackend backend = FockBackend::Cpu; } strategy;
  struct { double density_tolerance{}; } options;
  const auto eligible =
"""
        + predicate
        + r""";
  const generativeqc::solver::SelfConsistentProgress progress{};
  for (double tolerance : {1e-12, 1e-10, 1e-9, 1e-7}) {
    options.density_tolerance = tolerance;
    const double gate = std::min(1e-8, tolerance);
    for (bool alpha : {false, true}) {
      UksLoopEvaluation evaluation{Matrix(4096), Matrix(4096)};
      Matrix& residual = alpha ? evaluation.alpha_residual : evaluation.beta_residual;
      residual[1] = gate;
      residual[64] = -gate;
      if (!eligible(progress, evaluation)) throw std::runtime_error("equality rejected");
      residual[1] = std::nextafter(gate, 0.0);
      if (!eligible(progress, evaluation)) throw std::runtime_error("below gate rejected");
      residual[1] = std::nextafter(gate, std::numeric_limits<double>::infinity());
      const double rms = std::hypot(residual[1], residual[64]) / 64.0;
      if (!(rms < std::min(1e-9, tolerance)))
        throw std::runtime_error("fixture is not RMS-small/maximum-large");
      if (eligible(progress, evaluation)) throw std::runtime_error("above gate accepted");
      for (double value : {std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}) {
        residual[1] = value;
        if (eligible(progress, evaluation)) throw std::runtime_error("nonfinite accepted");
        strategy.backend = FockBackend::Cuda;
        if (!eligible(progress, evaluation)) throw std::runtime_error("non-CPU path changed");
        strategy.backend = FockBackend::Cpu;
      }
    }
  }
  // Integrated behavioral control: the original/default RMS-only decision
  // accepts iteration two, but the exact UKS predicate must retain the primary
  // loop until the physical maximum passes, or exhaust the original budget.
  options.density_tolerance = 1e-10;
  for (bool use_gate : {false, true}) {
    for (unsigned good_at : {3U, 99U}) {
      const generativeqc::solver::SelfConsistentPolicy policy{4, 1e-12, 1e-10, 1e-10, true};
      unsigned builds = 0, records = 0, accepts = 0;
      const auto evaluate = [&](double current, unsigned iteration) {
        ++builds;
        if (current != double(iteration - 1)) throw std::runtime_error("CURRENT mismatch");
        UksLoopEvaluation value{Matrix(4096), Matrix(4096)};
        value.alpha_residual[1] = iteration < good_at ? 2e-10 : 5e-11;
        value.residual_rms = value.alpha_residual[1] / 64.0;
        return value;
      };
      const auto record = [&](const generativeqc::solver::SelfConsistentProgress& progress,
                              const UksLoopEvaluation&) {
        ++records;
        if (records != progress.iteration || builds != records || accepts + 1 != records)
          throw std::runtime_error("build/history ordering changed");
        const bool expected = progress.iteration > 1 && (!use_gate || progress.iteration >= good_at);
        if (progress.converged != expected) throw std::runtime_error("wrong recorded decision");
      };
      const auto accept = [&](double current, UksLoopEvaluation,
                              const generativeqc::solver::SelfConsistentProgress& progress) {
        ++accepts;
        // Same terminal CURRENT policy used by UKS.
        if (progress.converged || progress.iteration == policy.max_iterations) return current;
        return current + 1.0;
      };
      const auto outcome = use_gate
          ? generativeqc::solver::run_self_consistent(0.0, policy, evaluate, accept, record, eligible)
          : generativeqc::solver::run_self_consistent(0.0, policy, evaluate, accept, record);
      const unsigned iterations = use_gate ? std::min(good_at, policy.max_iterations) : 2U;
      if (outcome.converged != (!use_gate || good_at <= policy.max_iterations) ||
          outcome.progress.iteration != iterations || builds != iterations || records != iterations ||
          accepts != iterations || outcome.state != double(iterations - 1))
        throw std::runtime_error("RMS-only versus UKS eligibility/budget regression");
    }
  }
  return 0;
}
"""
    )
    cpp = tmp_path / "eligibility.cpp"
    executable = tmp_path / "eligibility"
    cpp.write_text(program)
    env = {**os.environ, "CCACHE_BASEDIR": str(ROOT)}
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++17",
            "-O0",
            "-I",
            str(ROOT / "src"),
            str(cpp),
            "-o",
            str(executable),
        ],
        check=True,
        env=env,
    )
    subprocess.run([str(executable)], check=True, env=env)
