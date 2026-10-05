"""Execute composed screen/accelerator policy with real GMRES and small SPD operators."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from _cc_owner_test_support import compile_owner

ROOT = Path(__file__).resolve().parents[2]


def test_screened_acceleration_preserves_exact_gates_and_attempt_work(
    tmp_path: Path,
) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = (ROOT / "src/hf/rhf_frame_response.cu").read_text()
    start = source.index("    auto physical =")
    end = source.index("    for (std::size_t i = 0; i < o; ++i)", start)
    policy = source[start:end]
    probe = tmp_path / "composition.cpp"
    probe.write_text(PREFIX + policy + TAIL)
    executable = tmp_path / "composition"
    compile_owner(
        compiler,
        tmp_path,
        [probe, ROOT / "src/response/native_gmres.cpp"],
        executable,
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


PREFIX = r"""
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "hf/rhf_frame_response.hpp"
using namespace generativeqc;
using Clock = std::chrono::steady_clock;
constexpr double kResidualTolerance = 1e-10;
double seconds(Clock::time_point t) {
  return std::chrono::duration<double>(Clock::now()-t).count();
}
struct Inverse {
  bool refuse;
  void apply(std::span<const double> x, std::span<double> y) {
    if (refuse) throw std::runtime_error("optional inverse failed");
    y[0] = x[0]/2.; y[1] = x[1]/3.;
  }
};
struct Cache {
  std::vector<double> guess{.4,.6};
  auto initial_guess(std::span<const double>) { return std::span<const double>(guess); }
};
struct Owner {
  std::size_t actions{}, audits{}, screened{};
  bool fail_physical{}, fail_audit{};
  void apply(std::span<const double> x, std::span<double> y, bool scalar=false,
             double threshold=0.) {
    if (fail_physical) throw std::runtime_error("physical failure");
    ++actions; if(scalar) ++audits; if(threshold>0) ++screened;
    y[0]=(2.+threshold)*x[0]; y[1]=(3.+threshold)*x[1];
    if(scalar && fail_audit) y[0]+=.5;
  }
};
void run(double screen, int accelerator, bool cached, bool fail_physical=false,
         bool fail_audit=false) {
  const std::size_t o=1,v=2;
  hf::RHFFrameResponseResult result;
  result.applied_screening=screen;
  response::GmresOptions options;
  options.absolute_tolerance=1e-12; options.relative_tolerance=0;
  options.true_residual_every=options.restart;
  const auto zplan=response::prepare_gmres(2,options);
  std::vector<double> rhs{2.,6.},diagonal{2.,3.},exact_image;
  Owner owner;owner.fail_physical=fail_physical;owner.fail_audit=fail_audit;
  std::optional<Inverse> inverse;
  if(accelerator) inverse=Inverse{accelerator==2};
  Cache cache;auto* recycling=cached ? &cache : nullptr;
  auto phase=Clock::now();
"""
TAIL = r"""
  assert(std::abs(z.solution[0]-1.)<1e-10 && std::abs(z.solution[1]-2.)<1e-10);
  assert(z.operator_actions == owner.actions-owner.audits);
  assert(result.recycled_guess == cached);
  assert(result.orbital_residual<1e-10);
  assert(result.exact_refinements == (screen>0 ? 1 : 0));
  assert(result.preconditioner_fallback == (accelerator==2 || (accelerator && screen>0)));
  if(screen>0) {
    assert(result.screened_operator_actions==owner.screened);
    if(accelerator!=2) assert(owner.screened>0);
    assert(result.screened_operator_actions<z.operator_actions);
  }
  if(cached) {
    assert(exact_image.size()==2);
    assert(std::abs(exact_image[0]-2.)<1e-10 && std::abs(exact_image[1]-6.)<1e-10);
  }
}
int main() {
  for(double screen: {0.,.1}) for(int accelerator: {0,1,2}) for(bool cached: {false,true})
    run(screen,accelerator,cached);
  for(double screen: {0.,.1}) for(bool audit: {false,true}) {
    bool rejected=false;
    try {run(screen,1,true,!audit,audit);} catch(const std::runtime_error& e) {
      rejected=true;
      assert(std::string(e.what())==(audit ? "RHF independent Z residual failed" : "physical failure"));
    }
    assert(rejected);
  }
}
"""
