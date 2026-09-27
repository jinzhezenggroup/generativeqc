"""Execute production source handoffs with counted host owner stand-ins.

The numerical providers and solvers are not run here; this regression checks
which storage survives each stage and the retained-input budget handoff.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
int plan_live=0, view_live=0, raw_live=0;
std::size_t plan_bytes=64;
namespace integrals {
struct ElectronInteractionSource {
  virtual ~ElectronInteractionSource() = default;
  virtual std::size_t retained_numeric_bytes() const = 0;
};
}
namespace scf {
struct PreparedFockPlan {
  PreparedFockPlan() { ++plan_live; }
  ~PreparedFockPlan() { if (view_live) std::abort(); --plan_live; }
};
struct PreparedFockInteractionSourceView : integrals::ElectronInteractionSource {
  explicit PreparedFockInteractionSourceView(const PreparedFockPlan&) { ++view_live; }
  ~PreparedFockInteractionSourceView() { --view_live; }
  std::size_t retained_numeric_bytes() const override { return plan_bytes; }
};
}
namespace posthf {
struct RawSource : integrals::ElectronInteractionSource {
  explicit RawSource(int,const int* = nullptr) { ++raw_live; }
  ~RawSource() { --raw_live; }
  std::size_t retained_numeric_bytes() const override { return 16; }
};
std::size_t checked_add(std::size_t a,std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max()-a)
    throw std::overflow_error("retained-input overflow");
  return a+b;
}
}
struct Problem { std::size_t reference_retained_bytes=100, provider_peak_bytes{}; };
struct State { Problem problem; };
struct Execution { int device_id() const { return 0; } };
Problem build_problem(const integrals::ElectronInteractionSource& source,
                      int,int,bool,int,int&,int&) {
  // The provider already charges the source exactly once during its own phase.
  return {100, source.retained_numeric_bytes()+110};
}
"""


@pytest.fixture(scope="module")
def source_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    mp2 = (ROOT / "src/methods/mp2_method.cpp").read_text()
    cc = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    mp2_setup = mp2.split(
        "      std::unique_ptr<posthf::RawSource> raw_source;", 1
    )[1].split("      const auto corr =", 1)[0]
    force_prefix = mp2.split("      if (compute_forces) {", 1)[1].split(
        "        response::GmresOptions response_options;", 1
    )[0]
    cc_handoff = cc.split(
        "    std::unique_ptr<posthf::RawSource> raw_source;", 1
    )[1].split('    allocation_stage = "CC resident solve";', 1)[0]
    program = (
        PREFIX
        + r"""
int mp2_case(bool prepared,bool compute_forces) {
  std::unique_ptr<scf::PreparedFockPlan> cpu_exact_plan_;
  if (prepared) cpu_exact_plan_=std::make_unique<scf::PreparedFockPlan>();
  auto* prepared_exact=cpu_exact_plan_.get();
  const bool density_fitted_=false;
  int system_=0;
  std::optional<int> auxiliary_;
  std::unique_ptr<posthf::RawSource> raw_source;
"""
        + mp2_setup
        + r"""
  if (!conventional_source) return 1;
  if (prepared && (raw_live || !plan_live || !view_live)) return 2;
  if (compute_forces) {
"""
        + force_prefix
        + r"""
    if (plan_live || view_live || raw_live != 1 || !raw_source) return 3;
  } else if (prepared && (!plan_live || raw_live)) return 4;
  return 0;
}
int cc_case(bool prepared) {
  std::unique_ptr<scf::PreparedFockPlan> owner;
  if (prepared) owner=std::make_unique<scf::PreparedFockPlan>();
  auto* prepared_exact=owner.get();
  State state;
  int system=0, reference_value=0, solver_options=0, provider_work=0, provider_metrics=0;
  const auto* reference=&reference_value;
  const bool cuda=!prepared;
  Execution execution;
  std::unique_ptr<posthf::RawSource> raw_source;
"""
        + cc_handoff
        + r"""
  if (state.problem.reference_retained_bytes != (prepared ? 164u : 100u)) return 5;
  if (state.problem.provider_peak_bytes != (prepared ? 174u : 126u)) return 6;
  if (raw_live || view_live || plan_live != int(prepared)) return 7;
  return 0;
}
int main(int argc,char** argv) {
  if (argc != 2) return 8;
  const int mode=std::atoi(argv[1]);
  int result=0;
  if (mode < 4) result=mp2_case(mode < 2,mode%2);
  else if (mode < 6) result=cc_case(mode==4);
  else {
    plan_bytes=std::numeric_limits<std::size_t>::max();
    try { (void)cc_case(true); return 9; }
    catch (const std::overflow_error&) {}
  }
  if (plan_live || view_live || raw_live) return 10;
  return result;
}
"""
    )
    directory = tmp_path_factory.mktemp("posthf-source-lifetime")
    path, executable = directory / "probe.cpp", directory / "probe"
    path.write_text(program)
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable


@pytest.mark.parametrize("mode", range(7))
def test_posthf_source_lifetime_matches_retained_budget(
    source_probe: Path, mode: int
) -> None:
    process = subprocess.run(
        [str(source_probe), str(mode)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert process.returncode == 0, (mode, process.returncode, process.stderr)
