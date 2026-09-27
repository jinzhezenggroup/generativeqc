"""Host-execute the shared preparation gate without allocating molecular tensors."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r"""
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
int allocations=0, executions=0;
constexpr int VIBEQC_BACKEND_CPU_REFERENCE=1, VIBEQC_STATUS_OUT_OF_MEMORY=2;
namespace core { struct System {}; }
namespace runtime {
struct ExecutionContext {
  bool cuda=false;
  int backend() const { return cuda ? 3 : VIBEQC_BACKEND_CPU_REFERENCE; }
  bool cuda_requested() const { return cuda; }
};
}
struct vibeqc_method_descriptor { std::size_t budget=100; bool valid=true; };
struct MethodError : std::runtime_error {
  MethodError(int,const char* msg) : std::runtime_error(msg) {}
};
struct Reference { int diis_history=8; double screening_tolerance=0; };
namespace scf {
enum class FockSpin { Restricted };
enum class FockBackend { Cpu };
int make_hf_fock_spec(FockSpin) { return 0; }
int resolve_fock_build(int,FockBackend,double) { return 0; }
struct PreparedFockPlan {
  PreparedFockPlan(const core::System&,std::nullptr_t,int) { ++allocations; }
};
}
namespace posthf {
std::size_t rhf_reference_capacity(const core::System&,int,bool) { return 80; }
}
struct RccsdNativeState { bool cached; };
void validate_descriptor(const vibeqc_method_descriptor& d,const runtime::ExecutionContext&) {
  if (!d.valid) throw std::invalid_argument("invalid descriptor");
}
std::size_t correlation_budget(const vibeqc_method_descriptor& d) { return d.budget; }
int cc_options(const vibeqc_method_descriptor&,std::size_t) { return 0; }
Reference reference_options(const vibeqc_method_descriptor&,std::size_t) { return {}; }
RccsdNativeState execute_rccsd_prepared(runtime::ExecutionContext&,const core::System&,
                                      Reference,int,std::size_t,scf::PreparedFockPlan* p) {
  ++executions;
  return {p != nullptr};
}
"""

MAIN = r"""
int main(int argc,char** argv) {
  if (argc != 2) return 1;
  const int mode=std::atoi(argv[1]);
  runtime::ExecutionContext execution;
  core::System system;
  vibeqc_method_descriptor descriptor;
  std::unique_ptr<scf::PreparedFockPlan> cache;
  if (mode == 0) descriptor.budget=79;
  if (mode == 1) descriptor.valid=false;
  if (mode == 2) execution.cuda=true;
  try {
    auto result=run_rccsd_native_state(execution,system,descriptor,mode==3 ? nullptr : &cache);
    if (mode < 2) return 2;
    if (result.cached != (mode >= 4)) return 3;
    if (allocations != (mode >= 4 ? 1 : 0) || executions != 1) return 4;
    if (mode >= 4) {
      auto* first=cache.get();
      result=run_rccsd_native_state(execution,system,descriptor,&cache);
      if (!result.cached || cache.get()!=first || allocations!=1 || executions!=2) return 5;
      descriptor.budget=79;
      try { (void)run_rccsd_native_state(execution,system,descriptor,&cache); return 6; }
      catch (const MethodError&) {}
      if (cache.get()!=first || allocations!=1 || executions!=2) return 7;
    }
  } catch (const std::exception&) {
    if (mode >= 2 || allocations || executions || cache) return 8;
  }
}
"""


def test_rccsd_admits_before_creating_or_reusing_exact_cache(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    start = source.index("RccsdNativeState run_rccsd_native_state(")
    end = source.index("\nvibeqc_status validate_rccsd_system", start)
    program = PREFIX + source[start:end] + MAIN
    path, executable = tmp_path / "probe.cpp", tmp_path / "probe"
    path.write_text(program)
    compiled = subprocess.run(
        [compiler, "-std=c++20", "-O0", str(path), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    for mode in range(5):
        result = subprocess.run(
            [str(executable), str(mode)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, (mode, result.returncode, result.stderr)
    consumer = (ROOT / "src/methods/rccsdt_method.cpp").read_text()
    assert (
        "run_rccsd_native_state(execution_, system_, descriptor_, &cpu_exact_plan_)"
        in consumer
    )
    assert "make_unique<scf::PreparedFockPlan>" not in consumer
