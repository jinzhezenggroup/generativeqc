"""Exercise real GFN2 planning/mixing after shared workspace-owner cutover."""

import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src/xtb/native/src"
CONSUMERS = (
    ROOT / "src/methods/gfn2_electronic_update.cpp",
    ROOT / "src/solver/cpu/prepared_spectral.cpp",
    "model/gfn2/scc_driver.cpp",
    "model/gfn2/d4.cpp",
    "backends/cuda/gfn2_scc_setup_eigensolver.cu",
    "backends/cuda/gfn2_scc_setup_inputs.cu",
    "backends/cuda/gfn2_scc_setup_topology.cu",
    "runtime/gfn2_cuda_topology_staging.cu",
)


@pytest.mark.parametrize("path", CONSUMERS)
def test_runtime_planners_use_shared_checked_sizes(path: str | Path) -> None:
    source = (NATIVE / path).read_text()
    assert '#include "runtime/bounded_workspace.hpp"' in source
    assert "using ::generativeqc::runtime::checked_multiply;" in source
    assert not re.search(
        r"bool checked_(?:add|multiply)(?:_size)?\(std::size_t \w+, "
        r"std::size_t \w+, std::size_t&",
        source,
    )


def test_real_ragged_mixer_planning_and_execution(
    tmp_path: Path, required_native_cxx: "NativeCxx"
) -> None:
    """Compile the complete production consumer, not a copied planner surrogate."""
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_ordered_history_native.py"),
            "--output-directory",
            str(tmp_path),
            "--backend",
            "cpu",
        ],
        check=True,
        timeout=30,
    )
    probe = tmp_path / "probe.cpp"
    probe.write_text(r"""
#include "solver/cpu/johnson_broyden.hpp"
#include "runtime/bounded_workspace.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
using namespace generativeqc::solver::cpu;
using generativeqc::solver::BroydenPolicy;
using generativeqc::solver::BroydenStatusEncoding;
using generativeqc::solver::BroydenResult;
int main() {
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  for (std::size_t a : {std::size_t(0),std::size_t(1),std::size_t(63),std::size_t(64),max/2,max-1,max}) {
    for (std::size_t b : {std::size_t(0),std::size_t(1),std::size_t(2),std::size_t(64),max}) {
      std::size_t result = 123;
      bool ok = generativeqc::runtime::checked_add(a,b,result);
      assert(ok == (a <= max-b));
      assert(result == (ok ? a+b : 123));
      result = 123;
      ok = generativeqc::runtime::checked_multiply(a,b,result);
      assert(ok == (!b || a <= max/b));
      assert(result == (ok ? a*b : 123));
      result = a;
      ok = generativeqc::runtime::checked_add(result,b,result);
      assert(result == (ok ? a+b : a));
    }
  }
  std::int64_t offsets[] = {0,2,5};
  BroydenVectorLayoutView layout;
  layout.batch_size=2; layout.workspace_size_bytes=64;
  layout.workspace_alignment=64; layout.field_count=1;
  layout.fields[0]={0,40,5,offsets,3};
  BroydenPlan plan; std::string error;
  assert(make_broyden_plan(layout,BroydenPolicy{4,0.4,1e-8,1e-7},BroydenStatusEncoding{},plan,error)==BroydenResult::success);
  assert(plan.vector_offsets()==std::vector<std::int64_t>({0,2,5}));
  assert(plan.total_vector_elements()==5 && plan.maximum_vector_elements()==3);
  using Buffer = std::unique_ptr<void,decltype(&std::free)>;
  Buffer state_memory(std::aligned_alloc(64,plan.state_size_bytes()),std::free);
  Buffer scratch(std::aligned_alloc(64,plan.workspace_size_bytes()),std::free);
  BroydenState state; BroydenWorkspace workspace;
  assert(bind_broyden_state(plan,state_memory.get(),plan.state_size_bytes(),state,error)==BroydenResult::success);
  assert(bind_broyden_workspace(plan,scratch.get(),plan.workspace_size_bytes(),workspace,error)==BroydenResult::success);
  BroydenWorkspace bad;
  assert(bind_broyden_workspace(plan,scratch.get(),plan.workspace_size_bytes()-1,bad,error)!=BroydenResult::success);
  alignas(64) double values[8] = {};
  BroydenVectorView vector;
  vector.workspace_base=values; vector.workspace_size_bytes=64;
  vector.fields[0]=values; vector.field_count=1;
  assert(initialize_broyden_state(plan,vector,state,error)==BroydenResult::success);
  for (int i=0;i<5;++i) values[i]=double(i+1);
  assert(mix_broyden_batch(plan,vector,state,workspace,error)==BroydenResult::success);
  for(int i=0;i<5;++i) assert(std::abs(values[i]-0.4*double(i+1))<1e-14);
  assert(state.iterations[0]==1 && state.iterations[1]==1);
  const auto old_identity=plan.identity();
  layout.fields[0].element_count=std::numeric_limits<std::int64_t>::max();
  offsets[2]=layout.fields[0].element_count;
  assert(make_broyden_plan(layout,BroydenPolicy{4,0.4,1e-8,1e-7},BroydenStatusEncoding{},plan,error)!=BroydenResult::success);
  assert(plan.identity()==old_identity);
}
""")
    executable = required_native_cxx.build_executable(
        [probe, ROOT / "src/solver/cpu/johnson_broyden.cpp"],
        tmp_path / "probe",
        compile_args=[
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(tmp_path),
        ],
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_shared_johnson_planner_uses_shared_checked_sizes() -> None:
    source = (ROOT / "src/solver/cpu/johnson_broyden.cpp").read_text()
    assert '#include "runtime/bounded_workspace.hpp"' in source
    assert "using ::generativeqc::runtime::checked_multiply;" in source


def test_method_free_johnson_owner_contract(
    tmp_path: Path, required_native_cxx: "NativeCxx"
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_ordered_history_native.py"),
            "--output-directory",
            str(tmp_path),
            "--backend",
            "cpu",
        ],
        check=True,
        timeout=30,
    )
    executable = required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_johnson_broyden_owner.cpp",
            ROOT / "src/solver/cpu/johnson_broyden.cpp",
        ],
        tmp_path / "johnson-owner-contract",
        # Deliberately no src/xtb/native include directories or method owners.
        compile_args=[
            "-std=c++17",
            "-O2",
            "-I" + str(ROOT / "src"),
            "-I" + str(tmp_path),
        ],
    )
    subprocess.run([str(executable)], check=True, timeout=30)
