"""Host executable coverage and wiring guards for homogeneous force passes."""

import itertools
import shutil
import subprocess
from pathlib import Path

import pytest
from generativeqc_compiler.integral.cuda_schedule import ScheduleKind
from generativeqc_compiler.integral.direct_bounded_force_schedule import (
    emit_direct_bounded_force_schedule_header,
    homogeneous_bounded_force_passes,
)

ROOT = Path(__file__).resolve().parents[2]


def test_all_through_f_quartets_have_exactly_one_owner() -> None:
    passes = homogeneous_bounded_force_passes()
    assert len(passes) == 5
    for shells in itertools.product(range(4), repeat=4):
        assert sum(item.accepts(sum(shells)) for item in passes) == 1
    assert not any(item.accepts(13) for item in passes)
    assert not any(item.accepts(-1) for item in passes)
    assert passes[0].schedule.kind == ScheduleKind.THREAD_TASKS
    assert passes[0].schedule.tasks_per_warp == 32
    assert all(item.schedule.tasks_per_warp == 1 for item in passes[1:])
    assert all(item.schedule.kind == ScheduleKind.SHELL_TASK for item in passes[1:])
    assert all(item.cta_threads == 256 for item in passes)


def test_generated_membership_executes_without_cuda(tmp_path: Path) -> None:
    compiler, ccache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or ccache is None:
        pytest.skip("host compiler and ccache required")
    subprocess.run([ccache, "--version"], check=True, capture_output=True, timeout=10)
    header = emit_direct_bounded_force_schedule_header()
    assert header == emit_direct_bounded_force_schedule_header()
    (tmp_path / "policy.hpp").write_text(header)
    source = tmp_path / "policy.cpp"
    source.write_text(
        r"""
#include "policy.hpp"
#include <cassert>
using namespace generativeqc::scf::cuda_execution;
int main() {
  static_assert(kHomogeneousBoundedForceThreads == 256);
  static_assert(!BoundedForcePassPolicy<0>::warp);
  static_assert(!BoundedForcePassPolicy<4>::scalar);
  static_assert(BoundedForcePassPolicy<4>::fixed);
  static_assert(BoundedForcePassPolicy<5>::fixed);
  static_assert(BoundedForcePassPolicy<6>::fixed);
  static_assert(!BoundedForcePassPolicy<7>::fixed);
  for (unsigned order = 0; order <= 13; ++order) {
    unsigned owners = 0;
#define COUNT_PASS(tag) owners += BoundedForcePassPolicy<tag>::accepts(order)
    GENERATIVEQC_FOR_EACH_HOMOGENEOUS_FORCE_PASS(COUNT_PASS)
#undef COUNT_PASS
    assert(owners == (order <= 12 ? 1U : 0U));
    assert(BoundedForcePassPolicy<-1>::accepts(order));
  }
}
"""
    )
    binary = tmp_path / "policy"
    subprocess.run(
        [
            ccache,
            compiler,
            "-std=c++20",
            "-Wall",
            "-Werror",
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_fixed_workers_bypass_runtime_dispatch_without_changing_math() -> None:
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text()
    admission = source.index("!generated_class && PassPolicy::accepts(angular_order)")
    assert admission < source.index("profile_bounded_direct_shell_quartet(")
    assert "if constexpr (PassPolicy::scalar)" in source
    assert "if constexpr (PassPolicy::warp)" in source
    start = source.index("if constexpr (PassPolicy::fixed)")
    fixed = source[start : source.index("} else if (radial_operator", start)]
    assert "contract_two_electron_force_quartet_subtile_scaled<" in fixed
    assert "Unrestricted, ForcePass, true>" in fixed
    assert "contract_bounded_direct_force_subtile_scaled" not in fixed
    assert "if (separate_sources && purpose == DirectScreeningPurpose::Force)" in source
    assert "GENERATIVEQC_FOR_EACH_HOMOGENEOUS_FORCE_PASS(" in source


def test_selection_is_opt_in_and_codegen_has_a_build_owner() -> None:
    policy = (ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    assert 'selected("GENERATIVEQC_BOUNDED_FORCE_SCHEDULE", "homogeneous")' in policy
    owner = (ROOT / "src/scf/cuda/direct_coulomb.cpp").read_text()
    assert "homogeneous ? kHomogeneousBoundedForcePasses.size() : 1U" in owner
    assert "homogeneous ? kHomogeneousBoundedForcePasses[pass] : -1" in owner
    cmake = (ROOT / "cmake/GenerativeQCGeneratedSources.cmake").read_text()
    assert "tools/generate_direct_bounded_force_schedule.py" in cmake
    assert "integral/direct_bounded_force_schedule.py" in cmake
    device_link = (ROOT / "cmake/GenerativeQCCuda.cmake").read_text()
    assert '"${GENERATIVEQC_DIRECT_BOUNDED_FORCE_SCHEDULE_HEADER}"' in device_link
