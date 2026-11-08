"""Regression guards for compiler-owned resident-PSSS schedule policy."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.cuda_schedule import ScheduleKind
from generativeqc_compiler.integral.direct_resident_schedule import (
    DirectPersistentScheduleProfile,
    emit_direct_resident_psss_schedule_header,
    resident_psss_schedule_profile,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("force_count,fock_count", [(7, 6), (3, 3), (4, 5)])
def test_persistent_queue_profile_drives_compiled_arena(
    tmp_path: Path, required_native_cxx: NativeCxx, force_count: int, fock_count: int
) -> None:
    """Queue storage follows the emitted profile, including the independent FP32 head."""
    (tmp_path / "generated_direct_resident_psss_schedule.cuh").write_text(
        emit_direct_resident_psss_schedule_header(
            DirectPersistentScheduleProfile(force_count, fock_count)
        ),
        encoding="utf-8",
    )
    source = tmp_path / "queue_layout.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include "generated_direct_resident_psss_schedule.cuh"
#include "scf/cuda/arena.hpp"
int main() {
  using namespace generativeqc::scf::cuda_execution;
  ArenaLayout layout{};
  assert(make_layout(1, 7, 8, 1, 3, 6, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                     3, 2, 0, 1, false, true, false, false, false, false, true, false, layout));
  assert(layout.fp32_persistent_fock_task_heads - layout.persistent_fock_task_heads ==
         kPersistentFockAngularOrderCount * sizeof(std::uint32_t));
  assert(layout.persistent_force_task_heads - layout.fp32_persistent_fock_task_heads ==
         kPersistentFockAngularOrderCount * sizeof(std::uint32_t));
  const auto force_end = layout.persistent_force_task_heads +
                        kPersistentForceAngularOrderCount * sizeof(std::uint32_t);
  assert(layout.generated_shell_tasks >= force_end && layout.bytes >= force_end);
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "queue_layout"
    required_native_cxx.build_executable(
        [source, REPOSITORY_ROOT / "src/scf/cuda/arena.cpp"],
        executable,
        compile_args=(
            "-std=c++20",
            "-UNDEBUG",
            f"-I{tmp_path}",
            f"-I{REPOSITORY_ROOT / 'src'}",
            f"-I{REPOSITORY_ROOT / 'include'}",
        ),
    )
    subprocess.run([str(executable)], check=True, timeout=30)


@pytest.mark.parametrize(
    "field", ["force_angular_order_count", "fock_angular_order_count"]
)
@pytest.mark.parametrize("value", [True, 2, 8, 3.5])
def test_persistent_queue_profile_rejects_unsupported_domains(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        DirectPersistentScheduleProfile(**{field: value})


def test_resident_psss_profile_uses_shared_schedule_ir() -> None:
    profile = resident_psss_schedule_profile()
    schedule = profile.schedule

    assert schedule.kind == ScheduleKind.THREAD_TASKS
    assert schedule.block_threads == 128
    assert schedule.component_tile == 3
    assert schedule.tasks_per_warp == 32
    assert schedule.minimum_blocks_per_sm == 4
    assert profile.maximum_bra_primitive_pairs == 64


def test_resident_psss_header_is_compiler_owned() -> None:
    source = emit_direct_resident_psss_schedule_header()

    assert "kResidentPsssThreads = 128U;" in source
    assert "kResidentPsssMinimumBlocksPerSm =" in source
    assert "kResidentPsssMaximumBraPrimitivePairs =" in source
    assert "static_assert(kResidentPsssThreads % 32U == 0U);" in source


def test_native_resident_psss_consumers_have_no_duplicate_schedule_constants() -> None:
    constants = (REPOSITORY_ROOT / "src/scf/cuda/direct_constants.hpp").read_text(
        encoding="utf-8"
    )
    assert "constexpr unsigned kResidentPsssThreads = 128;" not in constants
    assert (
        "constexpr std::size_t kResidentPsssMaximumBraPrimitivePairs = 64;"
        not in constants
    )

    for path in (
        "src/scf/cuda/topology.cpp",
        "src/scf/cuda_rhf.cpp",
        "src/scf/cuda/direct_angular_force.cu",
    ):
        source = (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
        assert '#include "generated_direct_resident_psss_schedule.cuh"' in source

    angular_force = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_angular_force.cu"
    ).read_text(encoding="utf-8")
    assert (
        "__launch_bounds__(kResidentPsssThreads, kResidentPsssMinimumBlocksPerSm)"
        in angular_force
    )
