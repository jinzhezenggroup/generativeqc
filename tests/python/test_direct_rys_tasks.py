"""Task-parallel Rys capability, deterministic emission and fallback contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral.cuda_schedule import ScheduleKind
from generativeqc_compiler.integral.fused_schedule import build_fused_shell_plan
from generativeqc_compiler.integral.k_block import (
    PACKED_RESTRICTED_K_BLOCK_MAX_DOUBLES,
    packed_restricted_k_block_doubles,
    packed_restricted_k_block_eligible,
)
from generativeqc_compiler.integral.lowering.fock_tiled import (
    _packed_restricted_k_block,
)
from generativeqc_compiler.integral.production_emission import emit_production_shard
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_source,
    emit_registry_source,
)
from generativeqc_compiler.integral.production_rys_tasks import (
    direct_rys_task_candidates,
    preferred_rys_task_candidates,
)
from generativeqc_compiler.integral.production_rys_values import (
    direct_rys_value_candidates,
)

if TYPE_CHECKING:
    from generativeqc_compiler.integral.production_selection import KernelSelection

MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "python/generativeqc_compiler/integral/production_shell_classes.json"
)
PROFILE = resolve_production_profile(MANIFEST, "sm_120")
CANDIDATES = direct_rys_task_candidates(PROFILE)


def test_task_inventory_is_bounded_and_does_not_replace_old_rys() -> None:
    """Both execution families remain independently selectable experiments."""
    assert {item.spec.name for item in CANDIDATES} == {
        "ssss",
        "psss",
        "ppss",
        "psps",
        "dsss",
        "dpss",
        "dsps",
        "ppps",
    }
    assert all(item.schedule.kind == ScheduleKind.PACKED_TASKS for item in CANDIDATES)
    assert all(item.schedule.tasks_per_warp == 32 for item in CANDIDATES)
    assert all(item.schedule.block_threads == 128 for item in CANDIDATES)
    assert all(item.integral.derivative is None for item in CANDIDATES)
    assert all(item.spec.component_count <= 27 for item in CANDIDATES)
    assert all(
        item.schedule.kind == ScheduleKind.COMPONENT_LANES
        for item in direct_rys_value_candidates(PROFILE)
    )
    portable = resolve_production_profile(MANIFEST, "sm_120", "portable_cuda")
    assert direct_rys_task_candidates(portable) == ()


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda item: item.spec.name)
def test_quartet_worker_has_no_primitive_collectives(
    candidate: KernelSelection,
) -> None:
    """Only queue/launch plumbing may synchronize independent quartet lanes."""
    source = emit_production_shard((candidate,))
    name = candidate.spec.name
    worker = source.split(f"void generated_{name}_packed_fock_lane(", 1)[1]
    worker = worker.split('extern "C" __global__', 1)[0]
    assert "__syncthreads" not in worker
    assert "__syncwarp" not in worker
    assert "volatile double trr" not in worker
    assert "generated_" + name + "_component_value" not in worker
    assert "retained_components" in worker
    assert "component_integrals" in worker
    assert source == emit_production_shard((candidate,))
    assert "exchange_queue_pairs" in source
    assert "GeneratedExchangeTaskSchedule::Incumbent" in source
    assert "shell_class_fock_rhf_persistent_kernel" in source
    assert "shell_class_force_task" not in source


def test_task_schedule_rejects_incompatible_lane_ownership() -> None:
    candidate = next(item for item in CANDIDATES if item.spec.name == "ppps")
    with pytest.raises(ValueError, match="quartet"):
        build_fused_shell_plan(
            candidate.spec,
            integral=candidate.integral,
            schedule=replace(candidate.schedule, tasks_per_warp=16),
            target=PROFILE.target,
        )


def test_multiwarp_queue_has_independent_retirement_and_scratch() -> None:
    """An exhausted warp cannot strand another warp at a CTA barrier."""
    candidate = next(item for item in CANDIDATES if item.spec.name == "ppps")
    source = emit_production_shard((candidate,))
    worker = source.split("void generated_ppps_streaming_fock(", 1)[1]
    worker = worker.split('extern "C" __global__', 1)[0]
    assert "warp_exchange_queue_pairs[4][64]" in worker
    assert "warp_bra_ordinal[4]" in worker
    assert "warp_lane == 0U" in worker
    assert "__syncthreads" not in worker
    assert candidate.schedule.tasks_per_block == 128


def test_registry_keeps_task_masks_behind_incumbent_coverage() -> None:
    source = emit_multi_registry_source((PROFILE,))
    assert (
        '"GENERATIVEQC_AOT_RYS_TASK_FOCK_SHELL_CLASSES", kernels->rys_task_fock_mask,\n'
        "      kernels->fock_names, kernels->fock_name_count)" in source
    )
    assert "& enabled_fock_shell_class_mask()" in source
    assert "UINT64_C(1247), UINT64_C(1236)" in source
    assert (
        "kernels->preferred_rys_task_fock_mask\n      & enabled_rys_task_fock_shell_class_mask()"
        in source
    )
    for candidate in CANDIDATES:
        assert (
            "generativeqc_launch_sm120_rys_task_generated_"
            f"{candidate.spec.name}_streaming_fock"
        ) in source


def test_legacy_single_profile_registry_retains_zero_preference() -> None:
    """A legacy bundle does not acquire a dependency on absent task workers."""
    source = emit_registry_source(PROFILE.selections)
    assert "preferred_rys_task_fock_shell_class_mask() noexcept { return 0; }" in source


@pytest.mark.parametrize("architecture", ("sm_120", "sm_90", "sm_80"))
def test_only_qualified_profile_prefers_the_five_winners(architecture: str) -> None:
    """Capability is not performance qualification on a different target."""
    profile = resolve_production_profile(MANIFEST, architecture, "portable_cuda")
    assert preferred_rys_task_candidates(profile) == ()
    if architecture == "sm_120":
        assert {item.spec.name for item in preferred_rys_task_candidates(PROFILE)} == {
            "psps",
            "ppps",
            "dsss",
            "dpss",
            "dsps",
        }
    source = emit_multi_registry_source((profile,))
    assert "UINT64_C(0), UINT64_C(0)" in source


def test_larger_local_contraction_does_not_expand_old_block_candidates() -> None:
    """The 64-double bound is local to the Rys task emitter, not a default."""
    candidate = next(item for item in CANDIDATES if item.spec.name == "ppps")
    assert PACKED_RESTRICTED_K_BLOCK_MAX_DOUBLES == 32
    assert packed_restricted_k_block_doubles(candidate.spec) == 48
    assert not packed_restricted_k_block_eligible(candidate.spec)
    assert packed_restricted_k_block_eligible(candidate.spec, maximum_doubles=64)
    assert _packed_restricted_k_block(candidate.spec, "", ("a", "b", "c", "d")) == (
        "",
        "",
    )
    source = emit_production_shard((candidate,))
    assert "exchange_block[48]" in source
    assert "raw_exchange_only" in source
