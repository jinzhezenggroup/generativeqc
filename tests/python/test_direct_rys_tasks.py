"""Task-parallel Rys capability, deterministic emission and fallback contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
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
from generativeqc_compiler.integral.rys import emit_rys3_roots_cuda

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
        "ddss",
        "dsds",
        "dpps",
        "dspp",
    }
    assert all(item.schedule.kind == ScheduleKind.PACKED_TASKS for item in CANDIDATES)
    assert all(item.schedule.tasks_per_warp == 32 for item in CANDIDATES)
    assert all(item.schedule.block_threads == 128 for item in CANDIDATES)
    assert all(item.integral.derivative is None for item in CANDIDATES)
    assert all(item.spec.component_count <= 64 for item in CANDIDATES)
    assert all(item.integral.required_rys_roots <= 3 for item in CANDIDATES)
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
    assert "UINT64_C(36831), UINT64_C(36820)" in source
    assert (
        "kernels->preferred_rys_task_fock_mask\n      & enabled_rys_task_fock_shell_class_mask()"
        in source
    )
    for candidate in CANDIDATES:
        assert (
            "generativeqc_launch_sm120_rys_task_generated_"
            f"{candidate.spec.name}_streaming_fock"
        ) in source


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda item: item.spec.name)
def test_task_work_bins_have_private_warp_retirement(
    candidate: KernelSelection,
) -> None:
    """Four independently exhausted warps must never meet at a CTA barrier."""
    source = emit_production_shard((candidate,))
    worker = source.split(
        f"void generated_{candidate.spec.name}_work_streaming_fock(", 1
    )[1]
    worker = worker.split('extern "C" __global__', 1)[0]
    assert "warp_work_pairs[4][8][64]" in worker
    assert "warp_work_bounds[4][8][64]" in worker
    assert "warp_work_counts[4][8]" in worker
    assert "warp_selected_bucket[4]" in worker
    assert "warp_bra_ordinal[4]" in worker
    assert "const unsigned warp_lane = threadIdx.x % 32U;" in worker
    assert "__syncthreads" not in worker
    assert "__syncwarp(full_warp_mask)" in worker
    assert 4 * 8 * 64 * 12 + 4 * 8 * 4 + 2 * 4 * 4 == 24736


def test_legacy_single_profile_registry_retains_zero_preference() -> None:
    """A legacy bundle does not acquire a dependency on absent task workers."""
    source = emit_registry_source(PROFILE.selections)
    assert "preferred_rys_task_fock_shell_class_mask() noexcept { return 0; }" in source


@pytest.mark.parametrize("architecture", ("sm_120", "sm_90", "sm_80"))
def test_only_qualified_profile_prefers_the_selected_classes(architecture: str) -> None:
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
            "ddss",
            "dsds",
            "dpps",
            "dspp",
        }
    source = emit_multi_registry_source((profile,))
    assert "UINT64_C(0), UINT64_C(0)" in source


def test_qualified_rys_task_preference_is_not_tied_to_an_sm_name() -> None:
    """An equivalently qualified target uses its own capability, not a CUDA label."""
    other = replace(PROFILE, target=cuda_target_info("sm_90"), profile="measured_gpu")
    assert {item.spec.name for item in preferred_rys_task_candidates(other)} == {
        item.spec.name for item in preferred_rys_task_candidates(PROFILE)
    }
    assert preferred_rys_task_candidates(replace(other, tuned=False)) == ()


def test_qualified_rys_task_preference_must_be_compiled() -> None:
    """A stale tuning entry fails closed, rather than silently dropping a class."""
    invalid = replace(PROFILE, preferred_rys_task_fock_shell_classes=("dddd",))
    with pytest.raises(ValueError, match="lack generated capability"):
        preferred_rys_task_candidates(invalid)


@pytest.mark.parametrize(
    "invalid", (["psps", "psps"], ["psps", 10], "psps")
)
def test_qualified_rys_task_preference_manifest_validation(
    invalid: object, tmp_path: Path
) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["architectures"]["sm_120"]["preferred_rys_task_fock_shell_classes"] = invalid
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match="Rys-task Fock classes"):
        resolve_production_profile(path, "sm_120")


def test_larger_local_contraction_does_not_expand_old_block_candidates() -> None:
    """The larger local bound does not expand the old 32-double experiment."""
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


@pytest.mark.parametrize("name", ("ddss", "dsds", "dpps", "dspp"))
def test_three_root_tasks_keep_high_component_bits(name: str) -> None:
    """36/54-component masks must not use undefined 32-bit shifts."""
    candidate = next(item for item in CANDIDATES if item.spec.name == name)
    source = emit_production_shard((candidate,))
    assert candidate.integral.required_rys_roots == 3
    assert "std::uint64_t retained_components" in source
    assert "retained_components |= std::uint64_t{1} << component" in source
    assert f"std::uint64_t{{1}} << {candidate.spec.component_count - 1}U" in source
    assert "#pragma unroll 1\n    for (int polynomial_degree" in source
    footprint = packed_restricted_k_block_doubles(candidate.spec)
    if footprint <= 80:
        assert f"exchange_block[{footprint}]" in source
    else:
        assert "exchange_block[" not in source


@pytest.mark.parametrize("hint", (0, -1, True, 1.5))
def test_polynomial_unrolling_rejects_non_positive_integer_hints(hint: object) -> None:
    """A scheduling hint cannot produce an invalid CUDA pragma silently."""
    with pytest.raises(ValueError, match="positive integer"):
        emit_rys3_roots_cuda(polynomial_unroll=hint)
