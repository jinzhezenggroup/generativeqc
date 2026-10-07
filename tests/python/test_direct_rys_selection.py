"""Compiler inventory and native dispatch contracts for independent J/K choices."""

from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.integral import (
    KernelConsumer,
    TranslationInvariant,
    specialize_fock_integral,
)
from generativeqc_compiler.integral.capabilities import (
    CAPABILITY_K_BLOCK_FOCK,
    CAPABILITY_LOCAL_PACKED_STREAMING_FOCK,
    CAPABILITY_MIXED_FOCK,
    CAPABILITY_STREAMING_FOCK,
    query_integral_capability,
)
from generativeqc_compiler.integral.production_emission import (
    _streaming_fock_source,
    emit_profile_shard,
)
from generativeqc_compiler.integral.production_k_block import direct_k_block_candidates
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_source,
)
from generativeqc_compiler.integral.production_rys_values import (
    direct_rys_value_candidates,
)
from generativeqc_compiler.integral.production_selection import _selection_integral

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"


def test_candidates_keep_value_identity_and_incumbent_coverage() -> None:
    """Legal root/component mappings alone determine the optional inventory."""
    profile = resolve_production_profile(MANIFEST, "sm_120")
    candidates = direct_rys_value_candidates(profile)
    assert candidates
    compiled = {item.spec.name for item in profile.selections}
    for item in candidates:
        assert item.spec.name in compiled
        assert item.consumers == (KernelConsumer.FOCK,)
        assert item.integral.derivative is None
        assert item.integral.required_rys_roots == sum(item.spec.angular) // 2 + 1
        assert query_integral_capability(item.integral).supported
        assert item.schedule.block_threads >= item.spec.component_count
        assert not item.tuned
    names = {item.spec.name for item in candidates}
    assert {"ppps", "dddp"} <= names
    assert {"ssss", "psss", "dddd"}.isdisjoint(names)
    portable = resolve_production_profile(MANIFEST, "sm_120", "portable_cuda")
    assert direct_rys_value_candidates(portable) == ()


def test_registry_resolves_alternatives_beside_each_device_profile() -> None:
    profile = resolve_production_profile(MANIFEST, "sm_120")
    source = emit_multi_registry_source((profile,))
    assert "rys_fock_mask & enabled_fock_shell_class_mask()" in source
    for item in direct_rys_value_candidates(profile):
        assert (
            f"generativeqc_launch_sm120_rys_value_generated_{item.spec.name}_streaming_fock"
            in source
        )
    assert "launch_shell_class_rys_streaming_fock" in source


def test_k_block_candidates_are_bounded_packed_value_alternatives() -> None:
    profile = resolve_production_profile(MANIFEST, "sm_120")
    candidates = direct_k_block_candidates(profile)
    assert candidates
    compiled = {item.spec.name for item in profile.selections}
    incumbents = {item.spec.name: item for item in profile.selections}
    names = {item.spec.name for item in candidates}
    assert {"psss", "ppss", "psps", "dsss"} <= names
    for item in candidates:
        incumbent = incumbents[item.spec.name]
        expected = specialize_fock_integral(_selection_integral(incumbent))
        assert item.spec.name in compiled
        assert item.consumers == (KernelConsumer.FOCK,)
        assert item.integral.derivative is None
        assert item.integral == expected
        assert item.recurrence == expected.recurrence == "subset_wick"
        assert item.schedule == (incumbent.fock_schedule or incumbent.schedule)
        assert query_integral_capability(item.integral).supported
        assert item.schedule.kind.value == "packed_tasks"
        assert item.has_capability("k_block_fock")
        assert item.has_capability("streaming_fock")
        assert item.has_capability(CAPABILITY_LOCAL_PACKED_STREAMING_FOCK) == (
            incumbent.has_capability(CAPABILITY_LOCAL_PACKED_STREAMING_FOCK)
        )
        assert not item.tuned


@pytest.mark.parametrize("name", ("psss", "psps", "ssss", "ppss", "dsss"))
def test_k_block_streaming_preserves_incumbent_lane_storage(name: str) -> None:
    """Inspect streaming only; generic paged kernels still use shared lane arrays."""
    profile = resolve_production_profile(MANIFEST, "sm_120")
    candidate = next(
        item for item in direct_k_block_candidates(profile) if item.spec.name == name
    )
    local_lane_state = name in {"psss", "psps"}
    assert candidate.has_capability(CAPABILITY_LOCAL_PACKED_STREAMING_FOCK) == (
        local_lane_state
    )
    source = _streaming_fock_source(candidate)
    class_name = name[0].upper() + name[1:]
    private_task = f"Generated{class_name}ShellTask stream_task;"
    private_storage = f"Generated{class_name}PackedFockLaneStorage lane_storage;"
    shared_tasks = f"__shared__ Generated{class_name}ShellTask stream_tasks[32];"
    shared_storage = (
        f"__shared__ Generated{class_name}PackedFockLaneStorage lane_storage[32];"
    )
    assert (private_task in source) == local_lane_state
    assert (private_storage in source) == local_lane_state
    assert (shared_tasks in source) != local_lane_state
    assert (shared_storage in source) != local_lane_state
    assert "__shared__ std::uint32_t compact_bra_pairs[32];" in source
    assert "__shared__ double compact_contribution_bounds[32];" in source


@pytest.mark.parametrize("name", ("psss", "ppss"))
@pytest.mark.parametrize("local_lane_state", (False, True))
def test_k_block_candidates_inherit_only_applicable_storage_capability(
    name: str, local_lane_state: bool
) -> None:
    """Custom profiles opt in to private storage, not unrelated mixed-Fock policy."""
    profile = resolve_production_profile(MANIFEST, "sm_120")
    incumbent = next(item for item in profile.selections if item.spec.name == name)
    capabilities = {CAPABILITY_STREAMING_FOCK, CAPABILITY_MIXED_FOCK}
    expected = {CAPABILITY_STREAMING_FOCK, CAPABILITY_K_BLOCK_FOCK}
    if local_lane_state:
        capabilities.add(CAPABILITY_LOCAL_PACKED_STREAMING_FOCK)
        expected.add(CAPABILITY_LOCAL_PACKED_STREAMING_FOCK)
    incumbent = replace(incumbent, capabilities=frozenset(capabilities))
    profile = replace(profile, selections=(incumbent,))
    (candidate,) = direct_k_block_candidates(profile)
    assert candidate.capabilities == frozenset(expected)
    assert candidate.consumers == (KernelConsumer.FOCK,)
    assert candidate.integral.derivative is None


def test_k_block_candidates_preserve_explicit_integral_records() -> None:
    """Specialize the supplied scientific IR instead of rebuilding defaults."""
    profile = resolve_production_profile(MANIFEST, "sm_120")
    incumbent = next(item for item in profile.selections if item.spec.name == "ppss")
    integral = _selection_integral(incumbent)
    operator = replace(
        integral.operator, invariants=(TranslationInvariant(dependent_center=1),)
    )
    integral = replace(
        integral, operator=operator, derivative=operator.nuclear_derivative()
    )
    incumbent = replace(incumbent, integral=integral)
    profile = replace(profile, selections=(incumbent,))
    (candidate,) = direct_k_block_candidates(profile)
    assert candidate.integral.operator is operator
    assert candidate.integral.contractions == tuple(
        item
        for item in integral.contractions
        if item.kernel_consumer == KernelConsumer.FOCK
    )
    assert candidate.integral.derivative is None
    assert candidate.integral.recurrence == "subset_wick"


def test_k_block_shard_emits_incumbent_value_producers() -> None:
    """Mixed scalar-Rys force plans retain the accepted packed Fock producer."""
    profile = resolve_production_profile(MANIFEST, "sm_120")
    candidates = direct_k_block_candidates(profile)
    source = emit_profile_shard(profile, candidates, variant="_k_block")
    for item in candidates:
        assert (
            f"generativeqc_launch_sm120_k_block_generated_{item.spec.name}_streaming_fock"
            in source
        )


def test_registry_resolves_k_block_alternatives_separately() -> None:
    profile = resolve_production_profile(MANIFEST, "sm_120")
    source = emit_multi_registry_source((profile,))
    assert "k_block_fock_mask & enabled_fock_shell_class_mask()" in source
    for item in direct_k_block_candidates(profile):
        assert (
            f"generativeqc_launch_sm120_k_block_generated_{item.spec.name}_streaming_fock"
            in source
        )
    assert "launch_shell_class_k_block_streaming_fock" in source


def test_native_k_lowering_selector_keeps_block_k_only() -> None:
    source = (ROOT / "src/scf/cuda/direct_fock_lowering.hpp").read_text()
    assert '"GENERATIVEQC_DIRECT_K_FOCK_LOWERING"' in source
    assert '"GENERATIVEQC_DIRECT_J_FOCK_LOWERING"' in source
    assert 'std::strcmp(value, "block") == 0' in source
    assert "prepare_direct_fock_k_block_mask()" in source
    assert "enabled_k_block_fock_shell_class_mask()" in source
    assert "Direct J Fock lowering must be incumbent or rys" in source
