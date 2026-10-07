"""Compiler inventory and native dispatch contracts for independent J/K choices."""

from pathlib import Path

from generativeqc_compiler.integral import KernelConsumer
from generativeqc_compiler.integral.capabilities import query_integral_capability
from generativeqc_compiler.integral.production_k_block import direct_k_block_candidates
from generativeqc_compiler.integral.production_profile import resolve_production_profile
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_source,
)
from generativeqc_compiler.integral.production_rys_values import (
    direct_rys_value_candidates,
)

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
    names = {item.spec.name for item in candidates}
    assert {"psss", "ppss", "psps", "dsss"} <= names
    for item in candidates:
        assert item.spec.name in compiled
        assert item.consumers == (KernelConsumer.FOCK,)
        assert item.integral.derivative is None
        assert item.schedule.kind.value == "packed_tasks"
        assert item.has_capability("k_block_fock")
        assert item.has_capability("streaming_fock")
        assert not item.tuned


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
