"""Compiler inventory and native dispatch contracts for independent J/K choices."""

from pathlib import Path

from generativeqc_compiler.integral import KernelConsumer
from generativeqc_compiler.integral.capabilities import query_integral_capability
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
