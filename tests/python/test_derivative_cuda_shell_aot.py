"""CUDA derivative shell AOT package inventory and selector tests."""

from __future__ import annotations

import json
from pathlib import Path

from generativeqc_compiler.integral.derivative_aot_registry import (
    radial_inventory_from_payload,
)
from generativeqc_compiler.integral.derivative_cuda_shell_aot import (
    cuda_derivative_shell_packages,
    emit_cuda_derivative_shell_aot_header,
)
from generativeqc_compiler.integral.ir import KernelConsumer
from generativeqc_compiler.integral.production_profile import resolve_production_profile

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = (
    ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"
)
RADIALS = ROOT / "manifests/derivative_aot_radials.json"


def _inputs():
    profile = resolve_production_profile(PRODUCTION, "sm_120")
    payload = json.loads(RADIALS.read_text(encoding="utf-8"))
    radials = radial_inventory_from_payload(payload, backend="cuda")
    return profile, radials


def test_cuda_shell_inventory_covers_same_force_classes_for_full_sr_lr() -> None:
    profile, radials = _inputs()
    packages = cuda_derivative_shell_packages(profile, radials)
    force_classes = {
        selection.spec.name
        for selection in profile.selections
        if KernelConsumer.FORCE in selection.consumers
    }
    assert force_classes
    assert len(packages) == 3 * len(force_classes)
    assert {package.key.shell_name for package in packages} == force_classes
    assert {package.key.radial.family.value for package in packages} == {
        "full_range",
        "short_range",
        "long_range",
    }
    assert len({package.key.identity for package in packages}) == len(packages)
    assert len({package.package_key.identity for package in packages}) == len(packages)


def test_generated_selector_binds_exact_shell_and_fails_closed_off_target() -> None:
    profile, radials = _inputs()
    source = emit_cuda_derivative_shell_aot_header(profile, radials)
    assert "__CUDA_ARCH__ == 1200" in source
    assert (
        "contract_two_electron_force_quartet_subtile_range_shell_aot_scaled" in source
    )
    assert "generated-full-force" in source
    assert "bounded-range-shell-aot" in source
    assert "DirectRangeOperator::Short" in source
    assert "DirectRangeOperator::Long" in source
    assert "default:\n        return false;" in source
    lowered = source.lower()
    assert "wb97" not in lowered
    assert "b3lyp" not in lowered
    assert "pbe0" not in lowered


def test_bounded_runtime_prefers_exact_shell_package_before_radial_fallback() -> None:
    source = (ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text(
        encoding="utf-8"
    )
    packaged = source.index("contract_packaged_derivative_shell_aot<Unrestricted>")
    radial = source.index("contract_bounded_direct_force_subtile_range_aot_scaled<")
    assert packaged < radial
    assert "generated_derivative_cuda_shell_aot.cuh" in source
