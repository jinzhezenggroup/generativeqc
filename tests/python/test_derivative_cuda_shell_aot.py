"""CUDA derivative shell AOT package inventory and selector tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc_compiler.integral.derivative_aot_registry import (
    radial_inventory_from_payload,
)
from generativeqc_compiler.integral.derivative_cuda_shell_aot import (
    cuda_derivative_shell_packages,
    emit_cuda_derivative_shell_aot_bundle,
    emit_cuda_derivative_shell_aot_header,
)
from generativeqc_compiler.integral.ir import KernelConsumer
from generativeqc_compiler.integral.production_profile import resolve_production_profile

if TYPE_CHECKING:
    from generativeqc_compiler.integral.production_profile import (
        ResolvedProductionProfile,
    )
    from generativeqc_compiler.integral.range_separation import CoulombKernel

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = (
    ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json"
)
RADIALS = ROOT / "manifests/derivative_aot_radials.json"


def _inputs() -> tuple[ResolvedProductionProfile, tuple[CoulombKernel, ...]]:
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


@pytest.mark.parametrize("architecture", [800, 860, 890, 900, 1000, 1200, 1300])
@pytest.mark.parametrize("accepted_target", ["sm_90", "sm_120"])
def test_compiled_fatbinary_selector_keeps_portable_targets_bounded(
    architecture: int, accepted_target: str, tmp_path: Path, native_cxx: Any
) -> None:
    """Execute generated dispatch with stub arithmetic, including an unknown SM.

    This tests target admission/ABI only; existing oracle tests own the integral
    mathematics. No GPU is touched by this host-compiled selector probe.
    """
    measured, radials = _inputs()
    if accepted_target == "sm_90":
        # A synthetic accepted profile tests custom-manifest routing without
        # claiming a numerical/performance qualification for Hopper.
        manifest = tmp_path / "profile.json"
        payload = json.loads(PRODUCTION.read_text())
        kernel = payload["architectures"]["sm_120"]["kernels"][0]
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "default_architecture": accepted_target,
                    "architectures": {
                        accepted_target: {"kind": "tuned", "kernels": [kernel]}
                    },
                }
            )
        )
        measured = resolve_production_profile(manifest, accepted_target)
    profiles = tuple(
        resolve_production_profile(PRODUCTION, target, "portable")
        for target in ("sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120")
        if target != accepted_target
    ) + (measured,)
    generated = emit_cuda_derivative_shell_aot_bundle(profiles, radials)
    assert generated == emit_cuda_derivative_shell_aot_bundle(
        tuple(reversed(profiles)), radials
    )
    (tmp_path / "registry.hpp").write_text(generated)
    includes = tmp_path / "scf/cuda"
    includes.mkdir(parents=True)
    (includes / "direct_bounded_fallback.hpp").write_text("#pragma once\n")
    (includes / "direct_force_quartet.cuh").write_text("""
#pragma once
namespace generativeqc::integrals { enum class CoulombRange { Short, Long }; }
namespace generativeqc::scf::cuda_execution {
enum class DirectRangeOperator { Short, Long };
struct DeviceBatch {};
struct ActiveShellQuartetTile {};
inline unsigned calls = 0;
template<bool, unsigned, generativeqc::integrals::CoulombRange, int, class... Args>
void contract_two_electron_force_quartet_subtile_range_shell_aot_scaled(Args...) {
  ++calls;
}
}
""")
    package = next(
        package
        for package in cuda_derivative_shell_packages(measured, radials)
        if package.key.radial.family.value == "short_range"
    )
    source = tmp_path / "selector.cpp"
    source.write_text(f"""
#include "registry.hpp"
using namespace generativeqc::scf::cuda_execution;
int main() {{
  bool selected = contract_packaged_derivative_shell_aot<false>(
      {package.key.shell_class}U, DirectRangeOperator::Short, {package.key.radial.omega:.17g},
      DeviceBatch{{}}, nullptr, nullptr, 0., nullptr, nullptr, nullptr, nullptr, 1., 0, 0);
  bool expected = {str(architecture == int(accepted_target[3:]) * 10).lower()};
  if (selected != expected || calls != unsigned(expected)) return 1;
  if ((kDerivativeShellAotInventoryCount != 0) != expected) return 2;
  return 0;
}}
""")
    binary = native_cxx.build_executable(
        [source],
        tmp_path / "selector",
        compile_args=[
            "-std=c++20",
            "-pedantic-errors",
            f"-I{tmp_path}",
            f"-D__CUDA_ARCH__={architecture}",
            "-D__device__=",
            "-D__noinline__=",
        ],
    )
    subprocess.run([str(binary)], check=True, timeout=10)


def test_generator_requires_target_and_honors_portable_profile(tmp_path: Path) -> None:
    """The command-line path cannot silently restore the historical 5090 target."""
    command = [
        sys.executable,
        str(ROOT / "tools/generate_derivative_cuda_shell_aot.py"),
        "--output",
        str(tmp_path / "registry.cuh"),
    ]
    missing = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=30
    )
    assert missing.returncode != 0 and "--target-architecture" in missing.stderr
    subprocess.run(
        command + ["--target-architecture", "86-real", "--profile", "portable"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    expected = emit_cuda_derivative_shell_aot_bundle(
        (resolve_production_profile(PRODUCTION, "sm_86", "portable"),), _inputs()[1]
    )
    assert (tmp_path / "registry.cuh").read_text() == expected
