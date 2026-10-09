"""Protect the generated Fock sink and compensated reference routing contract."""

import re
from pathlib import Path

import pytest
from generativeqc_compiler.integral.production_emission import (
    _streaming_fock_internal_signature,
    emit_profile_shard,
)
from generativeqc_compiler.integral.production_profile import (
    ResolvedProductionProfile,
    resolve_production_profile,
)
from generativeqc_compiler.integral.production_registry import (
    emit_multi_registry_source,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def profile() -> ResolvedProductionProfile:
    """Use the actual production schedules rather than a synthetic allowlist."""
    return resolve_production_profile(
        ROOT / "python/generativeqc_compiler/integral/production_shell_classes.json",
        "sm_120",
    )


@pytest.mark.parametrize("name", ("ssss", "ppps", "dpps", "dddd", "ddds"))
def test_generated_schedules_keep_both_fock_planes(
    profile: ResolvedProductionProfile, name: str
) -> None:
    """Packed, component, subgroup and high-order workers share the sink ABI."""
    selection = next(item for item in profile.selections if item.spec.name == name)
    source = emit_profile_shard(profile, (selection,))
    assert '#include "runtime/compensated_atomic.cuh"' in source
    assert source.count('#include "runtime/compensated_atomic.cuh"') == 1
    assert source.index('#include "runtime/compensated_atomic.cuh"') < source.index(
        "namespace generativeqc::scf::generated::profile_"
    )
    assert "generativeqc::runtime::CompensatedOutput fock" in source
    assert "double* fock" not in source
    assert "typename Output = double*" in source
    signature = _streaming_fock_internal_signature(selection)
    assert "generativeqc::runtime::CompensatedOutput fock" in signature.parameter_list()


def test_generated_registry_distinguishes_force_and_fock_sinks(
    profile: ResolvedProductionProfile,
) -> None:
    """Force output remains a plain pointer; Fock forwards its two-plane value."""
    source = emit_multi_registry_source((profile,))
    assert "using FockLaunchFunction" in source
    assert "FockLaunchFunction launch_fock" in source
    assert "FockLaunchFunction launch_mixed_fock" in source
    assert "LaunchFunction launch_force" in source
    assert "runtime::CompensatedOutput output" in source


def test_work_bucket_launchers_keep_the_compensated_abi(
    profile: ResolvedProductionProfile,
) -> None:
    """Optional upstream work schedules must not narrow the output to one plane."""
    source = emit_multi_registry_source((profile,))
    declarations = re.findall(
        r'extern "C" cudaError_t [^(]*_work_streaming_fock\([^;]*\);', source
    )
    assert declarations
    assert all("runtime::CompensatedOutput" in item for item in declarations)
    header = (ROOT / "src/scf/aot_shell_registry.hpp").read_text()
    declaration = re.search(
        r"cudaError_t launch_shell_class_work_streaming_fock\((.*?)\) noexcept;",
        header,
        re.DOTALL,
    )
    assert declaration is not None
    assert "runtime::CompensatedOutput fock" in declaration.group(1)


def test_compensated_reference_does_not_skip_generated_classes() -> None:
    """Pages, retries and streams share one plane; only uncovered classes recur."""
    source = (ROOT / "src/scf/cuda_rhf.cpp").read_text()
    assert source.count("{quartet_fock, resources.reference_fock_correction_}") == 3
    assert "resources.reference_fock_correction_ ? 0U" not in source
    assert (
        "if (resources.reference_fock_correction_)\n"
        "      return launch_bounded_generic_fock"
    ) not in source
    assert "cudaMemsetAsync(resources.reference_fock_correction_, 0" in source
    assert "launch_jk_compensation_fold(resources.stream_, quartet_fock" in source
