"""Typed opt-in identity and executable exact coefficient-bit admission."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.integral import (
    FUSED_SHELL_SPEC_BY_NAME,
    FusedShellPlan,
    KernelConsumer,
    KernelSelection,
    build_fused_shell_plan,
    cuda_target_info,
    emit_shell_class_fused_cuda,
)
from generativeqc_compiler.integral.capabilities import (
    CAPABILITY_K_BLOCK_FOCK,
    CAPABILITY_MIXED_FOCK,
    CAPABILITY_STREAMING_FOCK,
)
from generativeqc_compiler.integral.lowering.fock_component import (
    _emit_weighted_z_match_cuda,
)
from generativeqc_compiler.integral.production_profile import _schedule_from_payload
from generativeqc_compiler.integral.tuning.emission import (
    schedule_execution_source_identity,
)
from generativeqc_compiler.integral.tuning.policy import ScheduleTrial, schedule_payload

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import NativeCxx


def weighted_plan(
    name: str, *, architecture: str = "sm_90", **changes: object
) -> FusedShellPlan:
    """Use the fixed restricted value mapping on the requested target."""
    plan = build_fused_shell_plan(
        FUSED_SHELL_SPEC_BY_NAME[name],
        consumers=(KernelConsumer.FOCK,),
        recurrence="rys3",
        target=cuda_target_info(architecture),
    )
    return build_fused_shell_plan(
        plan.spec,
        integral=plan.kernel.integral,
        target=plan.kernel.target,
        schedule=replace(
            plan.schedule,
            fock_shared_rys_xy=True,
            **{"fock_shared_rys_weighted_z": True, **changes},
        ),
    )


@pytest.mark.parametrize("value", (None, 0, 1, "false", [], {}))
def test_weighted_z_strict_boolean(value: object) -> None:
    with pytest.raises(TypeError, match="must be boolean"):
        weighted_plan("pppp", fock_shared_rys_weighted_z=value)
    payload = schedule_payload(weighted_plan("pppp").schedule)
    with pytest.raises(ValueError, match="invalid production kernel schedule"):
        _schedule_from_payload({**payload, "fock_shared_rys_weighted_z": value})


@pytest.mark.parametrize("name", ("pppp", "dppp"))
def test_weighted_z_roundtrip_identity_and_production_refusal(name: str) -> None:
    plan = weighted_plan(name)
    xy = replace(plan.schedule, fock_shared_rys_weighted_z=False)
    assert _schedule_from_payload(schedule_payload(plan.schedule)) == plan.schedule
    trial = ScheduleTrial(
        plan.spec,
        xy,
        plan.kernel.target,
        KernelConsumer.FOCK,
        integral=plan.kernel.integral,
    )
    weighted = replace(trial, schedule=plan.schedule)
    assert trial.schedule_id != weighted.schedule_id
    assert trial.entry_point != weighted.entry_point
    assert schedule_execution_source_identity(
        trial
    ) != schedule_execution_source_identity(weighted)
    with pytest.raises(ValueError, match="fixture-only"):
        KernelSelection(
            architecture="sm_90",
            spec=plan.spec,
            schedule=plan.schedule,
            integral=plan.kernel.integral,
            consumers=(KernelConsumer.FOCK,),
            recurrence="rys3",
        )
    with pytest.raises(ValueError, match="requires shared Rys x/y"):
        replace(plan.schedule, fock_shared_rys_xy=False)


@pytest.mark.parametrize(
    "architecture",
    (
        "sm_80",
        "sm_86",
        "sm_87",
        "sm_89",
        "sm_90",
        "sm_100",
        "sm_103",
        "sm_110",
        "sm_120",
        "sm_121",
    ),
)
@pytest.mark.parametrize("name,threads", (("pppp", 96), ("dppp", 192)))
def test_weighted_z_uses_legal_target_not_architecture_whitelist(
    architecture: str, name: str, threads: int
) -> None:
    """Known and conservative targets share an implementation, not evidence."""
    plan = weighted_plan(name, architecture=architecture)
    assert plan.kernel.target.architecture == architecture
    assert plan.block_threads == threads
    assert plan.schedule.fock_shared_rys_xy
    assert plan.schedule.fock_shared_rys_weighted_z
    assert _schedule_from_payload(schedule_payload(plan.schedule)) == plan.schedule
    assert emit_shell_class_fused_cuda(plan.spec, plan)


@pytest.mark.parametrize("architecture", ("sm_80", "sm_90", "sm_120"))
def test_weighted_z_retains_target_resource_validation(architecture: str) -> None:
    plan = weighted_plan("dppp", architecture=architecture)
    insufficient = replace(plan.kernel.target, maximum_threads_per_block=128)
    with pytest.raises(ValueError):
        replace(plan.kernel, target=insufficient)


@pytest.mark.parametrize("name", ("dpds", "ddps", "dpdp", "ddpp"))
def test_weighted_z_retains_mathematical_family_boundary(name: str) -> None:
    from tools.generativeqc_validation.shared_rys_xy_fixture import fixture_plan

    plan = fixture_plan(name, shared=True)
    with pytest.raises(ValueError, match="DPPP/PPPP"):
        replace(
            plan.kernel,
            schedule=replace(plan.schedule, fock_shared_rys_weighted_z=True),
        )


@pytest.mark.parametrize("architecture", ("sm_80", "sm_90", "sm_120"))
@pytest.mark.parametrize("weighted", (False, True))
def test_weighted_fixture_uses_requested_generation_target(
    architecture: str, weighted: bool
) -> None:
    from tools.generativeqc_validation.shared_rys_weighted_z_fixture import (
        companion_plan,
    )

    plan = companion_plan("pppp", weighted=weighted, architecture=architecture)
    assert plan.kernel.target.architecture == architecture
    assert plan.schedule.fock_shared_rys_xy
    assert plan.schedule.fock_shared_rys_weighted_z is weighted


@pytest.mark.parametrize(
    "capability",
    (CAPABILITY_MIXED_FOCK, CAPABILITY_STREAMING_FOCK, CAPABILITY_K_BLOCK_FOCK),
)
def test_weighted_z_rejects_other_owners(capability: str) -> None:
    plan = weighted_plan("pppp")
    with pytest.raises(ValueError, match="does not support"):
        emit_shell_class_fused_cuda(plan.spec, plan, capabilities=(capability,))


def test_actual_emitted_weighted_z_bit_guard(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Compile the emitted helper itself, including signed-zero/ULP refusal."""
    source = tmp_path / "guard.cpp"
    source.write_text(
        """
#include <bit>
#include <cmath>
#include <limits>
using std::isfinite;
#define __device__
#define __forceinline__ inline
long long __double_as_longlong(double x) {return std::bit_cast<long long>(x);}
"""
        + _emit_weighted_z_match_cuda()
        + """
int main() {
  const double one=1.0, next=std::nextafter(one,2.0);
  if(!generated_dppp_weighted_z_matches(one,one)) return 1;
  if(generated_dppp_weighted_z_matches(next,one)) return 2;
  if(generated_dppp_weighted_z_matches(-0.0,0.0)) return 3;
  if(!generated_dppp_weighted_z_matches(-0.0,-0.0)) return 4;
  if(generated_dppp_weighted_z_matches(INFINITY,INFINITY)) return 5;
  if(generated_dppp_weighted_z_matches(NAN,NAN)) return 6;
  return 0;
}
"""
    )
    executable = required_native_cxx.build_executable(
        [source], tmp_path / "guard", compile_args=("-std=c++20", "-O2")
    )
    subprocess.run([executable], check=True, timeout=10)
