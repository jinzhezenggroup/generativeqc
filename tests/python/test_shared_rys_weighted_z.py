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


def weighted_plan(name: str, **changes: object) -> FusedShellPlan:
    """Use the fixed materialized restricted SM90 value owner."""
    plan = build_fused_shell_plan(
        FUSED_SHELL_SPEC_BY_NAME[name],
        consumers=(KernelConsumer.FOCK,),
        recurrence="rys3",
        target=cuda_target_info("sm_90"),
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
    with pytest.raises(ValueError, match="requires the qualified SM90"):
        replace(plan.kernel, target=cuda_target_info("sm_120"))


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
