"""Reconstructed region descriptors must not change the source dataflow ABI."""

from __future__ import annotations

from dataclasses import replace

import pytest
from generativeqc_compiler.common.gpu_profitability import GpuProfitability
from generativeqc_compiler.common.liveness import EffectKind
from generativeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from generativeqc_compiler.common.program_region import (
    ProgramRegion,
    ProgramRegionCandidate,
    apply_program_region_candidate,
    derive_program_region,
)
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.schedule import (
    ScheduleContract,
    ScheduleResources,
    ScheduleTopology,
)


def _program() -> ProgramIR:
    return ProgramIR(
        "boundary-validation",
        tuple(ProgramBuffer(name, 8) for name in ("x", "y", "a", "b", "c", "out")),
        ("x", "y"),
        (
            PlanCall("first", "provider.first", "first-v1", ("x",), ("a",)),
            PlanCall("second", "provider.second", "second-v1", ("a",), ("b", "c")),
            PlanCall("finish", "provider.finish", "finish-v1", ("b", "c"), ("out",)),
        ),
        ("out",),
    )


def _region(program: ProgramIR) -> ProgramRegion:
    return derive_program_region(
        program,
        name="first-second",
        start_call="first",
        end_call="second",
        effects={
            "provider.first": EffectKind.PURE,
            "provider.second": EffectKind.PURE,
        },
    )


def _candidate(
    region: ProgramRegion, *, fallback: bool = False
) -> ProgramRegionCandidate:
    return ProgramRegionCandidate(
        name="original" if fallback else "fused",
        region=region,
        backend="original" if fallback else "cpu",
        provider=None if fallback else "provider.fused",
        implementation_identity=None if fallback else "fused-v1",
        schedule=ScheduleContract(
            consumer=region.consumer,
            schedule_hash=canonical_hash(
                {"region": region.identity, "fallback": fallback}
            ),
            workload_hash="1" * 64,
            profile_key="2" * 64,
            target_hash="3" * 64,
            precision_schedule_hash="4" * 64,
            fallback=fallback,
            legal=True,
            reasons=(),
            topology=ScheduleTopology(),
            resources=ScheduleResources(),
            profitability=GpuProfitability(endpoint_seconds=1.0),
        ),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        {"reads": ()},
        {"reads": ("x", "y")},
        {"writes": ("c", "b")},
        {"writes": ("a", "b", "c"), "internal_buffers": ()},
    ),
    ids=("omitted-input", "extra-input", "reordered-outputs", "exposed-internal"),
)
def test_reconstructed_region_revalidates_exact_boundary(mutation: dict) -> None:
    program = _program()
    region = replace(_region(program), **mutation)
    assert region.program_identity == program.identity
    with pytest.raises(ValueError, match="boundary.*source graph"):
        apply_program_region_candidate(program, _candidate(region))


def test_derived_region_preserves_ordered_multiple_outputs() -> None:
    program = _program()
    region = _region(program)
    fused = apply_program_region_candidate(program, _candidate(region))
    assert fused.calls[0].reads == ("x",)
    assert fused.calls[0].writes == ("b", "c")
    assert tuple(buffer.name for buffer in fused.buffers) == ("x", "y", "b", "c", "out")
    assert fused.inputs == program.inputs
    assert fused.outputs == program.outputs


def test_fallback_retains_original_program_object() -> None:
    program = _program()
    candidate = _candidate(_region(program), fallback=True)
    assert apply_program_region_candidate(program, candidate) is program
