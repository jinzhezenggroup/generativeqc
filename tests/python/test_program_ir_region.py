"""Whole-region ProgramIR fusion/backend-selection contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
from vibeqc_compiler.common.gpu_profitability import GpuProfitability
from vibeqc_compiler.common.liveness import EffectKind
from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.program_region import (
    ProgramRegion,
    ProgramRegionCandidate,
    apply_program_region_candidate,
    derive_program_region,
    select_program_region_candidate,
)
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.schedule import (
    ScheduleContract,
    ScheduleResources,
    ScheduleTopology,
)
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.program_ir import fixed_density_tile_program


def _program() -> ProgramIR:
    return ProgramIR(
        "region-test",
        (
            ProgramBuffer("x", 8),
            ProgramBuffer("a", 8),
            ProgramBuffer("b", 8),
            ProgramBuffer("out", 8),
        ),
        ("x",),
        (
            PlanCall("a", "provider.a", "a-v1", ("x",), ("a",)),
            PlanCall("b", "provider.b", "b-v1", ("a",), ("b",)),
            PlanCall("finish", "provider.finish", "finish-v1", ("b",), ("out",)),
        ),
        ("out",),
    )


def _region(program: ProgramIR | None = None) -> ProgramRegion:
    if program is None:
        program = _program()
    return derive_program_region(
        program,
        name="ab",
        start_call="a",
        end_call="b",
        effects={
            "provider.a": EffectKind.PURE,
            "provider.b": EffectKind.PURE,
        },
    )


def _candidate(
    region: ProgramRegion,
    name: str,
    seconds: float | None,
    *,
    fallback: bool = False,
    legal: bool = True,
) -> ProgramRegionCandidate:
    schedule = ScheduleContract(
        consumer=region.consumer,
        schedule_hash=canonical_hash(
            {
                "region": region.identity,
                "route": name,
            }
        ),
        workload_hash="1" * 64,
        profile_key="2" * 64,
        target_hash="3" * 64,
        precision_schedule_hash="4" * 64,
        fallback=fallback,
        legal=legal,
        reasons=() if legal else ("backend is not qualified",),
        topology=ScheduleTopology(
            fusion="unfused" if fallback else "whole-region",
            materialization="materialize" if fallback else "fused",
            residency="device",
        ),
        resources=ScheduleResources(),
        profitability=GpuProfitability(endpoint_seconds=seconds),
        provenance=(("route", name),),
    )
    return ProgramRegionCandidate(
        name=name,
        region=region,
        schedule=schedule,
        backend="original" if fallback else "generated-cuda",
        provider=None if fallback else "compiler.fused",
        implementation_identity=None if fallback else f"{name}-artifact-v1",
    )


def test_region_derives_boundary_and_eliminates_internal_materialization() -> None:
    program = _program()
    region = _region(program)
    assert region.call_names == ("a", "b")
    assert region.reads == ("x",)
    assert region.writes == ("b",)
    assert region.internal_buffers == ("a",)
    assert region.consumer.startswith("program-region:")

    candidate = _candidate(region, "fused_ab", 0.7)
    fused = apply_program_region_candidate(program, candidate)
    assert tuple(call.name for call in fused.calls) == ("fused_ab", "finish")
    assert fused.calls[0].provider == "compiler.fused"
    assert fused.calls[0].reads == ("x",)
    assert fused.calls[0].writes == ("b",)
    assert tuple(buffer.name for buffer in fused.buffers) == ("x", "b", "out")
    assert fused.identity != program.identity


def test_fallback_preserves_exact_original_program() -> None:
    program = _program()
    candidate = _candidate(_region(program), "original", 1.0, fallback=True)
    assert apply_program_region_candidate(program, candidate) is program


def test_region_requires_explicit_purity_and_contiguous_multi_call_scope() -> None:
    program = _program()
    with pytest.raises(ValueError, match="not proven pure"):
        derive_program_region(
            program,
            name="ab",
            start_call="a",
            end_call="b",
            effects={"provider.a": EffectKind.PURE},
        )
    with pytest.raises(ValueError, match="not proven pure"):
        derive_program_region(
            program,
            name="ab",
            start_call="a",
            end_call="b",
            effects={
                "provider.a": EffectKind.PURE,
                "provider.b": EffectKind.EFFECTFUL,
            },
        )
    with pytest.raises(ValueError, match="at least two"):
        derive_program_region(
            program,
            name="a",
            start_call="a",
            end_call="a",
            effects={"provider.a": EffectKind.PURE},
        )


def test_region_candidate_rejects_stale_or_illegal_application() -> None:
    program = _program()
    region = _region(program)
    candidate = _candidate(region, "fused_ab", 0.7)
    with pytest.raises(ValueError, match="stale"):
        apply_program_region_candidate(replace(program, name="changed"), candidate)

    illegal = _candidate(region, "unqualified", 0.5, legal=False)
    with pytest.raises(ValueError, match="illegal"):
        apply_program_region_candidate(program, illegal)


def test_measured_region_selection_reuses_shared_promotion_policy() -> None:
    region = _region()
    fallback = _candidate(region, "original", 1.0, fallback=True)
    faster = _candidate(region, "generated", 0.75)
    assert (
        select_program_region_candidate(
            (fallback, faster),
            minimum_speedup=1.02,
        )
        is faster
    )

    unmeasured = _candidate(region, "unmeasured", None)
    assert (
        select_program_region_candidate(
            (fallback, unmeasured),
            minimum_speedup=1.02,
        )
        is fallback
    )


def _pbe_program() -> ProgramIR:
    contract = ContractionProgram(functional("PBE", spin="polarized")).contract
    return fixed_density_tile_program(
        contract,
        nao=12,
        tile_points=7,
        basis_bytes=100,
        grid_bytes=1000,
        basis_identity="basis-v1",
        native_identity="native-v1",
        packed_features=True,
    )


def test_real_pbe_program_exposes_one_cross_subsystem_fusion_boundary() -> None:
    """Planning can collapse features -> scalar XC -> Vxc without changing science."""

    program = _pbe_program()
    region = derive_program_region(
        program,
        name="pbe-fixed-density",
        start_call="features",
        end_call="vxc",
        effects={
            "dft.density_feature_block": EffectKind.PURE,
            "xc.NativeContractionProgram.scalar_values_packed": EffectKind.PURE,
            "xc.NativeContractionProgram.potential_from_rows": EffectKind.PURE,
        },
    )
    assert region.reads == ("jets", "density", "quadrature")
    assert region.writes == ("contribution",)
    assert region.internal_buffers == (
        "feature_scalar",
        "feature_gradient",
        "xc_rows",
        "coefficients",
    )

    candidate = _candidate(region, "fused_pbe", 0.8)
    fused = apply_program_region_candidate(program, candidate)
    assert tuple(call.name for call in fused.calls) == (
        "collocation",
        "fused_pbe",
    )
    assert tuple(buffer.name for buffer in fused.buffers) == (
        "basis",
        "density",
        "quadrature",
        "jets",
        "contribution",
    )
    assert fused.calls[-1].reads == ("jets", "density", "quadrature")
    assert fused.calls[-1].writes == ("contribution",)


def test_region_schedule_identity_participates_in_replacement_identity() -> None:
    region = _region()
    first = _candidate(region, "generated", 0.75)
    second = _candidate(region, "generated-v2", 0.75)
    assert first.replacement_identity != second.replacement_identity


def test_region_selection_rejects_mixed_regions_and_duplicate_names() -> None:
    first_region = _region()
    changed = replace(_program(), name="other")
    second_region = _region(changed)
    first = _candidate(first_region, "original", 1.0, fallback=True)
    second = _candidate(second_region, "generated", 0.7)
    with pytest.raises(ValueError, match="one region"):
        select_program_region_candidate(
            (first, second),
            minimum_speedup=1.02,
        )

    duplicate = _candidate(first_region, "original", 0.7)
    with pytest.raises(ValueError, match="duplicate.*candidate name"):
        select_program_region_candidate(
            (first, duplicate),
            minimum_speedup=1.02,
        )
