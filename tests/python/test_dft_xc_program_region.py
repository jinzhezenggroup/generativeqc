"""ProgramIR binding for the real native CUDA-KS XC staging routes."""

from __future__ import annotations

import typing
from dataclasses import replace

import pytest
from generativeqc_compiler.common.cuda_resources import KernelResources
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.xc_compiled_resources import (
    GridXcCompiledRegionEvidence,
    GridXcCompiledResourceShape,
    native_grid_xc_compiled_region_evidence,
)
from generativeqc_compiler.dft.xc_program import (
    NativeKsXcSource,
    bind_native_ks_xc_region_candidates,
    native_ks_xc_region,
    select_native_ks_xc_region_program,
)
from generativeqc_compiler.dft.xc_schedule import (
    DEVICE_FUSED,
    HOST_UNFUSED,
    GridXcCandidateLimits,
    GridXcCandidateShape,
    GridXcScientificIdentity,
    assess_grid_xc_schedule,
)

if typing.TYPE_CHECKING:
    from generativeqc_compiler.common.program import ProgramIR
    from generativeqc_compiler.dft.xc_schedule import (
        GridXcCandidateAssessment,
        GridXcExecutionSchedule,
    )


def _shape(*, spins: int = 2) -> GridXcCandidateShape:
    return GridXcCandidateShape(
        npoint=48,
        tile_points=16,
        nao=7,
        max_active_ao=7,
        spins=spins,
        jet_components=4,
        device_workspace_bytes=1 << 20,
        generated_source_bytes=8192,
    )


def _limits() -> GridXcCandidateLimits:
    return GridXcCandidateLimits(
        device_bytes=1 << 30,
        live_values=1 << 28,
        source_bytes=1 << 20,
    )


def _scientific(*, spins: int = 2) -> GridXcScientificIdentity:
    return GridXcScientificIdentity(
        architecture="sm_120",
        functional="PBE",
        functional_identity="pbe-science-v1",
        ingredients=("rho", "gradient", "sigma"),
        jet_outputs=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        grid_identity="grid-v1",
        grid_model="grid-model-v1",
        screening_identity=None,
        precision="fp64",
        spin="polarized" if spins == 2 else "unpolarized",
        observable="potential",
        density_route="density_matrix",
        source_identity="xc-source-v1",
    )


def _assessment(
    schedule: GridXcExecutionSchedule,
    *,
    device_xc_available: bool = True,
    spins: int = 2,
    compiled_evidence: GridXcCompiledRegionEvidence | None = None,
) -> GridXcCandidateAssessment:
    return assess_grid_xc_schedule(
        schedule,
        _shape(spins=spins),
        _limits(),
        device_xc_available=device_xc_available,
        observable="potential",
        functional="PBE",
        scientific=_scientific(spins=spins),
        compiled_evidence=compiled_evidence,
    )


def _compiled_evidence(*, spins: int = 2) -> GridXcCompiledRegionEvidence:
    rows = (
        KernelResources("validate_density(double*)", 24, 0, 0, 0, 0),
        KernelResources("ao_kernel(double*)", 52, 0, 0, 0, 0),
        KernelResources("density_product<false>(double*)", 61, 0, 0, 0, 0),
        KernelResources("density_features<false>(double*)", 62, 0, 0, 0, 0),
        KernelResources("evaluate_points<1, false>(double*)", 80, 0, 16, 0, 0),
        KernelResources("assemble_potential(double*)", 64, 0, 0, 0, 0),
        KernelResources("accumulate_totals(double*)", 8, 0, 0, 0, 0),
    )
    return native_grid_xc_compiled_region_evidence(
        rows,
        shape=GridXcCompiledResourceShape(
            npoint=_shape(spins=spins).npoint,
            tile_points=_shape(spins=spins).tile_points,
            nao=_shape(spins=spins).nao,
            spins=spins,
        ),
        functional="PBE",
        target=cuda_target_info("sm_120"),
        source_identity=_scientific(spins=spins).source_identity,
    )


def _source(*, spins: int = 2) -> NativeKsXcSource:
    return NativeKsXcSource(
        _shape(spins=spins),
        _scientific(spins=spins),
        density_identity="cuda-ks-density-v1",
        host_xc_identity="host-pbe-v1",
        transfer_identity="cuda-ks-xc-transfer-v1",
    )


def _program(*, spins: int = 2) -> ProgramIR:
    return _source(spins=spins).program()


def test_host_unfused_uks_program_matches_cuda_ks_stage_xc_boundary() -> None:
    program = _program(spins=2)
    assert tuple(call.name for call in program.calls) == (
        "download_density",
        "split_spin_density",
        "host_xc_vxc",
        "upload_xc",
    )
    assert tuple(call.provider for call in program.calls) == (
        "runtime.cuda.ks_xc_density_d2h",
        "dft.CudaKsPlan.split_host_spin_density",
        "dft.CudaKsPlan.host_unfused_xc",
        "runtime.cuda.ks_xc_result_h2d",
    )
    buffers = {buffer.name: buffer for buffer in program.buffers}
    matrix_bytes = 8 * 7 * 7
    density_bytes = 2 * matrix_bytes
    assert buffers["density_device"].bytes == 0
    assert buffers["density_host"].bytes == density_bytes
    assert buffers["alpha_host"].bytes == matrix_bytes
    assert buffers["beta_host"].bytes == matrix_bytes
    assert buffers["vxc_host"].bytes == density_bytes
    assert buffers["totals_host"].bytes == 24
    assert buffers["error_host"].bytes == 4
    assert buffers["density_device"].space == "device:0"
    assert buffers["density_host"].space == "pageable"
    assert program.outputs == ("vxc_device", "totals_device", "error_device")


def test_host_unfused_rks_has_no_artificial_spin_split() -> None:
    program = _program(spins=1)
    assert tuple(call.name for call in program.calls) == (
        "download_density",
        "host_xc_vxc",
        "upload_xc",
    )
    assert "alpha_host" not in {buffer.name for buffer in program.buffers}
    assert "beta_host" not in {buffer.name for buffer in program.buffers}
    assert program.calls[1].reads == ("density_host",)


def test_native_ks_region_exposes_density_to_xc_view_boundary() -> None:
    region = native_ks_xc_region(_program())
    assert region.reads == ("density_device",)
    assert region.writes == ("vxc_device", "totals_device", "error_device")
    assert region.internal_buffers == (
        "density_host",
        "alpha_host",
        "beta_host",
        "vxc_host",
        "totals_host",
        "error_host",
    )


def test_host_region_cost_counts_exact_cuda_ks_bridge_payload() -> None:
    program = _program()
    candidates = bind_native_ks_xc_region_candidates(
        program,
        host_unfused=_assessment(HOST_UNFUSED),
        device_fused=_assessment(DEVICE_FUSED),
        source=_source(),
        device_xc_identity="cuda-xc-plan-pbe-v1",
    )
    buffers = {buffer.name: buffer.bytes for buffer in program.buffers}
    expected = (
        buffers["density_host"]
        + buffers["vxc_host"]
        + buffers["totals_host"]
        + buffers["error_host"]
    )
    profitability = candidates.host_unfused.schedule.profitability
    assert profitability.semantic_traffic_bytes == expected
    assert expected == 2 * (2 * 7 * 7 * 8) + 24 + 4
    assert candidates.host_unfused.schedule.resources.host_bytes == expected


def test_measured_device_fused_route_replaces_complete_host_region() -> None:
    program = _program()
    selected = select_native_ks_xc_region_program(
        program,
        host_unfused=_assessment(HOST_UNFUSED),
        device_fused=_assessment(DEVICE_FUSED),
        source=_source(),
        device_xc_identity="cuda-xc-plan-pbe-v1",
        endpoint_seconds={
            "host_unfused": 1.0,
            "device_fused": 0.75,
        },
        minimum_speedup=1.02,
    )
    assert selected.candidate.name == "device_fused"
    assert selected.candidate.provider == "dft.CudaXcPlan.device_fused"
    assert selected.candidate.backend == "cuda"
    assert tuple(call.name for call in selected.program.calls) == ("device_fused",)
    assert selected.program.calls[0].reads == ("density_device",)
    assert selected.program.calls[0].writes == (
        "vxc_device",
        "totals_device",
        "error_device",
    )
    assert tuple(buffer.name for buffer in selected.program.buffers) == (
        "density_device",
        "vxc_device",
        "totals_device",
        "error_device",
    )
    provenance = dict(selected.candidate.schedule.provenance)
    assert provenance["region_source_consumer"] == "dft.grid_xc"
    assert provenance["domain_schedule"] == "device_fused"


def test_region_selection_rejects_gpu_pressure_regression_inside_timing_noise() -> None:
    program = _program()
    kwargs = {
        "program": program,
        "host_unfused": _assessment(HOST_UNFUSED),
        "device_fused": _assessment(
            DEVICE_FUSED, compiled_evidence=_compiled_evidence()
        ),
        "source": _source(),
        "device_xc_identity": "cuda-xc-plan-pbe-v1",
        "minimum_speedup": 1.0,
    }

    tied = select_native_ks_xc_region_program(
        endpoint_seconds={"host_unfused": 1.0, "device_fused": 0.995},
        **kwargs,
    )
    assert tied.candidate.name == "host_unfused"
    assert tied.program is program

    faster = select_native_ks_xc_region_program(
        endpoint_seconds={"host_unfused": 1.0, "device_fused": 0.97},
        **kwargs,
    )
    assert faster.candidate.name == "device_fused"


def test_missing_endpoint_evidence_keeps_exact_host_fallback() -> None:
    program = _program()
    selected = select_native_ks_xc_region_program(
        program,
        host_unfused=_assessment(HOST_UNFUSED),
        device_fused=_assessment(DEVICE_FUSED),
        source=_source(),
        device_xc_identity="cuda-xc-plan-pbe-v1",
        endpoint_seconds=None,
        minimum_speedup=1.02,
    )
    assert selected.candidate.name == "host_unfused"
    assert selected.program is program


def test_unavailable_device_xc_cannot_be_promoted_by_fast_timing() -> None:
    program = _program()
    selected = select_native_ks_xc_region_program(
        program,
        host_unfused=_assessment(HOST_UNFUSED),
        device_fused=_assessment(DEVICE_FUSED, device_xc_available=False),
        source=_source(),
        device_xc_identity="cuda-xc-plan-pbe-v1",
        endpoint_seconds={
            "host_unfused": 1.0,
            "device_fused": 0.1,
        },
        minimum_speedup=1.02,
    )
    assert selected.candidate.name == "host_unfused"
    assert selected.program is program


def test_device_executable_identity_invalidates_replacement_program() -> None:
    program = _program()
    kwargs = {
        "program": program,
        "host_unfused": _assessment(HOST_UNFUSED),
        "device_fused": _assessment(DEVICE_FUSED),
        "endpoint_seconds": {
            "host_unfused": 1.0,
            "device_fused": 0.75,
        },
        "minimum_speedup": 1.02,
    }
    first = select_native_ks_xc_region_program(
        source=_source(),
        device_xc_identity="cuda-xc-plan-pbe-v1",
        **kwargs,
    )
    second = select_native_ks_xc_region_program(
        source=_source(),
        device_xc_identity="cuda-xc-plan-pbe-v2",
        **kwargs,
    )
    assert first.program.identity != second.program.identity
    assert first.candidate.replacement_identity != second.candidate.replacement_identity


@pytest.mark.parametrize(
    "timing",
    [
        {"host_unfused": 0.0, "device_fused": 0.5},
        {"host_unfused": 1.0, "device_fused": -1.0},
        {"host_unfused": 1.0, "device_fused": float("inf")},
        {"host_unfused": 1.0, "device_fused": True},
    ],
)
def test_region_binding_rejects_invalid_endpoint_timing(
    timing: typing.Any,
) -> None:
    with pytest.raises((TypeError, ValueError), match="endpoint timing"):
        bind_native_ks_xc_region_candidates(
            _program(),
            host_unfused=_assessment(HOST_UNFUSED),
            device_fused=_assessment(DEVICE_FUSED),
            source=_source(),
            device_xc_identity="cuda-xc-plan-pbe-v1",
            endpoint_seconds=timing,
        )


def test_region_binding_rejects_swapped_schedule_assessments() -> None:
    with pytest.raises(ValueError, match="host_unfused"):
        bind_native_ks_xc_region_candidates(
            _program(),
            host_unfused=_assessment(DEVICE_FUSED),
            device_fused=_assessment(HOST_UNFUSED),
            source=_source(),
            device_xc_identity="cuda-xc-plan-pbe-v1",
        )


def test_region_binding_rejects_inconsistent_assessment_legality() -> None:
    device = _assessment(DEVICE_FUSED)
    inconsistent = replace(device, legal=False)
    with pytest.raises(ValueError, match="legality disagrees"):
        bind_native_ks_xc_region_candidates(
            _program(),
            host_unfused=_assessment(HOST_UNFUSED),
            device_fused=inconsistent,
            source=_source(),
            device_xc_identity="cuda-xc-plan-pbe-v1",
        )


def test_host_fallback_must_remain_legal() -> None:
    host = _assessment(HOST_UNFUSED)
    illegal_host = replace(
        host,
        legal=False,
        reasons=("forced test rejection",),
        schedule_contract=replace(
            host.schedule_contract,
            legal=False,
            reasons=("forced test rejection",),
        ),
    )
    with pytest.raises(ValueError, match="fallback must be legal"):
        bind_native_ks_xc_region_candidates(
            _program(),
            host_unfused=illegal_host,
            device_fused=_assessment(DEVICE_FUSED),
            source=_source(),
            device_xc_identity="cuda-xc-plan-pbe-v1",
        )


@pytest.mark.parametrize("assessment_spins", [1])
def test_region_rejects_cross_spin_admission(assessment_spins: int) -> None:
    with pytest.raises(ValueError, match="source provenance"):
        select_native_ks_xc_region_program(
            _program(spins=2),
            source=_source(spins=2),
            host_unfused=_assessment(HOST_UNFUSED, spins=assessment_spins),
            device_fused=_assessment(DEVICE_FUSED, spins=assessment_spins),
            device_xc_identity="rks-plan",
            endpoint_seconds={"host_unfused": 1.0, "device_fused": 0.1},
        )


@pytest.mark.parametrize("field,value", [("nao", 8), ("npoint", 64)])
def test_region_rejects_other_shape_admission(field: str, value: int) -> None:
    shape = replace(_shape(), **{field: value})
    assessments = tuple(
        assess_grid_xc_schedule(
            schedule,
            shape,
            _limits(),
            device_xc_available=True,
            observable="potential",
            functional="PBE",
            scientific=_scientific(),
        )
        for schedule in (HOST_UNFUSED, DEVICE_FUSED)
    )
    with pytest.raises(ValueError, match="source provenance"):
        bind_native_ks_xc_region_candidates(
            _program(),
            source=_source(),
            host_unfused=assessments[0],
            device_fused=assessments[1],
            device_xc_identity="other-shape-plan",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("grid_identity", "other-grid"),
        ("source_identity", "other-source"),
        ("functional_identity", "other-functional"),
        ("architecture", "sm_90"),
    ],
)
def test_region_rejects_other_scientific_admission(field: str, value: str) -> None:
    science = replace(_scientific(), **{field: value})
    assessments = tuple(
        assess_grid_xc_schedule(
            schedule,
            _shape(),
            _limits(),
            device_xc_available=True,
            observable="potential",
            functional="PBE",
            scientific=science,
        )
        for schedule in (HOST_UNFUSED, DEVICE_FUSED)
    )
    with pytest.raises(ValueError, match="source provenance"):
        bind_native_ks_xc_region_candidates(
            _program(),
            source=_source(),
            host_unfused=assessments[0],
            device_fused=assessments[1],
            device_xc_identity="other-science-plan",
        )


def test_region_rejects_missing_scientific_admission() -> None:
    assessments = tuple(
        assess_grid_xc_schedule(
            schedule,
            _shape(),
            _limits(),
            device_xc_available=True,
            observable="potential",
            functional="PBE",
        )
        for schedule in (HOST_UNFUSED, DEVICE_FUSED)
    )
    with pytest.raises(ValueError, match="source provenance"):
        bind_native_ks_xc_region_candidates(
            _program(),
            source=_source(),
            host_unfused=assessments[0],
            device_fused=assessments[1],
            device_xc_identity="unbound-plan",
        )


@pytest.mark.parametrize(
    "field", ["density_identity", "host_xc_identity", "transfer_identity"]
)
def test_region_rejects_stale_source_binding(field: str) -> None:
    source = replace(_source(), **{field: "changed-identity"})
    with pytest.raises(ValueError, match="source program"):
        bind_native_ks_xc_region_candidates(
            _program(),
            source=source,
            host_unfused=_assessment(HOST_UNFUSED),
            device_fused=_assessment(DEVICE_FUSED),
            device_xc_identity="device-plan",
        )


def test_region_rejects_reconstructed_graph_with_stale_source() -> None:
    program = _program()
    calls = list(program.calls)
    calls[0] = replace(calls[0], identity="other-transfer")
    modified = replace(program, calls=tuple(calls))
    with pytest.raises(ValueError, match="source program"):
        bind_native_ks_xc_region_candidates(
            modified,
            source=_source(),
            host_unfused=_assessment(HOST_UNFUSED),
            device_fused=_assessment(DEVICE_FUSED),
            device_xc_identity="device-plan",
        )


def test_region_accepts_candidate_local_device_tiling() -> None:
    device = assess_grid_xc_schedule(
        replace(DEVICE_FUSED, point_tile=8),
        replace(_shape(), tile_points=8),
        _limits(),
        device_xc_available=True,
        observable="potential",
        functional="PBE",
        scientific=_scientific(),
    )
    selected = select_native_ks_xc_region_program(
        _program(),
        source=_source(),
        host_unfused=_assessment(HOST_UNFUSED),
        device_fused=device,
        device_xc_identity="retiled-plan",
        endpoint_seconds={"host_unfused": 1.0, "device_fused": 0.75},
    )
    assert selected.candidate.name == "device_fused"
