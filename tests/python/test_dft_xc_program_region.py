"""ProgramIR binding for the real native CUDA-KS XC staging routes."""

from __future__ import annotations

import typing
from dataclasses import replace

import pytest
from vibeqc_compiler.dft.xc_program import (
    bind_native_ks_xc_region_candidates,
    native_ks_host_unfused_xc_program,
    native_ks_xc_region,
    select_native_ks_xc_region_program,
)
from vibeqc_compiler.dft.xc_schedule import (
    DEVICE_FUSED,
    HOST_UNFUSED,
    GridXcCandidateAssessment,
    GridXcCandidateLimits,
    GridXcCandidateShape,
    GridXcExecutionSchedule,
    assess_grid_xc_schedule,
)

if typing.TYPE_CHECKING:
    from vibeqc_compiler.common.program import ProgramIR


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


def _assessment(
    schedule: GridXcExecutionSchedule,
    *,
    device_xc_available: bool = True,
    spins: int = 2,
) -> GridXcCandidateAssessment:
    return assess_grid_xc_schedule(
        schedule,
        _shape(spins=spins),
        _limits(),
        device_xc_available=device_xc_available,
        observable="potential",
        functional="PBE",
    )


def _program(*, spins: int = 2) -> ProgramIR:
    return native_ks_host_unfused_xc_program(
        _shape(spins=spins),
        density_identity="cuda-ks-density-v1",
        host_xc_identity="host-pbe-v1",
        transfer_identity="cuda-ks-xc-transfer-v1",
    )


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


def test_missing_endpoint_evidence_keeps_exact_host_fallback() -> None:
    program = _program()
    selected = select_native_ks_xc_region_program(
        program,
        host_unfused=_assessment(HOST_UNFUSED),
        device_fused=_assessment(DEVICE_FUSED),
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
        device_xc_identity="cuda-xc-plan-pbe-v1",
        **kwargs,
    )
    second = select_native_ks_xc_region_program(
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
            device_xc_identity="cuda-xc-plan-pbe-v1",
            endpoint_seconds=timing,
        )


def test_region_binding_rejects_swapped_schedule_assessments() -> None:
    with pytest.raises(ValueError, match="host_unfused"):
        bind_native_ks_xc_region_candidates(
            _program(),
            host_unfused=_assessment(DEVICE_FUSED),
            device_fused=_assessment(HOST_UNFUSED),
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
            device_xc_identity="cuda-xc-plan-pbe-v1",
        )
