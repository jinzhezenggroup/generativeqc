"""ProgramIR binding for the real CUDA grid/XC execution routes."""

from __future__ import annotations

from dataclasses import replace

import pytest
from vibeqc_compiler.dft.xc_program import (
    bind_grid_xc_region_candidates,
    host_unfused_grid_xc_program,
    host_unfused_grid_xc_region,
    select_grid_xc_region_program,
)
from vibeqc_compiler.dft.xc_schedule import (
    DEVICE_FUSED,
    HOST_UNFUSED,
    GridXcCandidateLimits,
    GridXcCandidateShape,
    assess_grid_xc_schedule,
    grid_xc_tile_capacities,
)


def _shape() -> GridXcCandidateShape:
    return GridXcCandidateShape(
        npoint=48,
        tile_points=16,
        nao=7,
        max_active_ao=7,
        spins=2,
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


def _assessment(schedule, *, device_xc_available: bool = True):
    return assess_grid_xc_schedule(
        schedule,
        _shape(),
        _limits(),
        device_xc_available=device_xc_available,
        observable="potential",
        functional="PBE",
    )


def _program():
    return host_unfused_grid_xc_program(
        _shape(),
        source_identity="cuda-density-source-v1",
        host_xc_identity="generated-host-pbe-v1",
        transfer_identity="cuda-transfer-v1",
    )


def test_host_unfused_program_matches_real_execution_boundary() -> None:
    program = _program()
    capacity = grid_xc_tile_capacities(_shape())
    assert tuple(call.name for call in program.calls) == (
        "collocate",
        "features",
        "download",
        "host_xc_vxc",
        "upload_vxc",
    )
    assert tuple(call.provider for call in program.calls) == (
        "dft.CudaDensityGrid.collocate",
        "dft.CudaDensityGrid.features",
        "runtime.cuda.download_grid_xc",
        "xc.GeneratedHost.grid_xc_vxc",
        "runtime.cuda.upload_grid_xc",
    )
    buffers = {buffer.name: buffer for buffer in program.buffers}
    assert buffers["density_source"].bytes == 0
    assert buffers["ao_jets_device"].bytes == capacity["ao_jets"]
    assert buffers["density_panel_device"].bytes == capacity["density_panel"]
    assert buffers["features_device"].bytes == capacity["features"]
    assert buffers["vxc_host"].bytes == capacity["vxc"]
    assert buffers["vxc_device"].bytes == capacity["vxc"]
    assert buffers["ao_jets_device"].space == "device:0"
    assert buffers["ao_jets_host"].space == "pageable"
    assert program.outputs == ("vxc_device",)


def test_host_region_exposes_only_density_source_to_device_vxc_boundary() -> None:
    region = host_unfused_grid_xc_region(_program())
    assert region.reads == ("density_source",)
    assert region.writes == ("vxc_device",)
    assert region.internal_buffers == (
        "ao_jets_device",
        "density_panel_device",
        "features_device",
        "ao_jets_host",
        "features_host",
        "vxc_host",
    )


def test_measured_device_fused_route_replaces_complete_host_region() -> None:
    program = _program()
    host = _assessment(HOST_UNFUSED)
    device = _assessment(DEVICE_FUSED)
    selected = select_grid_xc_region_program(
        program,
        host_unfused=host,
        device_fused=device,
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
    assert selected.program.calls[0].reads == ("density_source",)
    assert selected.program.calls[0].writes == ("vxc_device",)
    assert tuple(buffer.name for buffer in selected.program.buffers) == (
        "density_source",
        "vxc_device",
    )
    provenance = dict(selected.candidate.schedule.provenance)
    assert provenance["region_source_consumer"] == "dft.grid_xc"
    assert provenance["domain_schedule"] == "device_fused"


def test_missing_endpoint_evidence_keeps_exact_host_fallback() -> None:
    program = _program()
    selected = select_grid_xc_region_program(
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
    selected = select_grid_xc_region_program(
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
    first = select_grid_xc_region_program(
        device_xc_identity="cuda-xc-plan-pbe-v1",
        **kwargs,
    )
    second = select_grid_xc_region_program(
        device_xc_identity="cuda-xc-plan-pbe-v2",
        **kwargs,
    )
    assert first.program.identity != second.program.identity
    assert (
        first.candidate.replacement_identity
        != second.candidate.replacement_identity
    )


@pytest.mark.parametrize(
    "timing",
    [
        {"host_unfused": 0.0, "device_fused": 0.5},
        {"host_unfused": 1.0, "device_fused": -1.0},
        {"host_unfused": 1.0, "device_fused": float("inf")},
        {"host_unfused": 1.0, "device_fused": True},
    ],
)
def test_region_binding_rejects_invalid_endpoint_timing(timing) -> None:
    with pytest.raises((TypeError, ValueError), match="endpoint timing"):
        bind_grid_xc_region_candidates(
            _program(),
            host_unfused=_assessment(HOST_UNFUSED),
            device_fused=_assessment(DEVICE_FUSED),
            device_xc_identity="cuda-xc-plan-pbe-v1",
            endpoint_seconds=timing,
        )


def test_region_binding_rejects_swapped_schedule_assessments() -> None:
    with pytest.raises(ValueError, match="host_unfused"):
        bind_grid_xc_region_candidates(
            _program(),
            host_unfused=_assessment(DEVICE_FUSED),
            device_fused=_assessment(HOST_UNFUSED),
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
        bind_grid_xc_region_candidates(
            _program(),
            host_unfused=illegal_host,
            device_fused=_assessment(DEVICE_FUSED),
            device_xc_identity="cuda-xc-plan-pbe-v1",
        )
