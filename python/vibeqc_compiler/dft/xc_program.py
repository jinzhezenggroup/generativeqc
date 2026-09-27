"""Bind real grid/XC execution routes to ProgramIR whole-region selection.

The host-unfused CUDA path is described explicitly because it is the portable
fallback: device collocation/features, explicit host staging, generated host
XC/Vxc, then Vxc upload back to the CUDA SCF owner. The existing native
device-fused path can replace that whole region only when its DFT schedule
assessment is legal and measured evidence selects it.

No scientific XC equation is implemented here. Existing DFT schedule admission,
ProgramIR region legality, and shared measured promotion remain authoritative.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass, replace

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

from .xc_schedule import (
    GridXcCandidateAssessment,
    GridXcCandidateShape,
    grid_xc_tile_capacities,
)

if typing.TYPE_CHECKING:
    import collections.abc

    from vibeqc_compiler.common.schedule import ScheduleContract

_HOST_REGION_EFFECTS = {
    "dft.CudaDensityGrid.collocate": EffectKind.PURE,
    "dft.CudaDensityGrid.features": EffectKind.PURE,
    "runtime.cuda.download_grid_xc": EffectKind.PURE,
    "xc.GeneratedHost.grid_xc_vxc": EffectKind.PURE,
    "runtime.cuda.upload_grid_xc": EffectKind.PURE,
}


def _text(value: typing.Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _device_space(ordinal: int) -> str:
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("grid/XC device ordinal must be a non-negative integer")
    return f"device:{ordinal}"


def host_unfused_grid_xc_program(
    shape: GridXcCandidateShape,
    *,
    source_identity: str,
    host_xc_identity: str,
    transfer_identity: str,
    device_ordinal: int = 0,
) -> ProgramIR:
    """Describe the actual host-unfused CUDA grid/XC lowering boundary."""

    if not isinstance(shape, GridXcCandidateShape):
        raise TypeError("grid/XC ProgramIR requires GridXcCandidateShape")
    source_identity = _text(source_identity, "grid/XC source identity")
    host_xc_identity = _text(host_xc_identity, "host XC identity")
    transfer_identity = _text(transfer_identity, "transfer identity")
    device = _device_space(device_ordinal)
    capacity = grid_xc_tile_capacities(shape)
    topology = {
        "npoint": shape.npoint,
        "tile_points": shape.tile_points,
        "nao": shape.nao,
        "max_active_ao": shape.max_active_ao,
        "spins": shape.spins,
        "jet_components": shape.jet_components,
    }

    buffers = (
        # This is an execution dependency token. The real density/grid owner is
        # already charged by PreparedXCContractions/CudaDensityGrid.
        ProgramBuffer("density_source", 0, device),
        ProgramBuffer("ao_jets_device", capacity["ao_jets"], device),
        ProgramBuffer("density_panel_device", capacity["density_panel"], device),
        ProgramBuffer("features_device", capacity["features"], device),
        ProgramBuffer("ao_jets_host", capacity["ao_jets"]),
        ProgramBuffer("features_host", capacity["features"]),
        ProgramBuffer("vxc_host", capacity["vxc"]),
        ProgramBuffer("vxc_device", capacity["vxc"], device),
    )
    calls = (
        PlanCall(
            "collocate",
            "dft.CudaDensityGrid.collocate",
            canonical_hash(
                {
                    "source": source_identity,
                    "stage": "collocate",
                    "topology": topology,
                }
            ),
            ("density_source",),
            ("ao_jets_device",),
        ),
        PlanCall(
            "features",
            "dft.CudaDensityGrid.features",
            canonical_hash(
                {
                    "source": source_identity,
                    "stage": "features",
                    "topology": topology,
                }
            ),
            ("density_source", "ao_jets_device"),
            ("density_panel_device", "features_device"),
        ),
        PlanCall(
            "download",
            "runtime.cuda.download_grid_xc",
            transfer_identity,
            ("ao_jets_device", "density_panel_device", "features_device"),
            ("ao_jets_host", "features_host"),
        ),
        PlanCall(
            "host_xc_vxc",
            "xc.GeneratedHost.grid_xc_vxc",
            host_xc_identity,
            ("ao_jets_host", "features_host"),
            ("vxc_host",),
        ),
        PlanCall(
            "upload_vxc",
            "runtime.cuda.upload_grid_xc",
            transfer_identity,
            ("vxc_host",),
            ("vxc_device",),
        ),
    )
    return ProgramIR(
        "grid_xc_host_unfused",
        buffers,
        ("density_source",),
        calls,
        ("vxc_device",),
    )


def host_unfused_grid_xc_region(program: ProgramIR) -> ProgramRegion:
    """Return the exact portable region replaceable by native device fusion."""

    return derive_program_region(
        program,
        name="grid-xc",
        start_call="collocate",
        end_call="upload_vxc",
        effects=_HOST_REGION_EFFECTS,
    )


def _schedule_name(assessment: GridXcCandidateAssessment) -> str:
    if not isinstance(assessment, GridXcCandidateAssessment):
        raise TypeError("grid/XC region binding requires candidate assessments")
    provenance = dict(assessment.schedule_contract.provenance)
    name = provenance.get("domain_schedule")
    if name not in ("host_unfused", "device_fused"):
        raise ValueError("grid/XC assessment has an unknown execution schedule")
    return name


def _endpoint_seconds(
    measurements: collections.abc.Mapping[str, float | None] | None,
    name: str,
) -> float | None:
    if measurements is None or name not in measurements:
        return None
    value = measurements[name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("grid/XC endpoint timing must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("grid/XC endpoint timing must be a positive finite number")
    return numeric


def _region_schedule(
    assessment: GridXcCandidateAssessment,
    region: ProgramRegion,
    *,
    endpoint_seconds: float | None,
) -> ScheduleContract:
    contract = assessment.schedule_contract
    provenance = dict(contract.provenance)
    provenance["program_region"] = region.identity
    provenance["region_source_consumer"] = contract.consumer
    return replace(
        contract,
        consumer=region.consumer,
        profitability=replace(
            contract.profitability,
            endpoint_seconds=endpoint_seconds,
        ),
        provenance=tuple(provenance.items()),
    )


@dataclass(frozen=True, slots=True)
class GridXcRegionCandidates:
    """Original and device-fused implementations of one exact ProgramIR region."""

    region: ProgramRegion
    host_unfused: ProgramRegionCandidate
    device_fused: ProgramRegionCandidate

    def __post_init__(self) -> None:
        if self.host_unfused.region != self.region:
            raise ValueError("host grid/XC candidate is bound to another region")
        if self.device_fused.region != self.region:
            raise ValueError("device grid/XC candidate is bound to another region")

    @property
    def all(self) -> tuple[ProgramRegionCandidate, ProgramRegionCandidate]:
        return self.host_unfused, self.device_fused


def bind_grid_xc_region_candidates(
    program: ProgramIR,
    *,
    host_unfused: GridXcCandidateAssessment,
    device_fused: GridXcCandidateAssessment,
    device_xc_identity: str,
    endpoint_seconds: collections.abc.Mapping[str, float | None] | None = None,
) -> GridXcRegionCandidates:
    """Bind the existing DFT schedule pair to one exact ProgramIR region."""

    device_xc_identity = _text(device_xc_identity, "device XC executable identity")
    if _schedule_name(host_unfused) != "host_unfused":
        raise ValueError("host grid/XC assessment is not host_unfused")
    if _schedule_name(device_fused) != "device_fused":
        raise ValueError("device grid/XC assessment is not device_fused")
    if not host_unfused.schedule_contract.fallback:
        raise ValueError("host_unfused must remain the grid/XC fallback")
    if device_fused.schedule_contract.fallback:
        raise ValueError("device_fused must not be marked as fallback")
    if not host_unfused.legal:
        raise ValueError("host_unfused fallback must be legal")

    region = host_unfused_grid_xc_region(program)
    host_schedule = _region_schedule(
        host_unfused,
        region,
        endpoint_seconds=_endpoint_seconds(endpoint_seconds, "host_unfused"),
    )
    device_schedule = _region_schedule(
        device_fused,
        region,
        endpoint_seconds=_endpoint_seconds(endpoint_seconds, "device_fused"),
    )
    host_candidate = ProgramRegionCandidate(
        name="host_unfused",
        region=region,
        schedule=host_schedule,
        backend="cuda+host",
    )
    device_candidate = ProgramRegionCandidate(
        name="device_fused",
        region=region,
        schedule=device_schedule,
        backend="cuda",
        provider="dft.CudaXcPlan.device_fused",
        implementation_identity=device_xc_identity,
    )
    return GridXcRegionCandidates(
        region,
        host_candidate,
        device_candidate,
    )


@dataclass(frozen=True, slots=True)
class GridXcRegionSelection:
    """Selected executable candidate plus the resulting ProgramIR."""

    candidate: ProgramRegionCandidate
    program: ProgramIR


def select_grid_xc_region_program(
    program: ProgramIR,
    *,
    host_unfused: GridXcCandidateAssessment,
    device_fused: GridXcCandidateAssessment,
    device_xc_identity: str,
    endpoint_seconds: collections.abc.Mapping[str, float | None] | None = None,
    minimum_speedup: float = 1.0,
    endpoint_noise_fraction: float = 0.01,
) -> GridXcRegionSelection:
    """Select and materialize the measured grid/XC implementation plan."""

    candidates = bind_grid_xc_region_candidates(
        program,
        host_unfused=host_unfused,
        device_fused=device_fused,
        device_xc_identity=device_xc_identity,
        endpoint_seconds=endpoint_seconds,
    )
    selected = select_program_region_candidate(
        candidates.all,
        minimum_speedup=minimum_speedup,
        endpoint_noise_fraction=endpoint_noise_fraction,
    )
    return GridXcRegionSelection(
        selected,
        apply_program_region_candidate(program, selected),
    )
