"""Bind native CUDA-KS XC staging routes to ProgramIR region selection.

This module describes the actual `CudaKsPlan::stage_xc` execution boundary.
The portable host-unfused route downloads the current device density, evaluates
the audited CPU XC/Vxc implementation, then uploads Vxc plus scalar totals/error
back to the CUDA KS state. The existing resident CudaXcPlan is an alternate
implementation of exactly that boundary.

Scientific admission remains owned by DFT. ProgramIR owns only the replaceable
execution region, while ScheduleContract remains the shared measured-promotion
and fallback vocabulary. No XC equation or CUDA kernel is duplicated here.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass, replace

from generativeqc_compiler.common.liveness import EffectKind
from generativeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from generativeqc_compiler.common.program_region import (
    ProgramRegion,
    ProgramRegionCandidate,
    apply_program_region_candidate,
    derive_program_region,
    select_program_region_candidate,
)
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.common.resources import byte_product
from generativeqc_compiler.common.schedule import ScheduleResources

from .xc_schedule import (
    GridXcCandidateAssessment,
    GridXcCandidateShape,
    GridXcScientificIdentity,
    grid_xc_domain_identity,
    grid_xc_shape_identity,
    schedule_profile_key,
)

if typing.TYPE_CHECKING:
    import collections.abc

    from generativeqc_compiler.common.schedule import ScheduleContract

_HOST_REGION_EFFECTS = {
    "runtime.cuda.ks_xc_density_d2h": EffectKind.PURE,
    "dft.CudaKsPlan.split_host_spin_density": EffectKind.PURE,
    "dft.CudaKsPlan.host_unfused_xc": EffectKind.PURE,
    "runtime.cuda.ks_xc_result_h2d": EffectKind.PURE,
}

_TOTAL_BYTES = 3 * 8
_ERROR_BYTES = 4


def _text(value: typing.Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _device_space(ordinal: int) -> str:
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("grid/XC device ordinal must be a non-negative integer")
    return f"device:{ordinal}"


def _density_bytes(shape: GridXcCandidateShape) -> int:
    if not isinstance(shape, GridXcCandidateShape):
        raise TypeError("native KS XC ProgramIR requires GridXcCandidateShape")
    return byte_product(8, shape.spins, shape.nao, shape.nao)


def native_ks_host_unfused_xc_program(
    shape: GridXcCandidateShape,
    *,
    scientific: GridXcScientificIdentity,
    density_identity: str,
    host_xc_identity: str,
    transfer_identity: str,
    device_ordinal: int = 0,
) -> ProgramIR:
    """Describe the current host-unfused `CudaKsPlan::stage_xc` boundary."""

    if not isinstance(scientific, GridXcScientificIdentity):
        raise TypeError("native KS XC source requires scientific identity")
    if not isinstance(shape, GridXcCandidateShape):
        raise TypeError("native KS XC source requires candidate shape")
    if (
        shape.spins != (2 if scientific.spin == "polarized" else 1)
        or scientific.observable != "potential"
        or scientific.density_route != "density_matrix"
        or shape.jet_components != len(scientific.jet_outputs)
    ):
        raise ValueError("native KS XC source scientific domain mismatch")
    density_identity = _text(density_identity, "KS density identity")
    host_xc_identity = _text(host_xc_identity, "host XC identity")
    transfer_identity = _text(transfer_identity, "transfer identity")
    device = _device_space(device_ordinal)
    density_bytes = _density_bytes(shape)
    matrix_bytes = byte_product(8, shape.nao, shape.nao)
    topology = {
        "scientific": scientific.identity,
        "shape": grid_xc_shape_identity(shape),
        "npoint": shape.npoint,
        "tile_points": shape.tile_points,
        "nao": shape.nao,
        "spins": shape.spins,
    }

    # Device inputs/outputs are boundary tokens. Their physical owners live in
    # the prepared KS arena, so ProgramIR does not charge those owners again.
    buffers = [
        ProgramBuffer("density_device", 0, device),
        ProgramBuffer("density_host", density_bytes),
    ]
    calls = [
        PlanCall(
            "download_density",
            "runtime.cuda.ks_xc_density_d2h",
            canonical_hash(
                {
                    "transfer": transfer_identity,
                    "direction": "d2h",
                    "bytes": density_bytes,
                    "topology": topology,
                }
            ),
            ("density_device",),
            ("density_host",),
        )
    ]
    host_density_reads: tuple[str, ...] = ("density_host",)
    if shape.spins == 2:
        buffers.extend(
            (
                ProgramBuffer("alpha_host", matrix_bytes),
                ProgramBuffer("beta_host", matrix_bytes),
            )
        )
        calls.append(
            PlanCall(
                "split_spin_density",
                "dft.CudaKsPlan.split_host_spin_density",
                canonical_hash(
                    {
                        "density": density_identity,
                        "topology": topology,
                    }
                ),
                ("density_host",),
                ("alpha_host", "beta_host"),
            )
        )
        host_density_reads = ("alpha_host", "beta_host")

    buffers.extend(
        (
            ProgramBuffer("vxc_host", density_bytes),
            ProgramBuffer("totals_host", _TOTAL_BYTES),
            ProgramBuffer("error_host", _ERROR_BYTES),
            ProgramBuffer("vxc_device", 0, device),
            ProgramBuffer("totals_device", 0, device),
            ProgramBuffer("error_device", 0, device),
        )
    )
    calls.extend(
        (
            PlanCall(
                "host_xc_vxc",
                "dft.CudaKsPlan.host_unfused_xc",
                canonical_hash(
                    {
                        "xc": host_xc_identity,
                        "density": density_identity,
                        "topology": topology,
                    }
                ),
                host_density_reads,
                ("vxc_host", "totals_host", "error_host"),
            ),
            PlanCall(
                "upload_xc",
                "runtime.cuda.ks_xc_result_h2d",
                canonical_hash(
                    {
                        "transfer": transfer_identity,
                        "direction": "h2d",
                        "bytes": density_bytes + _TOTAL_BYTES + _ERROR_BYTES,
                        "topology": topology,
                    }
                ),
                ("vxc_host", "totals_host", "error_host"),
                ("vxc_device", "totals_device", "error_device"),
            ),
        )
    )
    return ProgramIR(
        "cuda_ks_host_unfused_xc",
        tuple(buffers),
        ("density_device",),
        tuple(calls),
        ("vxc_device", "totals_device", "error_device"),
    )


@dataclass(frozen=True, slots=True)
class NativeKsXcSource:
    """DFT-owned source binding used to reconstruct, not relabel, an XC region.

    ProgramIR remains method-neutral. The domain owner carries science and shape
    separately and reconstructs the complete graph before attaching admission
    records; matching just a supplied program digest is not sufficient.
    """

    shape: GridXcCandidateShape
    scientific: GridXcScientificIdentity
    density_identity: str
    host_xc_identity: str
    transfer_identity: str
    device_ordinal: int = 0

    def program(self) -> ProgramIR:
        return native_ks_host_unfused_xc_program(
            self.shape,
            scientific=self.scientific,
            density_identity=self.density_identity,
            host_xc_identity=self.host_xc_identity,
            transfer_identity=self.transfer_identity,
            device_ordinal=self.device_ordinal,
        )


def _validate_source_admissions(
    program: ProgramIR,
    source: NativeKsXcSource,
    host: GridXcCandidateAssessment,
    device: GridXcCandidateAssessment,
) -> None:
    if not isinstance(source, NativeKsXcSource):
        raise TypeError("native KS XC binding requires a typed source")
    if program.identity != source.program().identity:
        raise ValueError("native KS XC source program mismatch")
    scientific = source.scientific
    expected_target = canonical_hash(
        {"backend": "cuda", "architecture": scientific.architecture}
    )
    for assessment in (host, device):
        contract = assessment.schedule_contract
        provenance = dict(contract.provenance)
        if (
            contract.consumer != "dft.grid_xc"
            or assessment.schedule_hash != contract.schedule_hash
            or contract.workload_hash != scientific.identity
            or contract.profile_key != schedule_profile_key(scientific)
            or contract.target_hash != expected_target
            or provenance.get("candidate_domain")
            != grid_xc_domain_identity(source.shape)
        ):
            raise ValueError("grid/XC admission source provenance mismatch")
    if dict(host.schedule_contract.provenance).get(
        "candidate_shape"
    ) != grid_xc_shape_identity(source.shape):
        raise ValueError("host grid/XC admission source shape mismatch")
    if (
        host.schedule_contract.precision_schedule_hash
        != device.schedule_contract.precision_schedule_hash
    ):
        raise ValueError("grid/XC admission source precision mismatch")


def native_ks_xc_region(program: ProgramIR) -> ProgramRegion:
    """Return the exact `stage_xc` region replaceable by resident device XC."""

    return derive_program_region(
        program,
        name="cuda-ks-stage-xc",
        start_call="download_density",
        end_call="upload_xc",
        effects=_HOST_REGION_EFFECTS,
    )


def _schedule_name(assessment: GridXcCandidateAssessment) -> str:
    if not isinstance(assessment, GridXcCandidateAssessment):
        raise TypeError("grid/XC region binding requires candidate assessments")
    provenance = dict(assessment.schedule_contract.provenance)
    name = provenance.get("domain_schedule")
    if name not in ("host_unfused", "device_fused"):
        raise ValueError("grid/XC assessment has an unknown execution schedule")
    if assessment.legal != assessment.schedule_contract.legal:
        raise ValueError("grid/XC assessment legality disagrees with ScheduleContract")
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


def _host_region_resources(program: ProgramIR) -> ScheduleResources:
    """Project only ProgramIR-visible bridge storage, not the prepared KS arena."""

    peaks = program.storage_analysis().peak_by_space
    host_bytes = peaks.get("pageable", 0) + peaks.get("pinned", 0)
    device_bytes = sum(
        value for space, value in peaks.items() if space.startswith("device:")
    )
    return ScheduleResources(
        host_bytes=host_bytes,
        device_bytes=device_bytes,
    )


def _host_bridge_traffic(program: ProgramIR) -> int:
    """Exact transfer payload from `stage_xc`, excluding CPU-internal traffic."""

    buffers = {buffer.name: buffer for buffer in program.buffers}
    return (
        buffers["density_host"].bytes
        + buffers["vxc_host"].bytes
        + buffers["totals_host"].bytes
        + buffers["error_host"].bytes
    )


def _region_schedule(
    assessment: GridXcCandidateAssessment,
    region: ProgramRegion,
    *,
    endpoint_seconds: float | None,
    resources: ScheduleResources | None = None,
    semantic_traffic_bytes: int | None = None,
) -> ScheduleContract:
    contract = assessment.schedule_contract
    provenance = dict(contract.provenance)
    provenance["program_region"] = region.identity
    provenance["region_source_consumer"] = contract.consumer
    profitability_fields: dict[str, typing.Any] = {
        "endpoint_seconds": endpoint_seconds,
        **(
            {}
            if semantic_traffic_bytes is None
            else {"semantic_traffic_bytes": semantic_traffic_bytes}
        ),
    }
    if contract.fallback:
        # The host-unfused XC region owns copies plus CPU work, but no custom
        # GPU compute kernel. Keep registers/occupancy unknown while making the
        # absence of region-owned spill traffic explicit for relative gating.
        profitability_fields.update(spill_store_bytes=0, spill_load_bytes=0)
        provenance["region_gpu_compute"] = "none"
    profitability = replace(contract.profitability, **profitability_fields)
    return replace(
        contract,
        consumer=region.consumer,
        resources=contract.resources if resources is None else resources,
        profitability=profitability,
        provenance=tuple(provenance.items()),
    )


@dataclass(frozen=True, slots=True)
class GridXcRegionCandidates:
    """Host bridge and resident-device implementations of one native KS region."""

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


def bind_native_ks_xc_region_candidates(
    program: ProgramIR,
    *,
    source: NativeKsXcSource,
    host_unfused: GridXcCandidateAssessment,
    device_fused: GridXcCandidateAssessment,
    device_xc_identity: str,
    endpoint_seconds: collections.abc.Mapping[str, float | None] | None = None,
) -> GridXcRegionCandidates:
    """Bind existing DFT schedule evidence to one exact native KS XC region."""

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

    _validate_source_admissions(program, source, host_unfused, device_fused)
    region = native_ks_xc_region(program)
    host_schedule = _region_schedule(
        host_unfused,
        region,
        endpoint_seconds=_endpoint_seconds(endpoint_seconds, "host_unfused"),
        resources=_host_region_resources(program),
        semantic_traffic_bytes=_host_bridge_traffic(program),
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
    return GridXcRegionCandidates(region, host_candidate, device_candidate)


@dataclass(frozen=True, slots=True)
class GridXcRegionSelection:
    """Selected executable candidate plus the resulting ProgramIR."""

    candidate: ProgramRegionCandidate
    program: ProgramIR


def select_native_ks_xc_region_program(
    program: ProgramIR,
    *,
    source: NativeKsXcSource,
    host_unfused: GridXcCandidateAssessment,
    device_fused: GridXcCandidateAssessment,
    device_xc_identity: str,
    endpoint_seconds: collections.abc.Mapping[str, float | None] | None = None,
    minimum_speedup: float = 1.0,
    endpoint_noise_fraction: float = 0.01,
) -> GridXcRegionSelection:
    """Select and materialize one measured native-KS XC implementation."""

    candidates = bind_native_ks_xc_region_candidates(
        program,
        source=source,
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
