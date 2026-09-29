"""Compiled resource evidence for the complete native CUDA grid/XC region."""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass

from generativeqc_compiler.common.cuda_resources import KernelResources
from generativeqc_compiler.common.cuda_target import CudaTargetInfo
from generativeqc_compiler.common.gpu_profitability import GpuProfitability
from generativeqc_compiler.common.provenance import canonical_hash

from .xc_contraction_cuda import DEFAULT_XC_MATRIX_SCHEDULE

GRID_XC_COMPILED_RESOURCE_SCHEMA = "generativeqc.dft.grid-xc-compiled-region.v1"
GRID_XC_COMPILED_SCOPES = (
    "ao_jets",
    "density_product",
    "density_features",
    "xc_points",
    "vxc_contraction",
)
_FUNCTIONAL_CODES = {"LDA_XC_PW": 0, "PBE": 1}


@dataclass(frozen=True, slots=True)
class GridXcCompiledResourceShape:
    """Execution dimensions that decide which compiled native kernels are active."""

    npoint: int
    tile_points: int
    nao: int
    spins: int

    def __post_init__(self) -> None:
        for name in ("npoint", "tile_points", "nao", "spins"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.spins not in (1, 2):
            raise ValueError("grid/XC compiled resource spin count must be one or two")

    @property
    def identity(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class GridXcCompiledRegionEvidence:
    """Complete PTXAS pressure envelope for one native device-fused XC region."""

    architecture: str
    source_identity: str
    functional: str
    shape: GridXcCompiledResourceShape
    scopes: tuple[tuple[str, tuple[KernelResources, ...]], ...]
    profitability: GpuProfitability

    def __post_init__(self) -> None:
        if not self.architecture.startswith("sm_"):
            raise ValueError("compiled grid/XC evidence requires an sm_* architecture")
        if type(self.source_identity) is not str or not self.source_identity:
            raise ValueError("compiled grid/XC evidence requires a source identity")
        if self.functional not in _FUNCTIONAL_CODES:
            raise ValueError("compiled grid/XC evidence supports only LDA/PBE")
        if not isinstance(self.shape, GridXcCompiledResourceShape):
            raise TypeError("compiled grid/XC evidence requires a typed resource shape")
        if not isinstance(self.profitability, GpuProfitability):
            raise TypeError("compiled grid/XC evidence requires GpuProfitability")
        payload = self.profitability.to_payload()
        if (
            any(value is not None for value in payload["static"].values())
            or payload["endpoint_seconds"] is not None
        ):
            raise ValueError("compiled grid/XC evidence must contain compiled facts only")
        names = tuple(name for name, _ in self.scopes)
        if names != GRID_XC_COMPILED_SCOPES:
            raise ValueError("compiled grid/XC evidence is missing a required region scope")
        for name, resources in self.scopes:
            if not resources or any(
                not isinstance(resource, KernelResources) for resource in resources
            ):
                raise ValueError(f"compiled grid/XC scope {name!r} requires PTXAS resources")

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": GRID_XC_COMPILED_RESOURCE_SCHEMA,
            "architecture": self.architecture,
            "source_identity": self.source_identity,
            "functional": self.functional,
            "shape": asdict(self.shape),
            "scopes": {
                name: [asdict(resource) for resource in resources]
                for name, resources in self.scopes
            },
            "profitability": self.profitability.to_payload(),
        }


def _leaf(function: str) -> str:
    qualified = function.split("(", 1)[0]
    return qualified.rsplit("::", 1)[-1].replace(" ", "")


def _matching(
    resources: tuple[KernelResources, ...],
    predicate: typing.Callable[[str], bool],
    label: str,
) -> tuple[KernelResources, ...]:
    selected = tuple(resource for resource in resources if predicate(_leaf(resource.function)))
    if not selected:
        raise ValueError(f"compiled native grid/XC resources are missing {label}")
    return selected


def _active_scopes(
    resources: tuple[KernelResources, ...],
    shape: GridXcCompiledResourceShape,
    functional: str,
    target: CudaTargetInfo,
) -> tuple[tuple[str, tuple[KernelResources, ...]], ...]:
    code = _FUNCTIONAL_CODES[functional]
    count = min(shape.npoint, shape.tile_points)
    tiled = DEFAULT_XC_MATRIX_SCHEDULE.admitted(
        shape.nao,
        count,
        spins=shape.spins,
        work_jets=1,
        maximum_threads_per_block=target.maximum_threads_per_block,
        maximum_shared_bytes=target.shared_memory_per_block,
    )
    feature_token = "density_features<true>" if shape.nao >= 32 else "density_features<false>"
    density_token = (
        "tiled_density_product<false>" if tiled else "density_product<false>"
    )
    point_token = f"evaluate_points<{code}"

    ao = _matching(
        resources,
        lambda name: name.startswith("validate_density(")
        or (name.startswith("ao_kernel(") and "ao_kernel_fp32" not in name),
        "strict-FP64 AO/validation scope",
    )
    density = _matching(
        resources,
        lambda name: name.startswith(density_token),
        "strict-FP64 density-product scope",
    )
    features = _matching(
        resources,
        lambda name: name.startswith(feature_token),
        "active density-feature scope",
    )
    points = _matching(
        resources,
        lambda name: name.startswith(point_token)
        and name.endswith(",false>"),
        "functional-specific XC point scope",
    )
    if tiled:
        potential = _matching(
            resources,
            lambda name: name.startswith("compact_potential_panels(")
            or name.startswith("tiled_potential("),
            "tiled Vxc contraction scope",
        )
    else:
        potential = _matching(
            resources,
            lambda name: name.startswith("assemble_potential(")
            or name.startswith("accumulate_totals("),
            "scalar Vxc contraction scope",
        )
    return (
        ("ao_jets", ao),
        ("density_product", density),
        ("density_features", features),
        ("xc_points", points),
        ("vxc_contraction", potential),
    )


def _kernel_threads(function: str, shape: GridXcCompiledResourceShape, functional: str) -> int:
    name = _leaf(function)
    if name.startswith("tiled_density_product") or name.startswith("tiled_potential"):
        return DEFAULT_XC_MATRIX_SCHEDULE.threads
    if name.startswith("accumulate_totals"):
        return 32
    if name.startswith("evaluate_points"):
        return 32 if functional == "PBE" else 128
    return 128


def _occupancy(
    resource: KernelResources,
    *,
    threads: int,
    target: CudaTargetInfo,
) -> float:
    limits = [
        target.maximum_blocks_per_sm,
        target.maximum_threads_per_sm // threads,
    ]
    if resource.registers:
        limits.append(target.registers_per_sm // (resource.registers * threads))
    if resource.shared_bytes:
        limits.append(target.shared_memory_per_sm // resource.shared_bytes)
    resident = max(0, min(limits))
    return min(1.0, resident * threads / target.maximum_threads_per_sm)


def _pressure_envelope(
    rows: tuple[KernelResources, ...],
    *,
    shape: GridXcCompiledResourceShape,
    functional: str,
    target: CudaTargetInfo,
) -> GpuProfitability:
    spill_peak = max(rows, key=lambda row: row.spill_store_bytes + row.spill_load_bytes)
    local = tuple(row.local_bytes for row in rows if row.local_bytes is not None)
    occupancies = tuple(
        _occupancy(
            row,
            threads=_kernel_threads(row.function, shape, functional),
            target=target,
        )
        for row in rows
    )
    return GpuProfitability(
        compiled_registers_per_thread=max(row.registers for row in rows),
        spill_store_bytes=spill_peak.spill_store_bytes,
        spill_load_bytes=spill_peak.spill_load_bytes,
        local_bytes=max(local) if local else None,
        shared_bytes=max(row.shared_bytes for row in rows),
        compiled_occupancy_upper_bound=min(occupancies),
    )


def native_grid_xc_compiled_region_evidence(
    resources: typing.Iterable[KernelResources],
    *,
    shape: GridXcCompiledResourceShape,
    functional: str,
    target: CudaTargetInfo,
    source_identity: str,
    object_bytes: int | None = None,
    compile_seconds: float | None = None,
) -> GridXcCompiledRegionEvidence:
    """Select active native kernels and form one fail-closed region pressure record."""

    materialized = tuple(resources)
    if not materialized:
        raise ValueError("native grid/XC compiled evidence requires PTXAS resources")
    if any(not isinstance(resource, KernelResources) for resource in materialized):
        raise TypeError("native grid/XC compiled evidence requires KernelResources")
    if not isinstance(shape, GridXcCompiledResourceShape):
        raise TypeError("native grid/XC compiled evidence requires a typed resource shape")
    if not isinstance(target, CudaTargetInfo):
        raise TypeError("native grid/XC compiled evidence requires CudaTargetInfo")
    if functional not in _FUNCTIONAL_CODES:
        raise ValueError("native grid/XC compiled evidence supports only LDA/PBE")
    scopes = _active_scopes(materialized, shape, functional, target)
    scope_profitability = tuple(
        _pressure_envelope(
            rows,
            shape=shape,
            functional=functional,
            target=target,
        )
        for _, rows in scopes
    )
    spill_peak = max(
        scope_profitability,
        key=lambda value: -1 if value.spill_bytes is None else value.spill_bytes,
    )
    local = tuple(
        value.local_bytes
        for value in scope_profitability
        if value.local_bytes is not None
    )
    profitability = GpuProfitability(
        compiled_registers_per_thread=max(
            typing.cast("int", value.compiled_registers_per_thread)
            for value in scope_profitability
        ),
        spill_store_bytes=spill_peak.spill_store_bytes,
        spill_load_bytes=spill_peak.spill_load_bytes,
        local_bytes=max(local) if local else None,
        shared_bytes=max(
            typing.cast("int", value.shared_bytes) for value in scope_profitability
        ),
        compiled_occupancy_upper_bound=min(
            typing.cast("float", value.compiled_occupancy_upper_bound)
            for value in scope_profitability
        ),
        object_bytes=object_bytes,
        compile_seconds=compile_seconds,
    )
    return GridXcCompiledRegionEvidence(
        architecture=target.architecture,
        source_identity=source_identity,
        functional=functional,
        shape=shape,
        scopes=scopes,
        profitability=profitability,
    )
