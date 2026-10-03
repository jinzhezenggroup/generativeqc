"""Bounded AO/geometry schedule for the resident composite stationary consumer.

Only metadata and caller-provided native capacities enter this planner. It does
not load a runtime or probe a device. Full-grid nonlocal storage is invariant
under AO tiling; both stationary accumulators coexist with that storage.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from generativeqc_compiler.common.cuda_target import CudaTargetInfo
from generativeqc_compiler.method.stationary_resources import (
    StationaryCudaResources,
    plan_stationary_cuda_grid_schedule,
    plan_stationary_cuda_resources,
)


@dataclass(frozen=True, slots=True)
class CompositeStationaryCudaResources:
    """Admitted complete live capacity and selected point/Becke concurrency."""

    grid: Any
    sources: StationaryCudaResources
    native_bytes: int
    device_bound: int
    host_bound: int


def plan_composite_stationary_cuda_resources(
    basis: Any,
    *,
    grid_plan: Callable[[int], Any],
    grid_points: int,
    spins: int,
    source_count: int,
    nonlocal_bytes: int,
    target: CudaTargetInfo,
    max_device_bytes: int,
    max_host_bytes: int,
    tile_points: int | None = None,
) -> CompositeStationaryCudaResources:
    """Prefer 1024 points, retaining bounded smaller tiles under tight totals.

    Explicit tile requests are honored or rejected, never silently changed.
    The existing source planner selects a cooperative Becke block only when
    the target admits its shared memory; otherwise it keeps the scalar route.
    Native derivative and host allowances preserve the composite owner's prior
    conservative inventory, including both host and device grid capacity.
    The AO owner supplies its dry tile-capacity query; this method planner does
    not duplicate AO layout formulas or acquire a dependency on the DFT owner.
    """
    for name, value in (
        ("grid_points", grid_points),
        ("nonlocal_bytes", nonlocal_bytes),
        ("max_device_bytes", max_device_bytes),
        ("max_host_bytes", max_host_bytes),
    ):
        if type(value) is not int or not 0 < value < 2**64:
            raise ValueError(f"composite stationary {name} must be a positive uint64")
    n, na = basis.nao, basis.natom
    # The prepared Direct owner is already charged to SCF. This reserve covers
    # the existing one-electron/transient native route and final-state export.
    native_bytes = 256 * n * n + 1024 * (na + n + basis.nprimitive + len(basis.shells))

    def admit(points: int) -> CompositeStationaryCudaResources:
        grid = grid_plan(points)
        sources = plan_stationary_cuda_resources(
            atoms=na,
            aos=n,
            primitives=basis.nprimitive,
            points=points,
            tasks=1,
            spins=spins,
            sources=source_count,
            target=target,
            budget_bytes=(
                max_device_bytes
                - grid.peak_bytes
                - 48 * points
                - nonlocal_bytes
                - native_bytes
            )
            // 2,
            cooperative_becke=True,
        )
        device = (
            grid.peak_bytes
            + 2 * sources.allocation_bytes
            + 48 * points
            + nonlocal_bytes
            + native_bytes
        )
        host = (
            grid.host_bytes
            + 8 * (64 * grid_points + 8 * n * n + 128 * na)
            + nonlocal_bytes
            + native_bytes
        )
        if device > max_device_bytes or host > max_host_bytes:
            raise ValueError("composite stationary numeric capacity budget exceeded")
        return CompositeStationaryCudaResources(
            grid, sources, native_bytes, device, host
        )

    return plan_stationary_cuda_grid_schedule(
        grid_points=grid_points, tile_points=tile_points, admit=admit
    )
