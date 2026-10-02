"""Method-neutral resource schedule for the stationary CUDA source owner.

The compiler selects bounded point lanes; native code validates the contract and
owns allocation/launches. This schedule changes floating-point grouping, never
pointwise AO/Becke mathematics or the set of grid points visited.
"""

from dataclasses import dataclass

from generativeqc_compiler.common.cuda_target import CudaTargetInfo

GEOMETRY_MAX_LANES = 2048
GEOMETRY_MAX_SCRATCH_BYTES = 8 << 20
GEOMETRY_THREADS = 32
_SIZE_MAX = (1 << 64) - 1


def stationary_cuda_allocation_bytes(
    *,
    atoms: int,
    aos: int,
    primitives: int,
    points: int,
    tasks: int,
    spins: int,
    sources: int,
    geometry_lanes: int,
    cache_center_geometry: bool = False,
) -> int:
    """Exact arena bytes, including both lane panels and the error reserve."""
    for name, value, cap in (
        ("atoms", atoms, 128),
        ("aos", aos, 1024),
        ("primitives", primitives, 16384),
        ("points", points, 4096),
        ("tasks", tasks, 4096),
        ("spins", spins, 2),
        ("sources", sources, 128),
        ("geometry_lanes", geometry_lanes, min(points, GEOMETRY_MAX_LANES)),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"stationary CUDA {name} exceeds resource caps")
    if type(cache_center_geometry) is not bool:
        raise ValueError("stationary center geometry cache flag must be boolean")
    scratch = 18 * geometry_lanes * atoms * 8
    if scratch > GEOMETRY_MAX_SCRATCH_BYTES:
        raise ValueError("stationary geometry scratch budget exceeded")
    result = (
        8
        * (
            2 * primitives
            + 4 * aos
            + 22 * tasks
            + (3 + 3 * sources) * atoms
            + 3 * points
            + 2 * spins * aos * aos
        )
        + scratch
        + (48 * (atoms * (atoms - 1) // 2) if cache_center_geometry else 0)
        + 256
    )
    if result > _SIZE_MAX:
        raise ValueError("stationary CUDA allocation overflow")
    return result


@dataclass(frozen=True, slots=True)
class StationaryCudaResources:
    geometry_lanes: int
    geometry_threads: int
    geometry_scratch_bytes: int
    allocation_bytes: int
    center_geometry_bytes: int


def plan_stationary_cuda_resources(
    *,
    atoms: int,
    aos: int,
    primitives: int,
    points: int,
    tasks: int,
    spins: int,
    sources: int,
    target: CudaTargetInfo,
    budget_bytes: int,
) -> StationaryCudaResources:
    """Choose up to one lane per point within the admitted owner's byte budget.

    Small tiles and tight budgets naturally retain a single-block schedule.
    A tail uses only min(planned lanes, tail points) of the retained panels.
    No device probe or allocation is part of this compiler planning function.
    """
    if type(budget_bytes) is not int or not 0 <= budget_bytes <= _SIZE_MAX:
        raise ValueError("stationary CUDA byte budget is not representable")
    minimum = stationary_cuda_allocation_bytes(
        atoms=atoms,
        aos=aos,
        primitives=primitives,
        points=points,
        tasks=tasks,
        spins=spins,
        sources=sources,
        geometry_lanes=1,
    )
    per_lane = 18 * atoms * 8
    fixed = minimum - per_lane
    if budget_bytes < minimum:
        raise ValueError("stationary CUDA byte budget exceeded")
    lanes = min(
        points,
        GEOMETRY_MAX_LANES,
        min(GEOMETRY_MAX_SCRATCH_BYTES, budget_bytes - fixed) // per_lane,
    )
    threads = min(GEOMETRY_THREADS, target.maximum_threads_per_block, lanes)
    if threads < 1:
        raise ValueError("stationary CUDA target has no geometry threads")
    allocation = fixed + per_lane * lanes
    # Retain geometry only from spare budget after choosing point concurrency.
    # A tight budget must not lose lanes or revoke the bounded direct route.
    center_bytes = 48 * (atoms * (atoms - 1) // 2)
    if center_bytes > budget_bytes - allocation:
        center_bytes = 0
    return StationaryCudaResources(
        lanes, threads, per_lane * lanes, allocation + center_bytes, center_bytes
    )


def stationary_native_pair_reserve(*, atoms: int, aos: int, primitives: int) -> int:
    """Retain the existing paired one-electron provider's admission allowance.

    This is the conservative host-staging bound in
    src/scf/cuda/one_electron_gradient_bridge.cu, also covering its resident-D/W
    device scratch. A retained Direct owner only needs 48*atoms, but capability
    fallback must keep the full shared provider allowance. The current AO
    topology has at most three Cartesian expansion terms. Native arithmetic
    parity is independently host-compiled in the resource tests.
    """
    for name, value, cap in (
        ("atoms", atoms, 128),
        ("aos", aos, 1024),
        ("primitives", primitives, 16384),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"stationary CUDA {name} exceeds resource caps")
    return (
        2 * (58 * aos + 52 * atoms + 16 * primitives + 4 * aos * (aos + 1) + 32)
        + 48 * atoms
    )
