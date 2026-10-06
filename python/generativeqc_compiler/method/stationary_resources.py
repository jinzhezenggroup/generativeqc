"""Method-neutral resource schedule for the stationary CUDA source owner.

The compiler selects bounded point lanes; native code validates the contract and
owns allocation/launches. This schedule changes floating-point grouping, never
pointwise AO/Becke mathematics or the set of grid points visited.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TypeVar

from generativeqc_compiler.common.cuda_target import CudaTargetInfo
from generativeqc_compiler.xc.grid_phased import PhasedBeckePlan

GEOMETRY_MAX_LANES = 2048
GEOMETRY_MAX_SCRATCH_BYTES = 8 << 20
GEOMETRY_THREADS = 32
BECKE_COOPERATIVE_MAX_ATOMS = 128
BECKE_RETAINED_MAX_ATOMS = 32
BECKE_PAIR_TILE_ROWS = 4
BECKE_COOPERATIVE_THREADS = 32
BECKE_PAIR_STATE_BYTES = 64
BECKE_COOPERATIVE_CONTROL_BYTES = 16
STATIONARY_MAX_ATOMS = 128
# Dense D/W and AO panels are dynamically allocated. This finite bound admits
# 96-atom full def2-TZVPD (1856 AOs); byte budgets and the native-integral-only
# rule below still apply. Keep native allocation admission in parity tests.
STATIONARY_MAX_AOS = 2048
STATIONARY_MAX_PRIMITIVES = 16384
_SIZE_MAX = (1 << 64) - 1
_TileLayout = TypeVar("_TileLayout")


def plan_stationary_cuda_grid_schedule(
    *,
    grid_points: int,
    tile_points: int | None,
    admit: Callable[[int], _TileLayout],
) -> _TileLayout:
    """Select one complete, budget-admitted semilocal or composite grid tile.

    The callback is a dry capacity/work query: it must account for every live
    owner, raise ValueError on rejection, and never allocate or execute CUDA.
    AO and method owners retain their own layout formulas. The shared schedule
    prefers 1024 points, then the qualified 256-point fallback and smaller
    tiles; it cannot shrink whole-grid storage or change the scientific grid.
    Explicit requests are tried exactly once, including requests above the
    point count. This preserves caller-controlled capacity and tail tests.
    """
    if type(grid_points) is not int or not 1 <= grid_points <= _SIZE_MAX:
        raise ValueError("stationary CUDA grid_points must be a positive uint64")
    if tile_points is not None and (
        type(tile_points) is not int or not 1 <= tile_points <= 4096
    ):
        raise ValueError("stationary CUDA tile_points must be None or in [1,4096]")
    candidates = (
        (tile_points,)
        if tile_points is not None
        else tuple(
            dict.fromkeys(
                min(points, grid_points)
                for points in (1024, 256, 128, 64, 32, 16, 8, 4, 2, 1)
            )
        )
    )
    failure = None
    for points in candidates:
        try:
            return admit(points)
        except ValueError as error:
            failure = error
    raise ValueError(
        f"no admitted stationary CUDA grid schedule: {failure}"
    ) from failure


@dataclass(frozen=True, slots=True)
class StationaryCudaGridWork:
    """Finite grid work with bounded asynchronous submission windows.

    The whole-grid limits are optional diagnostic time/work guards. The
    per-window limits remain mandatory, including for public complete forces.
    No grid points, equations, or scientific accuracy settings are changed.
    """

    grid_points: int
    tile_points: int
    tile_count: int
    chunk_points: int
    chunk_count: int
    grid_pair_visits: int
    chunk_pair_visits: int

    def chunks(self) -> Iterator[tuple[int, int]]:
        """Yield half-open windows without materializing a full point/task list."""
        for begin in range(0, self.grid_points, self.chunk_points):
            yield begin, min(begin + self.chunk_points, self.grid_points)


def plan_stationary_cuda_grid_work(
    *,
    atoms: int,
    grid_points: int,
    tile_points: int,
    max_grid_points: int | None = 1_000_000,
    max_grid_pair_visits: int | None = 100_000_000,
    max_pending_tiles: int = 64,
    max_pending_pair_visits: int = 100_000_000,
) -> StationaryCudaGridWork:
    """Plan complete work and bounded error fences before compilation/allocation."""
    for name, value, cap in (
        ("atoms", atoms, STATIONARY_MAX_ATOMS),
        ("grid_points", grid_points, 1 << 40),
        ("tile_points", tile_points, 4096),
        ("max_pending_tiles", max_pending_tiles, 4096),
        ("max_pending_pair_visits", max_pending_pair_visits, 1 << 40),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"stationary CUDA {name} exceeds resource caps")
    for name, value in (
        ("max_grid_points", max_grid_points),
        ("max_grid_pair_visits", max_grid_pair_visits),
    ):
        if value is not None and (type(value) is not int or not 1 <= value <= 1 << 40):
            raise ValueError(f"{name} must be None or an integer in [1,{1 << 40}]")
    pairs = atoms * (atoms - 1) // 2
    visits = (1 + 2 * grid_points) * pairs
    if visits > _SIZE_MAX:
        raise ValueError("stationary grid work exceeds uint64 metric range")
    if max_grid_points is not None and grid_points > max_grid_points:
        raise ValueError("grid point work budget exceeded")
    if max_grid_pair_visits is not None and visits > max_grid_pair_visits:
        raise ValueError("grid work budget exceeded")
    tile_visits = 2 * min(grid_points, tile_points) * pairs
    if tile_visits > max_pending_pair_visits:
        raise ValueError("stationary grid tile exceeds pending pair-visit budget")
    pending = min(
        max_pending_tiles,
        max_pending_pair_visits // tile_visits if tile_visits else max_pending_tiles,
    )
    chunk_points = min(grid_points, pending * tile_points)
    return StationaryCudaGridWork(
        grid_points,
        tile_points,
        (grid_points + tile_points - 1) // tile_points,
        chunk_points,
        (grid_points + chunk_points - 1) // chunk_points,
        visits,
        2 * chunk_points * pairs,
    )


def stationary_cuda_requires_native_integrals(
    *, atoms: int, aos: int, primitives: int
) -> bool:
    """Enlarged domains must use the shared prepared HF/DFT derivative owner.

    The old AO-descriptor diagnostic fallback remains a small-domain route;
    bounded storage alone is no justification for silently executing AO^4 work.
    """
    for name, value, cap in (
        ("atoms", atoms, STATIONARY_MAX_ATOMS),
        ("aos", aos, STATIONARY_MAX_AOS),
        ("primitives", primitives, STATIONARY_MAX_PRIMITIVES),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"stationary CUDA {name} exceeds resource caps")
    return atoms > 32 or aos > 128 or primitives > 4096


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
        ("atoms", atoms, STATIONARY_MAX_ATOMS),
        ("aos", aos, STATIONARY_MAX_AOS),
        ("primitives", primitives, STATIONARY_MAX_PRIMITIVES),
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
    becke_threads_per_point: int = 1
    becke_shared_bytes: int = 0
    phased_becke_bytes: int = 0
    becke_primitive: bool = False


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
    cooperative_becke: bool | None = None,
    phased_becke: bool = False,
    becke_primitive: bool = False,
) -> StationaryCudaResources:
    """Choose up to one lane per point within the admitted owner's byte budget.

    Small tiles and tight budgets naturally retain a single-block schedule.
    A tail uses only min(planned lanes, tail points) of the retained panels.
    No device probe or allocation is part of this compiler planning function.
    Automatic cooperation is restricted to the measured sm_120 large-point
    domain. Explicit False retains the generic schedule; unqualified targets
    and small systems retain it automatically. Shared-memory admission below
    is mandatory even when cooperation is requested.
    """
    if cooperative_becke is None:
        cooperative_becke = (
            target.compute_capability == (12, 0)
            and type(atoms) is int
            and 12 <= atoms <= BECKE_COOPERATIVE_MAX_ATOMS
        )
    if type(cooperative_becke) is not bool:
        raise ValueError("cooperative Becke selection must be boolean")
    if type(phased_becke) is not bool:
        raise ValueError("phased Becke selection must be boolean")
    if type(becke_primitive) is not bool:
        raise ValueError("Becke primitive selection must be boolean")
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
    becke_threads, shared_bytes = 1, 0
    retained_rows = min(BECKE_PAIR_TILE_ROWS, atoms - 1)
    state_count = (
        atoms * (atoms - 1) // 2
        if atoms <= BECKE_RETAINED_MAX_ATOMS
        else retained_rows * (2 * atoms - retained_rows - 1) // 2
    )
    required_shared = (
        BECKE_COOPERATIVE_CONTROL_BYTES + BECKE_PAIR_STATE_BYTES * state_count
    )
    if (
        cooperative_becke
        and 1 < atoms <= BECKE_COOPERATIVE_MAX_ATOMS
        and target.maximum_threads_per_block >= BECKE_COOPERATIVE_THREADS
        and required_shared
        <= min(target.shared_memory_per_block, target.tuning_maximum_shared_bytes)
    ):
        becke_threads, shared_bytes = BECKE_COOPERATIVE_THREADS, required_shared
    phase_bytes = 0
    # Qualification-only opt-in. Preserve the lane count, integral reserve and
    # bounded route; never evict a concurrent owner to admit a pair cache.
    if (
        (phased_becke or becke_primitive)
        and atoms > BECKE_RETAINED_MAX_ATOMS
        and lanes == points
        and center_bytes
        and becke_threads > 1
        and target.compute_capability == (12, 0)
        and target.maximum_threads_per_block >= 128
    ):
        phase_plan = PhasedBeckePlan(atoms, points)
        required = phase_plan.scratch_bytes + 8 * points + 8 * phase_plan.pairs
        if required <= budget_bytes - allocation - center_bytes:
            phase_bytes = required
    return StationaryCudaResources(
        lanes,
        threads,
        per_lane * lanes,
        allocation + center_bytes + phase_bytes,
        center_bytes,
        becke_threads,
        shared_bytes,
        phase_bytes,
        bool(phase_bytes and becke_primitive),
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
        ("atoms", atoms, STATIONARY_MAX_ATOMS),
        ("aos", aos, STATIONARY_MAX_AOS),
        ("primitives", primitives, STATIONARY_MAX_PRIMITIVES),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"stationary CUDA {name} exceeds resource caps")
    return (
        2 * (58 * aos + 52 * atoms + 16 * primitives + 4 * aos * (aos + 1) + 32)
        + 48 * atoms
    )
