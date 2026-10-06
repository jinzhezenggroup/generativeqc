"""Guarded cross-functional active-AO policy for CUDA DFT forces.

The selector contains no method or functional names. Qualified profiles bind
source/device workload evidence; every miss retains the dense AO domain.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass

DEFAULT_FORCE_ACTIVE_AO_POLICY = "auto"
_SUPPORTED_DERIVATIVE_ORDERS = frozenset((1, 2))


@dataclass(frozen=True, slots=True)
class ForceActiveAoWorkload:
    architecture: str
    derivative_order: int
    spin_blocks: int
    composition: str
    hamiltonian: str
    density_fitted: bool
    atoms: int
    aos: int
    grid_points: int
    tile_policy: str
    tile_points: int | None
    max_device_bytes: int
    max_host_bytes: int
    resident_grid: bool

    def __post_init__(self) -> None:
        if not self.architecture:
            raise ValueError("force active-AO architecture must be nonempty")
        if type(self.derivative_order) is not int or self.derivative_order <= 0:
            raise ValueError("force active-AO derivative order must be positive")
        if self.spin_blocks not in (1, 2):
            raise ValueError("force active-AO spin blocks must be one or two")
        if self.composition not in ("ordinary", "composite"):
            raise ValueError("unknown stationary force composition")
        if type(self.density_fitted) is not bool:
            raise TypeError("force active-AO density-fitting flag must be boolean")
        if type(self.resident_grid) is not bool:
            raise TypeError("force active-AO resident-grid flag must be boolean")
        for name in (
            "atoms",
            "aos",
            "grid_points",
            "max_device_bytes",
            "max_host_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"force active-AO {name} must be positive")
        if self.tile_policy not in ("fixed", "budget-auto"):
            raise ValueError("unknown force active-AO tile policy")
        if self.tile_points is not None and (
            type(self.tile_points) is not int or self.tile_points <= 0
        ):
            raise ValueError("force active-AO tile size must be positive")

    def record(self) -> dict[str, typing.Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QualifiedForceActiveAoProfile:
    profile_id: str
    evidence: tuple[str, ...]
    architectures: tuple[str, ...]
    compositions: tuple[str, ...]
    derivative_orders: tuple[int, ...]
    spin_blocks: tuple[int, ...]
    density_fitted: bool | None
    min_atoms: int
    max_atoms: int
    min_aos: int
    max_aos: int
    min_grid_points: int
    max_grid_points: int
    tile_policy: str
    tile_points: int | None
    min_device_bytes: int
    min_host_bytes: int
    cutoff: float
    cache_bytes: int

    def __post_init__(self) -> None:
        if not self.profile_id or not self.evidence:
            raise ValueError("qualified force active-AO profile needs evidence")
        if not self.architectures or not self.compositions:
            raise ValueError("qualified force active-AO profile needs a device/domain")
        if any(
            value not in _SUPPORTED_DERIVATIVE_ORDERS
            for value in self.derivative_orders
        ):
            raise ValueError("profile derivative order is unsupported")
        if any(value not in (1, 2) for value in self.spin_blocks):
            raise ValueError("profile spin scope is invalid")
        if any(value not in ("ordinary", "composite") for value in self.compositions):
            raise ValueError("profile composition scope is invalid")
        if self.density_fitted is not None and type(self.density_fitted) is not bool:
            raise TypeError("profile density-fitting scope must be bool or None")
        for low, high, name in (
            (self.min_atoms, self.max_atoms, "atoms"),
            (self.min_aos, self.max_aos, "AOs"),
            (self.min_grid_points, self.max_grid_points, "grid points"),
        ):
            if type(low) is not int or type(high) is not int or low <= 0 or high < low:
                raise ValueError(f"invalid qualified {name} range")
        if self.tile_policy not in ("fixed", "budget-auto"):
            raise ValueError("qualified tile policy is invalid")
        if self.tile_points is not None and (
            type(self.tile_points) is not int or self.tile_points <= 0
        ):
            raise ValueError("qualified tile size is invalid")
        if (
            type(self.min_device_bytes) is not int
            or type(self.min_host_bytes) is not int
            or self.min_device_bytes <= 0
            or self.min_host_bytes <= 0
        ):
            raise ValueError("qualified resource minima must be positive")
        if (
            type(self.cutoff) not in (int, float)
            or isinstance(self.cutoff, bool)
            or not 0 < float(self.cutoff) < 1
        ):
            raise ValueError("qualified active-AO cutoff must be in (0,1)")
        if type(self.cache_bytes) is not int or self.cache_bytes < 0:
            raise ValueError("qualified active-AO cache budget must be nonnegative")

    def matches(self, workload: ForceActiveAoWorkload) -> bool:
        return (
            workload.architecture in self.architectures
            and workload.composition in self.compositions
            and workload.derivative_order in self.derivative_orders
            and workload.spin_blocks in self.spin_blocks
            and (
                self.density_fitted is None
                or workload.density_fitted is self.density_fitted
            )
            and self.min_atoms <= workload.atoms <= self.max_atoms
            and self.min_aos <= workload.aos <= self.max_aos
            and self.min_grid_points <= workload.grid_points <= self.max_grid_points
            and workload.tile_policy == self.tile_policy
            and (self.tile_points is None or workload.tile_points == self.tile_points)
            and workload.max_device_bytes >= self.min_device_bytes
            and workload.max_host_bytes >= self.min_host_bytes
        )


@dataclass(frozen=True, slots=True)
class ForceActiveAoDecision:
    workload: ForceActiveAoWorkload
    profile_id: str | None
    reason: str
    cutoff: float | None
    cache_bytes: int

    @property
    def selected(self) -> bool:
        return self.cutoff is not None


# #1598 / #1853 promotion registry. The first positive production profile is
# intentionally structural rather than functional-name based: it covers the
# sm_120 ordinary RKS second-jet workload envelope that has complete 48/96-atom
# cold/warm/moved evidence. Smaller, larger, spin-polarized, composite, DF, ECP,
# alternate-tile, or under-budget workloads remain dense until separately
# qualified.
QUALIFIED_FORCE_ACTIVE_AO_PROFILES: tuple[QualifiedForceActiveAoProfile, ...] = (
    QualifiedForceActiveAoProfile(
        profile_id="sm120-ordinary-rks-second-jet-v1",
        evidence=(
            "benchmarks/results/pbe0-public-force-policy-20261005/README.md",
            "benchmarks/results/pbe0-force-followups-20261005/README.md",
        ),
        architectures=("sm_120",),
        compositions=("ordinary",),
        derivative_orders=(2,),
        spin_blocks=(1,),
        density_fitted=False,
        min_atoms=48,
        max_atoms=96,
        min_aos=384,
        max_aos=768,
        min_grid_points=1_179_648,
        max_grid_points=2_359_296,
        tile_policy="fixed",
        tile_points=256,
        min_device_bytes=512 << 20,
        min_host_bytes=256 << 20,
        cutoff=1e-16,
        cache_bytes=16 << 20,
    ),
)


def resolve_force_active_ao_policy(
    workload: ForceActiveAoWorkload,
    *,
    profiles: tuple[QualifiedForceActiveAoProfile, ...] | None = None,
) -> ForceActiveAoDecision:
    if DEFAULT_FORCE_ACTIVE_AO_POLICY != "auto":
        raise RuntimeError("force active-AO production policy is not automatic")
    candidates = QUALIFIED_FORCE_ACTIVE_AO_PROFILES if profiles is None else profiles
    if not candidates:
        return ForceActiveAoDecision(workload, None, "no-qualified-profile", None, 0)
    if workload.hamiltonian != "all-electron":
        return ForceActiveAoDecision(workload, None, "unsupported-hamiltonian", None, 0)
    if workload.derivative_order not in _SUPPORTED_DERIVATIVE_ORDERS:
        return ForceActiveAoDecision(
            workload, None, "unsupported-derivative-order", None, 0
        )
    if not workload.resident_grid:
        return ForceActiveAoDecision(
            workload, None, "resident-grid-unavailable", None, 0
        )
    matched = tuple(profile for profile in candidates if profile.matches(workload))
    if not matched:
        return ForceActiveAoDecision(workload, None, "no-qualified-profile", None, 0)
    if len(matched) != 1:
        raise RuntimeError("overlapping qualified force active-AO profiles")
    profile = matched[0]
    return ForceActiveAoDecision(
        workload,
        profile.profile_id,
        "qualified-workload-profile",
        float(profile.cutoff),
        profile.cache_bytes,
    )


def force_active_ao_policy_record(
    decision: ForceActiveAoDecision,
    map_work: typing.Mapping[str, typing.Any] | None,
) -> dict[str, typing.Any]:
    observed = None if map_work is None else dict(map_work)
    selected_work = None if observed is None else observed.get("point_ao_square_sum")
    dense_work = None if observed is None else observed.get("dense_point_ao_square_sum")
    if not decision.selected:
        actual = "dense"
    elif observed is None:
        actual = "dense-fallback"
    elif (
        type(selected_work) is int
        and type(dense_work) is int
        and selected_work < dense_work
    ):
        actual = "selected"
    elif selected_work == dense_work:
        actual = "dense-identity"
    else:
        actual = "selected-unclassified"
    return {
        "schema": "generativeqc.force-active-ao-policy.v1",
        "requested_mode": DEFAULT_FORCE_ACTIVE_AO_POLICY,
        "decision": "selected" if decision.selected else "dense",
        "profile_id": decision.profile_id,
        "reason": decision.reason,
        "cutoff": decision.cutoff,
        "cache_bytes": decision.cache_bytes,
        "workload": decision.workload.record(),
        "actual_mode": actual,
        "selected_point_ao_square_sum": selected_work,
        "dense_point_ao_square_sum": dense_work,
    }
