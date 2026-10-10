"""Guarded cross-functional active-AO policy for CUDA DFT forces.

The selector contains no method or functional names. Qualified profiles bind
source/device workload evidence; every miss retains the dense AO domain.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass, replace

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
    device_name: str | None = None

    def __post_init__(self) -> None:
        if type(self.architecture) is not str or not self.architecture:
            raise ValueError("force active-AO architecture must be nonempty")
        if self.device_name is not None and (
            type(self.device_name) is not str or not self.device_name
        ):
            raise ValueError("force active-AO device name must be nonempty or absent")
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
    compositions: tuple[str, ...]
    derivative_orders: tuple[int, ...]
    spin_blocks: tuple[int, ...]
    density_fitted: bool | None
    min_dense_point_ao_square_work: int
    tile_policy: str
    tile_points: int | None
    min_device_bytes: int
    min_host_bytes: int
    cutoff: float
    cache_bytes: int
    producer: str = "sampled-jets"
    max_active_fraction: float = 1.0
    max_dense_point_ao_square_work: int | None = None

    def __post_init__(self) -> None:
        if not self.profile_id or not self.evidence:
            raise ValueError("qualified force active-AO profile needs evidence")
        if not self.compositions:
            raise ValueError(
                "qualified force active-AO profile needs an execution domain"
            )
        if type(self.producer) is not str or self.producer not in (
            "sampled-jets",
            "pre-ao-envelope-native-csr",
            "exact-jets-native-bitmask",
        ):
            raise ValueError("qualified force active-AO producer is unsupported")
        if type(self.max_active_fraction) not in (int, float) or not (
            0 < self.max_active_fraction <= 1
        ):
            raise ValueError("qualified AO occupancy limit must be in (0,1]")
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
        if (
            type(self.min_dense_point_ao_square_work) is not int
            or self.min_dense_point_ao_square_work <= 0
        ):
            raise ValueError("qualified dense-work crossover must be positive")
        if self.max_dense_point_ao_square_work is not None and (
            type(self.max_dense_point_ao_square_work) is not int
            or self.max_dense_point_ao_square_work < self.min_dense_point_ao_square_work
        ):
            raise ValueError("qualified dense-work ceiling must cover the crossover")
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
            workload.architecture.startswith("sm_")
            and workload.composition in self.compositions
            and workload.derivative_order in self.derivative_orders
            and workload.spin_blocks in self.spin_blocks
            and (
                self.density_fitted is None
                or workload.density_fitted is self.density_fitted
            )
            and workload.grid_points * workload.aos * workload.aos
            >= self.min_dense_point_ao_square_work
            and (
                self.max_dense_point_ao_square_work is None
                or workload.grid_points * workload.aos * workload.aos
                <= self.max_dense_point_ao_square_work
            )
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
    producer: str = "sampled-jets"
    max_active_fraction: float = 1.0

    @property
    def selected(self) -> bool:
        return self.cutoff is not None


# Preserve #2007's faster sampled route above its existing work crossover.
# #1893 extends previously dense work to the resident pre-AO producer below it;
# occupancy/resource misses remain dense, without product or shape whitelists.
_SAMPLED_DENSE_WORK_CROSSOVER = 173_946_175_488
QUALIFIED_FORCE_ACTIVE_AO_PROFILES: tuple[QualifiedForceActiveAoProfile, ...] = (
    QualifiedForceActiveAoProfile(
        profile_id="ordinary-direct-active-ao-cost-v3",
        evidence=(
            "benchmarks/results/pbe0-public-force-policy-20261005/README.md",
            "benchmarks/results/pbe0-force-followups-20261005/README.md",
        ),
        compositions=("ordinary",),
        derivative_orders=(1, 2),
        spin_blocks=(1, 2),
        density_fitted=False,
        min_dense_point_ao_square_work=_SAMPLED_DENSE_WORK_CROSSOVER,
        tile_policy="fixed",
        tile_points=256,
        min_device_bytes=512 << 20,
        min_host_bytes=256 << 20,
        cutoff=1e-16,
        cache_bytes=16 << 20,
    ),
    QualifiedForceActiveAoProfile(
        profile_id="cuda-resident-preao-native-csr-v1",
        evidence=(
            ".agents/notes/implemented/performance/2026-10-06-pre-ao-native-csr.md",
        ),
        compositions=("ordinary",),
        derivative_orders=(1, 2),
        spin_blocks=(1, 2),
        density_fitted=False,
        min_dense_point_ao_square_work=1,
        max_dense_point_ao_square_work=_SAMPLED_DENSE_WORK_CROSSOVER - 1,
        tile_policy="fixed",
        tile_points=256,
        min_device_bytes=512 << 20,
        min_host_bytes=256 << 20,
        cutoff=1e-16,
        cache_bytes=64 << 20,
        producer="pre-ao-envelope-native-csr",
        max_active_fraction=0.8,
    ),
    # DF changes the integral provider, not the moving-grid AO derivative.
    # Inspect every requested AO jet at the existing force cutoff; never reuse
    # the SCF's order-one masks for an order-two force. Discovery, rebinding and
    # optional storage are owned by the existing token-checked grid consumer.
    QualifiedForceActiveAoProfile(
        profile_id="fitted-exact-jet-bitmask-v1",
        evidence=(
            ".agents/notes/implemented/performance/2026-10-10-df-force-active-ao.md",
        ),
        compositions=("ordinary",),
        derivative_orders=(1, 2),
        spin_blocks=(1, 2),
        density_fitted=True,
        min_dense_point_ao_square_work=_SAMPLED_DENSE_WORK_CROSSOVER,
        tile_policy="fixed",
        tile_points=256,
        min_device_bytes=512 << 20,
        min_host_bytes=256 << 20,
        cutoff=1e-16,
        cache_bytes=64 << 20,
        producer="exact-jets-native-bitmask",
        max_active_fraction=0.8,
    ),
)


# Keep explicit 256-point callers and the same continuous producer crossover.
# Automatic tiles are admitted by the shared complete-owner byte-budget planner;
# do not let their policy label silently disable the existing indexed consumer.
QUALIFIED_FORCE_ACTIVE_AO_PROFILES += tuple(
    replace(
        profile,
        profile_id=f"{profile.profile_id}-budget-auto",
        evidence=(
            *profile.evidence,
            ".agents/notes/implemented/performance/2026-10-06-pbe0-budget-admitted-cuda-schedule.md",
        ),
        tile_policy="budget-auto",
        tile_points=None,
    )
    for profile in QUALIFIED_FORCE_ACTIVE_AO_PROFILES
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
        profile.producer,
        float(profile.max_active_fraction),
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
        "producer": decision.producer if decision.selected else None,
        "max_active_fraction": decision.max_active_fraction,
        "workload": decision.workload.record(),
        "actual_mode": actual,
        "selected_point_ao_square_sum": selected_work,
        "dense_point_ao_square_sum": dense_work,
    }
