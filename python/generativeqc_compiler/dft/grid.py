"""Versioned atom-centered quadrature and normalized Becke partitions.

The default policy uses pinned Treutler/Ahlrichs radial and Lebedev angular
rules, NWChem pruning and Treutler radius-adjusted Becke partitions. Explicit
versions 1--3 retain their rational-Legendre/tensor angular prescriptions.
No reference package or runtime download is needed.
"""

from __future__ import annotations

import hashlib
import json
import typing
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.provenance import canonical_hash


def checked_int(
    value: typing.Any,
    name: typing.Any,
    low: typing.Any = 1,
    high: typing.Any = 2**31 - 1,
) -> typing.Any:
    """Reject booleans, truncation and overflow at every public size boundary."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def owned_atoms(atoms: typing.Any) -> typing.Any:
    """Canonicalize all coordinate storage, including user-constructed Atoms."""
    # Reuse the public atom contract only when accepting molecular input.
    from generativeqc import Atom

    result = []
    for atom in atoms:
        a = Atom.from_value(atom)
        checked_int(a.atomic_number, "atomic number", high=118)
        xyz = immutable(a.position, shape=(3,))
        result.append(
            Atom(
                a.atomic_number,
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            )
        )
    if not result:
        raise ValueError("a grid requires atoms")
    return tuple(result)


def molecular_grid_identity(
    atoms: typing.Any,
    spec: GridSpec,
    *,
    charge: int = 0,
    multiplicity: int = 1,
) -> str:
    """Hash the exact molecular-grid scientific inputs without materializing quadrature."""

    atoms = owned_atoms(atoms)
    if not isinstance(spec, GridSpec):
        raise TypeError("expected GridSpec")
    checked_int(charge, "charge", low=-(2**31), high=2**31 - 1)
    checked_int(multiplicity, "multiplicity")
    radii = dict(spec.element_radii)
    if spec.version == 1:
        resolved = tuple(radii.get(a.atomic_number, 1.0) for a in atoms)
    else:
        missing = sorted(
            {a.atomic_number for a in atoms if a.atomic_number not in radii}
        )
        if missing:
            raise ValueError(
                "production GridSpec has no sourced radius for atomic number(s) "
                + ", ".join(map(str, missing))
            )
        resolved = tuple(radii[a.atomic_number] for a in atoms)
    return canonical_hash(
        {
            "atoms": [asdict(a) for a in atoms],
            "grid": asdict(spec),
            "resolved_radii_bohr": resolved,
            "charge": charge,
            "multiplicity": multiplicity,
            "charge_spin_policy": "independent-v1",
        }
    )


@dataclass(frozen=True)
class GridSpec:
    """Versioned, fully specified quadrature; all lengths are in Bohr.

    Version 1 is the historical reference contract: default radial radii are
    exactly one Bohr for every element. Version 2 is the resolved production
    contract: every element that is actually used must have an explicit radius.
    Overrides are explicit (atomic_number, radius) pairs, not an opaque accuracy level.
    Coincident centers within the declared distance use an equal pair split.
    Version 3 is an explicit-radius, opt-in ``radial-v1`` pruned contract.
    Its shell sizes depend only on the integer radial rank, never on
    geometry or density, so moving-grid derivatives retain fixed topology.
    Version 4 is Treutler/Lebedev with NWChem pruning and Treutler's Bragg-radius
    partition adjustment. Use ``GridSpec.treutler(level=3)`` to construct it.
    In that version the legacy wire field ``radial_points`` carries the grid
    level (0--9), and both angular fields are zero. Actual radial/angular counts
    depend on the element; they are never inferred from those wire fields.
    """

    radial_points: int = 48
    angular_polar: int = 16
    angular_azimuth: int = 32
    element_radii: tuple[tuple[int, float], ...] = ()
    partition_iterations: int = 3
    coincident_tolerance: float = 1e-12
    version: int = 1
    radial_rule: str = "rational-legendre"
    angular_rule: str = "legendre-trapezoid"
    partition: str = "becke-equal-radius"
    pruning: str = "none"
    units: str = "Bohr"
    ordering: str = "atom-radial-polar-azimuth"

    def __post_init__(self) -> None:
        """Validate the supported grid prescription and normalize unique element radii."""
        checked_int(self.version, "grid version", high=4)
        if self.version == 4:
            checked_int(self.radial_points, "Treutler grid level", low=0, high=9)
            if (self.angular_polar, self.angular_azimuth) != (0, 0) or (
                self.radial_rule,
                self.angular_rule,
                self.partition,
                self.pruning,
                self.units,
                self.ordering,
            ) != (
                "treutler-ahlrichs",
                "lebedev",
                "becke-treutler",
                "nwchem",
                "Bohr",
                "atom-radial-lebedev",
            ):
                raise ValueError("unsupported Treutler/Lebedev grid prescription")
            checked_int(self.partition_iterations, "partition iterations", high=5)
            if (
                not np.isfinite(self.coincident_tolerance)
                or self.coincident_tolerance < 0
            ):
                raise ValueError("invalid coincident-center tolerance")
            radii = tuple(sorted(self.element_radii))
            if any(
                type(z) is not int or not 1 <= z <= 103 or not np.isfinite(r) or r <= 0
                for z, r in radii
            ) or len({z for z, _ in radii}) != len(radii):
                raise ValueError("invalid or duplicate Treutler partition radius")
            object.__setattr__(self, "element_radii", radii)
            return
        checked_int(self.radial_points, "radial points", high=512)
        checked_int(self.angular_polar, "polar points", high=256)
        checked_int(self.angular_azimuth, "azimuth points", low=3, high=1024)
        checked_int(self.partition_iterations, "partition iterations", high=5)
        if (
            self.radial_rule,
            self.angular_rule,
            self.partition,
            self.units,
            self.ordering,
        ) != (
            "rational-legendre",
            "legendre-trapezoid",
            "becke-equal-radius",
            "Bohr",
            "atom-radial-polar-azimuth",
        ):
            raise ValueError("unsupported grid prescription/version")
        if self.pruning != ("radial-v1" if self.version == 3 else "none"):
            raise ValueError(
                "radial-v1 pruning requires grid version 3; v1/v2 are unpruned"
            )
        if self.version == 3 and (
            self.angular_polar > 128 or self.angular_azimuth > 512
        ):
            raise ValueError(
                "radial-v1 supports at most 128 polar and 512 azimuth points"
            )
        if not np.isfinite(self.coincident_tolerance) or self.coincident_tolerance < 0:
            raise ValueError("invalid coincident-center tolerance")
        radii = []
        for z, r in self.element_radii:
            checked_int(z, "radius atomic number", high=118)
            if not np.isfinite(r) or r <= 0:
                raise ValueError("element radii must be finite and positive")
            radii.append((z, float(r)))
        if len({z for z, _ in radii}) != len(radii):
            raise ValueError("duplicate element radius")
        object.__setattr__(self, "element_radii", tuple(sorted(radii)))

    @classmethod
    def treutler(cls, level: int = 3) -> GridSpec:
        """Construct the pinned GPU4PySCF/PySCF default quadrature family.

        Level 3 is the default for both XC and VV10. Changing a level changes
        quadrature accuracy, not SCF convergence or density screening.
        """
        from .treutler import tables, validate_element_level

        validate_element_level(1, level)
        data = tables()
        return cls(
            version=4,
            radial_points=level,
            angular_polar=0,
            angular_azimuth=0,
            radial_rule="treutler-ahlrichs",
            angular_rule="lebedev",
            partition="becke-treutler",
            pruning="nwchem",
            ordering="atom-radial-lebedev",
            element_radii=tuple(enumerate(data["bragg_radii_bohr"][1:104], start=1)),
        )

    @property
    def level(self) -> int | None:
        """Return the Treutler level, or None for explicit legacy tensor grids."""
        return self.radial_points if self.version == 4 else None

    def point_count(self, atomic_numbers: typing.Iterable[int]) -> int:
        """Count actual points without assuming equal-sized atomic quadratures."""
        if self.version == 4:
            from .treutler import atomic_point_count

            return sum(
                atomic_point_count(z, self.radial_points) for z in atomic_numbers
            )
        return sum(1 for _ in atomic_numbers) * self.points_per_atom

    def partition_radii(
        self, atoms: typing.Iterable[typing.Any]
    ) -> tuple[float, ...] | None:
        """Return per-center Bragg radii only for the adjusted partition family."""
        if self.version != 4:
            return None
        radii = dict(self.element_radii)
        return tuple(radii[a.atomic_number] for a in atoms)

    def atom_offsets(self, atoms: typing.Iterable[typing.Any]) -> tuple[int, ...]:
        """Return atom-major prefix offsets without materializing any points."""
        from itertools import accumulate

        return (0, *accumulate(self.point_count((a.atomic_number,)) for a in atoms))

    @property
    def angular_shells(self) -> tuple[tuple[int, int], ...]:
        """Return immutable (polar, azimuth) counts in increasing radial order."""
        if self.version == 4:
            raise ValueError("Lebedev shell orders depend on the element")
        return _angular_shells(
            self.version, self.radial_points, self.angular_polar, self.angular_azimuth
        )

    @property
    def points_per_atom(self) -> int:
        """Exact point count, including shell pruning but not density screening."""
        return sum(polar * azimuth for polar, azimuth in self.angular_shells)


@lru_cache(maxsize=128)
def _angular_shells(
    version: int, radial: int, polar: int, azimuth: int
) -> tuple[tuple[int, int], ...]:
    """Table-free radial-v1 policy, distinct from Lebedev/NWChem pruning.

    For u=(radial_index+1/2)/radial_points, retain full resolution on
    [1/3, 3/4], halve each angular order outside it, and quarter it for
    u<1/5 or u>6/7. Integer comparisons make point-count planning independent
    of quadrature materialization and floating-point root roundoff.
    Rounded-up orders keep small or odd prescriptions valid. Element radii
    set the physical shells without introducing geometry-dependent switches.
    """
    if version != 3:
        return ((polar, azimuth),) * radial
    result = []
    for index in range(radial):
        rank = 2 * index + 1
        divisor = (
            4
            if 5 * rank < 2 * radial or 7 * rank > 12 * radial
            else (2 if 3 * rank < 2 * radial or 4 * rank > 6 * radial else 1)
        )
        result.append(
            (
                (polar + divisor - 1) // divisor,
                max(3, (azimuth + divisor - 1) // divisor),
            )
        )
    return tuple(result)


# Canonical production radii. These are the pinned covalent_radius_bohr values
# extracted from xTBloom GFN1 data. Keep source identity beside the values.
GRID_POLICY_RADII_SOURCE = (
    "upstream/xtbloom/2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3/"
    "gfn1.json#elements[].covalent_radius_bohr@"
    "92b32fada844a337204b84f2d961473bad5737240765eb8d0727a62827de5111"
)
GRID_POLICY_UPSTREAM_REVISION = "2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3"
_PRODUCTION_RADII = (
    0.8062831465047213,
    1.1590320231005369,
    3.0235617993927044,
    2.368456742857618,
    1.9401188212769855,
    1.8897261246204402,
    1.7889407313073502,
    1.5873699446811698,
    1.6125662930094427,
    1.6881553379942602,
    3.5274887659581555,
    3.1495435410340673,
    2.8471873610947966,
    2.620420226140344,
    2.771598316109979,
    2.5700275294837986,
    2.494438484498981,
    2.4188494395141635,
    4.434557305775966,
    3.880237642553971,
    3.3511143276602473,
    3.0739544960492493,
    3.048758147720977,
    2.771598316109979,
    2.696009271125162,
    2.620420226140344,
    2.5196348328272538,
    2.494438484498981,
    2.5448311811555264,
    2.746401967781707,
    2.8219910127665244,
    2.746401967781707,
    2.8975800577513415,
    2.771598316109979,
    2.8723837094230693,
    2.9479727544078864,
    4.7621098340435095,
    4.2077901708215135,
    3.7038632042560633,
    3.5022924176298824,
    3.325917979331975,
    3.1243471927057946,
    2.8975800577513415,
    2.8471873610947966,
    2.8471873610947966,
    2.721205619453434,
    2.8975800577513415,
    3.0991508443775224,
    3.2251325860188853,
    3.1747398893623395,
    3.1747398893623395,
    3.0991508443775224,
    3.325917979331975,
    3.3007216310037024,
    5.26603680060896,
    4.434557305775966,
    4.0818084291801515,
    3.7038632042560633,
    3.9810230358670613,
    3.9558266875387886,
    3.9306303392105164,
    3.9054339908822433,
    3.8046485975691535,
    3.8298449458974257,
    3.8046485975691535,
    3.7794522492408804,
    3.754255900912608,
    3.754255900912608,
    3.7290595525843355,
    3.8550412942256984,
    3.6786668559277906,
    3.451899720973338,
    3.3007216310037024,
    3.0991508443775224,
    2.9731691027361595,
    2.922776406079614,
    2.796794664438252,
    2.8219910127665244,
    2.8471873610947966,
    3.325917979331975,
    3.27552528267543,
    3.27552528267543,
    3.4267033726450653,
    3.3007216310037024,
    3.4770960693016097,
    3.5778814626147004,
)


@dataclass(frozen=True)
class GridProfile:
    """Resolved production topology before element radii are attached."""

    name: str
    radial_points: int
    angular_polar: int
    angular_azimuth: int
    partition_iterations: int = 3
    pruning: str = "none"
    screening: str = "none"
    topology: str = "atom-radial-polar-azimuth"


@dataclass(frozen=True)
class GridPolicy:
    """Resolve Treutler defaults and explicit legacy LDA/PBE policies."""

    accuracy: str = "standard"
    pruning: str = "nwchem"

    def profile(self, method: str, *, derivative_order: int = 0) -> GridProfile:
        """Select a qualified LDA or GGA/PBE profile, using tight grids for derivatives."""
        if self.accuracy not in ("standard", "tight"):
            raise ValueError("grid accuracy must be 'standard' or 'tight'")
        if self.pruning not in ("none", "radial-v1", "nwchem"):
            raise ValueError("grid pruning must be 'none', 'radial-v1' or 'nwchem'")
        if self.pruning == "nwchem":
            if derivative_order not in (0, 1):
                raise NotImplementedError(
                    "production grid derivatives support orders 0 and 1"
                )
            level = 4 if self.accuracy == "tight" else 3
            return GridProfile(
                f"treutler-level{level}-v4",
                level,
                0,
                0,
                pruning="nwchem",
                topology="atom-radial-lebedev",
            )
        if derivative_order not in (0, 1):
            raise NotImplementedError(
                "production grid derivatives support orders 0 and 1"
            )
        name = str(method).lower().replace("_", "-")
        if name in ("lda", "lda-rks", "lda-uks"):
            family = "lda"
        elif name in ("gga", "pbe", "pbe-rks", "pbe-uks"):
            family = "gga"
        else:
            raise NotImplementedError(
                "production grid policy is qualified only for LDA and GGA/PBE"
            )
        tight = self.accuracy == "tight" or derivative_order == 1
        if family == "lda":
            shape = (64, 20, 40) if tight else (54, 16, 32)
        else:
            # Keep the qualified standard angular workload at the legacy
            # 16x32 topology while using the 54-point radial rule that meets
            # the retained independent energy/force convergence gate.
            shape = (72, 24, 48) if tight else (54, 16, 32)
        if not tight and self.pruning == "radial-v1":
            shape = (64, 20, 40) if family == "lda" else (64, 16, 32)
        version = 2 if self.pruning == "none" else 3
        return GridProfile(
            f"{family}-{'tight' if tight else 'standard'}-v{version}",
            *shape,
            pruning=self.pruning,
        )

    @property
    def provenance(self) -> dict[str, typing.Any]:
        """Return policy version, radii source, topology, and partition provenance."""
        if self.pruning == "nwchem":
            return grid_policy_provenance(self.resolve("pbe"))
        return {
            "policy_version": 2 if self.pruning == "none" else 3,
            "radii_source": GRID_POLICY_RADII_SOURCE,
            "upstream_revision": GRID_POLICY_UPSTREAM_REVISION,
            "supported_atomic_numbers": (1, len(_PRODUCTION_RADII)),
            "pruning": self.pruning,
            "screening": "none",
            "partition": "becke-equal-radius",
            "topology": "atom-radial-polar-azimuth",
            "derivative_topology": "fixed",
        }

    def resolve(self, method: str, *, derivative_order: int = 0) -> GridSpec:
        """Build the selected versioned quadrature with pinned element radii."""
        if self.pruning == "nwchem":
            self.profile(method, derivative_order=derivative_order)
            return GridSpec.treutler(4 if self.accuracy == "tight" else 3)
        profile = self.profile(method, derivative_order=derivative_order)
        return GridSpec(
            version=2 if self.pruning == "none" else 3,
            radial_points=profile.radial_points,
            angular_polar=profile.angular_polar,
            angular_azimuth=profile.angular_azimuth,
            partition_iterations=profile.partition_iterations,
            pruning=profile.pruning,
            element_radii=tuple(enumerate(_PRODUCTION_RADII, start=1)),
        )


def grid_policy_provenance(spec: GridSpec) -> dict[str, typing.Any]:
    """Describe the exact origin of a resolved grid without over-claiming policy provenance."""
    if spec.version == 4:
        from .treutler import tables

        return {
            "policy_version": 4,
            "contract": "treutler-lebedev-nwchem/v1",
            "level": spec.level,
            "upstream": tables()["upstream"],
            "source_sha256": dict(tables()["source_sha256"]),
            "partition": spec.partition,
            "pruning": spec.pruning,
            "screening": "none",
            "derivative_topology": "fixed",
        }
    if spec.version == 1:
        return {"policy_version": 1, "contract": "reference-grid-v1"}
    if spec.version not in (2, 3):
        raise ValueError("unsupported grid policy version")

    # Version identifies the representation, not the source of user data.
    # Only exact prescribed resolver outputs may claim the pinned policy source.
    for accuracy in ("standard", "tight"):
        policy = GridPolicy(accuracy, pruning=spec.pruning)
        if any(spec == policy.resolve(method) for method in ("lda", "pbe")):
            return policy.provenance
    return {
        "policy_version": spec.version,
        "contract": f"explicit-grid-v{spec.version}",
        **({"pruning": spec.pruning} if spec.version == 3 else {}),
    }


def partition_weights(
    points: typing.Any,
    centers: typing.Any,
    *,
    iterations: typing.Any = 3,
    coincident_tolerance: typing.Any = 1e-12,
    atomic_radii: typing.Any = None,
) -> typing.Any:
    """Return normalized ownership [point,atom] with stable log products.

    Becke's p(x)=(3x-x^3)/2 is composed ``iterations`` times. Identical centers
    share pair ownership equally, so duplicate atom grids do not double the
    molecular measure. Spatial, center and physical-atom derivatives of these
    weights are not supplied by this value-only partition routine.
    """
    checked_int(iterations, "partition iterations", high=5)
    if not np.isfinite(coincident_tolerance) or coincident_tolerance < 0:
        raise ValueError("invalid coincident-center tolerance")
    points, centers = immutable(points), immutable(centers)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or centers.ndim != 2
        or centers.shape[1] != 3
        or not len(centers)
    ):
        raise ValueError("points/centers require shape (n,3) and at least one center")
    if atomic_radii is not None:
        atomic_radii = immutable(atomic_radii, shape=(len(centers),))
        if np.any(atomic_radii <= 0):
            raise ValueError("partition radii must be positive")
        from .treutler import radii_adjustment
    distance = np.empty((len(points), len(centers)))
    # Avoid a point × atom × xyz temporary; this workspace is charged by plans.
    for a, center in enumerate(centers):
        delta = points - center
        distance[:, a] = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
    logs = np.zeros_like(distance)
    # Duplicate centers take the exact equal-split branch. Its zero input is
    # never modified in place, so one lazily allocated vector serves all pairs.
    coincident_zero: np.ndarray | None = None
    with np.errstate(divide="ignore"):
        for a in range(len(centers)):
            for b in range(a):
                delta = centers[a] - centers[b]
                separation = float(np.hypot(np.hypot(delta[0], delta[1]), delta[2]))
                if separation <= coincident_tolerance:
                    if coincident_zero is None:
                        coincident_zero = np.zeros(len(points))
                    mu = coincident_zero
                else:
                    mu = np.clip((distance[:, a] - distance[:, b]) / separation, -1, 1)
                    if atomic_radii is not None:
                        adjustment = radii_adjustment(atomic_radii[a], atomic_radii[b])
                        mu = mu + adjustment * (1 - mu * mu)
                for _ in range(iterations):
                    mu = 0.5 * mu * (3 - mu * mu)
                pair = np.clip(0.5 * (1 - mu), 0, 1)
                logs[:, a] += np.log(pair)
                logs[:, b] += np.log1p(-pair)
    logs -= np.max(logs, axis=1, keepdims=True)
    np.exp(logs, out=logs)
    logs /= np.sum(logs, axis=1, keepdims=True)
    return immutable(logs)


@dataclass(frozen=True, eq=False)
class GridTile:
    """Owned point/weight tile and global point offset; weights include r² dr dΩ."""

    begin: int
    points: np.ndarray
    weights: np.ndarray
    owners: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class MolecularGrid:
    """Prepared radial/angular topology; complete molecular points are streamed.

    Geometry, atom order, charge/spin policy and every grid parameter enter the
    identity. Construction retains no molecular grid-by-AO tensor.
    """

    if TYPE_CHECKING:
        centers: ClassVar[np.ndarray]
        directions: ClassVar[np.ndarray]
        angular_weights: ClassVar[np.ndarray]
        radii: ClassVar[np.ndarray]
        radial_weights: ClassVar[np.ndarray]
        shell_offsets: ClassVar[np.ndarray]
        shell_direction_offsets: ClassVar[np.ndarray]
        resolved_radii: ClassVar[tuple[float, ...]]
        npoint: ClassVar[int]
        identity: ClassVar[str]
        numeric_bytes: ClassVar[int]
        setup_scratch_bytes: ClassVar[int]

    atoms: tuple
    spec: GridSpec = field(default_factory=GridSpec.treutler)
    charge: int = 0
    multiplicity: int = 1

    def __post_init__(self) -> None:
        atoms = owned_atoms(self.atoms)
        if not isinstance(self.spec, GridSpec):
            raise TypeError("expected GridSpec")
        checked_int(self.charge, "charge", low=-(2**31), high=2**31 - 1)
        checked_int(self.multiplicity, "multiplicity")
        object.__setattr__(self, "atoms", atoms)
        centers = immutable([a.position for a in atoms])
        radii = dict(self.spec.element_radii)
        if self.spec.version == 1:
            resolved = tuple(radii.get(a.atomic_number, 1.0) for a in atoms)
        else:
            missing = sorted(
                {a.atomic_number for a in atoms if a.atomic_number not in radii}
            )
            if missing:
                raise ValueError(
                    "production GridSpec has no sourced radius for atomic number(s) "
                    + ", ".join(map(str, missing))
                )
            resolved = tuple(radii[a.atomic_number] for a in atoms)
        if self.spec.version == 4:
            from .treutler import atomic_rule

            rules = {
                a.atomic_number: atomic_rule(a.atomic_number, self.spec.radial_points)
                for a in atoms
            }
            offsets = np.cumsum([0, *(len(rules[a.atomic_number]) for a in atoms)])
            for name, value in (
                ("centers", immutable(centers)),
                ("resolved_radii", resolved),
                ("atomic_rules", MappingProxyType(rules)),
                ("atom_offsets", np.frombuffer(offsets.tobytes(), dtype=offsets.dtype)),
                ("npoint", int(offsets[-1])),
                (
                    "identity",
                    molecular_grid_identity(
                        atoms,
                        self.spec,
                        charge=self.charge,
                        multiplicity=self.multiplicity,
                    ),
                ),
                (
                    "numeric_bytes",
                    centers.nbytes
                    + offsets.nbytes
                    + 8 * len(atoms)
                    + sum(rule.nbytes for rule in rules.values()),
                ),
                (
                    "setup_scratch_bytes",
                    3 * max(rule.nbytes for rule in rules.values()),
                ),
            ):
                object.__setattr__(self, name, value)
            return
        z, wz = np.polynomial.legendre.leggauss(self.spec.angular_polar)
        phi = np.arange(self.spec.angular_azimuth) * (
            2 * np.pi / self.spec.angular_azimuth
        )
        zz, pp = np.meshgrid(z, phi, indexing="ij")
        rr = np.sqrt(1 - zz * zz)
        angular = np.stack((rr * np.cos(pp), rr * np.sin(pp), zz), axis=-1).reshape(
            -1, 3
        )
        angular_weights = np.repeat(
            wz * (2 * np.pi / self.spec.angular_azimuth), len(phi)
        )
        shell_bytes = 0
        if self.spec.pruning != "none":
            directions, measures = [], []
            rules = {}
            offsets, direction_offsets = [0], []
            direction_count = 0
            for polar, azimuth in self.spec.angular_shells:
                shape = (polar, azimuth)
                if shape not in rules:
                    nodes, weights = np.polynomial.legendre.leggauss(polar)
                    angles = np.arange(azimuth) * (2 * np.pi / azimuth)
                    heights, angles = np.meshgrid(nodes, angles, indexing="ij")
                    rings = np.sqrt(1 - heights * heights)
                    directions.append(
                        np.stack(
                            (rings * np.cos(angles), rings * np.sin(angles), heights),
                            axis=-1,
                        ).reshape(-1, 3)
                    )
                    measures.append(np.repeat(weights * (2 * np.pi / azimuth), azimuth))
                    rules[shape] = direction_count
                    direction_count += polar * azimuth
                direction_offsets.append(rules[shape])
                offsets.append(offsets[-1] + polar * azimuth)
            angular, angular_weights = (
                np.concatenate(directions),
                np.concatenate(measures),
            )
            for name, values in (
                ("shell_offsets", offsets),
                ("shell_direction_offsets", direction_offsets),
            ):
                storage = np.frombuffer(
                    np.asarray(values, dtype=np.int64).tobytes(), dtype=np.int64
                )
                object.__setattr__(self, name, storage)
                shell_bytes += storage.nbytes
        t, wt = np.polynomial.legendre.leggauss(self.spec.radial_points)
        t, wt = (t + 1) * 0.5, wt * 0.5
        r = t / (1 - t)
        radial_weights = wt * r * r / (1 - t) ** 2
        for name, value in (
            ("centers", centers),
            ("directions", angular),
            ("angular_weights", angular_weights),
            ("radii", r),
            ("radial_weights", radial_weights),
        ):
            object.__setattr__(self, name, immutable(value))
        object.__setattr__(self, "resolved_radii", resolved)
        object.__setattr__(self, "npoint", len(atoms) * self.spec.points_per_atom)
        object.__setattr__(
            self,
            "identity",
            molecular_grid_identity(
                atoms,
                self.spec,
                charge=self.charge,
                multiplicity=self.multiplicity,
            ),
        )
        object.__setattr__(
            self,
            "numeric_bytes",
            sum(
                getattr(self, n).nbytes
                for n in (
                    "centers",
                    "directions",
                    "angular_weights",
                    "radii",
                    "radial_weights",
                )
            )
            + 40 * len(atoms)
            + shell_bytes,
        )
        # NumPy's Legendre construction uses dense companion eigensolves.
        # Their numeric temporaries are setup-only and independent of grid
        # point count; conservatively charge them in prepared capacity bounds.
        object.__setattr__(
            self,
            "setup_scratch_bytes",
            64 * max(self.spec.radial_points, self.spec.angular_polar) ** 2
            + 128 * len(angular),
        )

    def _raw_tiles(self, tile_points: typing.Any = 256) -> typing.Any:
        """Shared atomic points/measures before molecular partitioning."""
        checked_int(tile_points, "tile points")
        if self.spec.version == 4:
            for owner, atom in enumerate(self.atoms):
                rule = self.atomic_rules[atom.atomic_number]
                for start in range(0, len(rule), tile_points):
                    block = rule[start : start + tile_points]
                    yield GridTile(
                        int(self.atom_offsets[owner]) + start,
                        immutable(block[:, :3] + self.centers[owner]),
                        immutable(block[:, 3]),
                        (owner,) * len(block),
                    )
            return
        na = len(self.directions)
        per_atom = self.spec.points_per_atom
        for begin in range(0, self.npoint, tile_points):
            indices = np.arange(begin, min(begin + tile_points, self.npoint))
            owner, local = np.divmod(indices, per_atom)
            if self.spec.pruning == "none":
                radial, angular = np.divmod(local, na)
            else:
                radial = np.searchsorted(self.shell_offsets, local, side="right") - 1
                angular = (
                    local
                    - self.shell_offsets[radial]
                    + self.shell_direction_offsets[radial]
                )
            scale = np.asarray(self.resolved_radii)[owner]
            points = (
                self.centers[owner]
                + self.directions[angular] * (self.radii[radial] * scale)[:, None]
            )
            raw = self.radial_weights[radial] * scale**3 * self.angular_weights[angular]
            yield GridTile(
                begin,
                immutable(points),
                immutable(raw),
                tuple(int(x) for x in owner),
            )

    def tiles(self, tile_points: typing.Any = 256) -> typing.Any:
        """Build moved points and fresh partition weights, including final tiles."""
        for raw in self._raw_tiles(tile_points):
            partition = partition_weights(
                raw.points,
                self.centers,
                iterations=self.spec.partition_iterations,
                coincident_tolerance=self.spec.coincident_tolerance,
                atomic_radii=self.resolved_radii if self.spec.version == 4 else None,
            )
            weights = raw.weights * partition[np.arange(len(raw.owners)), raw.owners]
            yield GridTile(raw.begin, raw.points, immutable(weights), raw.owners)

    def explicit(self, *, max_points: typing.Any = 200_000) -> typing.Any:
        """Materialize a guarded small reference grid for independent exporters."""
        checked_int(max_points, "explicit grid limit")
        if self.npoint > max_points:
            raise ValueError("explicit grid exceeds the small-grid limit; use tiles")
        points, weights, owners = [], [], []
        for tile in self.tiles():
            points.append(tile.points)
            weights.append(tile.weights)
            owners.extend(tile.owners)
        return ExplicitGrid(
            np.concatenate(points),
            np.concatenate(weights),
            tuple(owners),
            {
                "grid_spec": asdict(self.spec),
                "grid_identity": self.identity,
                "atoms": [asdict(a) for a in self.atoms],
                "units": "Bohr",
            },
        )


@dataclass(frozen=True, eq=False)
class ExplicitGrid:
    """Portable exact quadrature with versioned content identity.

    Version 1 retains the historical canonical-JSON hash. Version 2 hashes
    immutable typed buffers in bounded chunks; JSON is only materialized for
    explicit interchange. Both versions preserve signed zeros and normalize
    numerical inputs to finite FP64, independent of their original layout or
    endianness. Neither identity makes a grid-convergence claim.
    """

    points: np.ndarray
    weights: np.ndarray
    owners: tuple[int, ...]
    provenance: dict[str, object]
    identity_version: int = field(default=1, kw_only=True)

    if TYPE_CHECKING:
        _provenance_json: ClassVar[str]
        identity: ClassVar[str]

    def __post_init__(self) -> None:
        checked_int(self.identity_version, "explicit grid identity version", high=2)
        points, weights = immutable(self.points), immutable(self.weights)
        owners = tuple(self.owners)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or weights.shape != (len(points),)
            or len(owners) != len(points)
        ):
            raise ValueError("invalid explicit grid shapes")
        for owner in owners:
            checked_int(owner, "point owner", low=0)
        provenance = json.dumps(self.provenance, sort_keys=True, allow_nan=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "owners", owners)
        # Preserve identity even if the caller later mutates its metadata dict.
        object.__setattr__(self, "_provenance_json", provenance)
        identity = (
            canonical_hash(self.record())
            if self.identity_version == 1
            else self._typed_identity()
        )
        object.__setattr__(self, "identity", identity)

    def _typed_identity(self) -> str:
        """Hash owned FP64/i32 contents with at most 1 MiB conversion scratch.

        The fixed metadata digest domains the hash by schema, units, shapes,
        representation and frozen provenance. Payloads follow in point, weight,
        owner order. Little-endian encoding makes identity host-independent;
        byte hashing deliberately distinguishes positive and negative zero.
        """
        metadata = {
            "schema": "generativeqc.explicit-grid",
            "version": 2,
            "points": {"shape": self.points.shape, "dtype": "<f8", "unit": "bohr"},
            "weights": {
                "shape": self.weights.shape,
                "dtype": "<f8",
                "unit": "bohr^3",
            },
            "owners": {"shape": (len(self.owners),), "dtype": "<i4"},
            "order": "C",
            "provenance": json.loads(self._provenance_json),
        }
        digest = hashlib.sha256(canonical_hash(metadata).encode("ascii"))
        chunk_values = (1 << 20) // 8
        for array in (self.points, self.weights):
            flat = array.reshape(-1)
            for start in range(0, len(flat), chunk_values):
                chunk = flat[start : start + chunk_values].astype("<f8", copy=False)
                digest.update(memoryview(chunk).cast("B"))
                del chunk
        for start in range(0, len(self.owners), chunk_values):
            chunk = np.asarray(self.owners[start : start + chunk_values], dtype="<i4")
            digest.update(memoryview(chunk).cast("B"))
            del chunk
        return digest.hexdigest()

    def record(self) -> typing.Any:
        """Materialize the versioned JSON interchange record only on request."""
        return {
            "schema": "generativeqc.explicit-grid",
            "version": self.identity_version,
            "points_bohr": self.points.tolist(),
            "weights_bohr3": self.weights.tolist(),
            "owners": self.owners,
            "provenance": json.loads(self._provenance_json),
        }

    def write(self, path: typing.Any) -> None:
        """Write explicit data and a content hash; no external program is needed."""
        Path(path).write_text(
            json.dumps({**self.record(), "sha256": self.identity}, indent=2) + "\n"
        )

    @classmethod
    def read(cls, path: typing.Any) -> typing.Any:
        """Import only versioned, hash-verified Bohr point/weight arrays."""
        data = json.loads(Path(path).read_text())
        if data["schema"] != "generativeqc.explicit-grid" or data["version"] not in (
            1,
            2,
        ):
            raise ValueError("unsupported explicit-grid schema")
        result = cls(
            data["points_bohr"],
            data["weights_bohr3"],
            tuple(data["owners"]),
            data["provenance"],
            identity_version=data["version"],
        )
        if result.identity != data["sha256"]:
            raise ValueError("explicit grid hash mismatch")
        return result
