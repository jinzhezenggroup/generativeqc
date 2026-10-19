"""Explicit bounded real UHF/UKS root qualification and detached provenance.

These contracts describe one completed native CPU call, not complex/spin-flip
stability or a proof of the globally lowest electronic state. A geometry-step
receipt must be requalified before any new energy or force is published.
"""

from __future__ import annotations

import math
import typing
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from itertools import pairwise

import numpy as np
from generativeqc_compiler.common.arrays import immutable

from . import _native
from ._api_types import Atom, Result
from .profiles import canonical_hash


class SpinRootUnqualified(RuntimeError):
    """Requested electronic-root qualification failed; publish no partial result."""


@dataclass(frozen=True)
class SpinRootPolicy:
    """Finite CPU real-internal stability/restart and intended-root gates.

    Curvatures and energies are Hartree, angles radians, and projector distances
    dimensionless Hilbert-space Frobenius norms. Numeric bytes cover this
    controller's references/response/rotation scratch; native SCF arenas use the
    calculator's separate resource contract. No CUDA or default promotion.
    """

    recover: bool = True
    maximum_ao: int = 32
    maximum_dimension: int = 128
    maximum_actions: int = 4096
    memory_budget_bytes: int = 256 << 20
    angles: tuple[float, ...] = (0.15, 0.35, 0.65)
    curvature_tolerance: float = 1e-6
    symmetry_tolerance: float = 1e-9
    eigen_residual_tolerance: float = 1e-9
    stationarity_tolerance: float = 1e-8
    energy_tolerance: float = 2e-7
    downhill_tolerance: float = 1e-9
    root_tolerance: float = 1e-5
    tracking_tolerance: float = 0.25
    allow_root_switch: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous booleans, nonfinite gates and unbounded restarts."""
        for name in ("recover", "allow_root_switch"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        for name in (
            "maximum_ao",
            "maximum_dimension",
            "maximum_actions",
            "memory_budget_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        angles = tuple(self.angles)
        if (
            not angles
            or len(angles) > 6
            or any(
                type(x) not in (int, float)
                or not math.isfinite(x)
                or not 0 < x < math.pi / 2
                for x in angles
            )
            or any(a >= b for a, b in pairwise(angles))
        ):
            raise ValueError(
                "angles require 1-6 increasing finite rotations in (0, pi/2)"
            )
        object.__setattr__(self, "angles", angles)
        for name in (
            "curvature_tolerance",
            "symmetry_tolerance",
            "eigen_residual_tolerance",
            "stationarity_tolerance",
            "energy_tolerance",
            "downhill_tolerance",
            "root_tolerance",
            "tracking_tolerance",
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.stationarity_tolerance > 1e-7:
            raise ValueError("stationarity_tolerance cannot exceed 1e-7")

    @property
    def identity(self) -> str:
        """Deterministic identity of requested numerical/resource/root policy."""
        return canonical_hash(asdict(self))


@dataclass(frozen=True, eq=False)
class SpinRootReference:
    """Detached qualified two-spin canonical determinant and physical model.

    Occupations are 1/0, matrices use the public AO basis, coordinates are Bohr,
    and energy/orbital energies are Hartree. Density/orbital buffers are backed
    by immutable bytes. This is reference evidence, not an executable provider.
    """

    atoms: tuple[Atom, ...]
    overlap: np.ndarray
    density: np.ndarray
    coefficients: np.ndarray
    orbital_energies: np.ndarray
    occupations: np.ndarray
    fock: np.ndarray
    energy: float
    residual: float
    model_identity: str
    source_identity: str
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        """Own the fixed determinant; reject malformed/noncanonical snapshots."""
        atoms = tuple(Atom.from_value(atom) for atom in self.atoms)
        if not atoms or not all(
            isinstance(x, str) and x
            for x in (self.model_identity, self.source_identity)
        ):
            raise ValueError("reference requires atoms and model/source identities")
        object.__setattr__(self, "atoms", atoms)
        raw_overlap = np.asarray(self.overlap)
        if raw_overlap.ndim != 2 or raw_overlap.shape[0] != raw_overlap.shape[1]:
            raise ValueError("reference overlap must be square")
        n = raw_overlap.shape[0]
        shapes = {
            "overlap": (n, n),
            "density": (2, n, n),
            "coefficients": (2, n, n),
            "orbital_energies": (2, n),
            "occupations": (2, n),
            "fock": (2, n, n),
        }
        if n < 1:
            raise ValueError("reference AO domain must be nonempty")
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if (
                value.shape != shape
                or value.dtype != np.float64
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"invalid real FP64 {name}")
            object.__setattr__(self, name, immutable(value))
        if (
            type(self.energy) not in (int, float)
            or type(self.residual) not in (int, float)
            or not math.isfinite(self.energy)
            or not math.isfinite(self.residual)
            or self.residual < 0
        ):
            raise ValueError("invalid reference energy/residual")
        # Final acceptance uses the requested tighter stationarity gate as well.
        tolerance = 1e-7
        if np.max(np.abs(self.overlap - self.overlap.T)) > tolerance:
            raise ValueError("reference overlap is not symmetric")
        np.linalg.cholesky(self.overlap)
        for spin in range(2):
            c, f, d, occ = (
                self.coefficients[spin],
                self.fock[spin],
                self.density[spin],
                self.occupations[spin],
            )
            if not np.array_equal(occ, (occ > 0.5).astype(np.float64)):
                raise ValueError("spin occupations must be 1/0")
            count = int(occ.sum())
            if not np.array_equal(occ, np.r_[np.ones(count), np.zeros(n - count)]):
                raise ValueError("occupied columns must precede virtual columns")
            if (
                np.max(np.abs(c.T @ self.overlap @ c - np.eye(n))) > tolerance
                or np.max(np.abs((c * occ) @ c.T - d)) > tolerance
                or np.max(np.abs(f - f.T)) > tolerance
                or np.max(
                    np.abs(f @ c - self.overlap @ c * self.orbital_energies[spin])
                )
                > tolerance
            ):
                raise ValueError("reference is not a canonical stationary projector")
        object.__setattr__(
            self,
            "identity",
            canonical_hash(
                {
                    "schema": "generativeqc.spin-root-reference/v1",
                    "atoms": [asdict(a) for a in atoms],
                    "model": self.model_identity,
                    "source": self.source_identity,
                    "energy": self.energy,
                    "residual": self.residual,
                    "arrays": {
                        name: sha256(getattr(self, name).tobytes()).hexdigest()
                        for name in shapes
                    },
                }
            ),
        )

    @property
    def occupied(self) -> tuple[int, int]:
        """Ordered alpha and beta electron counts."""
        return int(self.occupations[0].sum()), int(self.occupations[1].sum())

    @property
    def numeric_bytes(self) -> int:
        """Owned numeric snapshot storage, excluding Python metadata."""
        return sum(
            getattr(self, name).nbytes
            for name in (
                "overlap",
                "density",
                "coefficients",
                "orbital_energies",
                "occupations",
                "fock",
            )
        )


@dataclass(frozen=True, eq=False)
class SpinRootResult:
    """Completed native energy/force plus independently checked root evidence.

    ``reference`` may seed a subsequent explicitly qualified call. Reusing this
    receipt does not qualify a changed geometry, model, provider or force.
    """

    result: Result
    reference: SpinRootReference
    initial_status: str
    recovered: bool
    eigenvalues: np.ndarray
    spin_squared: float
    alpha_populations: np.ndarray
    beta_populations: np.ndarray
    root_relation: str
    identity: str
    _diagnostics: typing.Mapping[str, typing.Any] = field(repr=False)

    @property
    def energy(self) -> float:
        """Qualified native total energy in Hartree."""
        return self.result.energy

    @property
    def forces(self) -> np.ndarray | None:
        """Qualified native analytic forces in Hartree/Bohr, or None."""
        return self.result.forces

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        """Detached numerical/work/root provenance from this completed call."""
        return deepcopy(dict(self._diagnostics))


def _has_uhf_reference(method: int) -> bool:
    return method in (
        _native.METHOD_UHF,
        _native.METHOD_UMP2,
    )
