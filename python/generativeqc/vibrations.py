"""Host analysis of explicitly supplied Cartesian energy Hessians.

This consumer owns units, mass weighting and rigid-motion projection, not the
electronic Hessian or a transition-state workflow. An array receipt cannot
verify that the supplied matrix belongs to a particular method or geometry.
"""

from __future__ import annotations

import hashlib
import math
import re
import typing
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np

__all__ = ["HarmonicAnalysis", "analyze_harmonic_modes"]

# Pinned 2022 CODATA SI values; conversion is angular frequency / (2*pi*c).
_HARTREE_JOULE = 4.3597447222060e-18
_BOHR_METRE = 5.29177210544e-11
_DALTON_KILOGRAM = 1.66053906892e-27
_LIGHT_METRE_SECOND = 299792458.0
_FREQUENCY_CM1 = math.sqrt(_HARTREE_JOULE / (_DALTON_KILOGRAM * _BOHR_METRE**2)) / (
    2 * math.pi * _LIGHT_METRE_SECOND * 100
)
_BOHR_ANGSTROM = _BOHR_METRE * 1e10


def _array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(array, dtype=np.float64, copy=True)


def _freeze(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("harmonic analysis produced nonfinite values")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real number")
    return float(value)


@dataclass(frozen=True)
class HarmonicAnalysis:
    """Unscaled supplied-Hessian modes, not a verified electronic/TS result.

    Arrays are detached, irreversibly read-only, and mode-major. ``frequencies_cm1``
    uses negative real values for imaginary frequencies, including soft negative
    modes. ``mass_weighted_modes`` have Euclidean unit norm; ``cartesian_modes``
    have unit mass norm with the supplied dalton masses. Degenerate eigenvectors
    are meaningful as subspaces, not individually reproducible directions.
    ``negative_mode_count`` includes every negative eigenvalue; the significant
    count separately excludes the declared soft band without deleting modes.
    ``input_identity`` identifies arrays/conventions, not scientific provenance.
    """

    frequencies_cm1: np.ndarray
    eigenvalues_hartree_per_bohr2_dalton: np.ndarray
    mass_weighted_modes: np.ndarray
    cartesian_modes: np.ndarray
    coordinates_bohr: np.ndarray
    masses_dalton: np.ndarray
    linear: bool
    rigid_mode_count: int
    negative_mode_count: int
    significant_imaginary_mode_count: int
    soft_mode_threshold_cm1: float
    input_identity: str
    diagnostics: typing.Mapping[str, object]

    def xyz_animation(
        self,
        symbols: typing.Sequence[str],
        mode_index: int,
        *,
        frames: int = 16,
        amplitude_angstrom: float = 0.2,
    ) -> str:
        """Export a mode as multi-frame XYZ with Angstrom coordinates.

        ``symbols`` follow the input atom order. ``mode_index`` is zero-based.
        ``amplitude_angstrom`` is the maximum per-atom excursion at unit phase,
        only a visualization normalization; result modes are not rescaled.
        ``frames`` includes one full sinusoidal cycle, without a repeated end.
        """
        count = len(self.masses_dalton)
        if len(symbols) != count or any(
            not isinstance(symbol, str) or re.fullmatch(r"[A-Z][a-z]?", symbol) is None
            for symbol in symbols
        ):
            raise ValueError("symbols must be atom-ordered element labels")
        if type(mode_index) is not int or not 0 <= mode_index < len(
            self.frequencies_cm1
        ):
            raise ValueError("mode_index is outside the vibrational mode array")
        if type(frames) is not int or not 4 <= frames <= 1000:
            raise ValueError("frames must be an integer between 4 and 1000")
        amplitude = _positive(amplitude_angstrom, "amplitude_angstrom")
        mode = self.cartesian_modes[mode_index]
        display = mode * (amplitude / np.max(np.linalg.norm(mode, axis=1)))
        output = []
        for frame in range(frames):
            phase = math.sin(2 * math.pi * frame / frames)
            coordinates = self.coordinates_bohr * _BOHR_ANGSTROM + phase * display
            output.extend(
                (
                    str(count),
                    (
                        f"supplied-Hessian mode={mode_index} frequency_cm-1="
                        f"{self.frequencies_cm1[mode_index]:.12g} phase={phase:.12g}"
                    ),
                )
            )
            output.extend(
                f"{symbol} {center[0]:.12g} {center[1]:.12g} {center[2]:.12g}"
                for symbol, center in zip(symbols, coordinates, strict=True)
            )
        return "\n".join(output) + "\n"


def analyze_harmonic_modes(
    hessian: object,
    *,
    coordinates: object,
    masses: object,
    hessian_units: str,
    coordinate_units: str,
    mass_units: str,
    symmetrize: bool = False,
    raw_symmetry_tolerance: float = 1e-7,
    linear_relative_tolerance: float = 1e-8,
    soft_mode_threshold_cm1: float = 1.0,
    maximum_coordinates: int = 600,
) -> HarmonicAnalysis:
    """Mass-weight and rigid-project a supplied energy Hessian on the host.

    ``hessian`` must be a complete real ``(3N, 3N)`` matrix in atom-major x/y/z
    ordering, not a force Jacobian. ``coordinates`` are matching ``(N, 3)`` values
    and ``masses`` are positive physical atomic/isotopic masses of length N,
    including for ECP atoms. No electron/core-count mass substitution is made.
    Explicit unit strings must be ``hartree/bohr^2``, ``bohr``, and ``dalton``;
    other units are rejected rather than guessed or silently converted.

    Raw asymmetry/rigid residuals are reported before projection. Asymmetry
    above ``raw_symmetry_tolerance`` always fails; any nonzero asymmetry also
    requires explicit ``symmetrize=True`` to average the admitted matrix.
    Linear detection uses the smallest/largest singular value of mass-weighted
    rotational vectors relative to ``linear_relative_tolerance``. A monatomic
    input has zero vibrations; distinct diatomics are linear. Coincident-only
    multiatom geometries fail. Normal counts are 3N-5 (linear), 3N-6 (nonlinear).

    ``soft_mode_threshold_cm1`` only labels a near-zero band; all signed
    frequencies are retained, unscaled. ``maximum_coordinates`` bounds dense
    analysis before matrix conversion and is not a native allocator/RSS budget.
    Constants use the pinned 2022 CODATA SI values. This function performs no
    SCF, optimization, Hessian generation, stationary-state verification, D3
    composition or TS classification. Geometry/method consistency is the
    caller's responsibility, even when rigid residuals are small.
    """
    if (hessian_units, coordinate_units, mass_units) != (
        "hartree/bohr^2",
        "bohr",
        "dalton",
    ):
        raise ValueError("required units: hartree/bohr^2, bohr, dalton")
    if type(symmetrize) is not bool:
        raise TypeError("symmetrize must be a bool")
    symmetry_gate = _positive(raw_symmetry_tolerance, "raw_symmetry_tolerance")
    linear_gate = _positive(linear_relative_tolerance, "linear_relative_tolerance")
    soft_gate = _positive(soft_mode_threshold_cm1, "soft_mode_threshold_cm1")
    if linear_gate >= 1:
        raise ValueError("linear_relative_tolerance must be below one")
    if type(maximum_coordinates) is not int or maximum_coordinates < 3:
        raise ValueError("maximum_coordinates must be an integer of at least three")
    positions = _array(coordinates, "coordinates")
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError("coordinates must be a nonempty (N, 3) array")
    count = positions.size
    if count > maximum_coordinates:
        raise ValueError("dense harmonic analysis exceeds maximum_coordinates")
    atomic_masses = _array(masses, "masses")
    if atomic_masses.shape != (len(positions),) or np.any(atomic_masses <= 0):
        raise ValueError("masses must have one positive physical mass per atom")
    matrix = _array(hessian, "hessian")
    if matrix.shape != (count, count):
        raise ValueError("hessian must be the complete atom-major (3N, 3N) matrix")
    asymmetry = float(np.max(abs(matrix - matrix.T)))
    if asymmetry > symmetry_gate:
        raise ValueError("raw Hessian exceeds raw_symmetry_tolerance")
    if asymmetry and not symmetrize:
        raise ValueError("nonzero raw asymmetry requires explicit symmetrize=True")
    weights = atomic_masses / np.max(atomic_masses)
    if np.any(weights == 0):
        raise ValueError("mass ratios exceed finite analysis precision")
    center = positions[0] + np.average(
        positions - positions[0], axis=0, weights=weights
    )
    centered = positions - center
    translations = np.tile(np.eye(3), (len(positions), 1))
    rotations = np.stack(
        [np.cross(axis, centered).reshape(-1) for axis in np.eye(3)], axis=1
    )
    root_mass = np.repeat(np.sqrt(atomic_masses), 3)
    relative_root = np.repeat(np.sqrt(weights), 3)
    translation_basis = relative_root[:, None] * translations
    translation_basis /= np.linalg.norm(translation_basis, axis=0)
    weighted_rotations = relative_root[:, None] * rotations
    weighted_rotations -= translation_basis @ (translation_basis.T @ weighted_rotations)
    rotation_vectors, rotation_singular, _ = np.linalg.svd(
        weighted_rotations, full_matrices=False
    )
    ratio = 0.0
    rotation_count = 0
    if len(positions) > 1:
        if not rotation_singular[0] > 0:
            raise ValueError("coincident-only geometry has undefined rotational rank")
        ratio = float(rotation_singular[-1] / rotation_singular[0])
        rotation_count = int(
            np.count_nonzero(rotation_singular > linear_gate * rotation_singular[0])
        )
        if rotation_count not in (2, 3):
            raise ValueError("geometry has an unresolved rotational rank")
    rigid = np.column_stack((translation_basis, rotation_vectors[:, :rotation_count]))
    complement = np.linalg.qr(rigid, mode="complete")[0][:, rigid.shape[1] :]
    admitted = (0.5 * matrix + 0.5 * matrix.T) if symmetrize else matrix
    weighted = admitted / root_mass[:, None] / root_mass[None, :]
    if not np.isfinite(weighted).all():
        raise ValueError("mass weighting exceeds finite analysis precision")
    reduced = complement.T @ weighted @ complement
    eigenvalues, vectors = np.linalg.eigh(reduced)
    modes = complement @ vectors
    for column in range(modes.shape[1]):
        if modes[np.argmax(abs(modes[:, column])), column] < 0:
            modes[:, column] *= -1
    frequencies = np.sign(eigenvalues) * np.sqrt(abs(eigenvalues)) * _FREQUENCY_CM1
    cartesian = modes / root_mass[:, None]
    digest = hashlib.sha256(b"generativeqc.supplied-harmonic-analysis/v1\0")
    for array in (matrix, positions, atomic_masses):
        digest.update(np.asarray(array, dtype="<f8", order="C").tobytes())
    digest.update(
        repr(
            (symmetrize, symmetry_gate, linear_gate, soft_gate, maximum_coordinates)
        ).encode()
    )
    diagnostics = MappingProxyType(
        {
            "analysis_kind": "supplied-hessian-only",
            "scientific_state_verified": False,
            "stationary_point_verified": False,
            "raw_asymmetry": asymmetry,
            "raw_translation_residual": float(np.max(abs(matrix @ translations))),
            "raw_rotation_residual": float(np.max(abs(matrix @ rotations))),
            "symmetrized": symmetrize,
            "projected_rigid_modes": rigid.shape[1],
            "rotation_singular_ratio": ratio,
            "rotation_singular_values": tuple(
                float(value) for value in rotation_singular
            ),
            "linear_relative_tolerance": linear_gate,
            "mass_weighted_rigid_coupling_norm": float(
                np.linalg.norm(rigid.T @ weighted @ complement)
            ),
            "frequency_scale_factor": 1.0,
            "constants": "2022-CODATA-SI",
            "hessian_units": hessian_units,
            "coordinate_units": coordinate_units,
            "mass_units": mass_units,
            "ordering": "atom-major-xyz",
        }
    )
    return HarmonicAnalysis(
        _freeze(frequencies),
        _freeze(eigenvalues),
        _freeze(modes.T.reshape(-1, len(positions), 3)),
        _freeze(cartesian.T.reshape(-1, len(positions), 3)),
        _freeze(positions),
        _freeze(atomic_masses),
        len(positions) > 1 and rotation_count == 2,
        rigid.shape[1],
        int(np.count_nonzero(eigenvalues < 0)),
        int(np.count_nonzero(frequencies < -soft_gate)),
        soft_gate,
        digest.hexdigest(),
        diagnostics,
    )
