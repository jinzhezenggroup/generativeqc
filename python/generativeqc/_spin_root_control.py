"""Shared bounded real spin-root certification/restart control, without SCF science."""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from scipy.linalg import expm

from ._response_spectrum import _qualify_real_spectrum, _RealSpectrum
from .spin_roots import SpinRootPolicy, SpinRootReference, SpinRootUnqualified


@dataclass
class _Work:
    maximum: int
    response_actions: int = 0
    fixed_density_evaluations: int = 0
    scf_restarts: int = 0

    def reserve(self, count: int) -> None:
        if (
            self.response_actions + self.fixed_density_evaluations + count
            > self.maximum
        ):
            raise SpinRootUnqualified("complete root action budget exhausted")


def _rotated_density(
    reference: SpinRootReference, direction: np.ndarray, angle: float
) -> np.ndarray:
    """Use C exp(-angle K) in the canonical alpha-then-beta OV coordinates."""
    counts = reference.occupied
    n = reference.overlap.shape[0]
    sizes = [count * (n - count) for count in counts]
    if direction.shape != (sum(sizes),) or not np.isfinite(direction).all():
        raise SpinRootUnqualified("invalid complete spin rotation")
    densities = []
    for c, count, block in zip(
        reference.coefficients, counts, np.split(direction, (sizes[0],)), strict=True
    ):
        rotations = block.reshape(count, n - count)
        generator = np.zeros((n, n))
        generator[:count, count:] = rotations
        generator[count:, :count] = -rotations.T
        occupied = (c @ expm(-angle * generator))[:, :count]
        densities.append(occupied @ occupied.T)
    return immutable(np.asarray(densities))


def _spin_diagnostics(
    reference: SpinRootReference,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Collinear determinant spin and basis-labelled Mulliken populations."""
    da, db = reference.density
    na, nb = reference.occupied
    sz = (na - nb) / 2
    squared = float(
        sz * (sz + 1) + nb - np.trace(da @ reference.overlap @ db @ reference.overlap)
    )
    if not math.isfinite(squared) or squared < -1e-10:
        raise SpinRootUnqualified("invalid spin expectation")
    populations = [np.diag(d @ reference.overlap) for d in (da, db)]
    if not all(np.isfinite(x).all() for x in populations):
        raise SpinRootUnqualified("nonfinite spin populations")
    return max(0.0, squared), immutable(populations[0]), immutable(populations[1])


def _root_distance(
    left: SpinRootReference,
    right: SpinRootReference,
    cross: np.ndarray,
    *,
    flip: bool = False,
) -> float:
    """Occupied-projector Hilbert distance using <left AO|right AO>.

    No AO coefficient norm or raw AO-density norm is a root metric. The cross
    overlap gives the same definition for nonsingular AO basis changes and
    changed geometry. A global spin flip is legal only for equal spin counts.
    """
    counts = left.occupied
    if counts != right.occupied or (flip and counts[0] != counts[1]):
        raise SpinRootUnqualified("incompatible root spin counts")
    if (
        cross.shape != (left.overlap.shape[0], right.overlap.shape[0])
        or not np.isfinite(cross).all()
    ):
        raise SpinRootUnqualified("invalid cross-geometry overlap")
    distances = []
    for spin, count in enumerate(counts):
        other = 1 - spin if flip else spin
        overlap = (
            left.coefficients[spin, :, :count].T
            @ cross
            @ right.coefficients[other, :, :count]
        )
        squared = float(2 * count - 2 * np.sum(overlap * overlap))
        if squared < -1e-7:
            raise SpinRootUnqualified("cross overlap violates projector norm")
        distances.append(math.sqrt(max(0.0, squared)))
    return max(distances)


def _evaluate(model: typing.Any, density: np.ndarray, work: _Work) -> typing.Any:
    work.reserve(1)
    model.validate_current()
    work.fixed_density_evaluations += 1
    result = model.evaluate(density)
    model.validate_current()
    if (
        not math.isfinite(result.energy)
        or result.fock.shape != density.shape
        or result.fock.dtype != np.float64
        or not np.isfinite(result.fock).all()
    ):
        raise SpinRootUnqualified("invalid nonlinear energy/Fock output")
    return result


def _certify(model: typing.Any, policy: SpinRootPolicy, work: _Work) -> _RealSpectrum:
    """Rebuild the physical nonlinear Fock, then qualify H=2*A, never Fxc[D]."""
    reference = model.reference
    n = model.operator.dimension
    if reference.overlap.shape[0] > policy.maximum_ao or n > policy.maximum_dimension:
        raise SpinRootUnqualified("complete spin domain exceeds policy")
    workspace = (
        8 * (12 * n * n + 20 * n) + reference.numeric_bytes + model.workspace_bytes
    )
    if workspace > policy.memory_budget_bytes:
        raise SpinRootUnqualified(
            "spin response numeric budget rejected before execution"
        )
    work.reserve(2 * n + 1)
    actual = _evaluate(model, reference.density, work)
    if abs(actual.energy - reference.energy) > policy.energy_tolerance:
        raise SpinRootUnqualified(
            "SCF energy does not match the physical nonlinear model"
        )
    defect = max(
        reference.residual, float(np.max(np.abs(actual.fock - reference.fock)))
    )
    uncertainty = 0.0
    for spin in range(2):
        c = reference.coefficients[spin]
        energies = reference.orbital_energies[spin]
        defect = max(
            defect,
            float(
                np.max(np.abs(actual.fock[spin] @ c - reference.overlap @ c * energies))
            ),
        )
        uncertainty = max(
            uncertainty,
            4 * float(np.linalg.norm(c.T @ actual.fock[spin] @ c - np.diag(energies))),
        )
    if not math.isfinite(defect) or defect > policy.stationarity_tolerance:
        raise SpinRootUnqualified("reference is not stationary under the actual model")

    def action(vector: np.ndarray) -> np.ndarray:
        work.reserve(1)
        model.validate_current()
        work.response_actions += 1
        result = np.asarray(model.operator.apply(vector))
        model.validate_current()
        if (
            result.shape != (n,)
            or result.dtype != np.float64
            or not np.isfinite(result).all()
        ):
            raise SpinRootUnqualified("invalid real spin response action")
        return 2.0 * result

    try:
        spectrum = _qualify_real_spectrum(
            action,
            n,
            max_dimension=policy.maximum_dimension,
            max_actions=policy.maximum_actions
            - work.response_actions
            - work.fixed_density_evaluations,
            budget_bytes=policy.memory_budget_bytes
            - reference.numeric_bytes
            - model.workspace_bytes,
            curvature_tolerance=policy.curvature_tolerance,
            symmetry_tolerance=policy.symmetry_tolerance,
            residual_tolerance=policy.eigen_residual_tolerance,
            reference_curvature_bound=uncertainty,
        )
    except ValueError as error:
        raise SpinRootUnqualified(str(error)) from error
    model.validate_current()
    return spectrum


def _recover_spin_root(
    initial: typing.Any,
    solve: typing.Callable,
    policy: SpinRootPolicy,
    *,
    preferred_distance: typing.Callable | None = None,
    work: _Work | None = None,
) -> tuple:
    """One shared restart controller; callbacks own their existing native SCF."""
    work = _Work(policy.maximum_actions) if work is None else work
    reference = initial.reference
    max_models = 2 * len(policy.angles) + 1 if policy.recover else 1
    reserved = max_models * (reference.numeric_bytes + initial.workspace_bytes) + 8 * (
        36 * reference.overlap.shape[0] ** 2
        + 12 * initial.operator.dimension**2
        + 20 * initial.operator.dimension
    )
    if (
        reference.overlap.shape[0] > policy.maximum_ao
        or reserved > policy.memory_budget_bytes
    ):
        raise SpinRootUnqualified(
            "complete recovery numeric budget rejected before execution"
        )
    first = _certify(initial, policy, work)
    if first.status in ("STABLE", "EMPTY"):
        return initial, first, first, work, 1, 0
    if not policy.recover or first.status != "UNSTABLE" or first.direction is None:
        raise SpinRootUnqualified(f"cannot publish {first.status} real-internal root")
    roots = []
    equivalents = 0
    for angle in policy.angles:
        for sign in (1.0, -1.0):
            initial.validate_current()
            for model, _ in roots:
                model.validate_current()
            seed = _rotated_density(reference, first.direction, sign * angle)
            trial = _evaluate(initial, seed, work)
            if trial.energy >= reference.energy - policy.downhill_tolerance:
                continue
            work.scf_restarts += 1
            try:
                candidate = solve(seed, trial.energy)
            except SpinRootUnqualified:
                raise
            except RuntimeError:
                # Nonconvergence of one finite restart does not admit a partial
                # root or excuse invalidating an already retained provider.
                initial.validate_current()
                for model, _ in roots:
                    model.validate_current()
                continue
            current = candidate.reference
            candidate.validate_current()
            if (
                current.model_identity != reference.model_identity
                or current.atoms != reference.atoms
                or current.occupied != reference.occupied
                or not np.array_equal(current.overlap, reference.overlap)
            ):
                raise SpinRootUnqualified("restart changed geometry, model, AO or spin")
            if candidate.workspace_bytes > initial.workspace_bytes:
                raise SpinRootUnqualified("restart exceeds reserved response workspace")
            if current.energy >= reference.energy - policy.downhill_tolerance:
                continue
            spectrum = _certify(candidate, policy, work)
            if spectrum.status != "STABLE":
                continue
            duplicate = None
            for index, (old, _) in enumerate(roots):
                same = _root_distance(current, old.reference, reference.overlap)
                if reference.occupied[0] == reference.occupied[1]:
                    same = min(
                        same,
                        _root_distance(
                            current, old.reference, reference.overlap, flip=True
                        ),
                    )
                if same < policy.root_tolerance:
                    if (
                        abs(current.energy - old.reference.energy)
                        > policy.energy_tolerance
                    ):
                        raise SpinRootUnqualified("equivalent roots disagree in energy")
                    duplicate = index
                    break
            if duplicate is None:
                roots.append((candidate, spectrum))
            else:
                equivalents += 1
                if current.energy < roots[duplicate][0].reference.energy:
                    roots[duplicate] = (candidate, spectrum)
    initial.validate_current()
    for model, _ in roots:
        model.validate_current()
    if not roots:
        raise SpinRootUnqualified(
            "no lower, converged and fully stable root after bounded native restarts"
        )
    eligible = roots
    if preferred_distance is not None and not policy.allow_root_switch:
        eligible = [
            item
            for item in roots
            if preferred_distance(item[0].reference)[0] <= policy.tracking_tolerance
        ]
        if not eligible:
            raise SpinRootUnqualified("recovery cannot retain the intended root")
    winner, spectrum = min(
        eligible,
        key=lambda item: (
            (preferred_distance(item[0].reference)[0], item[0].reference.energy)
            if preferred_distance is not None
            else (item[0].reference.energy,)
        ),
    )
    winner.validate_current()
    return winner, first, spectrum, work, len(roots), equivalents
