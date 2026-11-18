"""Explicit subspace equilibration for isolated independent Newton references.

PySCF's augmented-Hessian overlap contains an affine unit vector alongside
unnormalized orbital trial vectors. An absolute overlap cutoff can discard
small but independent trials as a physical SCF approaches convergence. A
diagonal congruence changes their units, not the augmented eigenproblem or any
physical Hessian action. This helper is never a production SCF implementation.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np


def equilibrated_augmented_inputs(
    hamiltonian: object, overlap: object, vectors: object
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Return a unit-diagonal congruence with the affine coordinate unchanged.

    Only the small projected problem and its reconstruction basis are scaled.
    Original physical response vectors/actions are untouched. Rank decisions
    therefore operate on a dimensionless overlap rather than vector amplitudes.
    """
    if any(np.iscomplexobj(value) for value in (hamiltonian, overlap, vectors)):
        raise ValueError("independent Newton equilibration requires real inputs")
    matrix = np.asarray(hamiltonian, dtype=float)
    metric = np.asarray(overlap, dtype=float)
    trials = np.asarray(vectors, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not 2 <= len(matrix) <= 129
        or metric.shape != matrix.shape
        or trials.ndim != 2
        or trials.shape[0] != len(matrix) - 1
        or not trials.shape[1]
    ):
        raise ValueError("independent Newton augmented dimensions are invalid")
    if not all(np.all(np.isfinite(value)) for value in (matrix, metric, trials)):
        raise ValueError("independent Newton augmented inputs must be finite")
    diagonal = np.diag(metric)
    if np.any(diagonal <= 0) or abs(metric[0, 0] - 1) > 1e-12:
        raise ValueError(
            "independent Newton requires a positive metric and unit affine coordinate"
        )
    if np.max(abs(metric[0, 1:])) > 1e-12 or np.max(abs(metric[1:, 0])) > 1e-12:
        raise ValueError("independent Newton affine coordinate must be orthogonal")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            scales = 1 / np.sqrt(diagonal)
            transformed = (matrix * scales[:, None]) * scales[None, :]
            normalized = (metric * scales[:, None]) * scales[None, :]
            basis = trials * scales[1:, None]
        except FloatingPointError as failure:
            raise ValueError(
                "independent Newton equilibration exceeds finite precision"
            ) from failure
    if not all(
        np.all(np.isfinite(value)) for value in (transformed, normalized, basis)
    ):
        raise ValueError("independent Newton equilibration must remain finite")
    if np.max(abs(transformed - transformed.T)) > 1e-12 * max(
        1.0, float(np.max(abs(transformed)))
    ):
        raise ValueError("independent Newton augmented Hamiltonian must be symmetric")
    if np.max(abs(normalized - normalized.T)) > 1e-12:
        raise ValueError("independent Newton augmented metric must be symmetric")
    if np.linalg.eigvalsh(normalized).min() < -1e-12 * len(normalized):
        raise ValueError(
            "independent Newton augmented metric must be positive semidefinite"
        )
    return transformed, normalized, list(basis), scales


@contextmanager
def equilibrated_newton_reference() -> Iterator[list[dict[str, float]]]:
    """Temporarily equilibrate PySCF's small AH solve in an isolated process.

    Callers must serialize reference SCFs; this scoped vendor binding is not
    suitable for a shared production process. The original binding is restored
    after success or exception. A zero inherited initial trial restarts from the
    unchanged current residual, rather than constructing a null projected metric.
    Neither physical Focks/Hessian actions nor SCF convergence flags, tolerances,
    occupations or orbital-step acceptance are replaced.
    """
    from pyscf.soscf import ciah

    original = ciah._regular_step
    original_davidson = ciah.davidson_cc
    records = []
    restarts = 0

    def davidson(
        hessian_action: Any,
        gradient: Any,
        preconditioner: Any,
        initial: object,
        *arguments: Any,
        **keywords: Any,
    ) -> Iterator:
        nonlocal restarts
        trial = np.asarray(initial)
        if not np.all(np.isfinite(trial)):
            raise ArithmeticError("independent Newton initial trial must be finite")
        if not np.any(trial):
            residual = np.asarray(gradient())
            if (
                residual.shape != trial.shape
                or not np.all(np.isfinite(residual))
                or not np.any(residual)
            ):
                raise ArithmeticError(
                    "independent Newton zero trial has no nonzero physical residual"
                )
            trial = residual.copy()
            restarts += 1
        yield from original_davidson(
            hessian_action, gradient, preconditioner, trial, *arguments, **keywords
        )

    def step(
        hamiltonian: object,
        overlap: object,
        vectors: object,
        relative_cutoff: float,
        logger: Any,
    ) -> tuple:
        transformed, normalized, basis, scales = equilibrated_augmented_inputs(
            hamiltonian, overlap, vectors
        )
        displacement, eigenvalue, coefficients, selected, spectrum = original(
            transformed, normalized, basis, relative_cutoff, logger
        )
        mapped = np.asarray(coefficients) * scales
        if not np.all(np.isfinite(mapped)) or not np.all(np.isfinite(displacement)):
            raise ArithmeticError("independent Newton mapped step must be finite")
        records.append(
            {
                "smallest_original_diagonal": float(np.diag(overlap).min()),
                "smallest_relative_overlap_eigenvalue": float(np.min(spectrum)),
                "physical_step_norm": float(np.linalg.norm(displacement)),
                "zero_initial_trial_restarts": restarts,
            }
        )
        return displacement, eigenvalue, mapped, selected, spectrum

    ciah._regular_step = step
    ciah.davidson_cc = davidson
    try:
        yield records
    finally:
        ciah._regular_step = original
        ciah.davidson_cc = original_davidson
