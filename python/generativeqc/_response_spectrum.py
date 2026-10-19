"""One bounded full real-orbital spectrum qualification for spin response owners.

Callers own reference/physics/lifetime validation. This owner performs no Fock,
XC, SCF or derivative algebra: it checks the supplied physical Hessian action.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from generativeqc_compiler.common.arrays import immutable


@dataclass(frozen=True, eq=False)
class _RealSpectrum:
    status: str
    eigenvalues: np.ndarray
    direction: np.ndarray | None
    symmetry_defect: float
    residual_bound: float
    uncertainty: float


def _qualify_real_spectrum(
    action: Callable[[np.ndarray], np.ndarray],
    dimension: int,
    *,
    max_dimension: int,
    max_actions: int,
    budget_bytes: int,
    curvature_tolerance: float,
    symmetry_tolerance: float,
    residual_tolerance: float,
    reference_curvature_bound: float,
) -> _RealSpectrum:
    """Check every column and every eigenpair before classifying the full space.

    A partial Krylov spectrum cannot establish positive definiteness. Admission
    precedes buffers/actions; the complete declared work is 2*n applications.
    Accounted numeric scratch excludes borrowed reference/provider storage,
    whose lifetime and budget remain the caller's responsibility.
    """
    for name, value in (
        ("dimension", dimension),
        ("max_dimension", max_dimension),
        ("max_actions", max_actions),
        ("budget_bytes", budget_bytes),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for value in (curvature_tolerance, symmetry_tolerance, residual_tolerance):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("spectrum tolerances must be finite and positive")
    if not math.isfinite(reference_curvature_bound) or reference_curvature_bound < 0:
        raise ValueError("invalid reference curvature uncertainty")
    n = dimension
    if n > max_dimension or 2 * n > max_actions:
        raise ValueError("dimension/action budget rejected before execution")
    if 8 * (12 * n * n + 20 * n) > budget_bytes:
        raise ValueError("spectrum storage budget rejected before allocation")

    def checked(vector: np.ndarray) -> np.ndarray:
        value = np.asarray(action(vector))
        if (
            value.shape != (n,)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
        ):
            raise ValueError("invalid finite real FP64 physical Hessian action")
        # An action may reuse its scratch on the next call.
        return np.array(value, dtype=np.float64, copy=True)

    hessian = np.empty((n, n), dtype=np.float64)
    column = np.zeros(n, dtype=np.float64)
    for index in range(n):
        column[index] = 1.0
        hessian[:, index] = checked(column)
        column[index] = 0.0
    symmetry = float(np.linalg.norm(hessian - hessian.T))
    if not math.isfinite(symmetry) or symmetry > symmetry_tolerance:
        raise ValueError("non-self-adjoint physical Hessian")
    eigenvalues, vectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    residual_squared = 0.0
    for index in range(n):
        residual = checked(vectors[:, index]) - eigenvalues[index] * vectors[:, index]
        residual_squared += float(np.dot(residual, residual))
    residual_bound = math.sqrt(residual_squared)
    if not math.isfinite(residual_bound) or residual_bound > residual_tolerance:
        raise ValueError("fresh-action eigen residual failed")
    if not np.isfinite(eigenvalues).all() or not np.isfinite(vectors).all():
        raise ValueError("nonfinite eigensystem")
    uncertainty = residual_bound + symmetry / 2.0 + reference_curvature_bound
    threshold = curvature_tolerance + uncertainty
    status = (
        "EMPTY"
        if n == 0
        else "UNSTABLE"
        if eigenvalues[0] < -threshold
        else "STABLE"
        if eigenvalues[0] > threshold
        else "NEAR_SINGULAR"
    )
    direction = None
    if status == "UNSTABLE":
        direction = vectors[:, 0].copy()
        if direction[int(np.argmax(np.abs(direction)))] < 0:
            direction *= -1
        direction = immutable(direction)
    return _RealSpectrum(
        status, immutable(eigenvalues), direction, symmetry, residual_bound, uncertainty
    )
