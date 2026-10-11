"""Bounded, full-domain real internal UHF diagnostics (internal CPU tool only).

No recovery, root selection, force publication, UKS, complex or spin-flip claim
is made. The shared UHF action is the only owner of Hessian/J/K algebra.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from generativeqc.profiles import canonical_hash

from tools.generativeqc_posthf.reference import immutable

from .uhf import UHFResponseOperator


class StabilityUnqualified(RuntimeError):
    """No certificate or recovery direction may be consumed after failure."""


@dataclass(frozen=True)
class StabilityOptions:
    """Admission limits and absolute physical-Hessian tolerances (hartree).

    Bytes bound accounted host numeric storage/workspace, including reference
    and provider declarations, not Python overhead or vendor LAPACK RSS.
    Work counts reference Fock evaluations plus shared response applications.
    """

    max_dimension: int = 128
    max_evaluations: int = 257
    budget_bytes: int = 64 << 20
    curvature_tolerance: float = 1e-6
    symmetry_tolerance: float = 1e-9
    residual_tolerance: float = 1e-9
    reference_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("max_dimension", "max_evaluations", "budget_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "curvature_tolerance",
            "symmetry_tolerance",
            "residual_tolerance",
            "reference_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.reference_tolerance > 1e-7:
            raise ValueError("reference_tolerance cannot exceed 1e-7")


def _source_identity(operator: UHFResponseOperator) -> str:
    """Bind Python owners; native/provider provenance remains in backend.identity."""
    owners = (UHFResponseOperator, type(operator.backend), StabilityOptions)
    files = [inspect.getsourcefile(owner) for owner in owners]
    if any(path is None for path in files):
        raise StabilityUnqualified("unavailable diagnostic/operator/provider source")
    return canonical_hash(
        [sha256(Path(path).read_bytes()).hexdigest() for path in files]
    )


def _binding(operator: UHFResponseOperator) -> tuple[str, ...]:
    operator.validate_current()
    problem = operator.problem
    return (
        problem.identity,
        problem.reference.identity,
        problem.reference.hamiltonian_id,
        operator.identity,
        operator.backend.identity,
        problem.layout.identity,
        _source_identity(operator),
    )


@dataclass(frozen=True, eq=False)
class UHFStabilityCertificate:
    """Immutable complete-real-OV evidence bound to one source/reference/action.

    STABLE means strictly positive curvature beyond the threshold and measured
    error bound in this domain only. EMPTY has no active rotations. UNSTABLE
    supplies a unit Euclidean packed direction for C exp(-t K(direction)).
    """

    status: str
    binding: tuple[str, ...]
    options: StabilityOptions
    dimension: int
    eigenvalues: np.ndarray
    direction: np.ndarray | None
    symmetry_defect: float
    residual_bound: float
    uncertainty: float
    reference_defect: float
    reference_curvature_bound: float
    response_actions: int
    reference_evaluations: int
    accounted_bytes: int
    elapsed_seconds: float
    identity: str

    def assert_current(self, operator: UHFResponseOperator) -> None:
        """Require this check immediately before consuming a recovery direction."""
        try:
            if _binding(operator) != self.binding:
                raise StabilityUnqualified("stale stability certificate binding")
        except StabilityUnqualified:
            raise
        except Exception as error:
            raise StabilityUnqualified("stale or failed provider") from error


def diagnose_uhf_stability(
    operator: UHFResponseOperator,
    *,
    options: StabilityOptions | None = None,
) -> UHFStabilityCertificate:
    """Qualify a tiny full real OV domain with 2*n actions and one Fock check.

    H=2*A for the shared density-symmetric-OV response Jacobian A. Matrix
    columns check self-adjointness; every eigenpair is checked by a fresh action.
    A Frobenius residual bound plus symmetrization defect guards classification.
    Admission precedes source validation, operator calls and diagnostic arrays.
    Any provider, identity, numerical or budget failure raises without a result.
    """
    started = time.perf_counter()
    if type(operator) is not UHFResponseOperator:
        raise StabilityUnqualified("requires the shared canonical real UHF operator")
    options = options or StabilityOptions()
    if not isinstance(options, StabilityOptions):
        raise TypeError("expected StabilityOptions")
    reference = operator.problem.reference
    n = operator.problem.dimension
    if operator.dimension != n or reference.algorithm != "UHF":
        raise StabilityUnqualified("incompatible UHF domain")
    provider_bytes = getattr(operator.backend, "host_workspace_bytes", None)
    if type(provider_bytes) is not int or provider_bytes < 0:
        raise StabilityUnqualified("provider must declare host workspace bytes")
    # Conservative numeric allowance for H, eigenvectors, eigensolver scratch,
    # fresh-action residuals and symmetrization; plus the shared AO action peak.
    accounted = (
        reference.numeric_bytes
        + provider_bytes
        + 8 * (12 * n * n + 20 * n + 16 * reference.nbf * reference.nbf)
    )
    if n > options.max_dimension or 2 * n + 1 > options.max_evaluations:
        raise StabilityUnqualified(
            "dimension/evaluation budget rejected before execution"
        )
    if accounted > options.budget_bytes:
        raise StabilityUnqualified("numeric storage budget rejected before allocation")
    try:
        binding = _binding(operator)

        canonical_defect = reference.scf_residual
        for spin in ("alpha", "beta"):
            c = getattr(reference, f"coefficients_{spin}")
            f = getattr(reference, f"fock_{spin}")
            energies = getattr(reference, f"orbital_energies_{spin}")
            canonical_defect = max(
                canonical_defect,
                float(np.max(np.abs(f @ c - reference.overlap @ c * energies))),
            )
        if (
            not math.isfinite(canonical_defect)
            or canonical_defect > options.reference_tolerance
        ):
            raise StabilityUnqualified(
                "reference stationarity exceeds diagnostic tolerance"
            )

        def check_current() -> None:
            if _binding(operator) != binding:
                raise StabilityUnqualified(
                    "provider/reference/operator changed during diagnostic"
                )

        def action(vector: np.ndarray) -> np.ndarray:
            check_current()
            raw = np.asarray(operator.apply(vector))
            if (
                raw.shape != (n,)
                or raw.dtype != np.float64
                or not np.isfinite(raw).all()
            ):
                raise StabilityUnqualified("invalid finite real FP64 operator output")
            check_current()
            result = 2.0 * raw
            if not np.isfinite(result).all():
                raise StabilityUnqualified("nonfinite physical Hessian action")
            return result

        # Snapshot canonicality alone does not establish that its Focks belong
        # to the actual Hamiltonian/provider. Reuse the shared spin Fock seam.
        densities = [
            getattr(reference, f"coefficients_{spin}")[:, : reference.nocc(spin)]
            for spin in ("alpha", "beta")
        ]
        focks = operator._fock_response(*(c @ c.T for c in densities))
        defect = canonical_defect
        reference_curvature_bound = 0.0
        for spin, fock in zip(("alpha", "beta"), focks, strict=True):
            raw = np.asarray(fock)
            if (
                raw.shape != reference.hcore.shape
                or np.iscomplexobj(raw)
                or not np.isfinite(raw).all()
            ):
                raise StabilityUnqualified("invalid reference Fock output")
            defect = max(
                defect,
                float(
                    np.max(
                        np.abs(
                            reference.hcore + raw - getattr(reference, f"fock_{spin}")
                        )
                    )
                ),
            )
            c = getattr(reference, f"coefficients_{spin}")
            energies = getattr(reference, f"orbital_energies_{spin}")
            # Replacing actual F_oo/F_vv by canonical diagonal energies changes
            # the physical action by 2*(delta_F_vv*x - x*delta_F_oo).
            # 4*||delta_F_MO||_F bounds that error for either spin block, even
            # for ill-conditioned AO overlap or very tight curvature thresholds.
            reference_curvature_bound = max(
                reference_curvature_bound,
                4
                * float(
                    np.linalg.norm(
                        c.T @ (reference.hcore + raw) @ c - np.diag(energies)
                    )
                ),
            )
        check_current()
        if defect > options.reference_tolerance:
            raise StabilityUnqualified(
                "reference Focks do not match operator Hamiltonian"
            )
        if not math.isfinite(reference_curvature_bound):
            raise StabilityUnqualified("nonfinite reference curvature bound")
        hessian = np.empty((n, n), dtype=np.float64)
        column = np.zeros(n, dtype=np.float64)
        for index in range(n):
            column[index] = 1.0
            hessian[:, index] = action(column)
            column[index] = 0.0
        symmetry = float(np.linalg.norm(hessian - hessian.T))
        if not math.isfinite(symmetry) or symmetry > options.symmetry_tolerance:
            raise StabilityUnqualified("non-self-adjoint physical Hessian")
        eigenvalues, vectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
        residual_squared = 0.0
        for index in range(n):
            residual = (
                action(vectors[:, index]) - eigenvalues[index] * vectors[:, index]
            )
            residual_squared += float(np.dot(residual, residual))
        residual_bound = math.sqrt(residual_squared)
        if (
            not math.isfinite(residual_bound)
            or residual_bound > options.residual_tolerance
        ):
            raise StabilityUnqualified("fresh-action eigen residual failed")
        if not np.isfinite(eigenvalues).all() or not np.isfinite(vectors).all():
            raise StabilityUnqualified("nonfinite eigensystem")
        uncertainty = residual_bound + symmetry / 2.0 + reference_curvature_bound
        threshold = options.curvature_tolerance + uncertainty
        status = (
            "EMPTY"
            if n == 0
            else (
                "UNSTABLE"
                if eigenvalues[0] < -threshold
                else "STABLE"
                if eigenvalues[0] > threshold
                else "NEAR_SINGULAR"
            )
        )
        direction = None
        if status == "UNSTABLE":
            direction = vectors[:, 0].copy()
            pivot = int(np.argmax(np.abs(direction)))
            if direction[pivot] < 0:
                direction *= -1
            direction = immutable(direction)
        eigenvalues = immutable(eigenvalues)
        check_current()
        identity = canonical_hash(
            {
                "kind": "full-real-internal-uhf-stability-v1",
                "binding": binding,
                "options": vars(options),
                "status": status,
                "eigenvalues": eigenvalues.tolist(),
                "direction": None if direction is None else direction.tolist(),
                "symmetry": symmetry,
                "residual": residual_bound,
                "reference": defect,
                "reference_curvature_bound": reference_curvature_bound,
            }
        )
        return UHFStabilityCertificate(
            status,
            binding,
            options,
            n,
            eigenvalues,
            direction,
            symmetry,
            residual_bound,
            uncertainty,
            defect,
            reference_curvature_bound,
            2 * n,
            1,
            accounted,
            time.perf_counter() - started,
            identity,
        )
    except StabilityUnqualified:
        raise
    except Exception as error:
        raise StabilityUnqualified(
            "diagnostic failed; no partial certificate"
        ) from error
