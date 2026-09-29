"""Installed owner for closed-shell stationary nuclear-response algebra.

This module owns the method-neutral closed-shell nuclear RHS, metric connection,
occupied-orbital reconstruction, and solver-consumer orchestration used by RHF
and semilocal RKS Hessian/HVP paths.  Solver algorithms and method-specific
Fock/CPKS physics remain injected by adapters; production code never imports
repository-only `tools.*` modules.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
from generativeqc_compiler.common.arrays import immutable

__all__ = [
    "RHFNuclearBatchResponse",
    "RHFNuclearResponse",
    "StationaryNuclearBatchResponse",
    "StationaryNuclearResponse",
    "build_rhf_nuclear_rhs",
    "build_stationary_nuclear_rhs",
    "metric_density_response_mo",
    "solve_stationary_nuclear_perturbation",
    "solve_stationary_nuclear_perturbations",
]


def _finite_matrix(values: typing.Any, *, name: str, nmo: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (nmo, nmo):
        raise ValueError(f"{name} must have shape ({nmo}, {nmo}), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _orbital_energies(values: typing.Any, *, nmo: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (nmo,):
        raise ValueError(
            f"orbital_energies must have shape ({nmo},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("orbital_energies must be finite")
    return array


def metric_density_response_mo(
    overlap_derivative_mo: typing.Any, *, nocc: int
) -> np.ndarray:
    """Return the known MO density connection `-1/2 (S_R D + D S_R)`."""
    overlap = np.asarray(overlap_derivative_mo, dtype=np.float64)
    if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("overlap_derivative_mo must be a square matrix")
    if not np.isfinite(overlap).all():
        raise ValueError("overlap_derivative_mo must be finite")
    nmo = overlap.shape[0]
    if not isinstance(nocc, (int, np.integer)) or not 0 < nocc < nmo:
        raise ValueError("nocc must be an integer between 1 and nmo-1")
    occupations = np.zeros(nmo, dtype=np.float64)
    occupations[:nocc] = 2.0
    return -0.5 * (overlap * occupations[None, :] + occupations[:, None] * overlap)


def build_stationary_nuclear_rhs(
    frozen_fock_derivative_mo: typing.Any,
    overlap_derivative_mo: typing.Any,
    metric_fock_response_mo: typing.Any,
    orbital_energies: typing.Any,
    *,
    nocc: int,
) -> np.ndarray:
    """Build one closed-shell nuclear RHS in occupied-major/virtual-minor order."""
    energies = _orbital_energies(orbital_energies, nmo=len(orbital_energies))
    nmo = energies.size
    frozen = _finite_matrix(
        frozen_fock_derivative_mo, name="frozen_fock_derivative_mo", nmo=nmo
    )
    overlap = _finite_matrix(
        overlap_derivative_mo, name="overlap_derivative_mo", nmo=nmo
    )
    metric = _finite_matrix(
        metric_fock_response_mo, name="metric_fock_response_mo", nmo=nmo
    )
    if not isinstance(nocc, (int, np.integer)) or not 0 < nocc < nmo:
        raise ValueError("nocc must be an integer between 1 and nmo-1")
    occupied = np.arange(nocc)
    virtual = np.arange(nocc, nmo)
    ai = (frozen + metric)[np.ix_(virtual, occupied)].T
    overlap_ia = overlap[np.ix_(occupied, virtual)]
    eps_i = energies[occupied, None]
    eps_a = energies[None, virtual]
    return ai - 0.5 * (eps_i + eps_a) * overlap_ia


def build_rhf_nuclear_rhs(
    frozen_fock_derivative_mo: typing.Any,
    overlap_derivative_mo: typing.Any,
    metric_fock_response_mo: typing.Any,
    orbital_energies: typing.Any,
    *,
    nocc: int,
) -> np.ndarray:
    """Backward-compatible RHF name for the common closed-shell RHS algebra."""
    return build_stationary_nuclear_rhs(
        frozen_fock_derivative_mo,
        overlap_derivative_mo,
        metric_fock_response_mo,
        orbital_energies,
        nocc=nocc,
    )


@dataclass(frozen=True, eq=False)
class StationaryNuclearResponse:
    """Detached response of occupied orbitals and densities to one perturbation."""

    rhs: np.ndarray
    coefficient_derivative: np.ndarray
    occupied_energy_derivative: np.ndarray
    density_derivative: np.ndarray
    energy_weighted_density_derivative: np.ndarray
    solve_result: typing.Any


@dataclass(frozen=True, eq=False)
class StationaryNuclearBatchResponse:
    """Responses to a bounded set of perturbations from one shared solve."""

    responses: tuple[StationaryNuclearResponse, ...]
    solve_result: typing.Any

    @property
    def converged(self) -> typing.Any:
        return self.solve_result.converged


@dataclass(frozen=True, eq=False)
class _PreparedNuclearPerturbation:
    rhs: np.ndarray
    frozen_mo: np.ndarray
    overlap_mo: np.ndarray


def _matrix(value: typing.Any, n: typing.Any, name: typing.Any) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != (n, n)
        or array.dtype.kind not in "iuf"
        or np.iscomplexobj(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real AO matrix")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be representable in FP64")
    if not np.allclose(array, array.T, atol=2e-10, rtol=2e-12):
        raise ValueError(f"{name} must be symmetric")
    return array


def _validate_operator(operator: typing.Any) -> tuple[typing.Any, typing.Any]:
    problem = getattr(operator, "problem", None)
    induced_fock = getattr(operator, "induced_fock", None)
    if (
        problem is None
        or problem.method not in ("rhf", "cpks")
        or not callable(induced_fock)
    ):
        raise TypeError(
            "expected a closed-shell RHF/CPKS response operator with induced_fock"
        )
    ref = problem.reference
    nmo, nocc = ref.nmo, ref.nocc
    layout = problem.layout
    if layout.occupied != tuple(range(nocc)) or layout.virtual != tuple(
        range(nocc, nmo)
    ):
        raise ValueError(
            "nuclear response requires the complete canonical closed-shell space"
        )
    validate_current = getattr(operator, "validate_current", None)
    if callable(validate_current):
        validate_current()
    backend = getattr(operator, "backend", None)
    validate = getattr(backend, "validate_reference", None)
    if callable(validate):
        validate(ref)
    return ref, layout


def _prepare_stationary_nuclear_perturbation(
    operator: typing.Any, frozen_fock: typing.Any, overlap: typing.Any
) -> _PreparedNuclearPerturbation:
    ref, _ = _validate_operator(operator)
    nmo, nocc = ref.nmo, ref.nocc
    frozen = _matrix(frozen_fock, nmo, "frozen Fock derivative")
    s1 = _matrix(overlap, nmo, "overlap derivative")
    coefficients, energies = ref.coefficients, ref.orbital_energies

    frozen_mo = coefficients.T @ frozen @ coefficients
    overlap_mo = coefficients.T @ s1 @ coefficients
    metric_dm_mo = metric_density_response_mo(overlap_mo, nocc=nocc)
    metric_fock_mo = (
        coefficients.T
        @ operator.induced_fock(coefficients @ metric_dm_mo @ coefficients.T)
        @ coefficients
    )
    rhs = build_stationary_nuclear_rhs(
        frozen_mo, overlap_mo, metric_fock_mo, energies, nocc=nocc
    )
    values = (rhs, frozen_mo, overlap_mo)
    if not all(np.isfinite(value).all() for value in values):
        raise FloatingPointError("nonfinite nuclear perturbation preparation")
    return _PreparedNuclearPerturbation(*(immutable(value) for value in values))


def _reconstruct_stationary_nuclear_response(
    operator: typing.Any,
    prepared: _PreparedNuclearPerturbation,
    result: typing.Any,
) -> StationaryNuclearResponse:
    if not isinstance(prepared, _PreparedNuclearPerturbation):
        raise TypeError("expected a prepared stationary nuclear perturbation")
    require_converged = getattr(result, "require_converged", None)
    if not callable(require_converged) or not hasattr(result, "solution"):
        raise TypeError("response solver result lacks solution/require_converged")
    require_converged()
    ref, layout = _validate_operator(operator)
    coefficients, energies = ref.coefficients, ref.orbital_energies
    nocc = ref.nocc
    occupied_coefficients, occupied_energies = (
        coefficients[:, :nocc],
        energies[:nocc],
    )

    x_ia = layout.as_ia(result.solution)
    mo1 = -0.5 * prepared.overlap_mo[:, :nocc]
    mo1[nocc:, :] += x_ia.T
    c1 = coefficients @ mo1
    density = 2.0 * (
        c1 @ occupied_coefficients.T + occupied_coefficients @ c1.T
    )
    hs = (
        prepared.frozen_mo[:, :nocc]
        - prepared.overlap_mo[:, :nocc] * occupied_energies[None, :]
    )
    hs += (
        coefficients.T
        @ operator.induced_fock(density)
        @ occupied_coefficients
    )
    e1 = (
        hs[:nocc, :]
        + mo1[:nocc, :]
        * (occupied_energies[:, None] - occupied_energies[None, :])
    )
    left = (c1 * occupied_energies[None, :]) @ occupied_coefficients.T
    energy_density = 2.0 * (
        left + left.T + occupied_coefficients @ e1 @ occupied_coefficients.T
    )
    values = (prepared.rhs, c1, e1, density, energy_density)
    if not all(np.isfinite(value).all() for value in values):
        raise FloatingPointError("nonfinite nuclear response; no result published")
    return StationaryNuclearResponse(
        *(immutable(value) for value in values),
        result,
    )


def solve_stationary_nuclear_perturbation(
    operator: typing.Any,
    frozen_fock: typing.Any,
    overlap: typing.Any,
    *,
    solver: typing.Callable[..., typing.Any],
    options: typing.Any,
    resident_reconstruction_consumer: typing.Any = None,
) -> StationaryNuclearResponse:
    """Solve one closed-shell nuclear perturbation through an injected solver."""
    if not callable(solver):
        raise TypeError("solver must be callable")
    if resident_reconstruction_consumer is not None and not callable(
        resident_reconstruction_consumer
    ):
        raise TypeError("resident_reconstruction_consumer must be callable")
    prepared = _prepare_stationary_nuclear_perturbation(operator, frozen_fock, overlap)
    _, layout = _validate_operator(operator)

    def consume_resident_solution(engine: typing.Any, solution: typing.Any) -> None:
        reconstruct = getattr(engine, "reconstruct_nuclear_response", None)
        if reconstruct is None:
            raise ValueError(
                "resident reconstruction requires a compatible resident response engine"
            )
        resident_reconstruction_consumer(
            reconstruct(solution, prepared.frozen_mo, prepared.overlap_mo)
        )

    result = solver(
        operator,
        layout.pack(-prepared.rhs),
        options=options,
        raise_on_failure=True,
        collect_basis=False,
        solution_consumer=(
            consume_resident_solution
            if resident_reconstruction_consumer is not None
            else None
        ),
    )
    return _reconstruct_stationary_nuclear_response(operator, prepared, result)


def solve_stationary_nuclear_perturbations(
    operator: typing.Any,
    frozen_focks: typing.Any,
    overlaps: typing.Any,
    *,
    solver: typing.Callable[..., typing.Any],
    strategy: str,
    options: typing.Any,
    resident_reconstruction_consumers: typing.Any = None,
) -> StationaryNuclearBatchResponse:
    """Solve a bounded set of perturbations through one injected multi-RHS solver."""
    if not callable(solver):
        raise TypeError("solver must be callable")
    if strategy not in ("sequential", "blocked", "recycled"):
        raise ValueError("strategy must be sequential, blocked or recycled")
    ref, layout = _validate_operator(operator)
    frozen_values = np.asarray(frozen_focks)
    overlap_values = np.asarray(overlaps)
    expected_tail = (ref.nmo, ref.nmo)
    if (
        frozen_values.ndim != 3
        or overlap_values.ndim != 3
        or frozen_values.shape != overlap_values.shape
        or frozen_values.shape[1:] != expected_tail
        or frozen_values.shape[0] < 1
    ):
        raise ValueError(
            "multi-RHS frozen Fock/overlap inputs must share shape "
            "(nrhs, nmo, nmo) with nrhs >= 1"
        )
    prepared = tuple(
        _prepare_stationary_nuclear_perturbation(operator, frozen, overlap)
        for frozen, overlap in zip(frozen_values, overlap_values, strict=True)
    )
    consumers = (
        (None,) * len(prepared)
        if resident_reconstruction_consumers is None
        else tuple(resident_reconstruction_consumers)
    )
    if len(consumers) != len(prepared) or any(
        consumer is not None and not callable(consumer) for consumer in consumers
    ):
        raise ValueError(
            "resident_reconstruction_consumers must match perturbation count"
        )

    def solution_consumer(
        item: _PreparedNuclearPerturbation, consumer: typing.Any
    ) -> typing.Any:
        if consumer is None:
            return None

        def consume(engine: typing.Any, solution: typing.Any) -> None:
            reconstruct = getattr(engine, "reconstruct_nuclear_response", None)
            if reconstruct is None:
                raise ValueError(
                    "resident reconstruction requires a compatible "
                    "resident response engine"
                )
            consumer(reconstruct(solution, item.frozen_mo, item.overlap_mo))

        return consume

    packed = np.column_stack([layout.pack(-item.rhs) for item in prepared])
    multi = solver(
        operator,
        packed,
        strategy=strategy,
        options=options,
        raise_on_failure=True,
        collect_basis=False,
        solution_consumers=tuple(
            solution_consumer(item, consumer)
            for item, consumer in zip(prepared, consumers, strict=True)
        ),
    )
    results = getattr(multi, "results", None)
    if results is None or len(results) != len(prepared):
        raise TypeError("multi-RHS solver result has invalid results collection")
    responses = tuple(
        _reconstruct_stationary_nuclear_response(operator, item, result)
        for item, result in zip(prepared, results, strict=True)
    )
    return StationaryNuclearBatchResponse(responses, multi)


RHFNuclearResponse = StationaryNuclearResponse
RHFNuclearBatchResponse = StationaryNuclearBatchResponse
