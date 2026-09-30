"""Compatibility adapter from the shared tools solver to production response math.

The closed-shell nuclear RHS, metric response and occupied/density
reconstruction live in :mod:`generativeqc.stationary_nuclear`.  This module
only binds the repository response solver used by existing qualification
consumers; it contains no second scientific implementation.
"""

import typing

from generativeqc.stationary_nuclear import (
    RHFNuclearBatchResponse,
    RHFNuclearResponse,
    StationaryNuclearBatchResponse,
    StationaryNuclearResponse,
    metric_density_response_mo,
)
from generativeqc.stationary_nuclear import (
    solve_stationary_nuclear_perturbation as _production_solve_one,
)
from generativeqc.stationary_nuclear import (
    solve_stationary_nuclear_perturbations as _production_solve_many,
)

from tools.generativeqc_response import GMRESOptions, solve, solve_many

__all__ = [
    "RHFNuclearBatchResponse",
    "RHFNuclearResponse",
    "StationaryNuclearBatchResponse",
    "StationaryNuclearResponse",
    "metric_density_response_mo",
    "solve_rhf_nuclear_perturbation",
    "solve_rhf_nuclear_perturbations",
    "solve_stationary_nuclear_perturbation",
    "solve_stationary_nuclear_perturbations",
]


def _checked_options(options: typing.Any) -> GMRESOptions:
    if options is None:
        return GMRESOptions()
    if not isinstance(options, GMRESOptions):
        raise TypeError("options must be GMRESOptions")
    return options


def solve_stationary_nuclear_perturbation(
    operator: typing.Any,
    frozen_fock: typing.Any,
    overlap: typing.Any,
    *,
    options: typing.Any = None,
    resident_reconstruction_consumer: typing.Any = None,
) -> StationaryNuclearResponse:
    """Bind one stationary perturbation to the existing shared GMRES solver."""
    return _production_solve_one(
        operator,
        frozen_fock,
        overlap,
        solver=solve,
        options=_checked_options(options),
        resident_reconstruction_consumer=resident_reconstruction_consumer,
        metric_response=metric_density_response_mo,
    )


def solve_stationary_nuclear_perturbations(
    operator: typing.Any,
    frozen_focks: typing.Any,
    overlaps: typing.Any,
    *,
    strategy: typing.Any = "recycled",
    options: typing.Any = None,
    resident_reconstruction_consumers: typing.Any = None,
) -> StationaryNuclearBatchResponse:
    """Bind a bounded perturbation block to the existing shared solve_many."""
    return _production_solve_many(
        operator,
        frozen_focks,
        overlaps,
        solver=solve_many,
        strategy=strategy,
        options=_checked_options(options),
        resident_reconstruction_consumers=resident_reconstruction_consumers,
        metric_response=metric_density_response_mo,
    )


def _require_rhf_operator(operator: typing.Any) -> None:
    problem = getattr(operator, "problem", None)
    if problem is None or problem.method != "rhf":
        raise TypeError("RHF nuclear response requires an RHFResponseOperator")


def solve_rhf_nuclear_perturbation(
    operator: typing.Any,
    frozen_fock: typing.Any,
    overlap: typing.Any,
    *,
    options: typing.Any = None,
    resident_reconstruction_consumer: typing.Any = None,
) -> RHFNuclearResponse:
    """Backward-compatible RHF wrapper over the stationary response consumer."""
    _require_rhf_operator(operator)
    return solve_stationary_nuclear_perturbation(
        operator,
        frozen_fock,
        overlap,
        options=options,
        resident_reconstruction_consumer=resident_reconstruction_consumer,
    )


def solve_rhf_nuclear_perturbations(
    operator: typing.Any,
    frozen_focks: typing.Any,
    overlaps: typing.Any,
    *,
    strategy: typing.Any = "recycled",
    options: typing.Any = None,
    resident_reconstruction_consumers: typing.Any = None,
) -> RHFNuclearBatchResponse:
    """Backward-compatible RHF multi-RHS wrapper over the stationary consumer."""
    _require_rhf_operator(operator)
    return solve_stationary_nuclear_perturbations(
        operator,
        frozen_focks,
        overlaps,
        strategy=strategy,
        options=options,
        resident_reconstruction_consumers=resident_reconstruction_consumers,
    )
