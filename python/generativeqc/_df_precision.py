"""Application-budget admission for the complete fitted stationary consumer.

This policy does not narrow the SCF or fitted integral producer. Only the shared
semilocal geometry consumer has independent mixed-arithmetic qualification.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import _native
from .force_aware_numerics import TargetErrorBudget
from .profiles import canonical_hash

DF_GRADIENT_QUALIFICATIONS = frozenset(
    f"{name}-df-rks-budget-fp32-v1"
    for name in ("pbe", "pbe0", "b3lyp", "r2scan", "m06-2x")
)


@lru_cache(maxsize=1)
def _qualified_models() -> dict[str, str]:
    from generativeqc_compiler.method import resolve_method

    return {
        resolve_method(name, spin="unpolarized").identity: (
            f"{name.lower()}-df-rks-budget-fp32-v1"
        )
        for name in ("PBE", "PBE0", "B3LYP", "R2SCAN", "M06-2X")
    }


def fitted_gradient_admission(calculator: Any, state: Any, basis: Any) -> str | None:
    """Select a bounded complete model, never a point-code/name-only ingredient.

    Called with the validated native state used by the stationary consumer.
    Missing optional provider evidence denies selection; stale evidence remains
    an error. Geometry and final-state validation still belong to the consumer.
    The constructor binds the canonical 3c basis; the facade additionally checks
    both typed correction results before publishing a complete 3c qualification.
    """
    from ._ks_snapshot import NativeKsSnapshot
    from .ks import GridSpec

    budget = getattr(calculator, "_application_precision_budget", None)
    if not isinstance(budget, TargetErrorBudget) or (
        budget.energy_abs < 1e-6
        or budget.force_max_abs < 1e-5
        or (budget.force_rms is not None and budget.force_rms < 3e-6)
    ):
        return None
    source = state._source
    if (
        type(source) is not NativeKsSnapshot
        or source.backend != "cuda"
        or source.hamiltonian != "all-electron"
        or state.identity.spin != "unpolarized"
        or calculator._precision_mode != _native.PRECISION_FP64
        or calculator._density_fitting_mode != _native.DENSITY_FITTING_CUDA
        or calculator._energy_tolerance > 1e-10
        or calculator._density_tolerance > 1e-8
        or calculator._screening_tolerance > 1e-12
        or not 24 <= basis.nao <= 96
        or not 2 <= basis.natom <= 12
        or any(shell.angular_momentum > 2 for shell in basis.shells)
        or source.grid_spec not in (GridSpec(24, 8, 16), GridSpec(48, 16, 32))
    ):
        return None
    qualification = _qualified_models().get(source.method_ir.identity)
    if qualification is None:
        return None
    correction_model = calculator._dispersion_method_ir
    if correction_model is not None:
        from generativeqc_compiler.method import resolve_method

        if (
            qualification != "r2scan-df-rks-budget-fp32-v1"
            or correction_model.identity
            != resolve_method("R2SCAN-3c", spin="unpolarized").identity
        ):
            return None
    try:
        coulomb, exchange, threshold = source.fock_provider_proof()
    except NotImplementedError:
        return None
    expected_exchange = "density-fitted" if source.coefficients[2] else None
    if (coulomb, exchange, threshold) != (
        "density-fitted",
        expected_exchange,
        1e-10,
    ):
        return None
    return qualification


def fitted_gradient_precision(qualification: str, result: Any) -> dict:
    """Publish completed consumer work without inventing native SCF counters.

    Candidate points count actual enqueues. Guarded XC tails may use FP64;
    they are not a census of interior point-graph arithmetic. A numerical retry
    keeps rejected work separate from the strictly returned gradient.
    """
    if qualification not in DF_GRADIENT_QUALIFICATIONS:
        raise ValueError("unqualified fitted gradient model")
    work = result.work
    points = work.get("fp32_geometry_points", 0)
    rejected = work.get("rejected_semilocal_work", {})
    attempts = points + rejected.get("fp32_geometry_points", 0)
    retries = work.get("fp64_geometry_retries", 0)
    if (
        retries not in (0, 1)
        or (retries and (points or not attempts))
        or work.get("xc_points", 0) <= 0
        or work.get("grid_density_source") != "exact-final-scf-device-binding"
        or work.get("native_integral_resources", {}).get("density_fitted_provider") != 1
        or (points and points != work["xc_points"])
    ):
        raise RuntimeError("fitted gradient precision lacks complete producer work")
    return {
        "schema_version": 4,
        "owner": "python-stationary-cuda",
        "complete": True,
        "qualification": qualification,
        "state_identity": canonical_hash(result.state_identity.to_payload()),
        "plan_identity": result.plan_identity,
        "returned_mode": "mixed" if points else "fp64",
        "attempted_fp32_geometry_points": attempts,
        "returned_fp32_geometry_points": points,
        "attempted_fp32_geometry_batches": work.get("fp32_geometry_batches", 0)
        + rejected.get("fp32_geometry_batches", 0),
        "attempted_fp32_grid_points": 0,
        "fp64_gradient_retries": retries,
        "capability_fallback": work.get("fp32_geometry_unavailable"),
        "point_count_scope": "enqueued-geometry; guarded-xc-tails-may-widen",
        "strict_components": (
            "scf",
            "ao-producer",
            "one-electron-and-overlap-pulay",
            "df-integrals-and-metric-response",
            "global-reductions",
        ),
    }
