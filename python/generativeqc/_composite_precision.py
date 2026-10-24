"""Budget admission and provenance for complete RSH/nonlocal gradients.

Only semilocal geometry narrows. Full-range/range exchange, VV10, AO generation
and the final SCF retain the incumbent strict owners and scientific parameters.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from . import _native
from .force_aware_numerics import TargetErrorBudget
from .profiles import canonical_hash

COMPOSITE_GRADIENT_QUALIFICATION = "wb97m-v-rks-budget-fp32-v1"
COMPOSITE_DF_GRADIENT_QUALIFICATION = "wb97m-v-df-rks-budget-fp32-v1"
COMPOSITE_GRADIENT_QUALIFICATIONS = frozenset(
    (COMPOSITE_GRADIENT_QUALIFICATION, COMPOSITE_DF_GRADIENT_QUALIFICATION)
)


@lru_cache(maxsize=1)
def _qualified_model() -> str:
    from generativeqc_compiler.method import resolve_method

    return resolve_method("WB97M-V", spin="unpolarized").identity


def composite_gradient_admission(calculator: Any, state: Any, basis: Any) -> str | None:
    """Check the whole model, including range exchange and both grid policies.

    The caller authenticates the current state. A missing optional provider
    proof denies admission; a stale/invalid proof remains an execution error.
    Dense-grid qualification does not authorize changing VV10's density cutoff.
    The provider proof names full-range J/K. For the fitted canonical model the
    composite owner separately retains exact long-range exchange and response.
    """
    from generativeqc_compiler.dft.nonlocal_policy import (
        MOLECULAR_VV10_DENSITY_POLICY,
    )

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
        or calculator._density_fitting_mode
        not in (_native.DENSITY_FITTING_NONE, _native.DENSITY_FITTING_CUDA)
        or calculator._dispersion_method_ir is not None
        or calculator._energy_tolerance > 1e-10
        or calculator._density_tolerance > 1e-8
        or calculator._screening_tolerance > 1e-12
        or not 24 <= basis.nao <= 96
        or not 2 <= basis.natom <= 12
        or any(shell.angular_momentum > 2 for shell in basis.shells)
        or source.grid_spec != GridSpec(48, 16, 32)
        or getattr(source, "nonlocal_grid_spec", source.grid_spec) != source.grid_spec
        or source.nonlocal_density_policy != MOLECULAR_VV10_DENSITY_POLICY
        or source.method_ir.identity != _qualified_model()
    ):
        return None
    try:
        provider = source.fock_provider_proof()
    except NotImplementedError:
        return None
    fitted = calculator._density_fitting_mode == _native.DENSITY_FITTING_CUDA
    expected = (
        ("density-fitted", "density-fitted", 1e-10)
        if fitted
        else ("exact", "exact", 0.0)
    )
    if provider != expected:
        return None
    return (
        COMPOSITE_DF_GRADIENT_QUALIFICATION
        if fitted
        else COMPOSITE_GRADIENT_QUALIFICATION
    )


def composite_gradient_precision(
    qualification: str, state_identity: Any, work: dict
) -> dict:
    """Record completed source work, not requested precision or SCF guesses.

    Both geometry consumers must visit the complete grid, but only semilocal
    points count as candidate work. A rejected candidate is never merged with
    the returned strict gradient. No per-pair VV10 arithmetic census is implied.
    """
    if qualification not in COMPOSITE_GRADIENT_QUALIFICATIONS:
        raise ValueError("unqualified composite gradient model")
    fitted = qualification == COMPOSITE_DF_GRADIENT_QUALIFICATION
    sources = work.get("stationary_source_work", {})
    semilocal = sources.get("semilocal", {})
    nonlocal_work = sources.get("nonlocal", {})
    points = semilocal.get("fp32_geometry_points", 0)
    rejected = work.get("rejected_semilocal_work", {})
    attempts = points + rejected.get("fp32_geometry_points", 0)
    retries = work.get("fp64_geometry_retries", 0)
    grid_points = work.get("grid_points", 0)
    if (
        retries not in (0, 1)
        or (retries and (points or not attempts))
        or grid_points <= 0
        or work.get("execution") != "cuda-complete-composite"
        or work.get("grid_density_source") != "exact-final-scf-device-binding"
        or work.get("geometry_point_visits") != 2 * grid_points
        or nonlocal_work.get("fp32_geometry_points") != 0
        or work.get("nonlocal_execution") != "resident-full-grid-device-seeds"
        or (points and points != grid_points)
        or not work.get("plan_identity")
        or (
            fitted
            and work.get("native_integral_resources", {}).get("mixed_rsh_df_provider")
            != 1
        )
    ):
        raise RuntimeError("composite gradient precision lacks complete producer work")
    return {
        "schema_version": 4,
        "owner": "python-composite-stationary-cuda",
        "complete": True,
        "qualification": qualification,
        "state_identity": canonical_hash(state_identity.to_payload()),
        "plan_identity": work["plan_identity"],
        "returned_mode": "mixed" if points else "fp64",
        "attempted_fp32_geometry_points": attempts,
        "returned_fp32_geometry_points": points,
        "attempted_fp32_geometry_batches": semilocal.get("fp32_geometry_batches", 0)
        + rejected.get("fp32_geometry_batches", 0),
        "attempted_fp32_grid_points": 0,
        "fp64_gradient_retries": retries,
        "capability_fallback": work.get("fp32_geometry_unavailable"),
        "point_count_scope": "enqueued-semilocal-geometry; guarded-xc-tails-may-widen",
        "strict_components": (
            "scf",
            "ao-producer",
            "one-electron-and-overlap-pulay",
            "df-integrals-metric-response-and-exact-long-range-exchange"
            if fitted
            else "full-range-and-range-separated-integrals",
            "vv10-pairs-and-nonlocal-geometry",
            "global-reductions",
        ),
    }
