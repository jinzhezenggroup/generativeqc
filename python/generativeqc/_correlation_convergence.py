"""CBS stopping limits mirrored by the native correlated convergence owner.

Budgets select stopping criteria, not an a posteriori observable certificate.
The native owner independently validates the transport and solver ceilings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .force_aware_numerics import TargetErrorBudget


def convergence_scale(budget: TargetErrorBudget | None) -> float:
    """Retain strict convergence below the supported relaxation bracket."""
    if budget is None:
        return 0.0
    scale = min(
        1.0,
        budget.energy_abs / 1e-5,
        budget.force_max_abs / 1e-4,
        budget.force_rms / 3e-5 if budget.force_rms is not None else 1.0,
    )
    return scale if scale >= 1e-3 else 0.0


def convergence_limits(budget: TargetErrorBudget | None) -> dict[str, float]:
    """Return native ceilings; explicit tighter SCF/CC controls still win."""
    scale = convergence_scale(budget)
    return {
        "energy_tolerance": max(1e-11, scale * 1e-8),
        "density_tolerance": max(1e-11, scale * 1e-9),
        "ccsd_energy_tolerance": 1e-8,
        "ccsd_residual_tolerance": max(1e-9, scale * 1e-6),
        "lambda_residual_tolerance": max(1e-9, scale * 1e-7),
        "response_absolute_tolerance": max(1e-12, scale * 1e-8),
        "orbital_residual_tolerance": max(1e-10, scale * 1e-7),
        "stationarity_tolerance": max(1e-8, scale * 1e-6),
    }
