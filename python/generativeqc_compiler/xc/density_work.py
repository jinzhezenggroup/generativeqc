"""Compose the declared density-only worker contract after differentiation.

The imported Graph remains the mathematical owner. This projection evaluates
its existing derivative roots at safe work densities, without differentiating
the clipping map, and weights energy by the original total density. The older
unclipped candidate domains deliberately retain their original behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from generativeqc_compiler.common.provenance import file_hash

if TYPE_CHECKING:
    from generativeqc_compiler.integral.expr import Expr, Graph

    from .libxc_work import LibxcWorkPolicy

DENSITY_WORK_DOMAIN = "libxc-7.0/work-density-only-v1"
DENSITY_WORK_DOMAIN_VERSION = 5


def density_work_payload(
    policy: LibxcWorkPolicy, *, projection_source_identity: str | None = None
) -> dict[str, Any]:
    """Bind the projection owner and exact worker policy into execution identity."""
    return {
        "schema": "generativeqc.libxc-density-work.v1",
        "domain": DENSITY_WORK_DOMAIN,
        "projection_owner_sha256": (
            file_hash(Path(__file__))
            if projection_source_identity is None
            else projection_source_identity
        ),
        "worker": policy.to_payload(),
        "derivative_contract": "interior-roots-at-work-inputs-no-clipping-derivative",
        "energy_contract": "original-density-weight",
    }


def apply_density_work(
    graph: Graph,
    roots: tuple[Expr, ...],
    outputs: tuple[tuple[int, ...], ...],
    features: tuple[str, ...],
    policy: LibxcWorkPolicy,
) -> tuple[Expr, ...]:
    """Project density-only roots while preserving unavailable product claims.

    Call after constructing and algebraically optimizing interior derivative roots. An
    energy-only projection followed by differentiation would silently create a
    different potential at floored spins. RKS total-density and UKS per-spin
    layouts retain their explicit, distinct normalization conventions.
    Divide energy by work density before reweighting, and do not flatten this
    quotient into a multiply-first n-ary product: that can square large finite
    densities and overflow even when the original energy is finite.
    """
    if policy.family != "lda" or features not in (("rho",), ("rho_a", "rho_b")):
        raise ValueError("density work projection requires a density-only LDA layout")
    if len(roots) != len(outputs):
        raise ValueError("density work roots and output contracts must align")
    variables = tuple(graph.variable(feature) for feature in features)
    work_variables = tuple(
        graph.select_le(
            variable, policy.density_threshold, policy.density_threshold, variable
        )
        for variable in variables
    )
    projected = graph.replace_subexpressions(
        roots, dict(zip(variables, work_variables, strict=True))
    )
    raw_total = variables[0] if len(variables) == 1 else variables[0] + variables[1]
    work_total = (
        work_variables[0]
        if len(work_variables) == 1
        else work_variables[0] + work_variables[1]
    )
    return tuple(
        (root / work_total) * raw_total if output == () else root
        for root, output in zip(projected, outputs, strict=True)
    )


__all__ = [
    "DENSITY_WORK_DOMAIN",
    "DENSITY_WORK_DOMAIN_VERSION",
    "apply_density_work",
    "density_work_payload",
]
