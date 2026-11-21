"""Smooth boundary primitives for energy-derived semilocal point domains."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from generativeqc_compiler.integral.expr import Expr


def smooth_floor(value: Expr, limit: Expr) -> Expr:
    """Join a positive floor to the identity with two continuous derivatives.

    The limit can depend on physical inputs. At each join the transition
    matches its neighboring branch through second order in both arguments.
    """
    graph = value.graph
    low = limit / 2
    high = 3 * low
    fraction = (value - low) / limit
    transition = limit + low * fraction.pow(3) * (2 - fraction)
    return graph.select_le(
        value, low, limit, graph.select_le(high, value, value, transition)
    )


def smooth_activation(value: Expr, cutoff: float, active: Expr) -> Expr:
    """Activate an energy lazily between its cutoff and twice that cutoff."""
    graph = value.graph
    low = graph.approximate_constant(cutoff)
    high = 2 * low
    fraction = (value - low) / low
    weight = fraction.pow(3) * (10 + fraction * (-15 + 6 * fraction))
    return graph.select_le(
        value, low, 0, graph.select_le(high, value, active, weight * active)
    )
