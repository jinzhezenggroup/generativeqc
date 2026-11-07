"""Neutral scalar quadrature primals shared by ECP and DFT consumers.

Only scalar transformations live here. Order bounds, Newton stopping rules,
final-slope conventions, tables and traversal remain consumer policies.
"""

import math
from functools import cache

from .expr import Expr, Graph


@cache
def legendre_step_roots() -> tuple[Graph, tuple[Expr, Expr]]:
    """Differentiate the Legendre recurrence with loop-carried tangents."""
    graph = Graph()
    order, coordinate, value, previous = (
        graph.variable(name) for name in ("k", "z", "p", "previous")
    )
    following = ((2 * order - 1) * coordinate * value - (order - 1) * previous) / order
    tangent = graph.differentiate(
        following,
        coordinate,
        {"p": graph.variable("dp"), "previous": graph.variable("dprevious")},
    )
    return graph, (following, tangent)


@cache
def rational_radial_roots() -> tuple[Graph, tuple[Expr, Expr]]:
    """Return the rational radius and its spherical quadrature measure.

    Differentiate the equivalent pole-coordinate primal for the Jacobian. Its
    constant offset vanishes under AD, avoiding the cancellation and extra
    arithmetic of differentiating a quotient numerator. Evaluate the original
    radius primal to avoid cancellation near the lower endpoint. Consumers
    admit only their finite Gauss-Legendre rules strictly inside (-1, 1).
    """
    graph = Graph()
    coordinate, scale, weight = (
        graph.variable(name) for name in ("node", "scale", "weight")
    )
    unit = (coordinate + 1) / 2
    radius = scale * unit / (1 - unit)
    pole_radius = scale * (1 / (1 - unit) - 1)
    jacobian = graph.differentiate(pole_radius, coordinate)
    return graph.lower_small_integer_powers(
        (radius, weight * jacobian * radius * radius)
    )


@cache
def treutler_radial_roots() -> tuple[Graph, tuple[Expr, Expr]]:
    """The retained Treutler M4 primal and its coordinate derivative."""
    graph = Graph()
    coordinate, scale = (graph.variable(name) for name in ("x", "xi"))
    radius = (
        -scale
        / math.log(2)
        * graph.power(1 + coordinate, 0.6)
        * graph.stable_unary("log", (1 - coordinate) / 2)
    )
    return graph, (radius, graph.differentiate(radius, coordinate))
