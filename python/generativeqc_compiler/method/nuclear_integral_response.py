"""Build-only AD of a projected, externally weighted integral energy term.

The integral owner supplies value/gradient/gradient-JVP from its existing AD
authority. The method owner supplies physical D/W-derived source weights. This
bilinear envelope does not choose HF factors, spin conventions or ECP policy.
"""

from functools import cache

from generativeqc_compiler.integral.expr import Graph


@cache
def projected_integral_program() -> tuple:
    """Value, right-direction term, left gradient and mixed weighted response.

    The projection is fixed in nuclear coordinates. Left weight motion is zero
    for a stationary nuclear partial; right weight motion is retained exactly.
    Each call represents one mathematical-center coordinate, before atom scatter.
    """
    graph = Graph()
    value, first, mixed, weight, delta_weight, projection, direction = (
        graph.variable(name)
        for name in (
            "value",
            "first",
            "mixed",
            "weight",
            "delta_weight",
            "projection",
            "direction",
        )
    )
    projected = projection * value
    left = graph.variable("left")
    right = graph.variable("right")
    first_value = graph.differentiate(projected, left, {"value": first})
    gradient = graph.differentiate(weight * projected, left, {"value": first})
    response = graph.differentiate(
        gradient, right, {"first": mixed, "weight": delta_weight}
    )
    return graph, (projected, first_value * direction, gradient, response)
