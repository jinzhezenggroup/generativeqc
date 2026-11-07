"""Graph AD lowered to bounded AO/density and AO-matrix contractions.

Only the linear contraction is opaque to the scalar graph. Its pointwise
integrand is differentiated first; exact monomials select existing GEMMs or
einsums without constructing point-by-AO-by-AO tensors at runtime.
"""

from collections.abc import Callable
from fractions import Fraction
from functools import cache

from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.scalar_python import (
    polynomial_terms,
    scalar_callable,
)


@cache
def geometry_feature_callable(family: str, direction: str) -> Callable:
    """Generate one complete rho/gradient panel response for symmetric D.

    ``left`` holds D fixed; ``right`` includes delta-D. ``mixed`` differentiates
    the left response in that right direction. AO Taylor jets (including their
    mixed response) are supplied independently by the collocation owner.
    Symmetry combines whole AO legs only, and is valid for indefinite D.
    """
    if family not in ("lda", "gga") or direction not in ("left", "right", "mixed"):
        raise ValueError("unsupported AO feature response")
    lines = [
        (
            "def evaluate(jets, left_jets, right_jets, mixed_jets, density, "
            "delta_density, rho, gradient, contract):"
        )
    ]
    for row in range(1 if family == "lda" else 4):
        graph = Graph()
        fields = {
            f"{motion}_{leg}{spatial}": (motion, spatial)
            for motion in ("base", "left", "right", "mixed")
            for leg in ("a", "b")
            for spatial in (0, 1)
        }
        primal = (
            graph.variable("base_a0")
            * graph.variable("density")
            * graph.variable("base_b0")
        )
        if row:
            primal = graph.differentiate(
                primal,
                graph.variable("point"),
                {f"base_{leg}0": graph.variable(f"base_{leg}1") for leg in ("a", "b")},
            )
        response = primal
        if direction in ("left", "mixed"):
            response = graph.differentiate(
                response,
                graph.variable("left_step"),
                {
                    f"base_{leg}{spatial}": graph.variable(f"left_{leg}{spatial}")
                    for leg in ("a", "b")
                    for spatial in (0, 1)
                },
            )
        if direction in ("right", "mixed"):
            seeds = {
                f"{motion}_{leg}{spatial}": graph.variable(f"{tangent}_{leg}{spatial}")
                for motion, tangent in (("base", "right"), ("left", "mixed"))
                for leg in ("a", "b")
                for spatial in (0, 1)
            }
            seeds["density"] = graph.variable("delta_density")
            response = graph.differentiate(
                response, graph.variable("right_step"), seeds
            )
        contractions = {}
        for coefficient, monomial in polynomial_terms(graph, response):
            matrices = [
                name for name in monomial if name in ("density", "delta_density")
            ]
            legs = tuple(sorted(fields[name] for name in monomial if name in fields))
            if len(matrices) != 1 or len(legs) != 2 or len(monomial) != 3:
                raise ValueError("feature AD no longer admits an AO bilinear schedule")
            key = matrices[0], legs
            contractions[key] = contractions.get(key, Fraction(0)) + coefficient
        groups = {}
        for (matrix, legs), coefficient in sorted(contractions.items()):
            operands = [
                f"{'jets' if motion == 'base' else motion + '_jets'}[{row if spatial else 0}]"
                for motion, spatial in legs
            ]
            groups.setdefault(coefficient, []).append(
                f"contract({operands[0]}, {matrix}, {operands[1]})"
            )
        expression = " + ".join(
            f"{float(coefficient)!r} * (" + " + ".join(terms) + ")"
            for coefficient, terms in groups.items()
        )
        output = "rho[:]" if row == 0 else f"gradient[:, {row - 1}]"
        lines.append(f"    {output} = {expression}")
    namespace = {"__builtins__": {}}
    exec(compile("\n".join(lines), "<generated-geometry-features>", "exec"), namespace)  # noqa: S102 -- only validated AD leaf metadata
    return namespace["evaluate"]


@cache
def weighted_pair_jvp_callable() -> Callable:
    """Return a weighted coefficient and its AD JVP before a GEMM.

    Retaining this common subexpression avoids expanding the two moving
    measure factors into separate matrix products.
    """
    import numpy as np

    graph = Graph()
    arguments = ("weight", "coefficient", "moving_weight", "moving_coefficient")
    primal = graph.variable("weight") * graph.variable("coefficient")
    response = graph.differentiate(
        primal,
        graph.variable("step"),
        {
            "weight": graph.variable("moving_weight"),
            "coefficient": graph.variable("moving_coefficient"),
        },
    )
    return scalar_callable(graph, (primal, response), arguments, np)


@cache
def spatial_panel_jvp_callable() -> Callable:
    """Generate the three-term spatial AO panel and its directional response."""
    import numpy as np

    graph = Graph()
    names = tuple(
        f"{kind}{axis}" for kind in ("coefficient", "orbital") for axis in range(3)
    )
    primal = graph.sum(
        graph.variable(f"coefficient{axis}") * graph.variable(f"orbital{axis}")
        for axis in range(3)
    )
    response = graph.differentiate(
        primal,
        graph.variable("step"),
        {name: graph.variable(f"moving_{name}") for name in names},
    )
    return scalar_callable(
        graph, (primal, response), (*names, *(f"moving_{name}" for name in names)), np
    )


@cache
def matrix_jvp_callable() -> Callable:
    """Differentiate left*measure*right, then lower its three linear GEMMs."""
    graph = Graph()
    names = ("left", "measure", "right")
    primal = (
        graph.variable("left") * graph.variable("measure") * graph.variable("right")
    )
    response = graph.differentiate(
        primal,
        graph.variable("step"),
        {name: graph.variable(f"moving_{name}") for name in names},
    )
    terms = []
    for coefficient, monomial in polynomial_terms(graph, response):
        operands = []
        for name in names:
            selected = [item for item in monomial if item in (name, f"moving_{name}")]
            if len(selected) != 1:
                raise ValueError("matrix AD lost its trilinear contraction")
            operands.append(selected[0])
        left, measure, right = operands
        term = f"({left}.T @ ({measure}[:, None] * {right}))"
        terms.append(term if coefficient == 1 else f"{float(coefficient)!r} * {term}")
    source = (
        "def evaluate(left, measure, right, moving_left, moving_measure, moving_right):\n"
        "    return " + " + ".join(terms)
    )
    namespace = {"__builtins__": {}}
    exec(compile(source, "<generated-matrix-jvp>", "exec"), namespace)  # noqa: S102 -- validated polynomial AD schedule
    return namespace["evaluate"]
