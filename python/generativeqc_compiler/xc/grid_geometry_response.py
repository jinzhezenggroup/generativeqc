"""Build-only mixed geometry jets for native moving-grid Hessian sources.

Local scalar derivatives reuse the canonical Becke/norm/ratio AD programs.
Product jets come from AD of multiplication, not a handwritten second product
rule. AO motion differentiates a Taylor polynomial in the incumbent ordinary
spatial-jet basis; its coefficients are supplied by the existing Gaussian AD.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from math import factorial

from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.xc.grid_response_ir import grid_mixed_response_program


def _mixed(graph: Graph, root: Expr, names: tuple[str, ...]) -> tuple[Expr, ...]:
    """Compose independent left/right tangents and upstream mixed leaves."""
    left = graph.differentiate(
        root,
        graph.variable("left_direction"),
        {name: graph.variable(f"l{name}") for name in names},
    )
    right = graph.differentiate(
        root,
        graph.variable("right_direction"),
        {name: graph.variable(f"r{name}") for name in names},
    )
    mixed = graph.differentiate(
        left,
        graph.variable("right_direction"),
        {
            **{name: graph.variable(f"r{name}") for name in names},
            **{f"l{name}": graph.variable(f"lr{name}") for name in names},
        },
    )
    return root, left, right, mixed


@cache
def local_program(kind: str) -> tuple[Graph, tuple[Expr, ...], tuple[str, ...]]:
    """Scalar jet arithmetic; binary rescaling remains native scheduling only."""
    if kind in ("norm", "ratio") or kind.startswith("becke_"):
        primitive = kind.split("_")[0]
        iterations = int(kind.split("_")[1]) if primitive == "becke" else 3
        program = grid_mixed_response_program(primitive, iterations)
        names = (
            ("x", "y", "z")
            if primitive == "norm"
            else (("a", "b") if primitive == "ratio" else ("mu",))
        )
        return program.graph, program.roots, names
    graph = Graph()
    names = ("a",) if kind == "complement" else ("a", "b")
    first = graph.variable("a")
    if kind == "multiply":
        root = first * graph.variable("b")
    elif kind == "add":
        root = first + graph.variable("b")
    elif kind == "subtract":
        root = first - graph.variable("b")
    elif kind == "complement":
        root = 1 - first
    else:
        raise ValueError("unknown mixed geometry jet primitive")
    return graph, _mixed(graph, root, names), names


@cache
def ao_motion_program() -> tuple[Graph, tuple[Expr, ...]]:
    """Return left/right/mixed order-one AO jets from ordinary order-three jets.

    Spatial Taylor variables are evaluated at zero by the emitter. Expanding
    monomials as multiplication avoids singular power-rule intermediates at
    zero. The xy coefficient occurs once, with the correct mixed multiplicity
    supplied by AD rather than manually attached to the stored spatial jet.
    """
    graph = Graph()
    coordinates = [graph.variable(f"x{axis}") for axis in range(3)]
    lookup = {index: offset for offset, index in enumerate(jet_indices(3))}
    roots = []
    for base in jet_indices(1):
        terms = []
        for moving in jet_indices(2):
            shifted = tuple(
                first + second for first, second in zip(base, moving, strict=True)
            )
            term = graph.variable(f"jet{lookup[shifted]}")
            denominator = 1
            for axis, degree in enumerate(moving):
                denominator *= factorial(degree)
                for _ in range(degree):
                    term *= coordinates[axis]
            terms.append(term * Fraction(1, denominator))
        polynomial = graph.sum(terms)
        first = graph.differentiate(
            polynomial,
            graph.variable("left"),
            {f"x{axis}": graph.variable(f"left{axis}") for axis in range(3)},
        )
        second = graph.differentiate(
            polynomial,
            graph.variable("right"),
            {f"x{axis}": graph.variable(f"right{axis}") for axis in range(3)},
        )
        mixed = graph.differentiate(
            first,
            graph.variable("right"),
            {f"x{axis}": graph.variable(f"right{axis}") for axis in range(3)},
        )
        roots.extend((first, second, mixed))
    return graph, tuple(roots)
