"""Runtime-independent scalar grid response IR shared by CUDA generation.

Keep NumPy evaluation lazy: uninstalled CMake generation only needs Graph and
standard-library metadata. The response consumer re-exports these canonical
objects so caches, scalar serialization and scientific identities stay shared.
"""

import typing
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral.expr import AlgebraForm, Expr, Graph


@dataclass(frozen=True)
class GridResponseProgram:
    """Shared scalar primal/JVP roots, independent of runtime or molecule size."""

    graph: Graph
    roots: tuple[Expr, ...]
    identity: str

    def evaluate(self, **variables: typing.Any) -> typing.Any:
        from generativeqc_compiler.common.array_graph import evaluate_array_graph

        return evaluate_array_graph(self.graph, self.roots, variables)


def grid_response_primal(
    graph: Graph, kind: str, variables: dict[str, Expr], iterations: int = 3
) -> Expr:
    """Compose the authoritative scalar mathematics into another graph.

    Bindings are expressions, not an opaque call or a trusted equation hash.
    This lets a structured partition graph include the actual Becke and ratio
    nodes before differentiating the complete normalized-product objective.
    """
    if type(iterations) is not int or not 1 <= iterations <= 5:
        raise ValueError("partition iterations must be an integer in [1, 5]")
    if kind == "norm":
        return graph.power(
            graph.sum(variables[name] * variables[name] for name in ("x", "y", "z")),
            0.5,
        )
    if kind == "ratio":
        return variables["a"] / variables["b"]
    if kind == "log":
        return graph.stable_unary("log", variables["p"])
    if kind == "becke_step":
        coordinate = variables["mu"]
        return 0.5 * coordinate * (3 - coordinate * coordinate)
    if kind == "becke_complement":
        return 0.5 * (1 - variables["mu"])
    if kind == "becke":
        coordinate = variables["mu"]
        for _ in range(iterations):
            coordinate = grid_response_primal(graph, "becke_step", {"mu": coordinate})
        return grid_response_primal(graph, "becke_complement", {"mu": coordinate})
    if kind == "treutler_adjust":
        mu = variables["mu"]
        return mu + variables["adjustment"] * (1 - mu * mu)
    raise ValueError("unknown grid response primitive")


def grid_response_graph_identity(
    graph: Graph,
    roots: tuple[Expr, ...],
    kind: str,
    *,
    schema: str = "generativeqc.grid-response-program/v1",
) -> str:
    """Authenticate actual reachable roots, including a composed primal/JVP.

    Cached metadata is not proof of recognition. Cross-graph roots are rejected
    before serialization, and unreachable scratch expressions are irrelevant.
    """
    if any(root.graph is not graph for root in roots):
        raise ValueError("Becke AD roots belong to a different graph")
    reachable = graph.topological_order(roots)
    indices = {node: index for index, node in enumerate(reachable)}
    return canonical_hash(
        {
            "schema": schema,
            "kind": kind,
            "nodes": [
                (
                    graph.nodes[node].operation,
                    [indices[child] for child in graph.nodes[node].arguments],
                    str(graph.nodes[node].payload),
                )
                for node in reachable
            ],
            "roots": [indices[root.identifier] for root in roots],
        }
    )


@lru_cache(maxsize=8, typed=True)
def grid_response_program(kind: typing.Any, iterations: typing.Any = 3) -> typing.Any:
    """Generate local mathematics once, not one graph per nuclear coordinate."""
    if type(iterations) is not int or not 1 <= iterations <= 5:
        raise ValueError("partition iterations must be an integer in [1, 5]")
    graph = Graph()
    if kind == "norm":
        names = ("x", "y", "z")
    elif kind == "ratio":
        names = ("a", "b")
    elif kind == "log":
        names = ("p",)
    elif kind in ("becke", "becke_step", "becke_complement"):
        names = ("mu",)
    elif kind == "treutler_adjust":
        names = ("mu", "adjustment")
    else:
        raise ValueError("unknown grid response primitive")
    primal = grid_response_primal(
        graph, kind, {name: graph.variable(name) for name in names}, iterations
    )
    tangent = graph.differentiate(
        primal,
        graph.variable("direction"),
        {name: graph.variable(f"d{name}") for name in names},
    )
    roots = (primal, tangent)
    identity = grid_response_graph_identity(graph, roots, kind)
    return GridResponseProgram(graph, roots, identity)


@lru_cache(maxsize=8, typed=True)
def grid_mixed_response_program(
    kind: typing.Any, iterations: typing.Any = 3
) -> typing.Any:
    """Generate primal, two JVPs and their mixed directional derivative.

    The left/right tangent leaves are independent. mixed_* inputs describe
    a mixed derivative of an upstream leaf; ordinary Cartesian nuclear
    directions bind them to zero. This lets composed primitives retain the
    complete chain rule without introducing a second handwritten derivative.
    """
    if type(iterations) is not int or not 1 <= iterations <= 5:
        raise ValueError("partition iterations must be an integer in [1, 5]")
    graph = Graph()
    if kind == "norm":
        names = ("x", "y", "z")
    elif kind == "ratio":
        names = ("a", "b")
    elif kind == "log":
        names = ("p",)
    elif kind == "becke":
        names = ("mu",)
    elif kind == "treutler_adjust":
        names = ("mu", "adjustment")
    else:
        raise ValueError("unknown grid response primitive")
    primal = grid_response_primal(
        graph, kind, {name: graph.variable(name) for name in names}, iterations
    )

    left = graph.differentiate(
        primal,
        graph.variable("left_direction"),
        {name: graph.variable(f"l{name}") for name in names},
    )
    right = graph.differentiate(
        primal,
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
    roots = (primal, left, right, mixed)
    identity = grid_response_graph_identity(
        graph, roots, kind, schema="generativeqc.grid-mixed-response-program/v1"
    )
    return GridResponseProgram(graph, roots, identity)


@lru_cache(maxsize=6, typed=True)
def partition_product_program(zeros: int, *, mixed: bool) -> GridResponseProgram:
    """AD of a log-scaled product with zero, one or two exact-zero factors.

    Native/array reductions retain the sum of zero-factor tangents and their
    diagonal left/right contraction. The degree-two elementary symmetric
    primal reconstructs distinct-factor pairs from those sufficient statistics.
    Its AD supplies the cross terms and multiplicities, including the square
    moment's upstream mixed response; no product-rule formula is provided.
    Three or more zeros cannot contribute through order two and stay omitted.
    """
    if type(zeros) is not int or zeros not in (0, 1, 2) or (zeros == 2 and not mixed):
        raise ValueError("unsupported zero-product response order")
    graph = Graph()
    logarithm = graph.variable("logarithm")
    zero_sum, square_sum = graph.variable("zero_sum"), graph.variable("square_sum")
    primal = graph.exponential(logarithm)
    if zeros == 1:
        primal *= zero_sum
    elif zeros == 2:
        primal *= Fraction(1, 2) * zero_sum * zero_sum - square_sum
    names = ("logarithm", "zero_sum", "square_sum")
    left = graph.differentiate(
        primal,
        graph.variable("left_direction"),
        {name: graph.variable(f"left_{name}") for name in names},
    )
    roots = (primal, left)
    if mixed:
        right_leaves = {name: graph.variable(f"right_{name}") for name in names}
        right = graph.differentiate(
            primal, graph.variable("right_direction"), right_leaves
        )
        cross = graph.differentiate(
            left,
            graph.variable("right_direction"),
            {
                **right_leaves,
                **{f"left_{name}": graph.variable(f"mixed_{name}") for name in names},
            },
        )
        roots = (*roots, right, cross)
    # At an exact zero, the square moment has only a mixed response. Derive
    # that response from one factor before lifting its bilinear contraction
    # through the reduction; the number of zero factors is runtime metadata.
    factor = graph.variable("factor")
    factor_left, factor_right = (
        graph.variable("factor_left"),
        graph.variable("factor_right"),
    )
    square_left = graph.differentiate(
        Fraction(1, 2) * factor * factor,
        graph.variable("left_direction"),
        {"factor": factor_left},
    )
    square_cross = graph.differentiate(
        square_left, graph.variable("right_direction"), {"factor": factor_right}
    )
    # Lift the AD-derived coefficient of each local left/right contraction
    # through a linear sum. This also cancels the half-square multiplicity
    # before runtime arithmetic, avoiding an otherwise needless doubled sum.
    square_coefficient = graph.differentiate(
        graph.differentiate(square_cross, factor_left), factor_right
    )
    square_cross = square_coefficient * graph.variable("zero_diagonal")
    replacements = {
        zero_sum: graph.constant(0),
        square_sum: graph.constant(0),
        graph.variable("left_square_sum"): graph.constant(0),
        graph.variable("right_square_sum"): graph.constant(0),
        graph.variable("mixed_square_sum"): square_cross,
    }
    roots = graph.replace_subexpressions(roots, replacements)
    graph, roots = graph.apply_algebra_form(roots, AlgebraForm.FACTORED_NARY)
    identity = grid_response_graph_identity(
        graph,
        roots,
        f"product-{zeros}-mixed-{mixed}",
        schema="generativeqc.grid-product-response-program/v1",
    )
    return GridResponseProgram(graph, roots, identity)
