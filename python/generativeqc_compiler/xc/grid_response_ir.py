"""Runtime-independent scalar grid response IR shared by CUDA generation.

Keep NumPy evaluation lazy: uninstalled CMake generation only needs Graph and
standard-library metadata. The response consumer re-exports these canonical
objects so caches, scalar serialization and scientific identities stay shared.
"""

import typing
from dataclasses import dataclass
from functools import lru_cache

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral.expr import Expr, Graph


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
    if kind == "becke":
        coordinate = variables["mu"]
        for _ in range(iterations):
            coordinate = 0.5 * coordinate * (3 - coordinate * coordinate)
        return 0.5 * (1 - coordinate)
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
    elif kind == "becke":
        names = ("mu",)
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
