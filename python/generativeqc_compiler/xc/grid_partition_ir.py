"""Canonical whole Becke normalized-product objective and its generated JVP.

Distance leaves bind the existing point/center norm owners. The graph contains
every ratio, clipped switch, complementary factor, atom product, denominator
and owner selection, rather than a list of companion scalar fingerprints.
The symbolic product expression defines the mathematics; the emitted stable
log/zero prescription remains independently qualified by the existing owners.
"""

from __future__ import annotations

from functools import lru_cache

from generativeqc_compiler.dft.grid import checked_int
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.xc.grid_response_ir import (
    GridResponseProgram,
    grid_response_graph_identity,
    grid_response_primal,
)


def partition_graph_kind(atoms: int) -> str:
    """Bind dimension and supported equal-radius semantics into recognition."""
    checked_int(atoms, "partition graph atoms", high=128)
    return f"becke-normalized-product/equal-radius/{atoms}"


@lru_cache(maxsize=4, typed=True)
def grid_partition_program(atoms: int, iterations: int = 3) -> GridResponseProgram:
    """Differentiate the complete objective, including all denominator products.

    The owner leaf is an integer in [0, atoms); it has no tangent. Distances
    belong to the validated smooth geometry domain. Clip mu at equality, but
    retain a switch's derivative at rounded exactly-zero/one factor values:
    only a strictly out-of-range switch value kills that derivative.
    Compiler graph size is explicitly bounded; larger unsupported dimensions
    must retain the existing generic route, not build unbounded scalar graphs.
    """
    kind = partition_graph_kind(atoms)
    if type(iterations) is not int or not 1 <= iterations <= 5:
        raise ValueError("partition iterations must be an integer in [1, 5]")
    graph = Graph()
    names = [f"distance_{atom}" for atom in range(atoms)]
    distances = [graph.variable(name) for name in names]
    factors = [[] for _ in range(atoms)]
    for first in range(atoms):
        for second in range(first):
            name = f"separation_{first}_{second}"
            names.append(name)
            coordinate = grid_response_primal(
                graph,
                "ratio",
                {"a": distances[first] - distances[second], "b": graph.variable(name)},
                iterations,
            )
            clipped = graph.select_le(
                coordinate, -1, -1, graph.select_le(1, coordinate, 1, coordinate)
            )
            raw = grid_response_primal(graph, "becke", {"mu": clipped}, iterations)
            # Equality must select raw, not a constant that would erase a
            # rounded-zero factor's still-nonzero scientific tangent.
            factor = graph.select_le(
                raw,
                0,
                graph.select_le(0, raw, raw, 0),
                graph.select_le(1, raw, graph.select_le(raw, 1, raw, 1), raw),
            )
            factors[first].append(factor)
            factors[second].append(1 - factor)
    products = [graph.multiply_many(values) for values in factors]
    owner = graph.variable("owner")
    numerator = products[-1]
    for atom in range(atoms - 2, -1, -1):
        numerator = graph.select_le(owner, atom, products[atom], numerator)
    primal = grid_response_primal(
        graph, "ratio", {"a": numerator, "b": graph.sum(products)}, iterations
    )
    tangent = graph.differentiate(
        primal,
        graph.variable("direction"),
        {name: graph.variable(f"d{name}") for name in names},
    )
    roots = (primal, tangent)
    return GridResponseProgram(
        graph, roots, grid_response_graph_identity(graph, roots, kind)
    )
