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


def partition_domain_graph_kind(atom_limit: int) -> str:
    """Identify a runtime prefix domain, not a representative fixed-size graph."""
    checked_int(atom_limit, "partition domain atom limit", high=128)
    return f"becke-normalized-product/equal-radius/active-prefix/1-{atom_limit}"


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
    return _partition_program(atoms, iterations, dynamic_domain=False)


@lru_cache(maxsize=2, typed=True)
def grid_partition_domain_program(
    atom_limit: int = 128, iterations: int = 3
) -> GridResponseProgram:
    """Differentiate the actual bounded family used by a dynamic native owner.

    The integer ``atoms`` leaf selects an active prefix in [1, atom_limit].
    Inactive pairs contribute one to BOTH incident products, and inactive atom
    products contribute zero to the denominator. Thus each valid prefix has
    exactly the canonical objective and JVP for its actual atom count. Neither
    ``atoms`` nor ``owner`` has a tangent. Runtime admission must enforce these
    integer domains; padding distances and tangents do not affect the roots.

    This finite compiler witness is built at source generation, never at a
    geometry, tile or seed bind. It is not a fixed-N witness reused at other N.
    """
    return _partition_program(atom_limit, iterations, dynamic_domain=True)


def _partition_program(
    atoms: int, iterations: int, *, dynamic_domain: bool
) -> GridResponseProgram:
    """Share the canonical expression between exact and active-prefix graphs."""
    kind = (
        partition_domain_graph_kind(atoms)
        if dynamic_domain
        else partition_graph_kind(atoms)
    )
    if type(iterations) is not int or not 1 <= iterations <= 5:
        raise ValueError("partition iterations must be an integer in [1, 5]")
    graph = Graph()
    names = [f"distance_{atom}" for atom in range(atoms)]
    distances = [graph.variable(name) for name in names]
    active = graph.variable("atoms") if dynamic_domain else None
    factors = [[] for _ in range(atoms)]
    for first in range(atoms):
        for second in range(first):
            name = f"separation_{first}_{second}"
            names.append(name)
            difference = distances[first] - distances[second]
            separation = graph.variable(name)
            if active is not None:
                # Guard the ratio operands as well as its consumers. Even an
                # eager scalar evaluator must not divide by padding's zero.
                difference = graph.select_le(active, first, 0, difference)
                separation = graph.select_le(active, first, 1, separation)
            coordinate = grid_response_primal(
                graph,
                "ratio",
                {"a": difference, "b": separation},
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
            complement = 1 - factor
            if active is not None:
                factor = graph.select_le(active, first, 1, factor)
                complement = graph.select_le(active, first, 1, complement)
            factors[first].append(factor)
            factors[second].append(complement)
    products = [graph.multiply_many(values) for values in factors]
    if active is not None:
        products = [
            graph.select_le(active, atom, 0, product)
            for atom, product in enumerate(products)
        ]
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
