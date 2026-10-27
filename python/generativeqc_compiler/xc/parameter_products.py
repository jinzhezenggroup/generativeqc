"""Recover coefficient-only multiplication folding for shared point workers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.expr import Expr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from generativeqc_compiler.integral.expr import Graph


def prepare_parameter_products(
    graph: Graph,
    roots: Sequence[Expr],
    parameters: tuple[tuple[str, float], ...],
) -> tuple[tuple[Expr, ...], tuple[tuple[str, float], ...]]:
    """Move finite parameter/literal products into immutable launch data.

    Run after differentiation, before any equal-spin projection, so both point
    paths have the same coefficient ABI. Only direct coefficient leaves and
    literal factors of an existing product are combined. Input-dependent
    arithmetic, predicates and lazy branch evaluation retain their Graph owner;
    no singular reciprocal or transcendental is speculated onto the host.

    This restores folding lost when registration literals become variables.
    Regrouping a floating-point product is numerically qualified, not promised
    bitwise identical to either the literal or unprepared parameter worker.
    """
    bindings = dict(parameters)
    prepared = list(parameters)
    slots: dict[Expr, Expr] = {}
    replacements: dict[Expr, Expr] = {}
    for identifier in tuple(graph.topological_order(roots)):
        node = graph.nodes[identifier]
        if node.operation != "multiply":
            continue
        coefficients = []
        remaining = []
        has_parameter = has_literal = False
        for argument in node.arguments:
            factor = graph.nodes[argument]
            parameter = factor.operation == "variable" and factor.payload in bindings
            literal = factor.operation == "constant"
            has_parameter |= parameter
            has_literal |= literal
            (coefficients if parameter or literal else remaining).append(
                Expr(graph, argument)
            )
        if not (has_parameter and has_literal):
            continue
        product = graph.multiply_many(coefficients)
        if product not in slots:
            value = graph.evaluate(product, bindings)
            if not math.isfinite(value):
                raise ValueError("prepared XC parameter product must be finite")
            name = f"prepared_parameter_{len(slots)}"
            if name in bindings:
                raise ValueError("prepared XC parameter name collides with an input")
            slots[product] = graph.variable(name)
            prepared.append((name, value))
        # Replacement subtrees are inserted verbatim by the Graph API. Apply
        # already prepared descendants first, retaining all nested rewrites.
        rewritten = graph.replace_subexpressions(remaining, replacements)
        replacements[Expr(graph, identifier)] = graph.multiply_many(
            (*rewritten, slots[product])
        )
    return graph.replace_subexpressions(roots, replacements), tuple(prepared)
