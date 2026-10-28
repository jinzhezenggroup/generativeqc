"""Discover coefficient slots in already specialized physical point Graphs.

Literal import, differentiation and algebra run first. Lifting leaves afterward
cannot restore zero terms, undo constant branch selection or turn a fixed power
into a general exponent. The package owner still compares complete emitted
bodies and ABIs, then restores coefficients common to every member as literals.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.expr import Expr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from generativeqc_compiler.integral.expr import Graph


def lift_point_constants(
    graph: Graph, roots: Sequence[Expr]
) -> tuple[tuple[Expr, ...], tuple[tuple[str, float], ...]]:
    """Expose arithmetic constants without changing predicates or operations.

    Predicate dependency closures stay literal, including thresholds nested in
    arithmetic. Power exponents are Graph payloads, never candidate leaves.
    Zero, unit and binary scaling constants stay native to preserve compiler
    strength reduction. Slot order follows the existing deterministic DAG walk;
    different CSE/alias structure deliberately produces different worker groups.
    This is a structural candidate, not a GPU performance qualification.
    """
    order = graph.topological_order(roots)
    protected: set[int] = set()
    for identifier in order:
        node = graph.nodes[identifier]
        if node.operation == "select_le":
            protected.update(
                graph.topological_order(
                    tuple(Expr(graph, child) for child in node.arguments[:2])
                )
            )
    replacements = {}
    parameters = []
    for identifier in order:
        node = graph.nodes[identifier]
        if node.operation != "constant" or identifier in protected:
            continue
        if node.payload is None or isinstance(node.payload, str):
            raise TypeError("point constant requires a numeric payload")
        value = float(node.payload)
        if not math.isfinite(value) or value == 0.0 or math.frexp(abs(value))[0] == 0.5:
            continue
        name = f"point_constant_{len(parameters)}"
        parameters.append((name, value))
        replacements[Expr(graph, identifier)] = graph.variable(name)
    return graph.replace_subexpressions(roots, replacements), tuple(parameters)
