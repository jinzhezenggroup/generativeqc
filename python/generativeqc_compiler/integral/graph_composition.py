"""Backend-neutral exact scalar DAG composition with simultaneous leaf binding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .expr import Node

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .expr import Expr, Graph


def import_roots(
    target: Graph,
    source: Graph,
    roots: Sequence[Expr],
    bindings: Mapping[str, Expr],
) -> tuple[Expr, ...]:
    """Rehome reachable operations verbatim with complete simultaneous binding.

    Roots must belong to the source and every replacement to the target. Validate
    the complete binding before appending nodes, including unused replacements.
    Replacement subtrees are never rewritten, preserving lazy branch operations
    and allowing names shared by source and target to be bound simultaneously.
    """
    for root in roots:
        source.node(root)
    for value in bindings.values():
        target.node(value)
    order = source.topological_order(roots)
    missing = {
        str(source.nodes[identifier].payload)
        for identifier in order
        if source.nodes[identifier].operation == "variable"
    } - bindings.keys()
    if missing:
        raise ValueError(f"missing scalar bindings: {', '.join(sorted(missing))}")
    imported: dict[int, Expr] = {}
    for identifier in order:
        node = source.nodes[identifier]
        imported[identifier] = (
            bindings[str(node.payload)]
            if node.operation == "variable"
            else target._intern(
                Node(
                    node.operation,
                    tuple(imported[child].identifier for child in node.arguments),
                    node.payload,
                )
            )
        )
    return tuple(imported[root.identifier] for root in roots)
