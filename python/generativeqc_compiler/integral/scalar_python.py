"""Straight-line Python/NumPy lowering of the shared scalar DAG.

This is an arithmetic backend, not another differentiation engine. Compiling
once removes per-node interpretation from production quadrature preparation;
NumPy inputs execute the same program over the complete radial rule at once.
"""

from collections.abc import Callable, Sequence
from fractions import Fraction
from types import ModuleType
from typing import Any, cast

from .expr import Expr, Graph


def polynomial_terms(
    graph: Graph, root: Expr
) -> tuple[tuple[Fraction, tuple[str, ...]], ...]:
    """Collect exact leaf monomials of an already differentiated polynomial.

    This is a contraction lowering, not an AD engine. Rejecting other scalar
    operations prevents accidentally treating nonlinear leaves as bilinear
    operands. Repeated leaf names remain repeated factors of a monomial.
    """
    by_node: dict[int, dict[tuple[str, ...], Fraction]] = {}
    for identifier in graph.topological_order((root,)):
        node = graph.nodes[identifier]
        current: dict[tuple[str, ...], Fraction]
        if node.operation == "constant":
            coefficient = Fraction(cast("Fraction | float", node.payload))
            current = {(): coefficient} if coefficient else {}
        elif node.operation == "variable":
            current = {(str(node.payload),): Fraction(1)}
        elif node.operation == "add":
            current = {}
            for child in node.arguments:
                for monomial, coefficient in by_node[child].items():
                    current[monomial] = current.get(monomial, Fraction(0)) + coefficient
        elif node.operation == "multiply":
            current = {(): Fraction(1)}
            for child in node.arguments:
                product = {}
                for first, coefficient in current.items():
                    for second, multiplier in by_node[child].items():
                        monomial = tuple(sorted((*first, *second)))
                        product[monomial] = (
                            product.get(monomial, Fraction(0))
                            + coefficient * multiplier
                        )
                current = product
        else:
            raise ValueError(f"nonpolynomial contraction operation {node.operation!r}")
        by_node[identifier] = current
    return tuple(
        (coefficient, monomial)
        for monomial, coefficient in sorted(by_node[root.identifier].items())
        if coefficient
    )


def scalar_callable(
    graph: Graph, roots: Sequence[Expr], arguments: Sequence[str], library: ModuleType
) -> Callable[..., tuple[Any, ...]]:
    """Lower a branch-free DAG to a callable using the supplied math library.

    All executable source comes from validated Graph operations and numeric
    payloads. Argument names must be identifiers; arbitrary source and lazy
    branches are deliberately unsupported by this bounded backend.
    """
    if len(set(arguments)) != len(arguments) or any(
        not name.isidentifier() or name.startswith("_") for name in arguments
    ):
        raise ValueError("invalid scalar callable arguments")
    variables = {name: f"_argument_{index}" for index, name in enumerate(arguments)}
    references: dict[int, str] = {}
    lines = [f"def _evaluate({', '.join(variables.values())}):"]
    for identifier in graph.topological_order(roots):
        node = graph.nodes[identifier]
        operands = [references[child] for child in node.arguments]
        if node.operation == "constant":
            references[identifier] = repr(float(cast("Fraction | float", node.payload)))
            continue
        if node.operation == "variable":
            references[identifier] = variables[str(node.payload)]
            continue
        if node.operation in ("add", "multiply"):
            operator = " + " if node.operation == "add" else " * "
            expression = operator.join(operands)
        elif node.operation == "reciprocal":
            expression = f"1.0 / {operands[0]}"
        elif node.operation == "power":
            expression = (
                f"{operands[0]} ** {float(cast('Fraction | float', node.payload))!r}"
            )
        elif node.operation in ("exp", "expm1", "log", "log1p", "atan", "asinh", "erf"):
            expression = f"_library.{node.operation}({operands[0]})"
        else:
            raise ValueError(f"unsupported Python scalar operation {node.operation!r}")
        reference = f"_value_{identifier}"
        lines.append(f"    {reference} = {expression}")
        references[identifier] = reference
    lines.append(
        "    return (" + ", ".join(references[root.identifier] for root in roots) + ",)"
    )
    namespace: dict[str, Any] = {"_library": library, "__builtins__": {}}
    exec(compile("\n".join(lines), "<generated-scalar-dag>", "exec"), namespace)  # noqa: S102 -- only validated DAG operations and float literals
    return namespace["_evaluate"]
