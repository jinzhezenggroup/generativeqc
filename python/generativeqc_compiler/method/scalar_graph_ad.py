"""Exact-polynomial AD lowering of scalar TensorIR energy primals.

TensorIR owns the actual energy; the shared Graph owns differentiation.
Collecting exact polynomial coefficients before native lowering avoids
redundant product-rule terms or rounded division-by-three coefficients.
"""

from collections import Counter
from fractions import Fraction

from generativeqc_compiler.integral.expr import Expr, Graph
from generativeqc_compiler.integral.scalar_python import polynomial_terms
from generativeqc_compiler.tensor.ir import Node, constant, multiply
from generativeqc_compiler.tensor.program import Program


def scalar_graph(primal: Program) -> tuple[Graph, dict[Node, Expr]]:
    """Convert admitted scalar arithmetic, rejecting opaque provider ops."""
    graph = Graph()
    values = {}
    for node in primal.live_nodes:
        if node.spec.shape or node.spec.dtype != "float64":
            raise ValueError("scalar Graph AD requires rank-zero FP64 TensorIR")
        operands = [values[child] for child in node.inputs]
        if node.op == "input":
            value = graph.variable(node.attrs["name"])
        elif node.op == "constant":
            (literal,) = node.attrs["values"]
            value = graph.constant(Fraction(*literal))
        elif node.op == "add":
            value = graph.sum(
                Fraction(*coefficient) * operand
                for coefficient, operand in zip(
                    node.attrs["coefficients"], operands, strict=True
                )
            )
        elif node.op == "multiply":
            value = operands[0] * operands[1]
        elif node.op == "divide":
            value = operands[0] / operands[1]
        elif node.op == "sqrt":
            value = operands[0].pow(Fraction(1, 2))
        else:
            raise ValueError(f"unsupported scalar TensorIR operation {node.op!r}")
        values[node] = value
    return graph, values


def polynomial_partial(
    primal: Program, output: str, coordinate: str, factor_order: tuple[str, ...]
) -> Node:
    """AD then lower a single collected monomial in a qualified factor order.

    The order controls only floating-point evaluation, never coefficients or
    exponents. A change in the energy that invalidates this schedule fails
    closed instead of silently reusing a historical potential formula.
    """
    graph, values = scalar_graph(primal)
    response = graph.differentiate(
        values[primal.outputs[output]], graph.variable(coordinate)
    )
    terms = polynomial_terms(graph, response)
    if len(terms) != 1:
        raise ValueError("scalar partial no longer admits a monomial schedule")
    coefficient, monomial = terms[0]
    if Counter(monomial) != Counter(factor_order):
        raise ValueError("scalar AD factors no longer fit the qualified schedule")
    inputs = {
        node.attrs["name"]: node for node in primal.live_nodes if node.op == "input"
    }
    factors = [inputs[name] for name in factor_order]
    if coefficient != 1:
        factors.insert(0, constant(coefficient))
    result = factors[0]
    for factor in factors[1:]:
        result = multiply(result, factor)
    return result
