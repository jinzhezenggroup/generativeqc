"""Derive cached softened-Coulomb weights from their scalar TensorIR primal.

The runtime has already evaluated a stable reciprocal distance. Factorization
is qualified on its positive domain; it never squares large coordinates again.
Interleaving cached-kernel factors with charge factors preserves the established
finite-range schedule rather than multiplying the two charges first.
"""

from fractions import Fraction
from typing import cast

from generativeqc_compiler.integral.expr import AlgebraForm, Expr
from generativeqc_compiler.method.scalar_graph_ad import scalar_graph
from generativeqc_compiler.tensor.ir import Node, constant, input_tensor, multiply
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import TensorSpec


def cached_radial_weight(primal: Program, scaled_inputs: tuple[str, ...]) -> Node:
    """Return -dE/dx divided symbolically by x, without a runtime x division.

    All exponents and coefficients come from AD of ``pair_energy``. The
    supplied input order is only a floating-point scaling schedule. Reject a
    primal outside the admitted reciprocal-square-root contract, rather than
    silently falling back to a supplied derivative formula.
    """
    graph, values = scalar_graph(primal)
    kernel = primal.outputs["kernel"]
    if (
        kernel.op != "divide"
        or kernel.inputs[0].op != "constant"
        or kernel.inputs[0].attrs["values"] != ((1, 1),)
        or kernel.inputs[1].op != "sqrt"
    ):
        raise ValueError("cached radial AD requires a reciprocal-square-root kernel")
    squared_radius = values[kernel.inputs[1].inputs[0]]
    coordinate = graph.variable("dx")
    response = (
        -graph.differentiate(values[primal.outputs["pair_energy"]], coordinate)
        / coordinate
    )
    cached_kernel = graph.variable("kernel")
    (response,) = graph.replace_subexpressions(
        (response,),
        {
            squared_radius.pow(0.5): graph.reciprocal(cached_kernel),
            squared_radius.pow(-0.5): cached_kernel,
        },
    )
    graph, (response,) = graph.apply_algebra_form(
        (response,), AlgebraForm.FACTORED_NARY
    )
    coefficient = Fraction(1)
    powers: dict[str, int] = {}

    def collect(root: Expr, exponent: int = 1) -> None:
        nonlocal coefficient
        node = graph.node(root)
        if node.operation == "constant":
            coefficient *= Fraction(cast("Fraction | float", node.payload)) ** exponent
        elif node.operation == "variable":
            powers[str(node.payload)] = powers.get(str(node.payload), 0) + exponent
        elif node.operation == "multiply":
            for child in node.arguments:
                collect(Expr(graph, child), exponent)
        elif node.operation == "reciprocal":
            collect(Expr(graph, node.arguments[0]), -exponent)
        elif (
            node.operation == "power"
            and float(cast("Fraction | float", node.payload)).is_integer()
        ):
            collect(
                Expr(graph, node.arguments[0]),
                exponent * int(cast("Fraction | float", node.payload)),
            )
        else:
            raise ValueError(f"unfactored cached radial AD operation {node.operation}")

    collect(response)
    powers = {name: exponent for name, exponent in powers.items() if exponent}
    if set(powers) != {*scaled_inputs, "kernel"} or any(
        exponent < 0 for exponent in powers.values()
    ):
        raise ValueError("cached radial AD acquired a different factor domain")
    inputs = {name: input_tensor(name, TensorSpec((), role="input")) for name in powers}
    factors = []
    if coefficient != 1:
        factors.append(constant(coefficient))
    remaining = powers["kernel"]
    for name in scaled_inputs:
        for _ in range(powers[name]):
            factors.append(inputs[name])
            if remaining:
                factors.append(inputs["kernel"])
                remaining -= 1
    factors.extend([inputs["kernel"]] * remaining)
    weight = factors[0]
    for factor in factors[1:]:
        weight = multiply(weight, factor)
    return weight
