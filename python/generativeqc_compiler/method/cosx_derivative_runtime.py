"""Compiler-owned scalar TensorIR for COSX derivative contractions.

Native CUDA owns tile/index traversal, validation, stream/error handling and
atom scatter. These programs own the pure FP64 contraction arithmetic reused by
the explicit-point and molecular derivative consumers.
"""

from __future__ import annotations

from generativeqc_compiler.tensor.ad_program import (
    _rebuild,
    linearize,
    transpose_program,
)
from generativeqc_compiler.tensor.ir import Node, add, constant, input_tensor, multiply
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import TensorSpec

COSX_DERIVATIVE_RUNTIME_VERSION = "cosx-derivative-contraction-ir-v1"


def _input(name: str, *, differentiable: bool = False) -> Node:
    return input_tensor(
        name, TensorSpec((), role="input", differentiable=differentiable)
    )


def _half() -> Node:
    return constant("1/2", TensorSpec((), role="constant"))


def _program(outputs: dict[str, Node], kind: str, derivative_hash: str = "") -> Program:
    return Program(
        outputs,
        provenance={
            "kind": kind,
            "version": COSX_DERIVATIVE_RUNTIME_VERSION,
            "source_issue": 246,
            **({"derivative_hash": derivative_hash} if derivative_hash else {}),
        },
    )


def _directional(value: Node, seeds: dict[str, str]) -> tuple[Node, str]:
    """Differentiate a scalar integrand and bind its runtime tangent ABI."""
    derivative = linearize(Program({"value": value}), tuple(seeds))
    replacements = {
        node: _input(seeds[node.attrs["name"][2:]])
        for node in derivative.program.nodes
        if node.op == "input" and node.attrs["name"] in {f"d_{name}" for name in seeds}
    }
    bound = _rebuild(derivative.program, replacements)
    return bound.outputs["d_value"], derivative.derivative_hash


def build_cosx_projection_update_program() -> Program:
    accumulator = _input("accumulator")
    left = _input("left")
    right = _input("right")
    return _program(
        {"updated": add(accumulator, multiply(left, right))},
        "cosx-projection-update",
    )


def build_cosx_esp_derivative_update_program() -> Program:
    """Generate the product JVP inside the existing linear ESP reduction."""
    accumulator = _input("accumulator")
    projected_value = _input("projected_value", differentiable=True)
    matrix_value = _input("matrix_value", differentiable=True)
    response, derivative_hash = _directional(
        multiply(matrix_value, projected_value),
        {
            "matrix_value": "matrix_derivative",
            "projected_value": "projected_derivative",
        },
    )
    return _program(
        {"updated": add(accumulator, response)},
        "cosx-esp-derivative-update",
        derivative_hash,
    )


def build_cosx_symmetric_projection_update_program() -> Program:
    accumulator = _input("accumulator")
    ao = _input("ao")
    density_rc = _input("density_rc")
    density_cr = _input("density_cr")
    averaged = multiply(multiply(ao, _half()), add(density_rc, density_cr))
    return _program(
        {"updated": add(accumulator, averaged)},
        "cosx-symmetric-projection-update",
    )


def build_cosx_bidirectional_update_program() -> Program:
    right = _input("right")
    left = _input("left")
    matrix_rc = _input("matrix_rc")
    matrix_cr = _input("matrix_cr")
    projected = _input("projected")
    symmetric_projection = _input("symmetric_projection")
    return _program(
        {
            "right_updated": add(right, multiply(matrix_rc, projected)),
            "left_updated": add(left, multiply(matrix_cr, symmetric_projection)),
        },
        "cosx-bidirectional-update",
    )


def build_cosx_point_gradient_update_program() -> Program:
    """Generate the fixed-density JVP of the symmetric point energy summand."""
    accumulator = _input("accumulator")
    density = _input("density")
    fields = ("phi_row", "phi_column", "potential_row", "potential_column")
    values = {name: _input(name, differentiable=True) for name in fields}
    primal = multiply(
        multiply(density, _half()),
        add(
            multiply(values["phi_row"], values["potential_column"]),
            multiply(values["phi_column"], values["potential_row"]),
        ),
    )
    contribution, derivative_hash = _directional(
        primal,
        {
            name: name.replace("_row", "_derivative_row").replace(
                "_column", "_derivative_column"
            )
            for name in fields
        },
    )
    return _program(
        {"updated": add(accumulator, contribution)},
        "cosx-point-gradient-update",
        derivative_hash,
    )


def build_cosx_molecular_ao_update_program() -> Program:
    """Generate both AO-leg cotangents of the factored molecular integrand."""
    from_left = _input("from_left")
    from_right = _input("from_right")
    density_rc = _input("density_rc")
    density_cr = _input("density_cr")
    potential = _input("potential")
    left_potential = _input("left_potential")
    left_ao = _input("left_ao", differentiable=True)
    right_ao = _input("right_ao", differentiable=True)
    primal = add(
        multiply(
            multiply(multiply(_half(), add(density_rc, density_cr)), left_ao), potential
        ),
        multiply(multiply(density_rc, right_ao), left_potential),
    )
    derivative = transpose_program(
        Program({"value": primal}), ("value",), inputs=("left_ao", "right_ao")
    )
    bound = _rebuild(
        derivative.program,
        {
            node: constant(1)
            for node in derivative.program.nodes
            if node.op == "input" and node.attrs["name"] == "bar_value"
        },
    )
    return _program(
        {
            "from_left_updated": add(from_left, bound.outputs["bar_left_ao"]),
            "from_right_updated": add(from_right, bound.outputs["bar_right_ao"]),
        },
        "cosx-molecular-ao-update",
        derivative.derivative_hash,
    )


def build_cosx_molecular_cotangent_program() -> Program:
    energy_factor = _input("energy_factor")
    weight = _input("weight")
    from_left = _input("from_left")
    from_right = _input("from_right")
    cotangent = multiply(
        multiply(energy_factor, weight),
        add(from_left, from_right),
    )
    return _program({"cotangent": cotangent}, "cosx-molecular-cotangent")


def build_cosx_scale_program() -> Program:
    factor = _input("factor")
    value = _input("value")
    return _program({"scaled": multiply(factor, value)}, "cosx-scale")


def build_cosx_pair_scale_program() -> Program:
    first = _input("first")
    second = _input("second")
    value = _input("value")
    return _program(
        {"scaled": multiply(multiply(first, second), value)},
        "cosx-pair-scale",
    )
