"""Compiler-owned GFN2 external point-charge force response.

The native runtimes keep topology, stable softened-distance evaluation,
screening/zero-displacement policy, accumulation, and error publication.
This module owns the scalar pair energy identity and the derivative algebra
used to project that energy into equal-and-opposite Cartesian forces.
"""

from __future__ import annotations

from generativeqc_compiler.tensor.ad_program import VJPProgram, transpose_program
from generativeqc_compiler.tensor.ir import (
    Node,
    add,
    constant,
    divide,
    input_tensor,
    multiply,
    sqrt,
)
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import TensorSpec

from .radial_kernel_ad import cached_radial_weight

GFN2_EXTERNAL_POINT_CHARGE_FORCE_VERSION = "gfn2-external-point-charge-force-ir-v1"


def _input(name: str, *, differentiable: bool = False) -> Node:
    return input_tensor(
        name,
        TensorSpec((), role="input", differentiable=differentiable),
    )


def build_gfn2_external_point_charge_pair_primal() -> Program:
    """One softened shell/point electrostatic pair energy."""

    dx = _input("dx", differentiable=True)
    dy = _input("dy", differentiable=True)
    dz = _input("dz", differentiable=True)
    inverse_average_hardness = _input("inverse_average_hardness")
    shell_charge = _input("shell_charge")
    point_charge = _input("point_charge")

    softened_squared = add(
        multiply(dx, dx),
        multiply(dy, dy),
        multiply(dz, dz),
        multiply(inverse_average_hardness, inverse_average_hardness),
    )
    kernel = divide(constant(1), sqrt(softened_squared))
    pair_energy = multiply(multiply(shell_charge, point_charge), kernel)
    return Program(
        {"kernel": kernel, "pair_energy": pair_energy},
        provenance={
            "kind": "gfn2-external-point-charge-pair-primal",
            "version": GFN2_EXTERNAL_POINT_CHARGE_FORCE_VERSION,
            "source": "softened shell/external-point electrostatics",
        },
    )


def build_gfn2_external_point_charge_pair_vjp() -> VJPProgram:
    """Generate Cartesian dE/d(delta-R) for one softened shell/point pair."""

    return transpose_program(
        build_gfn2_external_point_charge_pair_primal(),
        ("pair_energy",),
        inputs=("dx", "dy", "dz"),
    )


def build_gfn2_external_point_charge_force_weight_program() -> Program:
    """Factor the cached force weight from AD of the softened pair energy."""

    primal = build_gfn2_external_point_charge_pair_primal()
    weight = cached_radial_weight(primal, ("shell_charge", "point_charge"))
    return Program(
        {"weight": weight},
        provenance={
            "kind": "gfn2-external-point-charge-force-weight",
            "version": GFN2_EXTERNAL_POINT_CHARGE_FORCE_VERSION,
            "derived_from": build_gfn2_external_point_charge_pair_vjp().derivative_hash,
            "generated_from_primal": primal.logical_hash,
        },
    )


def build_gfn2_external_point_charge_force_projection_program() -> Program:
    """Project -dE/d(delta-R) into the force on the QM atom."""

    weight = _input("weight")
    dx = _input("dx")
    dy = _input("dy")
    dz = _input("dz")
    return Program(
        {
            "fx": multiply(weight, dx),
            "fy": multiply(weight, dy),
            "fz": multiply(weight, dz),
        },
        provenance={
            "kind": "gfn2-external-point-charge-force-projection",
            "version": GFN2_EXTERNAL_POINT_CHARGE_FORCE_VERSION,
            "derived_from": build_gfn2_external_point_charge_pair_vjp().derivative_hash,
            "sign": "force=-dE/dR",
        },
    )
