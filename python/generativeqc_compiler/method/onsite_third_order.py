"""Shared TensorIR owner for isotropic onsite third-order charge science."""

from __future__ import annotations

from typing import TYPE_CHECKING

from generativeqc_compiler.method.scalar_graph_ad import polynomial_partial
from generativeqc_compiler.tensor.ad_program import linearize
from generativeqc_compiler.tensor.ir import constant, divide, input_tensor, multiply
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import TensorSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

ONSITE_THIRD_ORDER_FP64_ORDER = "q2=q*q; potential=q2*gamma3; energy=(q2*q*gamma3)/3"


def build_onsite_third_order_primal(*, provenance: Mapping[str, object]) -> Program:
    """Build the actual energy and derive its factored potential with Graph AD."""

    charge_spec = TensorSpec((), role="input", differentiable=True)
    parameter_spec = TensorSpec((), role="input", differentiable=False)
    charge = input_tensor("charge", charge_spec)
    gamma3 = input_tensor("gamma3", parameter_spec)

    charge_squared = multiply(charge, charge)
    charge_cubed = multiply(charge_squared, charge)
    energy = divide(multiply(charge_cubed, gamma3), constant(3))
    potential = polynomial_partial(
        Program({"energy": energy}), "energy", "charge", ("charge", "charge", "gamma3")
    )
    return Program(
        {"energy": energy, "potential": potential},
        provenance=dict(provenance),
    )


def build_onsite_third_order_kernel(
    primal: Program, *, provenance: Mapping[str, object]
) -> Program:
    """Attach an AD witness that the energy derivative equals the potential."""

    jvp = linearize(primal, ("charge",), outputs=("energy",))
    definitions = tuple(dict.fromkeys([*primal.nodes, *jvp.program.nodes]))
    return Program(
        {
            "energy": primal.outputs["energy"],
            "potential": primal.outputs["potential"],
            "energy_charge_derivative": jvp.program.outputs["d_energy"],
        },
        definitions,
        provenance={
            **dict(provenance),
            "primal_logical_hash": primal.logical_hash,
            "charge_jvp_hash": jvp.derivative_hash,
            "generation": "TensorIR forward AD",
        },
    )
