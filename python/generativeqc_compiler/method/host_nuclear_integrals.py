"""Canonical projected first-integral and global-hybrid quartet Fock JVPs."""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.method.ordered_quartet import density_incidence
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    linearize,
    transpose_program,
)

if TYPE_CHECKING:
    from generativeqc_compiler.integral.expr import Expr
    from generativeqc_compiler.method.spec import MethodIR

INCIDENCES = ("coulomb_left", "coulomb_right", "exchange_left", "exchange_right")
POSITIONS = ((0, 1), (2, 3), (0, 2), (1, 3))


@cache
def integral_direction_program() -> tuple[Graph, tuple[Expr, ...]]:
    """Apply fixed projection and nuclear-slot motion to existing integral AD.

    Independent mathematical-center gradients come from the linked integral
    authority. Atom-slot binding and normalization are supplied by the native
    owner; repeated atoms must not merge independent slots before this action.
    """
    graph = Graph()
    projection = graph.variable("projection")
    gradient = [graph.variable(f"g{index}") for index in range(12)]
    direction = [graph.variable(f"u{index}") for index in range(12)]
    return graph, (
        projection
        * graph.sum(
            first * moving for first, moving in zip(gradient, direction, strict=True)
        ),
    )


@cache
def quartet_energy_program(method: MethodIR) -> Program:
    """Share the canonical ordered J/K energy between first and second sources."""
    if {primitive.kind for primitive in method.primitives} != {
        "semilocal_xc",
        "exact_exchange",
    }:
        raise ValueError("native quartet direction requires a full-range global hybrid")
    plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    return Program(
        {
            "energy": add(
                *(
                    density_incidence(
                        plan.integral_block(
                            source, terms=1, differentiate_densities=True
                        ).objective,
                        exchange=exchange,
                    )
                    for source, exchange in (
                        ("coulomb", False),
                        ("exact_exchange", True),
                    )
                )
            )
        }
    )


@cache
def integral_weight_program(
    method: MethodIR, source: str, *, differentiate_densities: bool = False
) -> Program:
    """Generate frozen D/W weights before traversing normalized primitive jets.

    Native consumers supply ordered public-AO incidences. All spin, occupation,
    exchange and overlap-constraint signs remain in the stationary objective.
    The quartet integral VJP exposes one energy weight, not Fock incidences.
    """
    if source == "quartet":
        reverse = transpose_program(
            quartet_energy_program(method), ("energy",), inputs=("integrals",)
        ).program
        return Program({"weights": reverse.outputs["bar_integrals"]})
    if source not in ("one_electron", "overlap_pulay"):
        raise ValueError("unknown frozen nuclear integral weight source")
    plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    return plan.integral_block(
        source, terms=1, differentiate_densities=differentiate_densities
    ).weights


@cache
def integral_weight_response_program(method: MethodIR, source: str) -> Program:
    """Pair canonical D/W weights with their explicitly supplied right JVP.

    The left nuclear derivative still holds D/W fixed. Only the right action
    moves density incidences; cotangents are intentionally not differentiated.
    This generates no SCF response and does not certify supplied D/W tangents.
    """
    primal = integral_weight_program(method, source, differentiate_densities=True)
    inputs = tuple(
        node.attrs["name"]
        for node in primal.nodes
        if node.op == "input" and node.spec.differentiable
    )
    tangent = linearize(primal, inputs).program
    return Program(
        {
            "weights": primal.outputs["weights"],
            "weights_right": tangent.outputs["d_weights"],
        }
    )


@cache
def quartet_direction_program(method: MethodIR) -> Program:
    """Differentiate the incumbent ordered Coulomb/exact-exchange objective.

    The Fock is a density VJP of the shared stationary energy, then a JVP only
    in the integral. Density stays fixed; spin/occupation/exchange factors come
    from MethodIR, never from a second native Fock equation.
    The sole integral tangent is linear by JVP construction. A coordinate-panel
    consumer may evaluate it at unit tangent and reuse these compiler-owned
    coefficients for every physical partial of that same integral quartet.
    """
    plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    objective = quartet_energy_program(method)
    reverse = transpose_program(objective, ("energy",), inputs=INCIDENCES).program
    fock = Program({name: reverse.outputs[f"bar_{name}"] for name in INCIDENCES})
    moving = linearize(fock, ("integrals",)).program
    return Program(
        {
            f"motion_{slot}": moving.outputs[f"d_{name}"]
            for slot, name in enumerate(INCIDENCES)
        },
        provenance={
            "stationary_plan": plan.identity,
            "source": "full-range-frozen-integral-direction",
        },
    )


@cache
def integral_density_response_program(method: MethodIR) -> Program:
    """Contract supplied D'/W' with coordinate first sources, not a new solve.

    Commuting the density VJP and nuclear JVP of the same ordered quadratic
    objective gives the complete moving-D integral term <D', F'_integral>.
    F' already contains the compiler's spin/exchange factors. The canonical
    overlap-constraint objective supplies the W' term without duplicating its
    sign or occupation convention. Full ordered symmetric AO pairs flatten to
    the existing ordered_terms domain; no AO4 derivative storage is needed.
    """
    plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    metric = plan.integral_block("overlap_pulay", terms=1).objective
    renamed = {}
    for node in metric.nodes:
        attrs = dict(node.attrs)
        if node.op == "input":
            attrs["name"] = {
                "integrals": "overlap_direction",
                "weighted_density": "weighted_density_right",
            }.get(attrs["name"], attrs["name"])
        renamed[node] = replace(
            node,
            inputs=tuple(renamed[value] for value in node.inputs),
            attributes=tuple(attrs.items()),
        )
    spin = Index("s", IndexSpace("density_spin", "spin", plan.spin_blocks))
    pair = Index("t", IndexSpace("ordered_terms", "batch", 1))
    spec = TensorSpec((spin, pair), role="parameter", differentiable=True)
    moving = einsum(
        "st,st->",
        input_tensor("density_right", spec),
        input_tensor("integral_fock_direction", spec),
    )
    return Program(
        {"curvature": add(moving, renamed[metric.outputs["energy"]])},
        provenance={
            "stationary_plan": plan.identity,
            "source": "integral-moving-density-coordinate-contraction",
            "fock_jvp": quartet_direction_program(method).logical_hash,
        },
    )
