"""Build-only ordered-quartet WB97M-V stationary energy/Fock/source AD.

The incumbent stationary gradient plan owns spin factors and SR/LR coefficients.
The full-primary plus LR-correction decomposition is the same physical operator
used by native exact/fitted Fock response; no density positivity is assumed.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.tensor import (
    Node,
    Program,
    add,
    linearize,
    reshape,
    slice_tensor,
    transpose_program,
)

PARTS = ("primary", "correction", "coulomb", "short", "long")
SLOTS = ("coulomb_left", "coulomb_right", "exchange_left", "exchange_right")


def _rename_density(program: Program, exchange: bool) -> Node:
    """Rebind SSA density leaves to their distinct ordered quartet incidences."""
    prefix = "exchange" if exchange else "coulomb"
    mapped = {}
    for node in program.nodes:
        attrs = dict(node.attrs)
        if node.op == "input" and attrs["name"] in ("density_left", "density_right"):
            attrs["name"] = attrs["name"].replace("density", prefix)
        mapped[node] = replace(
            node,
            inputs=tuple(mapped[value] for value in node.inputs),
            attributes=tuple(attrs.items()),
        )
    return mapped[program.outputs["energy"]]


@lru_cache(maxsize=None, typed=True)
def quartet_program(spins: int, part: str) -> Program:
    """Scalar weight/JVP and energy-density VJP/JVP for one ordered quartet.

    Integral motion is differentiated only in the Fock JVP, holding D fixed.
    The weight JVP instead retains every right D response for stationary mixed
    nuclear contraction. Native owners must not conflate these two responses.
    The native source supplies the already projected integral and its motion;
    compiler AD integral leaves remain the sole nuclear derivative authority.
    """
    if type(spins) is not int or spins not in (1, 2) or part not in PARTS:
        raise ValueError("invalid WB97M-V quartet spin/source")
    plan = StationaryGradientPlan(
        resolve_method("WB97M-V", spin="unpolarized" if spins == 1 else "polarized"),
        StationaryMeanField(SCF_POINT_MODEL),
    )
    roots = []
    if part in ("primary", "coulomb"):
        roots.append(
            _rename_density(
                plan.integral_block(
                    "coulomb", terms=1, differentiate_densities=True
                ).objective,
                False,
            )
        )
    if part != "coulomb":
        source = (
            "exchange_short_range"
            if part in ("primary", "short")
            else "exchange_long_range"
        )
        block = plan.integral_block(
            source,
            terms=1,
            differentiate_densities=True,
            mixed_density_fitted_correction=part == "correction",
        )
        roots.append(_rename_density(block.objective, True))
    objective = Program({"energy": add(*roots) if len(roots) > 1 else roots[0]})
    density_names = tuple(
        name
        for name in SLOTS
        if any(
            node.op == "input" and node.attrs["name"] == name
            for node in objective.nodes
        )
    )
    reverse = transpose_program(
        objective, ("energy",), inputs=("integrals", *density_names)
    ).program
    weights = Program({"weight": reverse.outputs["bar_integrals"]})
    delta_weight = linearize(weights, density_names).program.outputs["d_weight"]
    fock = Program({name: reverse.outputs[f"bar_{name}"] for name in density_names})
    moving = linearize(fock, ("integrals",)).program
    outputs = {
        "energy": objective.outputs["energy"],
        "weight": weights.outputs["weight"],
        "delta_weight": delta_weight,
    }
    for slot, name in enumerate(SLOTS):
        if name not in density_names:
            continue
        for spin in range(spins):
            for label, root in (
                ("fock", fock.outputs[name]),
                ("motion", moving.outputs[f"d_{name}"]),
            ):
                outputs[f"{label}_{slot}_{spin}"] = reshape(
                    slice_tensor(root, ((spin, spin + 1), (0, 1))), ()
                )
    return Program(
        outputs, provenance={"stationary_plan": plan.identity, "quartet_part": part}
    )


def quartet_operator(part: str) -> tuple[str, float]:
    """Explicit radial selectors for the shared full+LR physical decomposition."""
    if part not in PARTS:
        raise ValueError("invalid WB97M-V quartet source")
    if part in ("primary", "coulomb"):
        return "full_range", 0.0
    primitive = resolve_method("WB97M-V").primitives[1]
    return ("short_range" if part == "short" else "long_range"), float(primitive.omega)
