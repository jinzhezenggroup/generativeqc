"""Build-only moving-grid WB97M-V energy/Fock/second-contraction AD maps.

All response equations are derivatives of the incumbent feature, potential and
VV10 energy authorities. Runtime owners supply geometry jets, fixed reference
screening, bounded traversal, current-device execution and source lifetime.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.method.nonlocal_pair import vv10_grid_local_expression
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    _mixed_program,
    _pair_program,
    feature_programs,
)
from generativeqc_compiler.method.wb97mv_grid_response import grid_maps
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    linearize,
    reduce_sum,
)


def nuclear_grid_maps(points: int = 3, nao: int = 2) -> dict[str, Program]:
    """First/mixed AO-feature maps and the moving physical density cotangent.

    Both the AO bilinear and its density VJP are differentiated with TensorIR.
    The mixed map retains a right density response but has zero left/mixed
    density leaves at execution, as required by the stationary nuclear partial.
    No point-by-AO-pair nuclear Jacobian or handwritten product rule is used.
    """
    _, first = feature_programs(points, nao)
    mixed = _mixed_program(points, nao)
    potential = grid_maps(points, nao)["potential"]
    replacements = {}
    for node in potential.nodes:
        replacements[node] = replace(
            node,
            inputs=tuple(replacements[value] for value in node.inputs),
            spec=replace(node.spec, differentiable=True),
        )
    potential = Program(
        {name: replacements[value] for name, value in potential.outputs.items()}
    )
    names = tuple(node.attrs["name"] for node in potential.nodes if node.op == "input")
    moving = linearize(potential, names).program
    labels = ("rho", "g0", "g1", "g2", "tau")
    point = Index("point", IndexSpace("point", "batch", points))
    energy = input_tensor("energy", TensorSpec((point,), role="input"))
    value = Index("value", IndexSpace("grid_value", "batch", 27))
    vectors = [
        input_tensor(name, TensorSpec((value,), role="input"))
        for name in ("left", "mixed", "base_partial", "right_partial")
    ]
    left, second, base, right = vectors
    return {
        "feature_direction": Program(
            {name: first.outputs[f"d_{name}"] for name in labels}
        ),
        "feature_mixed": Program(
            {name: mixed.outputs[f"d_d_{name}"] for name in labels}
        ),
        "potential_direction": Program({"potential": moving.outputs["d_potential"]}),
        "energy_sum": Program({"energy": reduce_sum(energy, (0,))}),
        "mixed_contraction": Program(
            {
                "value": add(
                    einsum("p,p->", left, right),
                    einsum("p,p->", second, base),
                )
            }
        ),
    }


@cache
def nonlocal_field_program(kind: str) -> tuple:
    """Full energy/first-leg partials/JVP, including both VV10 source motions."""
    primitive = resolve_method("WB97M-V").primitives[-1]
    if kind == "pair":
        return _pair_program(primitive.spec)
    if kind != "local":
        raise ValueError("unknown WB97M-V nonlocal field program")
    graph = Graph()
    leg = [graph.variable(f"a{index}") for index in range(9)]
    energy = vv10_grid_local_expression(graph, leg, primitive.spec)
    gradient = tuple(graph.differentiate(energy, value) for value in leg)
    leaves = {f"a{index}": graph.variable(f"da{index}") for index in range(9)}
    direction = tuple(
        graph.differentiate(value, graph.variable("direction"), leaves)
        for value in gradient
    )
    return graph, (energy, *gradient, *direction)
