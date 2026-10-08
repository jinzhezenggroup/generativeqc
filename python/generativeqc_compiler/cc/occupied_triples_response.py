"""Shared-AD reverse programs for the occupied-tile DF triples decomposition.

These graphs keep panel/moment derivatives in packed BLAS-sized coordinates.
They do not scatter a tile into a complete rank-six tensor. The native owner
must sum every occupied permutation, including repeated indices, and bind each
local cotangent to the corresponding strided physical view.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from generativeqc_compiler.tensor import Node

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    input_tensor,
    multiply,
    optimize,
    transpose_program,
)

from .occupied_triples import (
    df_panel_program,
    energy_scalar_program,
    inverse,
    moment_program,
)
from .triples import _LABELS, VP


def _pullback(primal: Program, inputs: Iterable[str] | None = None) -> Program:
    """Keep the shared reverse algebra while pruning unrequested output paths."""
    result = optimize(
        transpose_program(primal, tuple(primal.outputs), inputs=inputs).program
    )
    return Program(
        result.outputs,
        provenance={**result.provenance, "native_execution_order": "dependencies"},
    )


def panel_vjp(virtuals: int, auxiliaries: int) -> Program:
    """Two BLAS contractions from one accumulated occupied-panel cotangent.

    bar_bov_i addresses a strided Q-by-v view; bar_bvv accumulates across all
    occupied panels. Bvv's physical symmetric tangent is an upstream boundary,
    so this map returns its unprojected dense Frobenius cotangent.
    """
    return _pullback(df_panel_program(virtuals, auxiliaries))


def moment_vjp(occupied: int, virtuals: int, output: str) -> Program:
    """Four packed contractions for a W or V cube, without global scatters.

    W produces panel/t2_kj/ovoo_ij/t2_mk cotangents. V produces
    ovov_ij/t1_k/t2_ij/fov_k cotangents. Repeated physical views accumulate;
    arbitrary local seed arrays need not themselves have pair symmetry.
    """
    if output not in ("w", "v"):
        raise ValueError("occupied triples response requires W or V")
    primal = moment_program(occupied, virtuals)
    return _pullback(Program({output: primal.outputs[output]}))


def energy_scalar_vjp(inputs: Iterable[str] | None = None) -> Program:
    """Demand-driven scalar pullback of the exact occupied-triangle epilogue.

    The denominator input already contains the forward 6/2/1 multiplicity.
    The caller composes its derivative with gap * multiplicity exactly once.
    Virtual-coordinate permutations belong to the source binding: gathering
    their inverse maps avoids nondeterministic atomic accumulation into W.
    This scalar boundary alone is not a full same-space Fock response.
    """
    return _pullback(energy_scalar_program(), inputs)


def fused_tile_program() -> Program:
    """Share primal and gathered W/V seeds at one virtual-cube coordinate.

    Derivatives come exclusively from the existing energy AD graph. Each W
    cotangent gathers its six inverse-coordinate permutations before the native
    packed reverse contractions. Renaming scalar boundary inputs makes common
    subexpressions visible to ordinary TensorIR value numbering; it does not
    rewrite the audited triples algebra or reassociate an individual derivative.

    The six denominators remain distinct inputs: their mathematical symmetry
    does not justify changing the ordered FP64 virtual-energy subtraction. V
    inputs carry both occupied and virtual permutations for the same reason.
    Epsilon outputs are deliberately absent; fixed-canonical consumers needing
    them must retain the unfused all-output schedule.
    """
    primal = energy_scalar_program()
    names = tuple(
        node.attrs["name"]
        for node in primal.live_nodes
        if node.op == "input" and node.attrs["name"] != "denominator"
    )
    reverse = energy_scalar_vjp(names)
    labels: dict[tuple[int, ...], str] = {VP[label]: label for label in _LABELS}

    def bind(program: Program, coordinates: tuple[int, ...]) -> dict[str, Node]:
        replacements = {}
        label = labels[coordinates]
        for node in program.live_nodes:
            if node.op == "input":
                name = node.attrs["name"]
                if name == "denominator":
                    name += "_" + label
                elif name.startswith("w_"):
                    _, occupied, virtual = name.split("_")
                    mapped = tuple(coordinates[axis] for axis in VP[virtual])
                    virtual = labels[mapped]
                    name = f"w_{occupied}_{virtual}"
                elif name.startswith("v_"):
                    name += "_" + label
                replacements[node] = input_tensor(name, node.spec)
            else:
                replacements[node] = replace(
                    node, inputs=tuple(replacements[child] for child in node.inputs)
                )
        return {name: replacements[node] for name, node in program.outputs.items()}

    direct = bind(reverse, (0, 1, 2))
    outputs = {"energy": bind(primal, (0, 1, 2))["energy"]}
    gathered = {virtual: bind(reverse, inverse(VP[virtual])) for virtual in _LABELS}
    for occupied in _LABELS:
        outputs[f"bar_v_{occupied}"] = direct[f"bar_v_{occupied}"]
        outputs[f"bar_w_{occupied}"] = add(
            *(gathered[virtual][f"bar_w_{occupied}_{virtual}"] for virtual in _LABELS)
        )
    return optimize(
        Program(
            outputs,
            provenance={
                **primal.provenance,
                "schedule": "occupied-tile-primal-w-v-scalar-fusion-v1",
                "denominator_order": "separate-inverse-permutation-bindings",
                "w_gather_order": _LABELS,
                "gap_cotangents": "unrequested",
            },
        )
    )


def gap_vjp(virtuals: int) -> Program:
    """Reduce a gap-cube seed into three occupied scalars and one virtual vector.

    Keeping each occupied slot independent lets the native scatter account for
    repeated physical indices without atomics or a changed multiplicity.
    """
    if type(virtuals) is not int or virtuals < 1:
        raise ValueError("positive virtual extent required")
    vir = IndexSpace("occupied_response_virtual", "virtual", virtuals)
    axes = tuple(Index(x, vir) for x in "abc")
    ev = input_tensor(
        "eps_v", TensorSpec((axes[0],), role="parameter", differentiable=True)
    )
    occupied = [
        input_tensor("eps_" + x, TensorSpec((), role="parameter", differentiable=True))
        for x in "ijk"
    ]
    gap = add(
        *(broadcast(x, axes, ()) for x in occupied),
        *(broadcast(ev, axes, (axis,)) for axis in range(3)),
        coefficients=(1, 1, 1, -1, -1, -1),
    )
    return _pullback(Program({"gap": gap}))


def scaled_denominator_vjp() -> Program:
    """Differentiate gap times the forward occupied multiplicity exactly once."""
    gap, multiplicity = [
        input_tensor(x, TensorSpec((), role="parameter", differentiable=True))
        for x in ("gap", "multiplicity")
    ]
    return _pullback(Program({"denominator": multiply(gap, multiplicity)}), ("gap",))
