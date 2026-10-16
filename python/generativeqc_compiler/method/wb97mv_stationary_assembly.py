"""Build-only composition of native stationary WB97M-V nuclear sources.

Primitive integral/grid/ECP differentiation and CPKS retain their existing
authorities. These maps supply physical spin totals, frozen nuclear Fock RHS,
signed pair incidence and final source addition, never a second Hessian algebra.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    input_tensor,
    multiply,
    reduce_sum,
)
from generativeqc_compiler.tensor.optimize import prepare_for_backend

if TYPE_CHECKING:
    from generativeqc_compiler.tensor.ir import Node


@cache
def assembly_maps(
    nao: int = 3, spins: int = 2, coordinates: int = 6, pairs: int = 1
) -> dict[str, Program]:
    """Runtime-sized physical sums and pair packing with explicit shared axes.

    Nuclear pair fields use [xyz,pair], matching the linked scalar-array ABI.
    W and dW are positive-sign CPKS convention; stationary overlap cotangents
    are their negative spin totals. RKS D already includes occupation two.
    """
    if any(type(value) is not int for value in (nao, spins, coordinates, pairs)) or (
        nao < 1
        or spins not in (1, 2)
        or coordinates < 3
        or coordinates % 3
        or pairs < 0
    ):
        raise ValueError("invalid stationary assembly extents")
    orbital = IndexSpace("assembly_ao", "ao", nao)
    row, column = Index("row", orbital), Index("column", orbital)
    spin = Index("spin", IndexSpace("assembly_spin", "spin", spins))
    coordinate = Index(
        "coordinate", IndexSpace("assembly_coordinate", "component", coordinates)
    )
    pair = Index("pair", IndexSpace("assembly_pair", "pair", pairs))
    axis = Index("axis", IndexSpace("assembly_axis", "cartesian", 3))
    neighbor = Index(
        "neighbor", IndexSpace("assembly_neighbor", "atom", coordinates // 3 - 1)
    )

    def field(name: str, *indices: Index) -> Node:
        return input_tensor(
            name, TensorSpec(indices, role="input", differentiable=True)
        )

    density = field("spin_density", spin, row, column)
    weights = field("spin_weights", spin, row, column)
    shared = add(
        *(
            field(name, row, column)
            for name in (
                "core_direction",
                "ecp_local_direction",
                "ecp_nonlocal_direction",
            )
        )
    )
    rhs = add(
        broadcast(shared, density.spec.indices, (1, 2)),
        *(
            field(name, spin, row, column)
            for name in ("exact_direction", "fitted_direction", "xc_direction")
        ),
    )
    vectors = tuple(
        field(name, coordinate)
        for name in (
            "core",
            "exact",
            "fitted",
            "ecp_local",
            "ecp_nonlocal",
            "grid",
            "nuclear",
        )
    )
    first_position, second_position = (
        field(name, axis, pair) for name in ("first_position", "second_position")
    )
    first_motion, second_motion = (
        field(name, axis, pair) for name in ("first_motion", "second_motion")
    )
    first_charge, second_charge = (
        field(name, pair) for name in ("first_charge", "second_charge")
    )
    pair_value = field("pair_value", axis, pair)
    programs = {
        "weights": Program(
            {
                "core": reduce_sum(density, (0,)),
                "overlap": add(reduce_sum(weights, (0,)), coefficients=(-1,)),
            }
        ),
        "rhs": Program({"rhs": rhs}),
        "hvp": Program({"hvp": add(*vectors)}),
        "nuclear_inputs": Program(
            {
                "charge_product": multiply(first_charge, second_charge),
                "displacement": add(
                    first_position, second_position, coefficients=(1, -1)
                ),
                "motion": add(first_motion, second_motion, coefficients=(1, -1)),
            }
        ),
        "nuclear_second": Program({"second": add(pair_value, coefficients=(-1,))}),
        "nuclear_atom": Program(
            {"atom": reduce_sum(field("incidences", axis, neighbor), (1,))}
        ),
    }
    return {
        name: prepare_for_backend(program, "cuda") for name, program in programs.items()
    }
