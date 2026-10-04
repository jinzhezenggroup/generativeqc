"""Occupied-domain views of the shared standard-(T) Fock resolvent.

X=PW/D and Y=R3(P(W+V/2))/D retain complete same-space Fock response,
including internal degeneracies. Fixed j>=k pages vary i and retain full
virtual cubes. Pair symmetry permits weight 2-delta_jk after integrating b,c;
no occupied-triangle multiplicity belongs in an individual resolvent vector.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    divide,
    einsum,
    input_tensor,
)

from .occupied_triples import inverse, occupied_label
from .triples import _LABELS, INVENTORY_HASH, OP, R3, SLOW_TABLE

if TYPE_CHECKING:
    from generativeqc_compiler.tensor import Node


def resolvent_scalar_program() -> Program:
    """Bind the audited pair projector and R3 to six occupied W/V cubes.

    w_OCC_VIR and v_OCC_VIR select a stored occupied seed and a virtual
    coordinate permutation. V is evaluated pointwise by the native binding.
    The inverse-transpose map follows the original virtual-page frontend;
    independent full-domain tests protect this change of storage coordinates.
    """
    scalar = TensorSpec((), role="parameter", differentiable=True)
    ws, zs = {}, {}
    for occ in _LABELS:
        for vir in _LABELS:
            key = occ, vir
            ws[key] = input_tensor(f"w_{occ}_{vir}", scalar)
            zs[key] = add(
                ws[key],
                input_tensor(f"v_{occ}_{vir}", scalar),
                coefficients=(1, Fraction(1, 2)),
            )

    def pair_sum(values: dict[tuple[str, str], Node], outer: tuple[int, ...]) -> Node:
        return add(
            *(
                values[occupied_label(outer, OP[order]), label]
                for label, order in SLOW_TABLE["abc"]
            )
        )

    gap = input_tensor("gap", scalar)
    return Program(
        {
            "x": divide(pair_sum(ws, (0, 1, 2)), gap),
            "y": divide(
                add(
                    *(pair_sum(zs, inverse(perm)) for _, perm in R3),
                    coefficients=tuple(coefficient for coefficient, _ in R3),
                ),
                gap,
            ),
        },
        provenance={
            "method": "RCCSD(T)",
            "inventory_hash": INVENTORY_HASH,
            "scope": "occupied-domain separable Fock resolvent vectors",
        },
    )


def moment_program(virtuals: int, capacity: int, block: str) -> Program:
    """Two direct products for a virtual moment or a cross-occupied-page block.

    The full response sums these products with half the occupied-pair weight.
    oo uses a negative sign; vv a positive sign. Keeping the products separate
    lets the native BLAS owner use beta accumulation without a temporary
    matrix sum. Padding is zero and must not contribute to either product.
    """
    if any(type(x) is not int or x < 1 for x in (virtuals, capacity)):
        raise ValueError("positive virtual/page sizes required")
    if block not in ("oo", "vv"):
        raise ValueError("occupied Fock moment block must be oo or vv")
    vir = IndexSpace("occupied_fock_virtual", "virtual", virtuals)
    page = IndexSpace("occupied_fock_page", "batch", capacity)
    a, b, c = (Index(label, vir) for label in "abc")
    p = Index("p", page)
    spec = TensorSpec(
        (p, a, b, c) if block == "oo" else (a, b, c),
        role="parameter",
        differentiable=True,
    )
    x, y = (input_tensor(name, spec) for name in ("x_left", "y_left"))
    if block == "oo":
        xr, yr = (input_tensor(name, spec) for name in ("x_right", "y_right"))
        xy, yx = einsum("pabc,qabc->pq", x, yr), einsum("pabc,qabc->pq", y, xr)
    else:
        xy, yx = einsum("abc,dbc->ad", x, y), einsum("abc,dbc->ad", y, x)
    return Program(
        {"xy": xy, "yx": yx},
        provenance={
            "inventory_hash": INVENTORY_HASH,
            "scope": "occupied-domain separable Fock moment products",
            "block": block,
        },
    )
