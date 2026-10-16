"""Compiler AD of full-rank fitted J/K stationary weights and Fock response.

The inverse is an explicit physical source. Its pullback and tangent are the
full-rank implicit-solve rules, expressed in TensorIR; differentiating the
cotangents therefore includes metric response, not a frozen fitting metric.
Rank-truncated spectral maps are deliberately outside this contract.
"""

from fractions import Fraction
from functools import cache

from ..tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
    linearize,
    reduce_sum,
    transpose_program,
)


@cache
def df_second_programs(nao: int, naux: int, spins: int, exchange: Fraction) -> tuple:
    """Return J/K energy, source weights/JVPs, Fock JVP, and inverse tangent."""
    orbital = IndexSpace("orbital", "ao", nao)
    auxiliary = IndexSpace("auxiliary", "auxiliary", naux)
    spin = Index("spin", IndexSpace("spin", "spin", spins))
    row, column = Index("row", orbital), Index("column", orbital)
    first, second = Index("first", auxiliary), Index("second", auxiliary)

    def field(name: str, indices: tuple) -> Node:
        return input_tensor(
            name, TensorSpec(indices, role="input", differentiable=True)
        )

    density = field("density", (spin, row, column))
    values = field("three_center", (row, column, first))
    inverse = field("inverse", (first, second))
    metric_direction = field("metric_direction", (first, second))
    total = reduce_sum(density, (0,))
    charge = einsum("ij,ijp->p", total, values)
    fitted_charge = einsum("pq,q->p", inverse, charge)
    coulomb = einsum("p,p->", charge, fitted_charge, coefficient=Fraction(1, 2))
    fitted = einsum("ijp,pq->ijq", values, inverse)
    exchange_energy = einsum(
        "sik,sjl,ijp,klp->",
        density,
        density,
        values,
        fitted,
        coefficient=-exchange * (Fraction(1, 2) if spins == 2 else Fraction(1, 4)),
    )
    energy = Program({"coulomb": coulomb, "exchange": exchange_energy})
    source_programs = []
    for name, root in energy.outputs.items():
        reverse = transpose_program(
            Program({name: root}), (name,), inputs=("three_center", "inverse")
        ).program
        weights = Program(
            {
                "three_center": reverse.outputs["bar_three_center"],
                "metric": einsum(
                    "pa,ab,bq->pq",
                    inverse,
                    reverse.outputs["bar_inverse"],
                    inverse,
                    coefficient=-1,
                ),
            }
        )
        source_programs.append(
            (
                weights,
                linearize(weights, ("density", "three_center", "inverse")).program,
            )
        )
    total_energy = Program({"energy": add(coulomb, exchange_energy)})
    fock = transpose_program(total_energy, ("energy",), inputs=("density",)).program
    fock_direction = linearize(fock, ("three_center", "inverse")).program
    inverse_direction = Program(
        {
            "direction": einsum(
                "pa,ab,bq->pq", inverse, metric_direction, inverse, coefficient=-1
            )
        }
    )
    return energy, tuple(source_programs), fock_direction, inverse_direction, fock
