"""Build-only physical WB97M-V J/K composition for native response owners.

The fitted map is the density VJP of the existing stationary DF energy, not
an occupied-factor approximation. Its argument may therefore be indefinite.
With exchange the owner retains a whitened AO-pair panel. For Coulomb alone it
retains raw values and applies the accepted whitening only to auxiliary charge
vectors, including when the metric is truncated at fixed geometry.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache

from generativeqc_compiler.method.stationary_df_second import (
    fitted_coulomb_energy,
    fitted_exchange_energy,
    solve_fitted_charge,
    symmetric_fitted_fock,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Node,
    Program,
    TensorSpec,
    add,
    broadcast,
    einsum,
    input_tensor,
    reduce_sum,
    transpose,
)
from generativeqc_compiler.tensor.optimize import prepare_for_backend


@cache
def fock_maps() -> dict[str, Program]:
    """Return exact/fitted spin maps and the shared physical source transform."""
    orbital = IndexSpace("orbital", "ao", 2)
    auxiliary = IndexSpace("auxiliary", "auxiliary", 3)
    row, column = Index("row", orbital), Index("column", orbital)
    first, second = Index("first", auxiliary), Index("second", auxiliary)
    whitening = input_tensor("whitening", TensorSpec((first, second), role="parameter"))
    raw = input_tensor(
        "three_center", TensorSpec((row, column, second), role="parameter")
    )
    programs = {"whiten": Program({"whitened": einsum("ijq,pq->ijp", raw, whitening)})}
    for prefix, spins, occupation in (("rks", 1, 2), ("uks", 2, 1)):
        spin = Index("spin", IndexSpace("spin", "spin", spins))
        matrix = (spin, row, column)

        def field(name: str, indices: tuple = matrix) -> Node:
            return input_tensor(name, TensorSpec(indices, role="parameter"))

        coulomb = field("coulomb", (row, column))
        exchange = field("exchange")
        exchange_coefficient = field("exchange_coefficient", ())
        programs[f"{prefix}_exact"] = Program(
            {
                "primary": add(
                    broadcast(coulomb, matrix, (1, 2)),
                    einsum(",sij->sij", exchange_coefficient, exchange),
                )
            }
        )
        density = input_tensor(
            "density", TensorSpec(matrix, role="input", differentiable=True)
        )
        whitened = field("whitened", (row, column, first))
        charge = einsum("ij,ijp->p", reduce_sum(density, (0,)), whitened)
        j_energy = fitted_coulomb_energy(charge, charge)
        k_energy = fitted_exchange_energy(
            density,
            whitened,
            whitened,
            -Fraction(1, 2 * occupation),
            field("exchange_fraction", ()),
        )
        programs[f"{prefix}_fitted"] = symmetric_fitted_fock(add(j_energy, k_energy))
        total = reduce_sum(density, (0,))
        # Project both legs onto symmetric AO coordinates without creating a
        # symmetrized copy of the three-center panel.
        symmetric_total = einsum(
            "ij->ij", add(total, transpose(total, (1, 0))), coefficient=Fraction(1, 2)
        )
        raw_charge = einsum("ij,ijq->q", symmetric_total, raw)
        fitted_charge = solve_fitted_charge(whitening, raw_charge)
        raw_fock = einsum("ijq,q->ij", raw, fitted_charge)
        sym_fock = einsum(
            "ij->ij",
            add(raw_fock, transpose(raw_fock, (1, 0))),
            coefficient=Fraction(1, 2),
        )
        programs[f"{prefix}_fitted_coulomb"] = Program(
            {"primary": broadcast(sym_fock, matrix, (1, 2))}
        )
        # Match the unsigned K panel produced by Direct. Assembly applies
        # the physical signed LR coefficient once for either provider.
        programs[f"{prefix}_fitted_range"] = symmetric_fitted_fock(
            fitted_exchange_energy(density, whitened, whitened, Fraction(1, 2), None)
        )
        programs[f"{prefix}_assemble"] = Program(
            {
                "fock": add(
                    field("primary"),
                    field("grid"),
                    einsum(
                        ",sij->sij", field("range_coefficient", ()), field("long_range")
                    ),
                    broadcast(field("hcore", (row, column)), matrix, (1, 2)),
                )
            }
        )
    return {
        name: prepare_for_backend(program, "cuda") for name, program in programs.items()
    }
