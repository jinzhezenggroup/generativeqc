"""Build-only physical WB97M-V J/K composition for native response owners.

The fitted map is the density VJP of the existing stationary DF energy, not
an occupied-factor approximation. Its argument may therefore be indefinite.
The source owner supplies raw three-centers and its actual whitening operator;
the inverse map respects that operator's out-auxiliary/source-auxiliary layout.
"""

from __future__ import annotations

from functools import cache

from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_df_second import df_second_programs
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
)


@cache
def fock_maps() -> dict[str, Program]:
    """Return exact/fitted spin maps and the shared physical source transform."""
    method = resolve_method("WB97M-V")
    short, long = method.primitives[1:3]
    orbital = IndexSpace("orbital", "ao", 2)
    auxiliary = IndexSpace("auxiliary", "auxiliary", 3)
    row, column = Index("row", orbital), Index("column", orbital)
    first, second = Index("first", auxiliary), Index("second", auxiliary)
    whitening = input_tensor("whitening", TensorSpec((first, second), role="parameter"))
    programs = {
        "inverse": Program({"inverse": einsum("pa,pb->ab", whitening, whitening)})
    }
    for prefix, spins, occupation in (("rks", 1, 2), ("uks", 2, 1)):
        spin = Index("spin", IndexSpace("spin", "spin", spins))
        matrix = (spin, row, column)

        def field(name: str, indices: tuple = matrix) -> Node:
            return input_tensor(name, TensorSpec(indices, role="parameter"))

        coulomb = field("coulomb", (row, column))
        exchange = field("exchange")
        programs[f"{prefix}_exact"] = Program(
            {
                "primary": add(
                    broadcast(coulomb, matrix, (1, 2)),
                    exchange,
                    coefficients=(1, -short.coefficient / occupation),
                )
            }
        )
        programs[f"{prefix}_fitted"] = df_second_programs(
            2, 3, spins, short.coefficient
        )[-1]
        programs[f"{prefix}_assemble"] = Program(
            {
                "fock": add(
                    field("primary"),
                    field("grid"),
                    field("long_range"),
                    broadcast(field("hcore", (row, column)), matrix, (1, 2)),
                    coefficients=(
                        1,
                        1,
                        -(long.coefficient - short.coefficient) / occupation,
                        1,
                    ),
                )
            }
        )
    return programs
