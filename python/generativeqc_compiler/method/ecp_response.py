"""Build-only residual ECP layer response from the stationary AO energy AD.

Local angular nodes and nonlocal projector rows share one fixed-measure AO
bilinear. Native owners supply normalized AO jets and fixed linear projections;
they do not implement a second ECP product rule or electronic response algebra.
"""

from __future__ import annotations

from functools import lru_cache

from generativeqc_compiler.method.stationary_grid_second import feature_programs
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
    linearize,
    reduce_sum,
    transpose_program,
)
from generativeqc_compiler.tensor.optimize import prepare_for_backend
from generativeqc_compiler.xc.grid_geometry_response import ao_motion_program


@lru_cache(maxsize=None, typed=True)
def ecp_layer_maps(points: int = 3, nao: int = 2) -> dict[str, Program]:
    """Physical matrix/motion and AO cotangents retaining signed right dD.

    The density is the full ordered total density, without an extra occupancy
    or half factor. The fixed-measure ECP-centered nodes/projectors depend on
    geometry only through the AO arguments. Reference AO cotangents and their
    joint density/AO JVP let a native source contract spatial jets once, rather
    than regenerating AOs or AO-pair work for every left nuclear coordinate.
    Matrix motion is a frozen-density partial, distinct from moving weights.
    """
    if type(points) is not int or type(nao) is not int or points < 1 or nao < 1:
        raise ValueError("ECP layer requires positive integer point/AO extents")
    primal, _ = feature_programs(points, nao)
    density = primal.outputs["rho"]
    measure = input_tensor("measure", TensorSpec(density.spec.indices, role="input"))
    objective = Program({"energy": einsum("p,p->", measure, density)})
    reverse = transpose_program(
        objective, ("energy",), inputs=("density", "ao0")
    ).program
    matrix = Program({"matrix": reverse.outputs["bar_density"]})
    matrix_motion = linearize(matrix, ("ao0",)).program
    weights = Program({"weights": reverse.outputs["bar_ao0"]})
    moving = linearize(weights, ("density", "ao0")).program
    return {
        name: prepare_for_backend(program, "cuda")
        for name, program in {
            "matrix": Program(
                {
                    "matrix": matrix.outputs["matrix"],
                    "direction": matrix_motion.outputs["d_matrix"],
                }
            ),
            "weights": Program(
                {
                    "weights": weights.outputs["weights"],
                    "direction": moving.outputs["d_weights"],
                }
            ),
        }.items()
    }


def ecp_ao_motion() -> tuple:
    """Only value left/right/mixed Taylor roots, requiring spatial jets 0..2.

    Reuse the existing generic AO Taylor AD; do not read order-three jets merely
    because the meta-GGA consumer also requests gradients of spatial gradients.
    """
    graph, roots = ao_motion_program()
    return graph, roots[:3]


def ecp_total_density(nao: int = 2, spins: int = 2) -> Program:
    """Sum physical spin densities and their signed JVP without occupancy factors."""
    if type(nao) is not int or nao < 1 or type(spins) is not int or spins not in (1, 2):
        raise ValueError(
            "ECP requires positive AO extent and one or two physical spin densities"
        )
    orbital = IndexSpace("orbital", "ao", nao)
    indices = (
        Index("spin", IndexSpace("spin", "spin", spins)),
        Index("row", orbital),
        Index("column", orbital),
    )
    density = input_tensor(
        "density", TensorSpec(indices, role="input", differentiable=True)
    )
    primal = Program({"total": reduce_sum(density, (0,))})
    tangent = linearize(primal, ("density",)).program
    return prepare_for_backend(
        Program(
            {"total": primal.outputs["total"], "direction": tangent.outputs["d_total"]}
        ),
        "cuda",
    )
