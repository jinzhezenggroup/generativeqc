"""Build-only CPU lowering of canonical moving-AO feature/potential maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.stationary_grid_second import (
    feature_mixed_program,
    feature_programs,
)
from generativeqc_compiler.method.wb97mv_nuclear_grid import nuclear_grid_maps
from generativeqc_compiler.tensor import Program, linearize, rewrite, transpose_program
from generativeqc_compiler.tensor.host_affine import emit_affine_host

from tools.generate_df_kernels import write_if_changed


def generate(*, ordered_binary_contractions: bool = True) -> str:
    """Reuse generic AO contractions only, never WB97M-V field policies."""
    primal, first = feature_programs(3, 2, shared_contractions=True)
    labels = ("rho", "g0", "g1", "g2", "tau")
    return emit_affine_host(
        {
            "features": primal,
            "feature_direction": Program(
                {name: first.outputs[f"d_{name}"] for name in labels}
            ),
            "potential_direction": nuclear_grid_maps(shared_contractions=True)[
                "potential_direction"
            ],
        },
        namespace="generativeqc_host_grid_maps",
        extents={"point": "points", "ao": "nao"},
        ordered_binary_contractions=ordered_binary_contractions,
    )


def generate_curvature(*, ordered_binary_contractions: bool = True) -> str:
    """Frozen-D Cartesian GGA fields reuse both canonical AO JVP legs."""
    primal, _ = feature_programs(3, 2, shared_contractions=True)
    first = linearize(primal, ("ao0", "ao1", "ao2", "ao3")).program
    mixed = feature_mixed_program(
        3, 2, density_response=False, shared_contractions=True
    )
    labels = ("rho", "g0", "g1", "g2")
    fields = Program({name: primal.outputs[name] for name in labels})
    aos = ("ao0", "ao1", "ao2", "ao3")
    pullback = transpose_program(fields, labels, inputs=aos).program
    pullback_direction = linearize(pullback, aos).program
    response_inputs = ("density", *aos)
    first_relaxed = linearize(fields, response_inputs).program
    pullback_relaxed = linearize(pullback, response_inputs).program
    programs = {
        "features": fields,
        "first": Program({name: first.outputs[f"d_{name}"] for name in labels}),
        "mixed": Program({name: mixed.outputs[f"d_d_{name}"] for name in labels}),
        "pullback": pullback,
        "pullback_direction": pullback_direction,
        "first_relaxed": Program(
            {name: first_relaxed.outputs[f"d_{name}"] for name in labels}
        ),
        "pullback_direction_relaxed": pullback_relaxed,
    }
    return emit_affine_host(
        {name: rewrite(program, "exact_cse") for name, program in programs.items()},
        namespace="generativeqc_host_curvature_maps",
        extents={"point": "points", "ao": "nao"},
        ordered_binary_contractions=ordered_binary_contractions,
    )


def main() -> None:
    """Emit deterministic checked maps without importing the public runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--curvature", action="store_true")
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(
        arguments.output, generate_curvature() if arguments.curvature else generate()
    )


if __name__ == "__main__":
    main()
