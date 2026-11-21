"""Emit runtime-sized native CUDA fitted-primary VJP/JVP maps at build time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method.wb97mv_df_response import df_response_maps
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda, emit_affine_work

from tools.generate_df_kernels import write_if_changed


def generate() -> str:
    """Generation imports no public runtime, GPU libraries or reference engine."""
    programs = df_response_maps()
    namespace = "generativeqc_wb97mv_df_maps"
    extents = {
        "orbital": "nao",
        "auxiliary": "naux",
        "spin": "spins",
        "symmetric_pair": "npair",
    }
    schedule = {
        "namespace": namespace,
        "extents": extents,
        "tiled_matrix_products": True,
    }
    # The spectral Gram only needs raw legs, so finish it before evaluating
    # fitted legs. In the implicit full-rank schedule, consume the residual
    # in the frozen Fock first, before building the dressed fitted tangent.
    response_order = {}
    for name, graph in programs.items():
        if "weight_direction_inverse" in graph.outputs:
            response_order[name] = ("weight_direction_inverse",)
        elif "weight_direction_metric" in graph.outputs and "coulomb" not in name:
            response_order[name] = ("fock_direction",)
    return (
        "\n".join(
            (
                emit_affine_cuda(
                    programs,
                    reuse_intermediate_storage=True,
                    inplace_add=True,
                    first_outputs=response_order,
                    **schedule,
                ),
                emit_affine_work(programs, **schedule),
            )
        )
        + "\n"
    )


def main() -> None:
    """Only explicitly selected CMake source generation calls this entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, generate())


if __name__ == "__main__":
    main()
