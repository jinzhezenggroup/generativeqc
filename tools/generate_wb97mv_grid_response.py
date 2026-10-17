"""Ahead-of-time scalar AD and feature/VJP maps for native WB97M-V response."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_THRESHOLD
from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.method.nonlocal_pair import vv10_grid_beta
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.stationary_grid_second import _pair_program
from generativeqc_compiler.method.wb97mv_grid_response import grid_maps, point_program
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda, emit_affine_work

from tools.generate_df_kernels import write_if_changed


def generate() -> str:
    """Emit immutable kernels; build-time imports never load a runtime or GPU."""
    text = [
        emit_affine_cuda(
            grid_maps(),
            namespace="generativeqc_wb97mv_grid_maps",
            extents={"point": "points", "ao": "nao"},
            fuse_elementwise=True,
            fuse_contraction_epilogues=True,
            parallel_scalar_reduction=True,
            parallel_output_reduction=True,
        ),
        emit_affine_work(
            grid_maps(),
            namespace="generativeqc_wb97mv_grid_maps",
            extents={"point": "points", "ao": "nao"},
        ),
        "namespace generativeqc_wb97mv_grid_ad {",
    ]
    for spins in (1, 2):
        graph, roots = point_program(spins)
        variables = {
            **{
                f"f{index}": f"fields[{index} * points + point]"
                for index in range(5 * spins)
            },
            **{
                f"d{index}": f"direction[{index} * points + point]"
                for index in range(5 * spins)
            },
            "weight": "weights[point]",
        }
        emitter = CudaEmitter(graph, variables, coalesce_branches=True)
        emitter.emit(roots)
        text.extend(
            [
                f"__global__ void point_{spins}(const double* fields, const double* direction,",
                " const double* weights, std::size_t points, bool response, double* coefficients, int* status) {",
                " for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x; point < points;",
                "      point += static_cast<std::size_t>(gridDim.x) * blockDim.x) {",
                *emitter.lines,
            ]
        )
        for index in range(5 * spins):
            text.extend(
                [
                    f" const double coefficient_{index} = response ? {emitter.reference(roots[index + 5 * spins])} : {emitter.reference(roots[index])};",
                    f" coefficients[{index} * points + point] = coefficient_{index};",
                    f" if (!isfinite(coefficient_{index})) atomicExch(status, 1);",
                ]
            )
        text.extend([" }", "}"])
    primitive = resolve_method("WB97M-V").primitives[-1]
    graph, roots = _pair_program(primitive.spec)
    roots = (*roots[1:5], *roots[10:14])
    variables = {}
    for side, point in (("a", "first"), ("b", "second")):
        for index in range(9):
            if index < 5:
                variables[f"{side}{index}"] = (
                    f"(fields[{index} * points + {point}] + (spins == 2 ? fields[({index} + 5) * points + {point}] : 0.0))"
                )
                variables[f"d{side}{index}"] = (
                    f"(direction[{index} * points + {point}] + (spins == 2 ? direction[({index} + 5) * points + {point}] : 0.0))"
                )
            else:
                variables[f"{side}{index}"] = (
                    f"weights[{point}]"
                    if index == 5
                    else f"positions[3 * {point} + {index - 6}]"
                )
                variables[f"d{side}{index}"] = "0.0"
    emitter = CudaEmitter(graph, variables)
    emitter.emit(roots)
    text.extend(
        [
            "__global__ void pairs(const double* fields, const double* direction, const double* weights,",
            " const double* positions, std::size_t points, unsigned spins, const std::size_t* active,",
            " std::size_t row_begin, std::size_t column_begin, std::size_t rows, std::size_t columns,",
            " bool response, double* output, int* status) {",
            " const auto lanes = rows * columns;",
            " for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;",
            "      lane += static_cast<std::size_t>(blockDim.x) * gridDim.x) {",
            " const auto first = active[row_begin + lane / columns];",
            " const auto second = active[column_begin + lane % columns];",
            *emitter.lines,
        ]
    )
    for index in range(4):
        factor = float(primitive.coefficient)
        text.extend(
            [
                f" const double value_{index} = {factor:.17g} * (response ? {emitter.reference(roots[index + 4])} : {emitter.reference(roots[index])});",
                f" output[{index} * lanes + lane] = value_{index};",
                f" if (!isfinite(value_{index})) atomicExch(status, 1);",
            ]
        )
    beta = vv10_grid_beta(primitive.spec) * float(primitive.coefficient)
    threshold = float(MOLECULAR_VV10_DENSITY_THRESHOLD)
    text.extend(
        [
            " }",
            "}",
            f"inline constexpr double vv10_beta = {beta:.17g};",
            f"inline constexpr double vv10_density_threshold = {threshold:.17g};",
            "}",
            "",
        ]
    )
    return "\n".join(text)


def main() -> None:
    """Write the linked native program, never a runtime/JIT artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, generate())


if __name__ == "__main__":
    main()
