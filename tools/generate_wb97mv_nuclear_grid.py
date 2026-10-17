"""Build-time full moving-grid WB97M-V field partial and contraction kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.wb97mv_grid_response import point_field_partials
from generativeqc_compiler.method.wb97mv_nuclear_grid import (
    nonlocal_field_program,
    nuclear_grid_maps,
)
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda, emit_affine_work

from tools.generate_df_kernels import write_if_changed


def _leg_variables(side: str, point: str) -> dict[str, str]:
    """Lift total-density VV10 fields without duplicating the pair mathematics."""
    result = {}
    for index in range(9):
        if index < 5:
            for prefix, source in (("", "fields"), ("d", "direction")):
                result[f"{prefix}{side}{index}"] = (
                    f"({source}[{index} * points + {point}] + "
                    f"(spins == 2 ? {source}[({index} + 5) * points + {point}] : 0.0))"
                )
        else:
            source = "weights" if index == 5 else "positions"
            offset = point if index == 5 else f"3 * {point} + {index - 6}"
            result[f"{side}{index}"] = f"{source}[{offset}]"
            result[f"d{side}{index}"] = f"(response ? delta_{source}[{offset}] : 0.0)"
    return result


def generate() -> str:
    """Emit native runtime-sized maps; no numerical/runtime discovery imports."""
    programs = nuclear_grid_maps()
    extents = {"point": "points", "ao": "nao", "grid_value": "values"}
    text = [
        emit_affine_cuda(
            programs,
            namespace="generativeqc_wb97mv_nuclear_maps",
            extents=extents,
            fuse_elementwise=True,
            fuse_contraction_epilogues=True,
            parallel_scalar_reduction=True,
            parallel_output_reduction=True,
        ),
        emit_affine_work(
            programs, namespace="generativeqc_wb97mv_nuclear_maps", extents=extents
        ),
        "namespace generativeqc_wb97mv_nuclear_ad {",
    ]
    for spins in (1, 2):
        graph, roots = point_field_partials(spins)
        count = 5 * spins + 1
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
            "dweight": "(response ? delta_weights[point] : 0.0)",
        }
        emitter = CudaEmitter(graph, variables, coalesce_branches=True)
        emitter.emit(roots)
        text.extend(
            [
                f"__global__ void point_{spins}(const double* fields, const double* direction,",
                " const double* weights, const double* delta_weights, std::size_t points,",
                " bool response, double* partials, double* energy, int* status) {",
                " for (std::size_t point = blockIdx.x * blockDim.x + threadIdx.x; point < points;",
                "      point += static_cast<std::size_t>(gridDim.x) * blockDim.x) {",
                *emitter.lines,
            ]
        )
        for index in range(count):
            text.extend(
                [
                    f" const double value_{index} = response ? {emitter.reference(roots[1 + count + index])} : {emitter.reference(roots[1 + index])};",
                    f" partials[{index} * points + point] = value_{index};",
                    f" if (!isfinite(value_{index})) atomicExch(status, 1);",
                ]
            )
        text.extend(
            [
                f" for (unsigned field = {count}; field < {count + 3}; ++field) partials[field * points + point] = 0.0;",
                f" if (!response) {{ energy[point] = {emitter.reference(roots[0])}; if (!isfinite(energy[point])) atomicExch(status, 1); }}",
                " }",
                "}",
            ]
        )
    factor = float(resolve_method("WB97M-V").primitives[-1].coefficient)
    for kind in ("pair", "local"):
        graph, roots = nonlocal_field_program(kind)
        variables = _leg_variables("a", "first")
        if kind == "pair":
            variables.update(_leg_variables("b", "second"))
        emitter = CudaEmitter(graph, variables)
        emitter.emit(roots)
        if kind == "pair":
            text.extend(
                [
                    "__global__ void pairs(const double* fields, const double* direction,",
                    " const double* weights, const double* positions, const double* delta_weights,",
                    " const double* delta_positions, std::size_t points, unsigned spins, const std::size_t* active,",
                    " std::size_t row_begin, std::size_t column_begin, std::size_t rows, std::size_t columns,",
                    " bool response, double* output, int* status) {",
                    " const auto lanes = rows * columns;",
                    " for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;",
                    "      lane += static_cast<std::size_t>(blockDim.x) * gridDim.x) {",
                    " const auto first = active[row_begin + lane / columns];",
                    " const auto second = active[column_begin + lane % columns];",
                    *emitter.lines,
                    f" output[lane] = response ? 0.0 : {factor * 0.5:.17g} * {emitter.reference(roots[0])};",
                    " if (!isfinite(output[lane])) atomicExch(status, 1);",
                ]
            )
        else:
            text.extend(
                [
                    "__global__ void local(const double* fields, const double* direction,",
                    " const double* weights, const double* positions, const double* delta_weights,",
                    " const double* delta_positions, std::size_t points, unsigned spins, const std::size_t* active,",
                    " std::size_t count, bool response, double* partials, double* energy, int* status) {",
                    " for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < count;",
                    "      lane += static_cast<std::size_t>(blockDim.x) * gridDim.x) {",
                    " const auto first = active[lane];",
                    *emitter.lines,
                    f" if (!response) energy[first] += {factor:.17g} * {emitter.reference(roots[0])};",
                    " if (!response && !isfinite(energy[first])) atomicExch(status, 1);",
                ]
            )
        for index in range(9):
            text.append(
                f" const double value_{index} = {factor:.17g} * (response ? {emitter.reference(roots[10 + index])} : {emitter.reference(roots[1 + index])});"
            )
            if kind == "pair":
                text.append(f" output[{1 + index} * lanes + lane] = value_{index};")
            else:
                if index < 5:
                    text.append(
                        f" for (unsigned spin = 0; spin < spins; ++spin) partials[(5 * spin + {index}) * points + first] += value_{index};"
                    )
                else:
                    text.append(
                        f" partials[(5 * spins + {index - 5}) * points + first] += value_{index};"
                    )
            text.append(f" if (!isfinite(value_{index})) atomicExch(status, 1);")
        text.extend([" }", "}"])
    text.extend(["}", ""])
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, generate())


if __name__ == "__main__":
    main()
