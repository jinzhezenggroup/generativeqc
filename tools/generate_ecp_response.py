"""Emit native residual ECP response maps and fixed two-grid constants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.integral.ecp_fixed_grid import ecp_fixed_quadratures
from generativeqc_compiler.integral.ecp_projector import emit_ecp_quadrature_cpp
from generativeqc_compiler.method.ecp_response import (
    ecp_ao_motion,
    ecp_layer_maps,
    ecp_total_density,
)
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda, emit_affine_work

from tools.generate_df_kernels import write_if_changed


def generate_maps() -> str:
    """Reuse build-only AD; no GPU, public runtime, reference engine or JIT."""
    programs = ecp_layer_maps()
    namespace = "generativeqc_ecp_response_maps"
    extents = {"point": "points", "ao": "nao"}
    text = [
        emit_ecp_quadrature_cpp(),
        emit_affine_cuda(programs, namespace=namespace, extents=extents),
        emit_affine_work(programs, namespace=namespace, extents=extents),
        "namespace generativeqc_ecp_response_maps {",
    ]
    graph, roots = ecp_ao_motion()
    emitter = CudaEmitter(
        graph,
        {
            **{f"x{axis}": "0.0" for axis in range(3)},
            **{f"jet{index}": f"jets[{index} * stride]" for index in range(10)},
            **{
                f"{name}{axis}": f"{name}[{axis}]"
                for name in ("left", "right")
                for axis in range(3)
            },
        },
    )
    emitter.emit(roots)
    text.extend(
        (
            "__device__ inline void ao_motion(const double* jets, std::size_t stride,",
            " const double* left, const double* right, double* output) {",
            *emitter.lines,
            *(
                f"output[{index}] = {emitter.reference(root)};"
                for index, root in enumerate(roots)
            ),
            "}",
            "}",
        )
    )
    return "\n".join(text) + "\n"


def generate_density_maps() -> str:
    """Emit the KS-only spin sum separately to keep CUDA definitions unique."""
    programs = {"total": ecp_total_density()}
    namespace = "generativeqc_ecp_density_maps"
    extents = {"orbital": "nao", "spin": "spins"}
    return (
        "\n".join(
            (
                emit_affine_cuda(programs, namespace=namespace, extents=extents),
                emit_affine_work(programs, namespace=namespace, extents=extents),
            )
        )
        + "\n"
    )


def generate_quadrature() -> str:
    """Host read-only constants are staged verbatim; no runtime root solving."""
    text = [
        "#pragma once",
        "#include <cstddef>",
        "namespace generativeqc_ecp_quadrature {",
        "struct Grid { std::size_t radial_count, points;",
        " const double *radial, *sphere, *angular, *projector; };",
    ]
    grids = ecp_fixed_quadratures()
    for index, grid in enumerate(grids):
        for name, array in grid.items():
            text.append(f"inline constexpr double {name}_{index}[] = {{")
            values = array.ravel()
            for begin in range(0, len(values), 4):
                text.append(
                    ", ".join(repr(float(value)) for value in values[begin : begin + 4])
                    + ","
                )
            text.append("};")
    text.append("inline constexpr Grid grids[] = {")
    for index, grid in enumerate(grids):
        text.append(
            f"{{{len(grid['radial'])}, {len(grid['sphere'])}, radial_{index}, sphere_{index}, "
            f"angular_{index}, projector_{index}}},"
        )
    return "\n".join((*text, "};", "}", ""))


def main() -> None:
    """Only source builds invoke generation; runtime executes linked artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quadrature", type=Path, required=True)
    parser.add_argument("--density-output", type=Path, required=True)
    arguments = parser.parse_args()
    for path, source in (
        (arguments.output, generate_maps()),
        (arguments.quadrature, generate_quadrature()),
        (arguments.density_output, generate_density_maps()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        write_if_changed(path, source)


if __name__ == "__main__":
    main()
