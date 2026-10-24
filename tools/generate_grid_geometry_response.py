"""Generate native mixed Becke/AO geometry maps at build time only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.xc.grid_geometry_response import (
    ao_motion_program,
    local_program,
)

from tools.generate_df_kernels import write_if_changed


def generate() -> str:
    """Emit scalar jet AD and AO Taylor pull-through; no runtime discovery."""
    text = [
        "#pragma once",
        "#include <cuda_runtime.h>",
        "#include <cmath>",
        "namespace generativeqc_grid_geometry_ad {",
        "struct Jet { double value{}, left{}, right{}, mixed{}; };",
    ]
    for kind in (
        "norm",
        "ratio",
        "multiply",
        "add",
        "subtract",
        "complement",
        "treutler_adjust",
        *(f"becke_{count}" for count in range(1, 6)),
    ):
        graph, roots, names = local_program(kind)
        variables = {
            f"{prefix}{name}": f"{name}.{field}"
            for name in names
            for prefix, field in (
                ("", "value"),
                ("l", "left"),
                ("r", "right"),
                ("lr", "mixed"),
            )
        }
        emitter = CudaEmitter(graph, variables, coalesce_branches=True)
        emitter.emit(roots)
        parameters = ", ".join(f"Jet {name}" for name in names)
        text.extend(
            [
                f"__device__ inline Jet {kind}({parameters}) {{",
                *emitter.lines,
                "return {"
                + ", ".join(emitter.reference(root) for root in roots)
                + "};",
                "}",
            ]
        )
    graph, roots = ao_motion_program()
    variables = {
        **{f"x{axis}": "0.0" for axis in range(3)},
        **{f"jet{index}": f"jets[{index} * stride]" for index in range(20)},
        **{
            f"{name}{axis}": f"{name}[{axis}]"
            for name in ("left", "right")
            for axis in range(3)
        },
    }
    emitter = CudaEmitter(graph, variables)
    emitter.emit(roots)
    text.extend(
        [
            "__device__ inline void ao_motion(const double* jets, std::size_t stride,",
            " const double* left, const double* right, double* output) {",
            *emitter.lines,
        ]
    )
    for index, root in enumerate(roots):
        text.append(f"output[{index}] = {emitter.reference(root)};")
    text.extend(["}", "}"])
    return "\n".join(text) + "\n"


def main() -> None:
    """Only the build invokes this source generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, generate())


if __name__ == "__main__":
    main()
