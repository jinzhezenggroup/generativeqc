"""Generate native mixed Becke/AO geometry maps at build time only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.dft.ao import jet_indices
from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter
from generativeqc_compiler.xc.grid_center_pair import emit_center_pair_layout
from generativeqc_compiler.xc.grid_geometry_response import (
    ao_motion_program,
    local_program,
    lower_cuda_geometry_powers,
    prepared_center_norm_program,
)

from tools.generate_df_kernels import write_if_changed


def generate(backend: str = "cuda") -> str:
    """Emit scalar jet AD and AO Taylor pull-through; no runtime discovery."""
    if backend not in ("cpu", "cuda"):
        raise ValueError("grid geometry backend must be cpu or cuda")
    emitter_type = ScalarCEmitter if backend == "cpu" else CudaEmitter
    qualifier = "inline" if backend == "cpu" else "__device__ inline"
    text = [
        "#pragma once",
        "#include <cstddef>" if backend == "cpu" else "#include <cuda_runtime.h>",
        "#include <array>",
        "#include <cstddef>",
        "#include <cmath>",
        "#define GENERATIVEQC_GRID_HD "
        + ("__host__ __device__" if backend == "cuda" else ""),
        emit_center_pair_layout(),
        "#undef GENERATIVEQC_GRID_HD",
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
        if backend == "cuda":
            roots = lower_cuda_geometry_powers(kind, graph, roots, names)
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
        emitter = emitter_type(graph, variables, coalesce_branches=True)
        emitter.emit(roots)
        parameters = ", ".join(f"Jet {name}" for name in names)
        text.extend(
            [
                f"{qualifier} Jet {kind}({parameters}) {{",
                *emitter.lines,
                "return {"
                + ", ".join(emitter.reference(root) for root in roots)
                + "};",
                "}",
            ]
        )
    graph, roots, names = prepared_center_norm_program()
    variables = {"prepared_distance": "pair.distance[0]"}
    for index, name in enumerate(names):
        variables[name] = f"pair.distance[{index + 1}]"
        variables[f"l{name}"] = f"left[{index}]"
        variables[f"r{name}"] = f"right[{index}]"
    emitter = emitter_type(graph, variables)
    emitter.emit(roots)
    text.extend(
        [
            f"{qualifier} Jet prepared_center_norm(const generativeqc_grid_pair_geometry::CenterPair& pair,",
            " const double* left, const double* right) {",
            *emitter.lines,
            "return {" + ", ".join(emitter.reference(root) for root in roots) + "};",
            "}",
        ]
    )
    graph, roots = ao_motion_program()
    if backend == "cpu":
        roots = graph.replace_subexpressions(
            roots,
            {graph.variable(f"x{axis}"): graph.constant(0) for axis in range(3)},
        )
        graph, roots = graph.canonicalize_associative(roots)
    variables = {
        **{f"x{axis}": "0.0" for axis in range(3)},
        **{f"jet{index}": f"jets[{index} * stride]" for index in range(20)},
        **{
            f"{name}{axis}": f"{name}[{axis}]"
            for name in ("left", "right")
            for axis in range(3)
        },
    }
    emitter = emitter_type(graph, variables)
    emitter.emit(roots)
    text.extend(
        [
            f"{qualifier} void ao_motion(const double* jets, std::size_t stride,",
            " const double* left, const double* right, double* output) {",
            *emitter.lines,
        ]
    )
    for index, root in enumerate(roots):
        text.append(f"output[{index}] = {emitter.reference(root)};")
    text.append("}")
    if backend == "cuda":
        # Bind unit directions to the same Taylor jet ordering as ao_motion.
        lookup = {index: offset for offset, index in enumerate(jet_indices(3))}
        first, mixed = [], []
        for base in jet_indices(1):
            for left in range(3):
                shifted = tuple(
                    value + (axis == left) for axis, value in enumerate(base)
                )
                first.append(lookup[shifted])
                for right in range(3):
                    mixed.append(
                        lookup[
                            tuple(
                                value + (axis == right)
                                for axis, value in enumerate(shifted)
                            )
                        ]
                    )
        for name, indices, arguments, index in (
            ("cartesian_first", first, "unsigned jet, unsigned left", "3 * jet + left"),
            (
                "cartesian_mixed",
                mixed,
                "unsigned jet, unsigned left, unsigned right",
                "9 * jet + 3 * left + right",
            ),
        ):
            text.append(
                f"static __device__ __constant__ unsigned char {name}_indices[] = "
                + "{"
                + ", ".join(map(str, indices))
                + "};"
            )
            text.append(
                f"{qualifier} unsigned {name}_index({arguments}) {{ return {name}_indices[{index}]; }}"
            )
    text.append("}")
    return "\n".join(text) + "\n"


def main() -> None:
    """Only the build invokes this source generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(arguments.output, generate(arguments.backend))


if __name__ == "__main__":
    main()
