"""Ahead-of-time scalar AD and feature/VJP maps for native WB97M-V response."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_THRESHOLD,
    MOLECULAR_VV10_WEIGHT_THRESHOLD,
)
from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.method.nonlocal_pair import nonlocal_grid_beta
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.wb97mv_grid_response import grid_maps, point_program
from generativeqc_compiler.method.wb97mv_nuclear_grid import (
    nonlocal_field_program,
    nuclear_grid_maps,
)
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda, emit_affine_work

from tools.generate_df_kernels import write_if_changed

if TYPE_CHECKING:
    from collections.abc import Iterator

    from generativeqc_compiler.tensor import Node, Program


def _potential_point_product(name: str, root: Node, *, moving: bool) -> str:
    """Lower the density VJP's sum of weighted AO products for one grid point.

    Distribute the point reduction across the sum, keeping the actual weights,
    slot order and scales from TensorIR. A changed graph must match this
    narrow contract or fail at generation rather than silently losing terms.
    """
    ao = [f"ao{axis}" for axis in range(4)]
    coefficients = ["bar_rho", "bar_g0", "bar_g1", "bar_g2", "bar_tau"]
    if moving:
        ao += [f"d_{name}" for name in ao]
        coefficients += [f"d_{name}" for name in coefficients]

    def additive_terms(node: Node, scale: Fraction) -> Iterator[tuple[Node, Fraction]]:
        if node.op == "add":
            for child, coefficient in zip(
                node.inputs, node.attrs["coefficients"], strict=True
            ):
                yield from additive_terms(child, scale * Fraction(*coefficient))
        else:
            yield node, scale

    def input_index(node: Node, names: list[str]) -> int:
        if node.op != "input" or node.attrs["name"] not in names:
            raise ValueError("unsupported AO potential operand")
        return names.index(node.attrs["name"])

    grouped: dict[tuple[int, Fraction], list[str]] = {}
    by_row: list[list[str]] = [[] for _ in ao]
    for term, scale in additive_terms(root, Fraction(1)):
        if (
            term.op != "einsum"
            or term.attrs["labels"] != ((0,), (0, 1), (0, 2))
            or term.attrs["output"] != (1, 2)
        ):
            raise ValueError("unsupported AO potential contraction")
        weights, row, column = term.inputs
        row_slot = input_index(row, ao)
        column_slot = input_index(column, ao)
        product = f"row[{row_slot}] * column[{column_slot}]"
        for weight, coefficient in additive_terms(weights, scale):
            key = (
                input_index(weight, coefficients),
                coefficient * Fraction(*term.attrs["coefficient"]),
            )
            grouped.setdefault(key, []).append(product)
            index, factor = key
            by_row[row_slot].append(
                f"({float(factor):.17g}) * weights[{index}] * column[{column_slot}]"
            )
    expression = "\n     + ".join(
        f"({float(scale):.17g}) * weights[{index}] * ({' + '.join(products)})"
        for (index, scale), products in grouped.items()
    )
    transformed = "\n".join(
        f"    result[{slot}] = {' + '.join(terms) or '0.0'};"
        for slot, terms in enumerate(by_row)
    )
    transition = ""
    if not moving:
        # Lower the same density adjoint on an occupied/virtual pair of MO jets.
        fields = [[] for _ in coefficients]
        for (index, scale), products in grouped.items():
            fields[index].append(f"({float(scale):.17g}) * ({' + '.join(products)})")
        assignments = "\n".join(
            f"    result[{index}] = {' + '.join(terms) or '0.0'};"
            for index, terms in enumerate(fields)
        )
        transition = (
            " __device__ static void transition(const double* row, const double* column, double* result) {\n"
            f"{assignments}\n }}\n"
        )
    return (
        f"struct {name} {{\n"
        f" static constexpr unsigned ao_count = {len(ao)}, weight_count = {len(coefficients)};\n"
        f" static constexpr unsigned transform_terms = {sum(map(len, by_row))};\n"
        " __device__ static double point(const double* row, const double* column, const double* weights) {\n"
        f"    return {expression};\n"
        " }\n"
        " __device__ static void right_transform(const double* column, const double* weights, double* result) {\n"
        f"{transformed}\n"
        " }\n"
        f"{transition}"
        "};"
    )


def _warp_features(
    name: str, program: Program, *, moving: bool, strips: int = 1
) -> str:
    """Keep the reassociated density contractions in one grid-point warp.

    Only lane-local linear combinations may move across the final AO sum.
    Retain each matrix's actual orientation, even for asymmetric inputs. The
    wider kernels give each lane one output from each strip of 32 AO rows.
    """
    wide = strips > 1
    matrices = {"density": "density", "d_density": "direction"}
    references = {}
    loads, accumulators, updates = [], [], []
    for index, node in enumerate(program.nodes):
        symbol = f"v{index}"
        references[node] = f"{symbol}[row]" if wide else symbol
        if node.op == "input":
            operand = node.attrs["name"]
            if operand in matrices:
                continue
            if operand not in {
                f"{prefix}ao{axis}" for prefix in ("", "d_") for axis in range(4)
            }:
                raise ValueError("unsupported warp feature input")
            panel = "motion" if operand.startswith("d_") else "ao"
            axis = int(operand[-1])
            offset = f"({axis} * points + point) * nao"
            if wide:
                entries = [f"{panel}[{offset} + lane]"]
                for strip in range(1, strips):
                    row = f"lane + {32 * strip}"
                    entries.append(f"{row} < nao ? {panel}[{offset} + {row}] : 0")
                loads.append(
                    f"const double {symbol}[{strips}] = {{ {', '.join(entries)} }};"
                )
            else:
                loads.append(
                    f"const double {symbol} = lane < nao ? {panel}[{offset} + lane] : 0;"
                )
        elif node.op == "einsum":
            left, right = node.inputs
            if node.attrs["coefficient"] != (1, 1):
                raise ValueError("unsupported warp feature contraction scale")
            labels, output = node.attrs["labels"], node.attrs["output"]
            if labels == ((0, 1), (2, 1)) and output == (2, 0):
                if (
                    left.op != "input"
                    or left.attrs["name"] not in matrices
                    or right.op != "input"
                ):
                    raise ValueError("unsupported density-vector contraction")
                matrix = matrices[left.attrs["name"]]
                if wide:
                    accumulators.append(f"double {symbol}[{strips}]{{}};")
                    operand = references[right].removesuffix("[row]")
                    updates.append(
                        f"const double rhs{index} = __shfl_sync(0xffffffffu, {operand}[column / 32], column % 32);"
                    )
                    updates.extend(
                        f"{symbol}[{strip}] += {matrix}_element_{strip} * rhs{index};"
                        for strip in range(strips)
                    )
                else:
                    accumulators.append(f"double {symbol} = 0;")
                    updates.append(
                        f"{symbol} += {matrix}_element * __shfl_sync(0xffffffffu, {references[right]}, column);"
                    )
            elif labels == ((0, 1), (0, 1)) and output == (0,):
                references[node] = f"({references[left]} * {references[right]})"
            else:
                raise ValueError("unsupported warp feature contraction indices")
        elif node.op == "add":
            references[node] = (
                "("
                + " + ".join(
                    f"({float(Fraction(*scale)):.17g}) * {references[child]}"
                    for child, scale in zip(
                        node.inputs, node.attrs["coefficients"], strict=True
                    )
                )
                + ")"
            )
        else:
            raise ValueError("unsupported operation in warp grid feature program")
    stores = []
    for index, field in enumerate(("rho", "g0", "g1", "g2", "tau")):
        value = references[program.outputs[field]]
        if wide:
            value = " + ".join(
                value.replace("[row]", f"[{strip}]") for strip in range(strips)
            )
        stores.extend(
            (
                f"double sum{index} = {value};",
                (
                    "for (unsigned delta = 16; delta; delta /= 2) "
                    f"sum{index} += __shfl_down_sync(0xffffffffu, sum{index}, delta);"
                ),
                (
                    f"if (!lane) {{ fields[{index} * points + point] = sum{index};"
                    f" if (!isfinite(sum{index})) atomicExch(status, 1); }}"
                ),
            )
        )
    panels = tuple(matrices.values()) if moving else ("density",)
    if wide:
        prelude = ()
        load_matrices = []
        for panel in panels:
            for strip in range(strips):
                row = f"lane + {32 * strip}" if strip else "lane"
                value = f"{panel}[column * nao + {row}]"
                if strip:
                    value = f"{row} < nao ? {value} : 0"
                load_matrices.append(f"const double {panel}_element_{strip} = {value};")
    else:
        prelude = (
            *(f"__shared__ double {panel}_tile[32][33];" for panel in panels),
            "for (unsigned pair = threadIdx.x; pair < nao * nao; pair += blockDim.x) {",
            *(
                f"{panel}_tile[pair % nao][pair / nao] = {panel}[pair];"
                for panel in panels
            ),
            "}",
            "__syncthreads();",
        )
        load_matrices = tuple(
            f"const double {panel}_element = lane < nao ? {panel}_tile[column][lane] : 0;"
            for panel in panels
        )
    return "\n".join(
        (
            f"inline constexpr unsigned {name}_products = {len(accumulators)};",
            f"inline constexpr unsigned {name}_epilogue = {len(program.nodes) + 25};",
            f"__global__ void {name}(std::size_t points, unsigned nao, const double* ao,",
            " const double* motion, const double* density, const double* direction, double* fields, int* status) {",
            *prelude,
            "const unsigned lane = threadIdx.x % 32;",
            "const auto warps = blockDim.x / 32;",
            "for (std::size_t point = std::size_t(blockIdx.x) * warps + threadIdx.x / 32;",
            "     point < points; point += std::size_t(gridDim.x) * warps) {",
            *loads,
            *accumulators,
            "for (unsigned column = 0; column < nao; ++column) {",
            *load_matrices,
            *updates,
            "}",
            *stores,
            "}",
            "}",
        )
    )


def generate() -> str:
    """Emit immutable kernels; build-time imports never load a runtime or GPU."""
    fixed_features = grid_maps(shared_contractions=True)["features"]
    moving_features = nuclear_grid_maps(shared_contractions=True)["feature_direction"]
    text = [
        emit_affine_cuda(
            grid_maps(),
            namespace="generativeqc_wb97mv_grid_maps",
            extents={"point": "points", "ao": "nao"},
            fuse_elementwise=True,
            fuse_contraction_epilogues=True,
            parallel_scalar_reduction=True,
            parallel_output_reduction=True,
            # Every AO feature is a symmetric bilinear in the density. Use
            # the same rounded adjoint for both entries of each AO pair.
            symmetric_outputs={"potential": ("potential",)},
        ),
        emit_affine_work(
            grid_maps(),
            namespace="generativeqc_wb97mv_grid_maps",
            extents={"point": "points", "ao": "nao"},
        ),
        "namespace generativeqc_wb97mv_grid_ad {",
        _potential_point_product(
            "FixedPotential",
            grid_maps()["potential"].outputs["potential"],
            moving=False,
        ),
        _potential_point_product(
            "MovingPotential",
            nuclear_grid_maps()["potential_direction"].outputs["potential"],
            moving=True,
        ),
        _warp_features("fixed_features", fixed_features, moving=False),
        _warp_features("moving_features", moving_features, moving=True),
        _warp_features("wide_fixed_features", fixed_features, moving=False, strips=2),
        _warp_features("wide_moving_features", moving_features, moving=True, strips=2),
        _warp_features("large_fixed_features", fixed_features, moving=False, strips=3),
        _warp_features("large_moving_features", moving_features, moving=True, strips=3),
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
    for revised in (False, True):
        graph, roots = nonlocal_field_program("pair", runtime_rvv10=revised)
        roots = (*roots[1:5], *roots[10:14])
        emitter = CudaEmitter(
            graph,
            {**variables, "damping": "damping", "gradient_scale": "gradient_scale"},
        )
        emitter.emit(roots)
        name = "rvv10_pairs" if revised else "pairs"
        parameters = (
            " double damping, double gradient_scale, double coefficient,"
            if revised
            else ""
        )
        factor = "coefficient" if revised else f"{float(primitive.coefficient):.17g}"
        text.extend(
            [
                f"__global__ void {name}(const double* fields, const double* direction, const double* weights,",
                " const double* positions, std::size_t points, unsigned spins, const std::size_t* active,",
                " std::size_t row_begin, std::size_t column_begin, std::size_t rows, std::size_t columns,",
                f"{parameters} bool response, double* output, int* status) {{",
                " const auto lanes = rows * columns;",
                " for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;",
                "      lane += static_cast<std::size_t>(blockDim.x) * gridDim.x) {",
                " const auto first = active[row_begin + lane / columns];",
                " const auto second = active[column_begin + lane % columns];",
                *emitter.lines,
            ]
        )
        for index in range(4):
            text.extend(
                [
                    f" const double value_{index} = {factor} * (response ? {emitter.reference(roots[index + 4])} : {emitter.reference(roots[index])});",
                    f" output[{index} * lanes + lane] = value_{index};",
                    f" if (!isfinite(value_{index})) atomicExch(status, 1);",
                ]
            )
        text.extend([" }", "}"])
    beta = nonlocal_grid_beta(primitive.spec) * float(primitive.coefficient)
    threshold = float(MOLECULAR_VV10_DENSITY_THRESHOLD)
    text.extend(
        [
            f"inline constexpr double vv10_beta = {beta:.17g};",
            f"inline constexpr double vv10_density_threshold = {threshold:.17g};",
            f"inline constexpr double vv10_weight_threshold = {float(MOLECULAR_VV10_WEIGHT_THRESHOLD):.17g};",
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
