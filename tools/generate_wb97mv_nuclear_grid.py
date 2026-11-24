"""Build-time full moving-grid WB97M-V field partial and contraction kernels."""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.method.wb97mv_grid_response import point_field_partials
from generativeqc_compiler.method.wb97mv_nuclear_grid import (
    nonlocal_field_program,
    nuclear_grid_ao_adjoints,
    nuclear_grid_maps,
)
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda, emit_affine_work

from tools.generate_df_kernels import write_if_changed

if TYPE_CHECKING:
    from collections.abc import Iterator

    from generativeqc_compiler.tensor import Node, Program


def _ao_adjoint_terms(
    node: Node, scale: Fraction = Fraction(1)
) -> Iterator[tuple[Node, Fraction]]:
    """Flatten only exact rational linear combinations from the AO VJP."""
    if node.op == "add":
        for child, coefficient in zip(
            node.inputs, node.attrs["coefficients"], strict=True
        ):
            yield from _ao_adjoint_terms(child, scale * Fraction(*coefficient))
    else:
        yield node, scale


def _fused_ao_adjoints(programs: dict[str, Program]) -> str:
    """Factor both generated AO transposes into shared matrix-vector products.

    The two density orientations must occur with exactly equal rational
    coefficients for each output. Only then can one contraction apply the sum
    of both orientations. Signed/asymmetric matrices still work; a changed IR
    outside this lowering contract fails at generation rather than dropping terms.
    """
    layouts = {
        "feature_direction": {"bar_d_ao": "first"},
        "feature_mixed": {"bar_d_left_ao": "second", "bar_left_ao": "left"},
    }
    fields = ("rho", "g0", "g1", "g2", "tau")
    ao_inputs = {f"{prefix}ao{jet}" for prefix in ("", "d_") for jet in range(4)}
    contractions: dict[tuple[str, str], str] = {}
    coefficients = []
    terms = 0
    for name, layout in layouts.items():
        program = programs[name]
        if set(program.outputs) != {
            f"{stem}{axis}" for stem in layout for axis in range(4)
        }:
            raise ValueError("unsupported fused AO adjoint outputs")
        weight = "first_partial" if name == "feature_direction" else "base_partial"
        for stem, panel in layout.items():
            for axis in range(4):
                orientations = (defaultdict(Fraction), defaultdict(Fraction))
                for term, scale in _ao_adjoint_terms(program.outputs[f"{stem}{axis}"]):
                    if term.op != "einsum" or len(term.inputs) != 3:
                        raise ValueError("unsupported fused AO adjoint term")
                    field_weights, first, second = term.inputs
                    labels = term.attrs["labels"], term.attrs["output"]
                    if labels == (((0,), (1, 2), (0, 2)), (0, 1)):
                        matrix, orbital, side = first, second, 0
                    elif labels == (((0,), (0, 1), (1, 2)), (0, 2)):
                        matrix, orbital, side = second, first, 1
                    else:
                        raise ValueError("unsupported fused AO adjoint indices")
                    matrix_name = matrix.attrs.get("name")
                    orbital_name = orbital.attrs.get("name")
                    if (
                        matrix.op != "input"
                        or matrix_name not in ("density", "d_density")
                        or orbital.op != "input"
                        or orbital_name not in ao_inputs
                    ):
                        raise ValueError("unsupported fused AO adjoint operand")
                    for field, factor in _ao_adjoint_terms(field_weights):
                        field_name = field.attrs.get("name", "")
                        if (
                            field.op != "input"
                            or not field_name.startswith("bar_")
                            or field_name[4:] not in fields
                        ):
                            raise ValueError("unsupported fused AO adjoint field")
                        index = fields.index(field_name[4:])
                        orientations[side][matrix_name, orbital_name, index] += (
                            scale * factor * Fraction(*term.attrs["coefficient"])
                        )
                if orientations[0] != orientations[1]:
                    raise ValueError(
                        "unpaired density orientations in fused AO adjoint"
                    )
                expression = []
                for (matrix, orbital, field), coefficient in orientations[0].items():
                    if not coefficient:
                        continue
                    symbol = contractions.setdefault(
                        (matrix, orbital), f"product_{len(contractions)}"
                    )
                    expression.append(
                        f"({float(coefficient):.17g}) * {weight}[{field} * points + point] * {symbol}"
                    )
                terms += len(expression)
                value = f"{panel}_{axis}"
                coefficients.extend(
                    (
                        f"    const double {value} = {' + '.join(expression) or '0.0'};",
                        f"    if (!isfinite({value})) atomicExch(status, 1);",
                    )
                )
    # Both branches multiply the same left jet in the stationary contraction.
    # Combine them once for both the full-panel and bounded point-tile layouts.
    outputs = [
        line
        for axis in range(4)
        for line in (
            f"    const double combined_{axis} = first_{axis} + left_{axis};",
            f"    if (!isfinite(combined_{axis})) atomicExch(status, 1);",
            f"    first[{axis} * size + index] = combined_{axis};",
            f"    second[{axis} * tile_size + lane] = second_{axis};",
        )
    ]
    lines = [
        "namespace generativeqc_wb97mv_nuclear_ao_adjoints {",
        "inline bool work_fused_ao(Extents shape, std::size_t& work) {",
        "  work = 0;",
        f"  return add_work(work, {{shape.points, shape.nao, shape.nao, {len(contractions)}}}) &&",
        f"         add_work(work, {{shape.points, shape.nao, {terms + 4}}});",
        "}",
        "template <bool Tiled>",
        "__global__ void fused_ao(Extents shape, std::size_t begin, std::size_t count,",
        "                        const double* ao,",
        "                        const double* right, const double* density, const double* d_density,",
        "                        const double* first_partial, const double* base_partial,",
        "                        double* first, double* second, int* status) {",
        "  if (atomicAdd(status, 0)) return;",
        "  const auto points = shape.points, size = points * shape.nao;",
        "  const auto tile_size = Tiled ? count * shape.nao : size;",
        "  for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < tile_size;",
        "       lane += static_cast<std::size_t>(gridDim.x) * blockDim.x) {",
        "    const auto index = Tiled ? lane + begin * shape.nao : lane;",
        "    const auto point = index / shape.nao, orbital = index % shape.nao;",
        *(f"    double {symbol} = 0;" for symbol in contractions.values()),
        "    for (std::size_t inner = 0; inner < shape.nao; ++inner) {",
        "      const auto position = point * shape.nao + inner;",
        "      const auto row = orbital * shape.nao + inner, column = inner * shape.nao + orbital;",
        "      const double base = density[row] + density[column];",
        "      const double delta = d_density[row] + d_density[column];",
    ]
    for (matrix, orbital), symbol in contractions.items():
        operand = "right" if orbital.startswith("d_") else "ao"
        matrix_value = "delta" if matrix == "d_density" else "base"
        lines.append(
            f"      {symbol} += {matrix_value} * {operand}[{orbital[-1]} * size + position];"
        )
    return "\n".join([*lines, "    }", *coefficients, *outputs, "  }", "}", "}"])


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
    adjoints = nuclear_grid_ao_adjoints()
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
        emit_affine_cuda(
            adjoints,
            namespace="generativeqc_wb97mv_nuclear_ao_adjoints",
            extents={"point": "points", "ao": "nao"},
            fuse_elementwise=True,
            fuse_contraction_epilogues=True,
        ),
        emit_affine_work(
            adjoints,
            namespace="generativeqc_wb97mv_nuclear_ao_adjoints",
            extents={"point": "points", "ao": "nao"},
        ),
        _fused_ao_adjoints(adjoints),
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
    canonical = float(resolve_method("WB97M-V").primitives[-1].coefficient)
    for revised, kind in itertools.product((False, True), ("pair", "local")):
        graph, roots = nonlocal_field_program(kind, runtime_rvv10=revised)
        factor = "coefficient" if revised else f"{canonical:.17g}"
        name = f"{'rvv10_' if revised else ''}{kind}{'s' if kind == 'pair' else ''}"
        variables = _leg_variables("a", "first")
        if kind == "pair":
            variables.update(_leg_variables("b", "second"))
        variables.update(
            damping="damping", gradient_scale="gradient_scale", beta="beta"
        )
        emitter = CudaEmitter(graph, variables)
        emitter.emit(roots)
        if kind == "pair":
            text.extend(
                [
                    f"__global__ void {name}(const double* fields, const double* direction,",
                    " const double* weights, const double* positions, const double* delta_weights,",
                    " const double* delta_positions, std::size_t points, unsigned spins, const std::size_t* active,",
                    " std::size_t row_begin, std::size_t column_begin, std::size_t rows, std::size_t columns,",
                    (
                        " double damping, double gradient_scale, double coefficient,"
                        if revised
                        else ""
                    )
                    + " bool response, double* output, int* status) {",
                    " const auto lanes = rows * columns;",
                    " for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < lanes;",
                    "      lane += static_cast<std::size_t>(blockDim.x) * gridDim.x) {",
                    " const auto first = active[row_begin + lane / columns];",
                    " const auto second = active[column_begin + lane % columns];",
                    *emitter.lines,
                    f" output[lane] = response ? 0.0 : ({factor} * 0.5) * {emitter.reference(roots[0])};",
                    " if (!isfinite(output[lane])) atomicExch(status, 1);",
                ]
            )
        else:
            text.extend(
                [
                    f"__global__ void {name}(const double* fields, const double* direction,",
                    " const double* weights, const double* positions, const double* delta_weights,",
                    " const double* delta_positions, std::size_t points, unsigned spins, const std::size_t* active,",
                    (" double beta, double coefficient," if revised else "")
                    + " std::size_t count, bool response, double* partials, double* energy, int* status) {",
                    " for (std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x; lane < count;",
                    "      lane += static_cast<std::size_t>(blockDim.x) * gridDim.x) {",
                    " const auto first = active[lane];",
                    *emitter.lines,
                    f" if (!response) energy[first] += {factor} * {emitter.reference(roots[0])};",
                    " if (!response && !isfinite(energy[first])) atomicExch(status, 1);",
                ]
            )
        for index in range(9):
            text.append(
                f" const double value_{index} = {factor} * (response ? {emitter.reference(roots[10 + index])} : {emitter.reference(roots[1 + index])});"
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
