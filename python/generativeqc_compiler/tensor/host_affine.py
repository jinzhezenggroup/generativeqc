"""Build-time, runtime-sized FP64 TensorIR lowering into a borrowed CPU arena.

This materializing schedule is deliberately explicit about work and storage.
It preserves every definition (including AD primal validation), never allocates
numeric arrays, and does not select a method, compile at runtime or dispatch to
an interpreter. Scientific equations remain in the supplied TensorIR programs.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import isfinite
from typing import TYPE_CHECKING

from .cuda_affine import emit_affine_work

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .ir import Node
    from .program import Program


def emit_affine_host(
    programs: Mapping[str, Program],
    *,
    namespace: str,
    extents: Mapping[str, str],
    ordered_binary_contractions: bool = False,
) -> str:
    """Emit checked host kernels with exact runtime shapes and no aliasing.

    Only full-range FP64 input, uniform constant, add/multiply, einsum,
    sum-reduction, transpose and broadcast are admitted. Empty reductions are
    zero. One workspace contains all non-input nodes; outputs publish only
    after every input/intermediate passes its finite check. Inputs, outputs
    and workspace must not overlap (distinct inputs may share storage).

    The explicit ordered-binary schedule packs independent output columns,
    never reduction terms. It admits only two rank-two operands, one reduced
    label and an exact unit coefficient; other nodes and unsupported CPUs keep
    the original serial lowering. Equations, work and arena layouts are shared.
    """
    if type(ordered_binary_contractions) is not bool:
        raise TypeError("ordered_binary_contractions must be an explicit boolean")
    if not programs or any(
        re.fullmatch(r"[A-Za-z_]\w*", name, flags=re.ASCII) is None
        for name in (namespace, *programs, *extents.values())
    ):
        raise ValueError(
            "host affine lowering requires nonempty identifier-named bindings"
        )

    def dimensions(node: Node) -> list[str]:
        result = []
        for axis in node.spec.indices:
            if axis.start or axis.stop != axis.space.size or axis.selection is not None:
                raise ValueError(
                    "host affine lowering requires full-range unselected axes"
                )
            if axis.space.name not in extents:
                raise ValueError(f"unbound runtime index space: {axis.space.name}")
            result.append("shape." + extents[axis.space.name])
        return result

    def product(values: list[str]) -> str:
        return " * ".join(values) or "1"

    def address(coordinates: list[str], shape: list[str]) -> str:
        expression = "0"
        for coordinate, extent in zip(coordinates, shape, strict=True):
            expression = f"({expression} * {extent} + {coordinate})"
        return expression

    def coefficient(pair: tuple[int, int]) -> str:
        value = Fraction(*pair)
        literal = float(value)
        if not isfinite(literal):
            raise ValueError("host affine coefficient is not finite FP64")
        return literal.hex()

    def ordered_call(node: Node, index: int, positions: Mapping[Node, int]) -> str:
        if (
            not ordered_binary_contractions
            or node.op != "einsum"
            or len(node.inputs) != 2
            or len(node.spec.indices) != 2
            or node.attrs["coefficient"] != (1, 1)
        ):
            return ""
        labels, output = node.attrs["labels"], node.attrs["output"]
        reduced = set().union(*map(set, labels)) - set(output)
        if (
            len(set(output)) != 2
            or len(reduced) != 1
            or any(len(source) != 2 or len(set(source)) != 2 for source in labels)
        ):
            return ""
        reduction = next(iter(reduced))
        if any(
            reduction not in source or len(set(source) & set(output)) != 1
            for source in labels
        ):
            return ""
        label_extents: dict[int, str] = {}
        for source, source_labels in zip(node.inputs, labels, strict=True):
            for label, extent in zip(source_labels, dimensions(source), strict=True):
                if label in label_extents and label_extents[label] != extent:
                    raise ValueError(
                        "host affine contraction binds one label to different runtime spaces"
                    )
                label_extents[label] = extent
        arguments = [label_extents[label] for label in (*output, reduction)]
        for source, source_labels in zip(node.inputs, labels, strict=True):
            shape = dimensions(source)
            strides = {
                label: product(shape[axis + 1 :])
                for axis, label in enumerate(source_labels)
            }
            arguments.extend(
                (
                    f"value_{positions[source]}",
                    strides.get(output[0], "0"),
                    strides.get(output[1], "0"),
                    strides[reduction],
                )
            )
        arguments.append(f"value_{index}")
        return (
            "::generativeqc::tensor::ordered_binary_contraction::apply("
            + ", ".join(arguments)
            + ");"
        )

    text = [
        "#pragma once",
        *(
            ['#include "tensor/ordered_binary_contraction.hpp"']
            if ordered_binary_contractions
            else []
        ),
        "#include <algorithm>",
        "#include <array>",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <initializer_list>",
        "#include <limits>",
        "#include <span>",
        "#include <string_view>",
        f"namespace {namespace} {{",
        "struct Extents { "
        + " ".join(
            "std::size_t " + field + "{};" for field in sorted(set(extents.values()))
        )
        + " };",
        "enum class Status { success, invalid_binding, workspace_limit, nonfinite };",
        "struct Input { const double* data{}; std::size_t elements{}; };",
        "struct Output { double* data{}; std::size_t elements{}; };",
        r"""
inline bool append(std::initializer_list<std::size_t> dimensions, bool allocate,
                   std::size_t& size, std::size_t& offset, std::size_t& elements) noexcept {
  size = 1;
  for (auto extent : dimensions) {
    if (extent && size > std::numeric_limits<std::size_t>::max() / extent) return false;
    size *= extent;
  }
  if (size > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / sizeof(double))
    return false;
  offset = elements;
  if (allocate) {
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / sizeof(double) - elements)
      return false;
    elements += size;
  }
  return true;
}
inline bool overlap(const double* first, std::size_t first_count,
                    const double* second, std::size_t second_count) noexcept {
  if (!first_count || !second_count) return false;
  const auto left = reinterpret_cast<std::uintptr_t>(first);
  const auto right = reinterpret_cast<std::uintptr_t>(second);
  return left <= right ? right - left < first_count * sizeof(double)
                       : left - right < second_count * sizeof(double);
}
inline bool finite(const double* values, std::size_t size) noexcept {
  for (std::size_t position = 0; position < size; ++position)
    if (!std::isfinite(values[position])) return false;
  return true;
}
inline bool aligned(const double* values) noexcept {
  return reinterpret_cast<std::uintptr_t>(values) % alignof(double) == 0;
}
""",
    ]
    metadata = []
    for name, program in sorted(programs.items()):
        nodes = program.nodes
        positions = {node: index for index, node in enumerate(nodes)}
        inputs = sorted(
            (node for node in nodes if node.op == "input"),
            key=lambda node: node.attrs["name"],
        )
        outputs = tuple(program.outputs.items())
        if len({node.attrs["name"] for node in inputs}) != len(inputs):
            raise ValueError("host affine lowering requires unique input definitions")
        for node in nodes:
            if (
                node.spec.dtype != "float64"
                or node.spec.symmetries
                or node.op
                not in (
                    "input",
                    "constant",
                    "add",
                    "multiply",
                    "einsum",
                    "reduce",
                    "transpose",
                    "broadcast",
                )
            ):
                raise ValueError(
                    f"unqualified host affine primitive/dtype/symmetry: {node.op}"
                )
            dimensions(node)
            if node.op == "reduce" and node.attrs.get("operation", "sum") != "sum":
                raise ValueError("host affine lowering supports only sum reductions")
        text.extend(
            [
                f"inline bool layout_{name}(Extents shape, std::size_t* sizes, std::size_t* offsets, std::size_t& count) noexcept {{",
                " (void)shape; count = 0;",
            ]
        )
        for index, node in enumerate(nodes):
            text.append(
                f" if (!append({{{', '.join(dimensions(node))}}}, "
                f"{'false' if node.op == 'input' else 'true'}, sizes[{index}], offsets[{index}], count)) return false;"
            )
        text.extend(
            [
                " return true; }",
                f"inline bool workspace_{name}(Extents shape, std::size_t& bytes) noexcept {{",
                f" std::size_t sizes[{len(nodes)}]{{}}, offsets[{len(nodes)}]{{}}, count{{}}; bytes = 0;",
                f" if (!layout_{name}(shape, sizes, offsets, count)) return false;",
                " bytes = count * sizeof(double); return true; }",
                f"inline Status evaluate_{name}(Extents shape, std::span<const Input> inputs, std::span<const Output> outputs, std::span<double> arena) noexcept {{",
                f" std::size_t sizes[{len(nodes)}]{{}}, offsets[{len(nodes)}]{{}}, count{{}};",
                f" if (inputs.size() != {len(inputs)} || outputs.size() != {len(outputs)} ||",
                f"     !layout_{name}(shape, sizes, offsets, count)) return Status::invalid_binding;",
                " if (arena.size() < count || (count && !arena.data())) return Status::workspace_limit;",
                " if (count && !aligned(arena.data())) return Status::invalid_binding;",
            ]
        )
        for slot, node in enumerate(inputs):
            text.append(
                f" if (inputs[{slot}].elements != sizes[{positions[node]}] || "
                f"(inputs[{slot}].elements && (!inputs[{slot}].data || !aligned(inputs[{slot}].data)))) return Status::invalid_binding;"
            )
        for slot, (_, node) in enumerate(outputs):
            text.append(
                f" if (outputs[{slot}].elements != sizes[{positions[node]}] || "
                f"(outputs[{slot}].elements && (!outputs[{slot}].data || !aligned(outputs[{slot}].data)))) return Status::invalid_binding;"
            )
        text.append(r"""
 for (const auto& input : inputs)
   if (overlap(input.data, input.elements, arena.data(), count)) return Status::invalid_binding;
 for (std::size_t slot = 0; slot < outputs.size(); ++slot) {
   const auto& output = outputs[slot];
   if (overlap(output.data, output.elements, arena.data(), count)) return Status::invalid_binding;
   for (const auto& input : inputs)
     if (overlap(input.data, input.elements, output.data, output.elements)) return Status::invalid_binding;
   for (std::size_t prior = 0; prior < slot; ++prior)
     if (overlap(output.data, output.elements, outputs[prior].data, outputs[prior].elements))
       return Status::invalid_binding;
 }
 for (const auto& input : inputs)
   if (!finite(input.data, input.elements)) return Status::nonfinite;
""")
        for slot, node in enumerate(inputs):
            text.append(
                f" const double* value_{positions[node]} = inputs[{slot}].data;"
            )
        for index, node in enumerate(nodes):
            if node.op == "input":
                continue
            text.append(f" double* value_{index} = arena.data() + offsets[{index}];")
            ordered = ordered_call(node, index, positions)
            if ordered:
                text.extend(
                    [
                        " if (::generativeqc::tensor::ordered_binary_contraction::available()) {",
                        ordered,
                        f" if (!finite(value_{index}, sizes[{index}])) return Status::nonfinite;",
                        " } else {",
                    ]
                )
            text.append(
                f" for (std::size_t position = 0; position < sizes[{index}]; ++position) {{"
            )
            shape = dimensions(node)
            coordinates = [f"coordinate_{axis}" for axis in range(len(shape))]
            for axis, extent in enumerate(shape):
                text.append(
                    f" [[maybe_unused]] const auto {coordinates[axis]} = "
                    f"(position / ({product(shape[axis + 1 :])})) % {extent};"
                )
            operands = [f"value_{positions[source]}" for source in node.inputs]
            if node.op == "constant":
                values = node.attrs["values"]
                if not values or len(set(values)) != 1:
                    raise ValueError(
                        "host affine constants require uniform nonempty witness literals"
                    )
                expression = coefficient(values[0])
            elif node.op == "add":
                expression = "0.0 + " + " + ".join(
                    f"{coefficient(weight)} * {operand}[position]"
                    for weight, operand in zip(
                        node.attrs["coefficients"], operands, strict=True
                    )
                )
            elif node.op == "multiply":
                expression = " * ".join(operand + "[position]" for operand in operands)
            elif node.op == "transpose":
                source_coordinates = [""] * len(shape)
                for axis, source_axis in enumerate(node.attrs["axes"]):
                    source_coordinates[source_axis] = coordinates[axis]
                expression = f"{operands[0]}[{address(source_coordinates, dimensions(node.inputs[0]))}]"
            elif node.op == "broadcast":
                source_coordinates = [coordinates[axis] for axis in node.attrs["axes"]]
                expression = f"{operands[0]}[{address(source_coordinates, dimensions(node.inputs[0]))}]"
            else:
                if node.op == "reduce":
                    labels = (tuple(range(len(node.inputs[0].spec.indices))),)
                    output_labels = tuple(
                        axis for axis in labels[0] if axis not in node.attrs["axes"]
                    )
                else:
                    labels, output_labels = node.attrs["labels"], node.attrs["output"]
                label_dimensions = {}
                for source, source_labels in zip(node.inputs, labels, strict=True):
                    for label, extent in zip(
                        source_labels, dimensions(source), strict=True
                    ):
                        if (
                            label in label_dimensions
                            and label_dimensions[label] != extent
                        ):
                            raise ValueError(
                                "host affine contraction binds one label to different runtime spaces"
                            )
                        label_dimensions[label] = extent
                label_coordinates = dict(zip(output_labels, coordinates, strict=True))
                reduced = sorted(label_dimensions.keys() - label_coordinates.keys())
                text.append(" double sum = 0.0;")
                for ordinal, label in enumerate(reduced):
                    coordinate = f"reduction_{ordinal}"
                    label_coordinates[label] = coordinate
                    text.append(
                        f" for (std::size_t {coordinate} = 0; {coordinate} < {label_dimensions[label]}; ++{coordinate}) {{"
                    )
                terms = [
                    coefficient(
                        (1, 1) if node.op == "reduce" else node.attrs["coefficient"]
                    )
                ]
                terms.extend(
                    f"{operand}[{address([label_coordinates[label] for label in source_labels], dimensions(source))}]"
                    for operand, source, source_labels in zip(
                        operands, node.inputs, labels, strict=True
                    )
                )
                text.append(" sum += " + " * ".join(terms) + ";")
                text.extend(" }" for _ in reduced)
                expression = "sum"
            text.extend(
                [
                    f" value_{index}[position] = {expression};",
                    f" if (!std::isfinite(value_{index}[position])) return Status::nonfinite;",
                    " }",
                ]
            )
            if ordered:
                text.append(" }")
        for slot, (_, node) in enumerate(outputs):
            text.append(
                f" if (sizes[{positions[node]}]) std::copy_n(value_{positions[node]}, sizes[{positions[node]}], outputs[{slot}].data);"
            )
        text.append(" return Status::success; }")
        for label, names in (
            ("inputs", [node.attrs["name"] for node in inputs]),
            ("outputs", list(program.outputs)),
        ):
            text.append(
                f"inline constexpr std::array<std::string_view, {len(names)}> {name}_{label}{{{{"
                + ", ".join('"' + item + '"' for item in names)
                + "}};"
            )
        metadata.append(
            f' {{"{name}", "{program.logical_hash}", {name}_inputs, {name}_outputs, '
            f"workspace_{name}, work_{name}, evaluate_{name}}}"
        )
    text.append("}")
    text.append(emit_affine_work(programs, namespace=namespace, extents=extents))
    text.extend(
        [
            f"namespace {namespace} {{",
            "struct Program { std::string_view name, equation; std::span<const std::string_view> inputs, outputs;",
            " bool (*workspace)(Extents, std::size_t&) noexcept; bool (*work)(Extents, std::size_t&);",
            " Status (*evaluate)(Extents, std::span<const Input>, std::span<const Output>, std::span<double>) noexcept; };",
            "inline std::span<const Program> programs() noexcept {",
            f" static const std::array<Program, {len(metadata)}> entries{{{{",
            ",\n".join(metadata),
            " }}; return entries; }",
            "inline const Program* find(std::string_view name, std::string_view equation) noexcept {",
            " for (const auto& program : programs()) if (program.name == name && program.equation == equation) return &program;",
            " return nullptr; }",
            "}",
            "",
        ]
    )
    return "\n".join(text)
