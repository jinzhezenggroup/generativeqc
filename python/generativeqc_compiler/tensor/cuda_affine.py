"""Build-only FP64 CUDA lowering for full-range, runtime-sized affine tensors.

This deliberately narrow materializing schedule preserves every definition of
the supplied program, including AD primal validation. It is not the optimized
TensorIR planner, a runtime compiler, or an implicit backend fallback.
"""

from __future__ import annotations

import re
import typing
from collections import Counter
from fractions import Fraction

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from .ir import Node
    from .program import Program


def emit_affine_work(
    programs: Mapping[str, Program], *, namespace: str, extents: Mapping[str, str]
) -> str:
    """Count scheduled scalar term visits, including AD primal validation.

    This companion to the materializing affine schedule is resource admission,
    not a FLOP or endpoint-performance estimate. Reductions count their input
    domain; einsums count the union of label domains. Repeated operands sharing
    an index do not introduce an extra contraction loop.
    """
    if not programs or any(
        not re.fullmatch(r"[A-Za-z_]\w*", value)
        for value in (namespace, *programs, *extents.values())
    ):
        raise ValueError(
            "affine work requires nonempty identifier-named programs/extents"
        )
    text = [
        f"namespace {namespace} {{",
        "inline bool add_work(std::size_t& total, std::initializer_list<std::size_t> factors) {",
        " std::size_t term = 1;",
        " for (auto factor : factors) {",
        "  if (factor && term > std::numeric_limits<std::size_t>::max() / factor) return false;",
        "  term *= factor;",
        " }",
        " if (total > std::numeric_limits<std::size_t>::max() - term) return false;",
        " total += term; return true;",
        "}",
    ]
    for name, program in programs.items():
        text.append(
            f"inline bool work_{name}(Extents shape, std::size_t& terms) {{ terms = 0;"
        )
        for node in program.nodes:
            if node.op == "einsum":
                spaces = {}
                for source, labels in zip(
                    node.inputs, node.attrs["labels"], strict=True
                ):
                    spaces.update(zip(labels, source.spec.indices, strict=True))
                indices = spaces.values()
            elif node.op == "reduce":
                indices = node.inputs[0].spec.indices
            else:
                indices = node.spec.indices
            factors = [f"shape.{extents[index.space.name]}" for index in indices]
            text.append(
                " if (!add_work(terms, {" + ", ".join(factors) + "})) return false;"
            )
        text.append(" return true; }")
    text.append("}")
    return "\n".join(text)


def emit_affine_cuda(
    programs: Mapping[str, Program],
    *,
    namespace: str,
    extents: Mapping[str, str],
    fuse_elementwise: bool = False,
    fuse_contraction_epilogues: bool = False,
    parallel_scalar_reduction: bool = False,
    parallel_output_reduction: bool = False,
) -> str:
    """Emit checked borrowed-arena launchers, binding spaces by identity/name.

    Only FP64 input, uniform constant, add, multiply, einsum, sum-reduce, transpose and
    broadcast are admitted. Slices, selections, nonuniform witness literals and
    other operations fail at generation time. Empty contractions sum to zero.
    All device arithmetic and finite checks run on the caller's CUDA stream.
    Optional single-consumer elementwise fusion preserves each intermediate
    finite check and arithmetic order. Optional scalar reductions use a fixed
    128-lane FP64 tree: this deliberately changes summation order, never dtype,
    and requires consumer-specific independent numerical qualification.
    Single-consumer contraction epilogues may share the consumer's output
    coordinates, retaining the original inner loops and finite checks without
    recomputing any contraction or changing its reduction order.
    """
    identifiers = (namespace, *programs, *extents.values())
    if not programs or any(
        not re.fullmatch(r"[A-Za-z_]\w*", value) for value in identifiers
    ):
        raise ValueError(
            "affine CUDA requires nonempty, identifier-named programs/extents"
        )
    fields = sorted(set(extents.values()))
    text = [
        "#pragma once",
        "#include <cuda_runtime.h>",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <initializer_list>",
        "#include <limits>",
        f"namespace {namespace} {{",
        "struct Extents { "
        + " ".join(f"std::size_t {field};" for field in fields)
        + " };",
        """
inline bool append(std::initializer_list<std::size_t> dimensions, bool allocate,
                   std::size_t& size, std::size_t& offset, std::size_t& elements) {
  size = 1;
  for (auto extent : dimensions) {
    if (extent && size > std::numeric_limits<std::size_t>::max() / extent) return false;
    size *= extent;
  }
  offset = elements;
  if (allocate) {
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(double) - elements)
      return false;
    elements += size;
  }
  return size <= std::numeric_limits<std::size_t>::max() / sizeof(double);
}
inline unsigned blocks(std::size_t size) {
  return static_cast<unsigned>(size / 128 + (size % 128 != 0) > 65535
                                  ? 65535 : size / 128 + (size % 128 != 0));
}
__global__ void finite_input(const double* input, std::size_t size, int* status) {
  for (std::size_t position = blockIdx.x * blockDim.x + threadIdx.x; position < size;
       position += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    if (!isfinite(input[position])) atomicExch(status, 1);
}
__global__ void publish(const double* input, double* output, std::size_t size, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t position = blockIdx.x * blockDim.x + threadIdx.x; position < size;
       position += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    output[position] = input[position];
}
""",
    ]

    def dims(node: Node) -> list[str]:
        result = []
        for axis in node.spec.indices:
            if axis.start or axis.stop != axis.space.size or axis.selection is not None:
                raise ValueError("affine CUDA requires full-range unselected axes")
            if axis.space.name not in extents:
                raise ValueError(f"unbound runtime index space: {axis.space.name}")
            result.append(f"shape.{extents[axis.space.name]}")
        return result

    def product(values: list[str]) -> str:
        return " * ".join(values) or "1"

    def address(coordinates: list[str], dimensions: list[str]) -> str:
        expression = "0"
        for coordinate, extent in zip(coordinates, dimensions, strict=True):
            expression = f"({expression} * {extent} + {coordinate})"
        return expression

    def factor(pair: tuple) -> str:
        value = Fraction(*pair)
        return f"({value.numerator}.0 / {value.denominator}.0)"

    for name, program in sorted(programs.items()):
        nodes = program.nodes
        positions = {node: index for index, node in enumerate(nodes)}
        inputs = sorted(
            (node for node in nodes if node.op == "input"),
            key=lambda node: node.attrs["name"],
        )
        if len({node.attrs["name"] for node in inputs}) != len(inputs):
            raise ValueError("affine CUDA requires unique input definitions")
        for node in nodes:
            if node.spec.dtype != "float64" or node.op not in (
                "input",
                "constant",
                "add",
                "multiply",
                "einsum",
                "reduce",
                "transpose",
                "broadcast",
            ):
                raise ValueError(f"unqualified affine CUDA primitive/dtype: {node.op}")
            dims(node)
        uses = Counter(source for node in nodes for source in node.inputs)
        outputs = set(program.outputs.values())
        cooperative = {}
        for node in nodes:
            if (
                (
                    parallel_output_reduction
                    if node.spec.indices
                    else parallel_scalar_reduction
                )
                and node.op in ("einsum", "reduce")
                and 0 < len(node.inputs) <= 4
            ):
                labels = (
                    (tuple(range(len(node.inputs[0].spec.indices))),)
                    if node.op == "reduce"
                    else node.attrs["labels"]
                )
                output_labels = (
                    tuple(axis for axis in labels[0] if axis not in node.attrs["axes"])
                    if node.op == "reduce"
                    else node.attrs["output"]
                )
                dimensions_by_label = {
                    label: extent
                    for source, source_labels in zip(node.inputs, labels, strict=True)
                    for label, extent in zip(source_labels, dims(source), strict=True)
                }
                reduced = dimensions_by_label.keys() - set(output_labels)
                if len(reduced) == 1:
                    cooperative[node] = (
                        labels,
                        output_labels,
                        dimensions_by_label,
                        next(iter(reduced)),
                    )
        absorbed = {
            source
            for node in nodes
            for source in node.inputs
            if fuse_elementwise
            and node.op in ("add", "multiply")
            and (
                source.op in ("constant", "add", "multiply")
                or (
                    fuse_contraction_epilogues
                    and source.op in ("einsum", "reduce")
                    and source not in cooperative
                )
            )
            and source not in outputs
            and uses[source] == 1
            and dims(source) == dims(node)
        }

        def leaves(node: Node, absorbed: set[Node]) -> list[Node]:
            return (
                [leaf for source in node.inputs for leaf in leaves(source, absorbed)]
                if node in absorbed
                else [node]
            )

        kernel_inputs = {
            node: (
                tuple(
                    dict.fromkeys(
                        leaf
                        for source in node.inputs
                        for leaf in leaves(source, absorbed)
                    )
                )
                if any(source in absorbed for source in node.inputs)
                else node.inputs
            )
            for node in nodes
        }

        for index, node in enumerate(nodes):
            if node.op == "input" or node in absorbed:
                continue
            operands = ", ".join(
                f"const double* operand_{number}"
                for number in range(len(kernel_inputs[node]))
            )
            parameters = f", {operands}" if operands else ""
            if node in cooperative:
                labels, output_labels, dimensions_by_label, reduced = cooperative[node]
                extent = dimensions_by_label[reduced]
                text.extend(
                    [
                        f"__global__ void {name}_node_{index}(Extents shape, std::size_t size{parameters}, double* output, int* status) {{",
                        " __shared__ int failed;",
                        " __shared__ double partials[128];",
                        " if (!threadIdx.x) failed = atomicAdd(status, 0);",
                        " __syncthreads(); if (failed) return;",
                        " for (std::size_t position = blockIdx.x; position < size; position += gridDim.x) {",
                    ]
                )
                dimensions = dims(node)
                coordinates = [f"coordinate_{axis}" for axis in range(len(dimensions))]
                for axis, dimension in enumerate(dimensions):
                    text.append(
                        f" const auto {coordinates[axis]} = (position / ({product(dimensions[axis + 1 :])})) % {dimension};"
                    )
                label_coordinates = dict(zip(output_labels, coordinates, strict=True))
                label_coordinates[reduced] = "radial"
                text.extend(
                    [
                        " double sum = 0.0;",
                        f" for (std::size_t radial = threadIdx.x; radial < {extent}; radial += 128) {{",
                    ]
                )
                term = " * ".join(
                    [
                        factor(
                            (1, 1) if node.op == "reduce" else node.attrs["coefficient"]
                        ),
                        *(
                            f"operand_{operand}[{address([label_coordinates[label] for label in source_labels], dims(source))}]"
                            for operand, (source, source_labels) in enumerate(
                                zip(node.inputs, labels, strict=True)
                            )
                        ),
                    ]
                )
                text.extend(
                    [
                        f"  sum += {term};",
                        " }",
                        " if (!isfinite(sum)) atomicExch(status, 1);",
                        " partials[threadIdx.x] = sum; __syncthreads();",
                        " for (unsigned stride = 64; stride; stride /= 2) {",
                        "  if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];",
                        "  __syncthreads();",
                        " }",
                        " if (!threadIdx.x) { output[position] = partials[0]; if (!isfinite(partials[0])) atomicExch(status, 1); }",
                        " __syncthreads(); }",
                        "}",
                    ]
                )
                continue
            text.append(
                f"__global__ void {name}_node_{index}(Extents shape, std::size_t size"
                f"{parameters}, double* output, int* status) {{\n"
                "  if (atomicAdd(status, 0)) return;\n"
                "  for (std::size_t position = blockIdx.x * blockDim.x + threadIdx.x; position < size;\n"
                "       position += static_cast<std::size_t>(blockDim.x) * gridDim.x) {"
            )
            dimensions = dims(node)
            coordinates = [f"coordinate_{axis}" for axis in range(len(dimensions))]
            indexed_dimensions = (
                dimensions
                if node.op in ("einsum", "reduce", "transpose", "broadcast")
                or fuse_contraction_epilogues
                else []
            )
            for axis, extent in enumerate(indexed_dimensions):
                text.append(
                    f"    [[maybe_unused]] const std::size_t {coordinates[axis]} = "
                    f"(position / ({product(dimensions[axis + 1 :])})) % {extent};"
                )
            if node.op in ("constant", "add", "multiply"):
                local_values = {
                    source: f"operand_{operand}[position]"
                    for operand, source in enumerate(kernel_inputs[node])
                }

                def elementwise(
                    source: Node,
                    root: Node,
                    values: dict[Node, str],
                    positions: dict[Node, int],
                    pointers: dict[Node, str],
                    coordinates: list[str],
                ) -> str:
                    if source in values:
                        return values[source]
                    identifier = f"fused_{positions[source]}"
                    if source.op in ("einsum", "reduce"):
                        if source.op == "reduce":
                            labels = (tuple(range(len(source.inputs[0].spec.indices))),)
                            output_labels = tuple(
                                axis
                                for axis in labels[0]
                                if axis not in source.attrs["axes"]
                            )
                        else:
                            labels, output_labels = (
                                source.attrs["labels"],
                                source.attrs["output"],
                            )
                        dimensions_by_label = {
                            label: extent
                            for operand, source_labels in zip(
                                source.inputs, labels, strict=True
                            )
                            for label, extent in zip(
                                source_labels, dims(operand), strict=True
                            )
                        }
                        label_coordinates = dict(
                            zip(output_labels, coordinates, strict=True)
                        )
                        reduced = sorted(
                            dimensions_by_label.keys() - label_coordinates.keys()
                        )
                        text.append(f"    double {identifier} = 0.0;")
                        for label in reduced:
                            coordinate = f"{identifier}_reduction_{label}"
                            label_coordinates[label] = coordinate
                            text.append(
                                f"    for (std::size_t {coordinate} = 0; {coordinate} < {dimensions_by_label[label]}; ++{coordinate}) {{"
                            )
                        terms = [
                            factor(
                                (1, 1)
                                if source.op == "reduce"
                                else source.attrs["coefficient"]
                            )
                        ]
                        for operand, source_labels in zip(
                            source.inputs, labels, strict=True
                        ):
                            offset = address(
                                [label_coordinates[label] for label in source_labels],
                                dims(operand),
                            )
                            terms.append(f"{pointers[operand]}[{offset}]")
                        text.append(f"      {identifier} += " + " * ".join(terms) + ";")
                        text.extend("    }" for _ in reduced)
                        text.append(
                            f"    if (!isfinite({identifier})) atomicExch(status, 1);"
                        )
                        values[source] = identifier
                        return identifier
                    arguments = [
                        elementwise(
                            value, root, values, positions, pointers, coordinates
                        )
                        for value in source.inputs
                    ]
                    if source.op == "constant":
                        literals = source.attrs["values"]
                        if not literals or len(set(literals)) != 1:
                            raise ValueError(
                                "affine CUDA requires a uniform, nonempty witness literal"
                            )
                        value = factor(literals[0])
                    elif source.op == "multiply":
                        value = " * ".join(arguments)
                    else:
                        value = "0.0 + " + " + ".join(
                            f"{factor(coefficient)} * {argument}"
                            for coefficient, argument in zip(
                                source.attrs["coefficients"], arguments, strict=True
                            )
                        )
                    if source is root:
                        return value
                    text.append(f"    const double {identifier} = {value};")
                    text.append(
                        f"    if (!isfinite({identifier})) atomicExch(status, 1);"
                    )
                    values[source] = identifier
                    return identifier

                expression = elementwise(
                    node,
                    node,
                    local_values,
                    positions,
                    {
                        source: f"operand_{operand}"
                        for operand, source in enumerate(kernel_inputs[node])
                    },
                    coordinates,
                )
            elif node.op == "transpose":
                input_coordinates = [""] * len(coordinates)
                for axis, source in enumerate(node.attrs["axes"]):
                    input_coordinates[source] = coordinates[axis]
                expression = (
                    f"operand_0[{address(input_coordinates, dims(node.inputs[0]))}]"
                )
            elif node.op == "broadcast":
                input_coordinates = [coordinates[axis] for axis in node.attrs["axes"]]
                expression = (
                    f"operand_0[{address(input_coordinates, dims(node.inputs[0]))}]"
                )
            else:
                if node.op == "reduce":
                    labels = (tuple(range(len(node.inputs[0].spec.indices))),)
                    output_labels = tuple(
                        axis for axis in labels[0] if axis not in node.attrs["axes"]
                    )
                else:
                    labels = node.attrs["labels"]
                    output_labels = node.attrs["output"]
                label_dimensions = {}
                for source, source_labels in zip(node.inputs, labels, strict=True):
                    label_dimensions.update(
                        zip(source_labels, dims(source), strict=True)
                    )
                label_coordinates = dict(zip(output_labels, coordinates, strict=True))
                reduced = sorted(label_dimensions.keys() - label_coordinates.keys())
                text.append("    double sum = 0.0;")
                for label in reduced:
                    coordinate = f"reduction_{label}"
                    label_coordinates[label] = coordinate
                    text.append(
                        f"    for (std::size_t {coordinate} = 0; {coordinate} < "
                        f"{label_dimensions[label]}; ++{coordinate}) {{"
                    )
                terms = [
                    factor((1, 1) if node.op == "reduce" else node.attrs["coefficient"])
                ]
                for operand, (source, source_labels) in enumerate(
                    zip(node.inputs, labels, strict=True)
                ):
                    terms.append(
                        f"operand_{operand}[{address([label_coordinates[label] for label in source_labels], dims(source))}]"
                    )
                text.append("      sum += " + " * ".join(terms) + ";")
                text.extend("    }" for _ in reduced)
                expression = "sum"
            text.extend(
                [
                    f"    const double value = {expression};",
                    "    output[position] = value;",
                    "    if (!isfinite(value)) atomicExch(status, 1);",
                    "  }",
                    "}",
                ]
            )

        text.append(
            f"inline bool layout_{name}(Extents shape, std::size_t* sizes, "
            "std::size_t* offsets, std::size_t& bytes) {\n  std::size_t elements = 0;"
        )
        for index, node in enumerate(nodes):
            text.append(
                f"  if (!append({{{', '.join(dims(node))}}}, "
                f"{'false' if node.op == 'input' else 'true'}, sizes[{index}], offsets[{index}], elements)) return false;"
            )
        text.append("  bytes = elements * sizeof(double);\n  return true;\n}")
        text.append(
            f"inline cudaError_t workspace_{name}(Extents shape, std::size_t& bytes) {{\n"
            f"  std::size_t sizes[{len(nodes)}]{{}}, offsets[{len(nodes)}]{{}};\n"
            "  bytes = 0;\n"
            f"  return layout_{name}(shape, sizes, offsets, bytes) ? cudaSuccess : cudaErrorInvalidValue;\n}}"
        )
        parameters = [f"const double* input_{node.attrs['name']}" for node in inputs]
        parameters += [f"double* output_{output}" for output in program.outputs]
        text.append(
            f"inline cudaError_t launch_{name}(Extents shape, {', '.join(parameters)}, "
            "void* workspace, std::size_t workspace_bytes, int* status, cudaStream_t stream) {\n"
            f"  std::size_t sizes[{len(nodes)}]{{}}, offsets[{len(nodes)}]{{}}, bytes{{}};\n"
            f"  if (!layout_{name}(shape, sizes, offsets, bytes) || bytes > workspace_bytes || !status ||\n"
            "      (bytes && (!workspace || reinterpret_cast<std::size_t>(workspace) % alignof(double))))\n"
            "    return cudaErrorInvalidValue;\n  auto* base = static_cast<double*>(workspace);"
        )
        pointers = {}
        for node in inputs:
            index = positions[node]
            pointer = f"input_{node.attrs['name']}"
            pointers[node] = pointer
            text.append(
                f"  if (sizes[{index}] && !{pointer}) return cudaErrorInvalidValue;"
            )
        for output, node in program.outputs.items():
            text.append(
                f"  if (sizes[{positions[node]}] && !output_{output}) return cudaErrorInvalidValue;"
            )
        text.append("  cudaError_t error = cudaSuccess;")
        for node in inputs:
            index = positions[node]
            text.append(
                f"  if (sizes[{index}]) finite_input<<<blocks(sizes[{index}]), 128, 0, stream>>>"
                f"({pointers[node]}, sizes[{index}], status);\n"
                "  if ((error = cudaGetLastError()) != cudaSuccess) return error;"
            )
        for index, node in enumerate(nodes):
            if node.op == "input":
                continue
            pointers[node] = f"base + offsets[{index}]"
            if node in absorbed:
                continue
            operands = ", ".join(pointers[source] for source in kernel_inputs[node])
            grid = (
                f"static_cast<unsigned>(sizes[{index}] < 65535 ? sizes[{index}] : 65535)"
                if node in cooperative
                else f"blocks(sizes[{index}])"
            )
            text.append(
                f"  if (sizes[{index}]) {name}_node_{index}<<<{grid}, 128, 0, stream>>>"
                f"(shape, sizes[{index}], {operands + ', ' if operands else ''}{pointers[node]}, status);\n"
                "  if ((error = cudaGetLastError()) != cudaSuccess) return error;"
            )
        for output, node in program.outputs.items():
            index = positions[node]
            text.append(
                f"  if (sizes[{index}]) publish<<<blocks(sizes[{index}]), 128, 0, stream>>>"
                f"({pointers[node]}, output_{output}, sizes[{index}], status);\n"
                "  if ((error = cudaGetLastError()) != cudaSuccess) return error;"
            )
        text.extend(["  return cudaSuccess;", "}"])
    text.append(f"}}  // namespace {namespace}\n")
    return "\n".join(text)
