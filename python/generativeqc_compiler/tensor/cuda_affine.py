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

from .program import _topological

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from .ir import Node
    from .program import Program


_TILE = 16


def _matrix_axes(node: Node) -> tuple[tuple[int, ...], ...] | None:
    """Flatten a two-panel contraction into batch, row, column and inner axes."""
    if node.op != "einsum":
        return None
    operands = [axes for axes in node.attrs["labels"] if axes]
    if len(operands) != 2 or any(len(set(axes)) != len(axes) for axes in operands):
        return None
    lhs, rhs = map(set, operands)
    output = node.attrs["output"]
    free = set(output)
    if (lhs ^ rhs) - free:
        return None  # One-sided reductions are not matrix products.
    axes = (
        tuple(axis for axis in output if axis in lhs & rhs),
        tuple(axis for axis in output if axis in lhs - rhs),
        tuple(axis for axis in output if axis in rhs - lhs),
        tuple(sorted((lhs & rhs) - free)),
    )
    return axes if all(axes[1:]) else None


def _matrix_extents(
    node: Node, axes: tuple[tuple[int, ...], ...], extents: Mapping[str, str]
) -> tuple[str, ...]:
    """Use identical runtime dimensions for layout, dispatch and admission."""
    dimensions = {
        label: f"shape.{extents[index.space.name]}"
        for operand, labels in zip(node.inputs, node.attrs["labels"], strict=True)
        for label, index in zip(labels, operand.spec.indices, strict=True)
    }
    return tuple(
        " * ".join(dimensions[label] for label in group) or "1" for group in axes
    )


def emit_affine_work(
    programs: Mapping[str, Program],
    *,
    namespace: str,
    extents: Mapping[str, str],
    tiled_matrix_products: bool = False,
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
        tiled_arg = ", std::size_t& tiled" if tiled_matrix_products else ""
        reset_tiled = " tiled = 0;" if tiled_matrix_products else ""
        text.append(
            f"inline bool work_{name}(Extents shape, std::size_t& terms{tiled_arg}) {{ terms = 0;{reset_tiled}"
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
            matrix = _matrix_axes(node) if tiled_matrix_products else None
            if matrix:
                text.append(" { std::size_t batch{}, rows{}, columns{}, inner{};")
                for name_, axes in zip(
                    ("batch", "rows", "columns", "inner"), matrix, strict=True
                ):
                    dimensions = _matrix_extents(
                        node, tuple((axis,) for axis in axes), extents
                    )
                    text.append(
                        f"  if (!add_work({name_}, {{{', '.join(dimensions)}}})) return false;"
                    )
                text.extend(
                    [
                        "  if (use_matrix_tiles(rows, columns, inner)) {",
                        "   if (!add_work(tiled, {batch, matrix_tile_count(rows), matrix_tile_count(columns),",
                        "                        matrix_tile_count(inner), matrix_tile, matrix_tile, matrix_tile}) ||",
                        "       !add_work(terms, {batch, rows, columns})) return false;",
                        "  } else",
                    ]
                )
            text.append(
                " if (!add_work(terms, {" + ", ".join(factors) + "})) return false;"
            )
            if matrix:
                text.append(" }")
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
    tiled_matrix_products: bool = False,
    reuse_intermediate_storage: bool = False,
    inplace_add: bool = False,
    first_outputs: Mapping[str, tuple[str, ...]] | None = None,
    symmetric_outputs: Mapping[str, tuple[str, ...]] | None = None,
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
    Optional slot reuse is restricted to equal runtime extents and ends a
    value's lifetime only after its final consumer has executed on the stream.
    With ``inplace_add``, a lane-local add may overwrite a dead internal
    operand at the same address it reads. Inputs and outputs are not donors.
    Specifying first outputs evaluates their dependencies depth-first before
    the remaining graph. Publication still follows all finite checks.
    A consumer may declare a provably symmetric square output. Its publication
    reads one canonical triangle so transposed coordinates have identical FP64
    values even if the intermediate reduction order differs. This is a storage
    contract, not a symmetry check on a general matrix.
    """
    identifiers = (namespace, *programs, *extents.values())
    if tiled_matrix_products and (fuse_elementwise or parallel_output_reduction):
        raise ValueError("matrix tiles require separately materialized contractions")
    if reuse_intermediate_storage and fuse_elementwise:
        raise ValueError("storage reuse requires separately materialized dependencies")
    if inplace_add and not reuse_intermediate_storage:
        raise ValueError("in-place adds require intermediate storage reuse")
    first_outputs = first_outputs or {}
    if first_outputs.keys() - programs.keys():
        raise ValueError("affine evaluation order references an unknown program")
    symmetric_outputs = symmetric_outputs or {}
    if symmetric_outputs.keys() - programs.keys():
        raise ValueError("affine symmetric publication references an unknown program")
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
    if symmetric_outputs:
        text.append("""
__global__ void publish_symmetric(const double* input, double* output, std::size_t size,
                                  std::size_t extent, int* status) {
  if (atomicAdd(status, 0)) return;
  for (std::size_t position = blockIdx.x * blockDim.x + threadIdx.x; position < size;
       position += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const auto row = position / extent, column = position % extent;
    output[position] = input[row <= column ? position : column * extent + row];
  }
}
""")

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

    if tiled_matrix_products:
        text.append(f"inline constexpr unsigned matrix_tile = {_TILE};")
        text.append("""
__host__ __device__ inline std::size_t matrix_tile_count(std::size_t extent) {
  return extent / matrix_tile + (extent % matrix_tile != 0);
}
inline bool use_matrix_tiles(std::size_t rows, std::size_t columns, std::size_t inner) {
  return rows >= matrix_tile && columns >= matrix_tile && inner >= matrix_tile;
}
""")

    def matrix_kernel(node: Node, symbol: str, parameters: str, axes: tuple) -> None:
        labels = node.attrs["labels"]
        panel_indices = [index for index, indices in enumerate(labels) if indices]
        lhs, rhs = panel_indices
        extents_by_label = {
            label: dimension
            for operand, operand_labels in zip(node.inputs, labels, strict=True)
            for label, dimension in zip(operand_labels, dims(operand), strict=True)
        }

        def addresses(operand_labels: tuple, flattened: tuple[str, ...]) -> str:
            coordinates = {
                label: f"(({coordinate}) / ({product([extents_by_label[tail] for tail in group[index + 1 :]])})) % {extents_by_label[label]}"
                for group, coordinate in zip(axes, flattened, strict=True)
                for index, label in enumerate(group)
            }
            return address(
                [coordinates[label] for label in operand_labels],
                [extents_by_label[label] for label in operand_labels],
            )

        lhs_address = addresses(
            labels[lhs], ("batch", "row", "column_inner", "column_inner")
        )
        rhs_address = addresses(
            labels[rhs], ("batch", "row_inner", "column", "row_inner")
        )
        output_address = addresses(
            node.attrs["output"], ("batch", "row", "column", "0")
        )
        term = " * ".join(
            (
                factor(node.attrs["coefficient"]),
                *(
                    "lhs_tile[lane_row][inner_lane]"
                    if index == lhs
                    else "rhs_tile[inner_lane][lane_column]"
                    if index == rhs
                    else f"operand_{index}[0]"
                    for index in range(len(labels))
                ),
            )
        )
        text.append(f"""
__global__ void {symbol}_matrix(Extents shape, std::size_t tiles, std::size_t rows,
    std::size_t columns, std::size_t inner{parameters}, double* output, int* status) {{
  __shared__ double lhs_tile[matrix_tile][matrix_tile + 1];
  __shared__ double rhs_tile[matrix_tile][matrix_tile + 1];
  __shared__ int failed;
  if (!threadIdx.x) failed = atomicAdd(status, 0);
  __syncthreads();
  if (failed) return;
  const auto lane_row = threadIdx.x / matrix_tile, lane_column = threadIdx.x % matrix_tile;
  const auto row_tiles = matrix_tile_count(rows), column_tiles = matrix_tile_count(columns);
  for (std::size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {{
    const auto batch = tile / row_tiles / column_tiles;
    const auto row = (tile / column_tiles % row_tiles) * matrix_tile + lane_row;
    const auto column = (tile % column_tiles) * matrix_tile + lane_column;
    double sum = 0.0;
    for (std::size_t begin = 0; begin < inner; begin += matrix_tile) {{
      const auto row_inner = begin + lane_row, column_inner = begin + lane_column;
      lhs_tile[lane_row][lane_column] = row < rows && column_inner < inner
          ? operand_{lhs}[{lhs_address}] : 0.0;
      rhs_tile[lane_row][lane_column] = row_inner < inner && column < columns
          ? operand_{rhs}[{rhs_address}] : 0.0;
      __syncthreads();
      const auto valid = inner - begin < matrix_tile ? inner - begin : matrix_tile;
      for (unsigned inner_lane = 0; inner_lane < valid; ++inner_lane)
        sum += {term};
      __syncthreads();
    }}
    if (row < rows && column < columns) {{
      output[{output_address}] = sum;
      if (!isfinite(sum)) atomicExch(status, 1);
    }}
  }}
}}
""")

    for name, program in sorted(programs.items()):
        nodes = program.nodes
        symmetric = set(symmetric_outputs.get(name, ()))
        if symmetric - program.outputs.keys():
            raise ValueError(
                "affine symmetric publication references an unknown output"
            )
        for output in symmetric:
            axes = program.outputs[output].spec.indices
            if len(axes) != 2 or axes[0].space != axes[1].space:
                raise ValueError(
                    "affine symmetric publication requires a square matrix"
                )
        if name in first_outputs:
            priority = first_outputs[name]
            if set(priority) - program.outputs.keys():
                raise ValueError("affine evaluation order references an unknown output")
            roots = tuple(
                program.outputs[output] for output in (*priority, *program.outputs)
            )
            nodes = _topological((*roots, *nodes))
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
        matrices = {
            node: axes
            for node in nodes
            if tiled_matrix_products and (axes := _matrix_axes(node)) is not None
        }

        for index, node in enumerate(nodes):
            if node.op == "input" or node in absorbed:
                continue
            operands = ", ".join(
                f"const double* operand_{number}"
                for number in range(len(kernel_inputs[node]))
            )
            parameters = f", {operands}" if operands else ""
            if node in matrices:
                matrix_kernel(node, f"{name}_node_{index}", parameters, matrices[node])
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
        last_use = {node: index for index, node in enumerate(nodes)}
        for index, node in enumerate(nodes):
            for source in node.inputs:
                last_use[source] = index
        for node in outputs:
            last_use[node] = len(nodes)
        slots: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        slot_value: dict[int, int] = {}
        for index, node in enumerate(nodes):
            allocate = node.op != "input" and node not in absorbed
            previous = None
            if reuse_intermediate_storage and allocate:
                compatible = slots.setdefault(tuple(sorted(dims(node))), [])
                donors = (
                    {
                        positions[source]
                        for source in node.inputs
                        if source.op != "input" and last_use[source] == index
                    }
                    if inplace_add and node.op == "add"
                    else set()
                )
                for slot, (original, live_through) in enumerate(compatible):
                    if live_through < index or (
                        live_through == index and slot_value[original] in donors
                    ):
                        previous = original
                        compatible[slot] = (original, last_use[node])
                        break
                if previous is None:
                    compatible.append((index, last_use[node]))
                slot_value[index if previous is None else previous] = index
                allocate = previous is None
            text.append(
                f"  if (!append({{{', '.join(dims(node))}}}, "
                f"{'true' if allocate else 'false'}, sizes[{index}], offsets[{index}], elements)) return false;"
            )
            if previous is not None:
                text.append(f"  offsets[{index}] = offsets[{previous}];")
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
            if node in matrices:
                batch, rows, columns, inner = _matrix_extents(
                    node, matrices[node], extents
                )
                text.append(f"""
  if (sizes[{index}] && use_matrix_tiles({rows}, {columns}, {inner})) {{
    const auto tiles = ({batch}) * matrix_tile_count({rows}) * matrix_tile_count({columns});
    {name}_node_{index}_matrix<<<static_cast<unsigned>(tiles < 65535 ? tiles : 65535),
        matrix_tile * matrix_tile, 0, stream>>>(shape, tiles, {rows}, {columns}, {inner},
        {operands}, {pointers[node]}, status);
  }} else""")
            text.append(
                f"  if (sizes[{index}]) {name}_node_{index}<<<{grid}, 128, 0, stream>>>"
                f"(shape, sizes[{index}], {operands + ', ' if operands else ''}{pointers[node]}, status);\n"
                "  if ((error = cudaGetLastError()) != cudaSuccess) return error;"
            )
        for output, node in program.outputs.items():
            index = positions[node]
            kernel = "publish_symmetric" if output in symmetric else "publish"
            extent = f"{dims(node)[0]}, " if output in symmetric else ""
            text.append(
                f"  if (sizes[{index}]) {kernel}<<<blocks(sizes[{index}]), 128, 0, stream>>>"
                f"({pointers[node]}, output_{output}, sizes[{index}], {extent}status);\n"
                "  if ((error = cudaGetLastError()) != cudaSuccess) return error;"
            )
        text.extend(["  return cudaSuccess;", "}"])
    text.append(f"}}  // namespace {namespace}\n")
    return "\n".join(text)
