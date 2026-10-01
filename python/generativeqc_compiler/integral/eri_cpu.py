"""Bounded value-only CPU ERIs from the shared four-center scalar DAG.

All ordered Cartesian s/p/d components are covered by 313 representatives
under the eight ERI center symmetries and six simultaneous axis permutations.
This is a lowering-size reduction, not screening or a change to the operator.
Contraction, normalization, spherical transforms and derivative fallback remain
caller-owned. No new recurrence or geometry mathematics is maintained here.
"""

from __future__ import annotations

from functools import cache
from itertools import permutations, product
from typing import Any

from .blocks import RawBlock, TensorLayout
from .expr import AlgebraForm, PowerLowering
from .ir import IntegralIR, build_integral_ir
from .ir_serialization import integral_to_payload
from .one_electron_cuda import _emit_boys_support
from .scalar_c import ScalarCEmitter
from .shell_class import (
    ShellClassComponentKernel,
    build_coulomb_derivative_algebra,
    build_shell_class_component_kernel,
)
from .shell_signature import ShellSignature
from .shell_spec import ShellClassSpec, cartesian_components

COMPONENTS = tuple(c for angular in range(3) for c in cartesian_components(angular))
AXIS_PERMUTATIONS = tuple(permutations(range(3)))
CENTER_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


@cache
def eri_cpu_component_map() -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return representatives and packed function/center/axis lookup records.

    Each permutation maps an output slot to its original input slot. The low
    three record bits select axes, the next three select centers, and remaining
    bits select a representative. Canonical ties use the first fixed permutation.
    """
    indices = {label: index for index, label in enumerate(COMPONENTS)}
    axis_components = tuple(
        tuple(
            indices[
                "".join(
                    "xyz"[axis] * label.count("xyz"[original])
                    for axis, original in enumerate(permutation)
                )
            ]
            for label in COMPONENTS
        )
        for permutation in AXIS_PERMUTATIONS
    )
    canonical = []
    for components in product(range(len(COMPONENTS)), repeat=4):
        representative, centers, axes = min(
            (
                tuple(axis_map[components[slot]] for slot in center_map),
                center_index,
                axis_index,
            )
            for center_index, center_map in enumerate(CENTER_PERMUTATIONS)
            for axis_index, axis_map in enumerate(axis_components)
        )
        canonical.append((representative, centers, axes))
    representatives = tuple(sorted({entry[0] for entry in canonical}))
    representative_index = {entry: index for index, entry in enumerate(representatives)}
    records = tuple(
        (representative_index[representative] << 6) | (centers << 3) | axes
        for representative, centers, axes in canonical
    )
    if len(representatives) != 313 or max(records) >= 2**16:
        raise ValueError("CPU ERI symmetry inventory exceeded its declared bound")
    return representatives, records


@cache
def eri_cpu_coulomb_layout() -> tuple[
    tuple[tuple[int, int, int], ...], tuple[tuple[int, ...], ...]
]:
    """Graded roots and lossless physical-axis indices for each canonical view."""
    orders = tuple(order for order, _ in build_coulomb_derivative_algebra(8).roots)
    indices = {order: index for index, order in enumerate(orders)}
    permutations_ = []
    for axes in AXIS_PERMUTATIONS:
        row = []
        for order in orders:
            physical = [0, 0, 0]
            for axis, original in enumerate(axes):
                physical[original] = order[axis]
            row.append(indices[(physical[0], physical[1], physical[2])])
        permutations_.append(tuple(row))
    return orders, tuple(permutations_)


def _value_integral(angular: tuple[int, ...]) -> IntegralIR:
    if len(angular) != 4:
        raise ValueError("CPU ERI values require four angular momenta")
    spec = ShellClassSpec(
        "".join("spd"[item] for item in angular),
        (angular[0], angular[1], angular[2], angular[3]),
    )
    signature = ShellSignature.from_shell_class(spec)
    return build_integral_ir(
        spec,
        contractions=(
            RawBlock(
                TensorLayout(signature.tensor_indices, signature.component_shape),
                4 * 1024**2,
            ),
        ),
    )


def eri_cpu_inventory() -> dict[str, Any]:
    """Describe the exact unscreened Cartesian FP64 value-only promotion."""
    representatives, _ = eri_cpu_component_map()
    return {
        "schema": "generativeqc.eri_cpu",
        "version": 3,
        "precision": "fp64",
        "maximum_angular": 2,
        "maximum_coulomb_order": 8,
        "derivative_order": 0,
        "ordered_component_count": len(COMPONENTS) ** 4,
        "representative_count": len(representatives),
        "primitive_geometry": {
            "abi_version": 1,
            "reuse_scope": "primitive_quartet",
            "maximum_order": 8,
            "required_order": "sum_shell_angular_momenta",
            "scalar_count": len(_geometry_fields()) + 9,
            "boys_value_count": 9,
        },
        "component_schedule": {
            "kind": "prepared_geometry_coulomb_symmetry_representatives",
            "record_bits": 16,
            "maximum_shell_component_count": len(cartesian_components(2)) ** 4,
            "component_scratch_owner": "native_caller",
        },
        "coulomb_reuse": {
            "abi_version": 1,
            "reuse_scope": "primitive_quartet",
            "maximum_order": 8,
            "scalar_count": 165,
            "readiness": "matching_maximum_order",
            "order_prefix_counts": [
                (order + 1) * (order + 2) * (order + 3) // 6 for order in range(9)
            ],
            "scratch_owner": "native_caller",
            "initialization": "requested_graded_prefix_only",
            "compatibility_schedule": "component_local_factored_dag",
        },
        "components": list(COMPONENTS),
        "values": [
            integral_to_payload(_value_integral(angular))
            for angular in product(range(3), repeat=4)
        ],
    }


def _geometry_fields() -> dict[str, str]:
    """Storage for exact common roots exported by the shared component DAG."""
    return {
        "inverse_two_p": "inverse_two[0]",
        "inverse_two_q": "inverse_two[1]",
        "rho": "rho",
        **{
            f"{pair}_{axis}": f"pair_shifts[{slot}][{coordinate}]"
            for slot, pair in enumerate(("pa", "pb", "qc", "qd"))
            for coordinate, axis in enumerate("xyz")
        },
        **{
            f"difference_{axis}": f"difference[{coordinate}]"
            for coordinate, axis in enumerate("xyz")
        },
        "boys_argument": "boys_argument",
        "prefactor": "prefactor",
    }


def _emit_geometry() -> str:
    integral = _value_integral((0, 0, 0, 0))
    kernel = build_shell_class_component_kernel(
        integral.spec, ("", "", "", ""), integral=integral
    )
    graph, roots = kernel.graph.apply_algebra_form(
        tuple(expression for _, expression in kernel.geometry_roots),
        AlgebraForm.FACTORED_NARY,
        PowerLowering.SMALL_INTEGER,
    )
    variables = {
        name: f"exponents[{slot}]"
        for slot, name in enumerate(("alpha", "beta", "gamma", "delta"))
    }
    variables.update(
        {
            f"{center}_{axis}": f"centers[{slot}][{coordinate}]"
            for slot, center in enumerate(("first", "second", "third", "fourth"))
            for coordinate, axis in enumerate("xyz")
        }
    )
    variables["kPi"] = "3.141592653589793238462643383279502884"
    emitter = ScalarCEmitter(graph, variables)
    fields = _geometry_fields()
    for (name, _), root in zip(kernel.geometry_roots, roots, strict=True):
        emitter.emit_assignment(root, f"geometry.{fields[name]}")
    lines = [
        "/** Component-independent primitive quartet; evaluate Boys only once. */",
        "struct Geometry {",
        "  double inverse_two[2], rho, pair_shifts[4][3], difference[3];",
        "  double boys_argument, prefactor;",
        "  double boys[9];",
        "  unsigned maximum_order;",
        "};",
        "/** Positive finite inputs are caller-validated; maximum_order must be in [0,8]. */",
        "inline Geometry make_geometry(const double (&exponents)[4], const double (&centers)[4][3],",
        "                              unsigned maximum_order) {",
        "  Geometry geometry{};",
        "  geometry.maximum_order = maximum_order;",
        "  if (maximum_order > 8U) return geometry;",
        *emitter.lines,
        "  switch (maximum_order) {",
    ]
    lines.extend(
        f"    case {order}: boys_values<{order}>(geometry.boys_argument, geometry.boys); break;"
        for order in range(9)
    )
    lines.extend(("  }", "  return geometry;", "}"))
    return "\n".join(lines)


def _emit_component(index: int, component_indices: tuple[int, ...]) -> str:
    components = tuple(COMPONENTS[item] for item in component_indices)
    integral = _value_integral(tuple(map(len, components)))
    kernel = build_shell_class_component_kernel(
        integral.spec, components, integral=integral
    )
    shared = _emit_component_body(kernel, shared_coulomb=True)
    local = _emit_component_body(kernel, shared_coulomb=False)
    return "\n".join(
        [
            f"// Cartesian representative {components!r}; value-only IntegralIR.",
            "template <bool SharedCoulomb>",
            f"static inline double component_{index}(const Geometry& geometry,",
            "    [[maybe_unused]] const CoulombValues* coulomb,",
            "    [[maybe_unused]] const unsigned* center_order,",
            "    [[maybe_unused]] const unsigned* axis_order,",
            "    [[maybe_unused]] const std::uint8_t* coulomb_order,",
            "    [[maybe_unused]] double difference_sign) {",
            "  if constexpr (SharedCoulomb) {",
            *("  " + line for line in shared),
            "  } else {",
            *("  " + line for line in local),
            "  }",
            "}",
        ]
    )


def _emit_component_body(
    kernel: ShellClassComponentKernel, *, shared_coulomb: bool
) -> list[str]:
    # Cut the original DAG at its exact shared geometry nodes before algebraic
    # factoring. No recurrence or Gaussian-product equations live in this backend.
    replacements = {
        root: kernel.graph.variable(f"geometry_{name}")
        for name, root in kernel.geometry_roots
    }
    orders, _ = eri_cpu_coulomb_layout()
    order_indices = {order: index for index, order in enumerate(orders)}
    if shared_coulomb:
        replacements.update(
            {
                root: kernel.graph.variable(f"coulomb_{order_indices[order]}")
                for order, root in kernel.coulomb_roots
            }
        )
    prepared_value = kernel.graph.replace_subexpressions((kernel.value,), replacements)
    graph, roots = kernel.graph.apply_algebra_form(
        prepared_value,
        AlgebraForm.FACTORED_NARY,
        PowerLowering.SMALL_INTEGER,
    )
    variables = {
        f"geometry_{name}": f"geometry.{field}"
        for name, field in _geometry_fields().items()
    }
    variables.update(
        {
            "geometry_inverse_two_p": "geometry.inverse_two[center_order[0] / 2]",
            "geometry_inverse_two_q": "geometry.inverse_two[center_order[2] / 2]",
            **{
                f"geometry_{pair}_{axis}": f"geometry.pair_shifts[center_order[{slot}]][axis_order[{coordinate}]]"
                for slot, pair in enumerate(("pa", "pb", "qc", "qd"))
                for coordinate, axis in enumerate("xyz")
            },
            **{
                f"geometry_difference_{axis}": f"(difference_sign * geometry.difference[axis_order[{coordinate}]])"
                for coordinate, axis in enumerate("xyz")
            },
        }
    )
    variables.update({f"boys_{order}": f"geometry.boys[{order}]" for order in range(9)})
    if shared_coulomb:
        variables.update(
            {
                f"coulomb_{index}": (
                    "geometry.boys[0]"
                    if index == 0
                    else (
                        f"(difference_sign * coulomb->values[coulomb_order[{index}]])"
                        if sum(order) % 2
                        else f"coulomb->values[coulomb_order[{index}]]"
                    )
                )
                for index, order in enumerate(orders)
            }
        )
    required_variables = {
        str(graph.nodes[identifier].payload)
        for identifier in graph.topological_order(roots)
        if graph.nodes[identifier].operation == "variable"
    }
    if not required_variables.issubset(variables):
        raise ValueError(
            "prepared CPU ERI component retained primitive geometry inputs"
        )
    # Name each used view field once rather than expanding permutation indexing
    # throughout the recurrence. This also keeps the generated header bounded.
    references = {
        name: f"g{index}"
        for index, name in enumerate(variables)
        if name in required_variables
    }
    emitter = ScalarCEmitter(graph, references)
    emitter.emit(roots)
    return [
        *(
            f"  const double {reference} = {variables[name]};"
            for name, reference in references.items()
        ),
        *emitter.lines,
        f"  return {emitter.reference(roots[0])};",
    ]


def _emit_coulomb() -> str:
    algebra = build_coulomb_derivative_algebra(8)
    orders, permutations_ = eri_cpu_coulomb_layout()
    graph, roots = algebra.graph.apply_algebra_form(
        tuple(root for _, root in algebra.roots),
        AlgebraForm.FACTORED_NARY,
        PowerLowering.SMALL_INTEGER,
    )
    variables = {"rho": "geometry.rho"}
    variables.update(
        {
            f"difference_{axis}": f"geometry.difference[{i}]"
            for i, axis in enumerate("xyz")
        }
    )
    variables.update({f"boys_{order}": f"geometry.boys[{order}]" for order in range(9)})
    emitter = ScalarCEmitter(graph, variables)
    for index, (order, root) in enumerate(zip(orders, roots, strict=True)):
        emitter.emit_assignment(root, f"coulomb.values[{index}]")
        if index + 1 == len(orders) or sum(orders[index + 1]) != sum(order):
            emitter.lines.append(
                f"  if (geometry.maximum_order == {sum(order)}U) return;"
            )
    lines = [
        "/** Caller-owned scratch; only the requested graded prefix is initialized. */",
        "struct CoulombValues { double values[165]; unsigned maximum_order = 9U; };",
        "/** Fill once per geometry, before any shared-component consumers. */",
        "inline void prepare_coulomb(const Geometry& geometry, CoulombValues& coulomb) {",
        "  coulomb.maximum_order = geometry.maximum_order;",
        "  if (geometry.maximum_order > 8U) return;",
        *emitter.lines,
        "}",
        "inline constexpr std::uint8_t coulomb_axis_permutations[6][165] = {",
        *("  {" + ", ".join(map(str, row)) + "}," for row in permutations_),
        "};",
    ]
    return "\n".join(lines)


def emit_eri_cpu() -> str:
    """Emit one bounded header; native callers own all contraction policy."""
    representatives, records = eri_cpu_component_map()
    boys = _emit_boys_support(8).replace("__device__ __forceinline__", "inline")
    lines = [
        "// Generated by tools/generate_eri_cpu.py; do not edit.",
        "#ifndef GENERATIVEQC_GENERATED_ERI_CPU_HPP",
        "#define GENERATIVEQC_GENERATED_ERI_CPU_HPP",
        "#include <cmath>",
        "#include <cstdint>",
        "namespace generativeqc::integrals::generated_eri_cpu {",
        boys,
        _emit_geometry(),
        _emit_coulomb(),
        "/** CCA-ordered Cartesian s/p/d index; 10 denotes unsupported angular momentum. */",
        "constexpr unsigned component_index(unsigned x, unsigned y, unsigned z) {",
    ]
    for index, component in enumerate(COMPONENTS):
        x, y, z = (component.count(axis) for axis in "xyz")
        lines.append(f"  if (x == {x} && y == {y} && z == {z}) return {index};")
    lines.extend(("  return 10;", "}"))
    lines.extend(
        _emit_component(index, component)
        for index, component in enumerate(representatives)
    )
    for name, permutations_ in (
        ("center_permutations", CENTER_PERMUTATIONS),
        ("axis_permutations", AXIS_PERMUTATIONS),
    ):
        lines.append(
            f"inline constexpr unsigned {name}[{len(permutations_)}][{len(permutations_[0])}] = {{"
        )
        lines.extend("  {" + ", ".join(map(str, item)) + "}," for item in permutations_)
        lines.append("};")
    lines.append("inline constexpr std::uint16_t component_map[10000] = {")
    lines.extend(
        "  " + ", ".join(map(str, records[start : start + 20])) + ","
        for start in range(0, len(records), 20)
    )
    lines.extend(
        (
            "};",
            "inline constexpr unsigned representative_orders[313] = {",
            "  "
            + ", ".join(
                str(sum(len(COMPONENTS[item]) for item in component))
                for component in representatives
            ),
            "};",
            "/** Hoist this lossless lookup outside the primitive-quartet contraction loop. */",
            "inline unsigned component_record(const unsigned (&components)[4]) {",
            "  for (unsigned slot = 0; slot < 4; ++slot)",
            "    if (components[slot] >= 10) return 0xffffffffU;",
            "  const unsigned key = ((components[0] * 10 + components[1]) * 10 + components[2]) * 10 + components[3];",
            "  return component_map[key];",
            "}",
            "/** Unnormalized, unscreened component using previously prepared geometry. */",
            "template <bool SharedCoulomb>",
            "inline double primitive_dispatch(const Geometry& geometry, const CoulombValues* coulomb, unsigned record) {",
            "  if ((record >> 6) >= 313U || (record & 7U) >= 6U ||",
            "      geometry.maximum_order > 8U ||",
            "      representative_orders[record >> 6] > geometry.maximum_order) return NAN;",
            "  if constexpr (SharedCoulomb)",
            "    if (coulomb == nullptr || coulomb->maximum_order != geometry.maximum_order) return NAN;",
            "  const auto& center_order = center_permutations[(record >> 3) & 7U];",
            "  const auto& axis_order = axis_permutations[record & 7U];",
            "  const auto& coulomb_order = coulomb_axis_permutations[record & 7U];",
            "  const double difference_sign = center_order[0] < 2U ? 1.0 : -1.0;",
            "  switch (record >> 6) {",
        )
    )
    lines.extend(
        f"    case {index}: return component_{index}<SharedCoulomb>(geometry, coulomb, center_order, axis_order, coulomb_order, difference_sign);"
        for index in range(len(representatives))
    )
    lines.extend(
        (
            "  }",
            "  return NAN;",
            "}",
            "/** The scratch must have been prepared from this exact geometry. */",
            "inline double prepared_primitive(const Geometry& geometry, const CoulombValues& coulomb, unsigned record) {",
            "  return primitive_dispatch<true>(geometry, &coulomb, record);",
            "}",
            "/** Single-component compatibility preserves its component-local CSE schedule. */",
            "inline double prepared_primitive(const Geometry& geometry, unsigned record) {",
            "  return primitive_dispatch<false>(geometry, nullptr, record);",
            "}",
            "/** Compatibility wrapper for callers without a shell-quartet reuse schedule. */",
            "inline double primitive(const double (&exponents)[4], const double (&centers)[4][3],",
            "                        const unsigned (&components)[4]) {",
            "  const unsigned record = component_record(components);",
            "  if (record == 0xffffffffU) return NAN;",
            "  return prepared_primitive(make_geometry(exponents, centers, representative_orders[record >> 6]), record);",
            "}",
            "}  // namespace generativeqc::integrals::generated_eri_cpu",
            "#endif",
            "",
        )
    )
    return "\n".join(lines)
