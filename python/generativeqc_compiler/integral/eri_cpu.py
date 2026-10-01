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
from .shell_class import build_shell_class_component_kernel
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
        "version": 1,
        "precision": "fp64",
        "maximum_angular": 2,
        "maximum_coulomb_order": 8,
        "derivative_order": 0,
        "ordered_component_count": len(COMPONENTS) ** 4,
        "representative_count": len(representatives),
        "components": list(COMPONENTS),
        "values": [
            integral_to_payload(_value_integral(angular))
            for angular in product(range(3), repeat=4)
        ],
    }


def _emit_component(index: int, component_indices: tuple[int, ...]) -> str:
    components = tuple(COMPONENTS[item] for item in component_indices)
    integral = _value_integral(tuple(map(len, components)))
    kernel = build_shell_class_component_kernel(
        integral.spec, components, integral=integral
    )
    graph, roots = kernel.graph.apply_algebra_form(
        (kernel.boys_argument, kernel.value),
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
    variables.update({f"boys_{order}": f"boys[{order}]" for order in range(9)})
    emitter = ScalarCEmitter(graph, variables)
    emitter.emit((roots[0],))
    lines = [
        f"// Cartesian representative {components!r}; value-only IntegralIR.",
        f"static inline double component_{index}(const double* exponents, const double (*centers)[3]) {{",
        *emitter.lines,
        f"  double boys[{integral.maximum_coulomb_order + 1}];",
        f"  boys_values<{integral.maximum_coulomb_order}>({emitter.reference(roots[0])}, boys);",
    ]
    emitter.lines.clear()
    emitter.emit((roots[1],))
    lines.extend((*emitter.lines, f"  return {emitter.reference(roots[1])};", "}"))
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
            "/** Unnormalized, unscreened primitive. Positive finite inputs are caller-validated. */",
            "inline double primitive(const double (&exponents)[4], const double (&centers)[4][3],",
            "                        const unsigned (&components)[4]) {",
            "  for (unsigned slot = 0; slot < 4; ++slot)",
            "    if (components[slot] >= 10) return NAN;",
            "  const unsigned key = ((components[0] * 10 + components[1]) * 10 + components[2]) * 10 + components[3];",
            "  const unsigned record = component_map[key];",
            "  const auto& center_order = center_permutations[(record >> 3) & 7U];",
            "  const auto& axis_order = axis_permutations[record & 7U];",
            "  double ordered_exponents[4], ordered_centers[4][3];",
            "  for (unsigned slot = 0; slot < 4; ++slot) {",
            "    ordered_exponents[slot] = exponents[center_order[slot]];",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      ordered_centers[slot][axis] = centers[center_order[slot]][axis_order[axis]];",
            "  }",
            "  switch (record >> 6) {",
        )
    )
    lines.extend(
        f"    case {index}: return component_{index}(ordered_exponents, ordered_centers);"
        for index in range(len(representatives))
    )
    lines.extend(
        (
            "  }",
            "  return NAN;",
            "}",
            "}  // namespace generativeqc::integrals::generated_eri_cpu",
            "#endif",
            "",
        )
    )
    return "\n".join(lines)
