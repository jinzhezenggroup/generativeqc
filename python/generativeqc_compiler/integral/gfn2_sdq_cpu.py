"""Host lowering for compiler-owned primitive GFN2 S/D/Q integrals.

The generated artifact owns only primitive Cartesian mathematics. Contracted
coefficients, shell traversal, screening, Cartesian-to-spherical transforms,
matrix packing, and reverse origin translation remain native runtime policy.
"""

from __future__ import annotations

from functools import cache
from itertools import product

from .expr import Expr, Graph
from .gfn2_sdq import (
    GFN2_SDQ_COMPONENTS,
    Gfn2SdqPrimitiveKernel,
    build_gfn2_sdq_primitive_kernel,
)
from .scalar_c import ScalarCEmitter
from .shell_spec import AXES, cartesian_components

_MODES = (
    ("overlap", 1, False),
    ("overlap_gradient", 1, True),
    ("sdq_values", len(GFN2_SDQ_COMPONENTS), False),
    ("sdq", len(GFN2_SDQ_COMPONENTS), True),
)


def gfn2_sdq_cpu_inventory() -> dict[str, object]:
    """Describe the bounded native primitive contract."""
    component_pairs = sum(
        len(cartesian_components(bra)) * len(cartesian_components(ket))
        for bra, ket in product(range(3), repeat=2)
    )
    return {
        "schema": "generativeqc.gfn2_sdq_cpu",
        "version": 1,
        "precision": "fp64",
        "public_maximum_angular": 2,
        "operator_origin": "ket",
        "components": list(GFN2_SDQ_COMPONENTS),
        "gradient_center": "ket",
        "component_pairs": component_pairs,
        "entry_points": [mode[0] for mode in _MODES],
        "shell_block_entry_points": [f"{mode[0]}_shell_block" for mode in _MODES],
        "maximum_shell_block_pairs": 36,
        "shell_block_gaussian_prefactors_per_primitive_pair": 1,
    }


def _variables() -> dict[str, str]:
    variables = {"alpha": "bra_alpha", "beta": "ket_alpha"}
    for axis in AXES:
        variables[f"a_{axis}"] = "0.0"
    for index, axis in enumerate(AXES):
        variables[f"b_{axis}"] = f"vector[{index}]"
    return variables


@cache
def _shell_kernels(
    bra_angular: int,
    ket_angular: int,
) -> tuple[Gfn2SdqPrimitiveKernel, ...]:
    """Intern Cartesian alternatives together for exact cross-branch CSE."""

    graph = Graph()
    return tuple(
        build_gfn2_sdq_primitive_kernel(
            (bra_angular, ket_angular), (bra_component, ket_component), graph=graph
        )
        for bra_component in cartesian_components(bra_angular)
        for ket_component in cartesian_components(ket_angular)
    )


def _roots(
    kernel: Gfn2SdqPrimitiveKernel, value_count: int, gradient: bool
) -> tuple[Expr, ...]:
    values = tuple(kernel.values[:value_count])
    if not gradient:
        return values
    ket_gradients = tuple(
        root for axis in kernel.gradients[1] for root in axis[:value_count]
    )
    return (*values, *ket_gradients)


def _emit_shell_pair(
    tag: str,
    bra_angular: int,
    ket_angular: int,
    value_count: int,
    gradient: bool,
) -> str:
    ket_components = cartesian_components(ket_angular)
    ket_count = len(ket_components)
    kernels = _shell_kernels(bra_angular, ket_angular)
    graph = kernels[0].graph
    roots_by_case = tuple(_roots(kernel, value_count, gradient) for kernel in kernels)
    # Only expressions reachable from every alternative may be hoisted. This
    # preserves the mathematical DAG and avoids speculative unused arithmetic.
    # On CUDA it also evaluates the Gaussian prefactor before Cartesian lanes
    # diverge, instead of repeating exp/pow within each serialized switch arm.
    common = set.intersection(
        *(set(graph.topological_order(roots)) for roots in roots_by_case)
    )
    shared = ScalarCEmitter(graph, _variables())
    shared.emit(tuple(Expr(graph, identifier) for identifier in sorted(common)))
    name = f"evaluate_gfn2_{tag}_{bra_angular}{ket_angular}"
    lines = [
        f"inline bool {name}(unsigned bra_component, unsigned ket_component,",
        "    double bra_alpha, double ket_alpha, const double* vector,",
        "    Gfn2SdqPrimitive& result) noexcept {",
        *shared.lines,
        f"  switch (bra_component * {ket_count}U + ket_component) {{",
    ]
    for case, (kernel, roots) in enumerate(zip(kernels, roots_by_case, strict=True)):
        emitter = shared.fork()
        emitter.emit(roots)
        lines.append(f"    case {case}U: {{")
        lines.extend("    " + line for line in emitter.lines)
        for index, root in enumerate(kernel.values[:value_count]):
            lines.append(f"      result.values[{index}] = {emitter.reference(root)};")
        if gradient:
            for axis in range(3):
                for index, root in enumerate(kernel.gradients[1][axis][:value_count]):
                    lines.append(
                        f"      result.ket_gradient[{axis}][{index}] = "
                        f"{emitter.reference(root)};"
                    )
        lines += ["      return true;", "    }"]
    lines += ["  }", "  return false;", "}"]
    return "\n".join(lines)


def _emit_dispatch(tag: str) -> str:
    lines = [
        f"inline bool evaluate_gfn2_{tag}_primitive(",
        "    unsigned bra_angular, unsigned ket_angular, unsigned bra_component,",
        "    unsigned ket_component, double bra_alpha, double ket_alpha,",
        "    const double* vector, Gfn2SdqPrimitive& result) noexcept {",
        "  if (bra_angular > 2U || ket_angular > 2U) return false;",
        "  constexpr unsigned components[] = {1U, 3U, 6U};",
        "  if (bra_component >= components[bra_angular] || ket_component >= components[ket_angular]) return false;",
        "  if (!vector || !std::isfinite(bra_alpha) || !std::isfinite(ket_alpha) || bra_alpha <= 0.0 || ket_alpha <= 0.0) return false;",
        "  for (unsigned axis = 0; axis < 3U; ++axis) if (!std::isfinite(vector[axis])) return false;",
        "  switch (bra_angular * 3U + ket_angular) {",
    ]
    for bra, ket in product(range(3), repeat=2):
        lines.append(
            f"    case {bra * 3 + ket}U: return evaluate_gfn2_{tag}_{bra}{ket}("
            "bra_component, ket_component, bra_alpha, ket_alpha, vector, result);"
        )
    lines += ["  }", "  return false;", "}"]
    return "\n".join(lines)


def _emit_shell_block(
    tag: str, bra: int, ket: int, value_count: int, gradient: bool
) -> str:
    """Lower all Cartesian consumers of one primitive pair with shared CSE.

    CPU shell traversal consumes the entire block sequentially. Retaining one
    emitter across its outputs avoids repeating the Gaussian prefactor and
    one-dimensional recurrence terms for every Cartesian pair. Store each
    completed output promptly to keep unrelated output live ranges short.
    """
    kernels = _shell_kernels(bra, ket)
    emitter = ScalarCEmitter(kernels[0].graph, _variables())
    lines = [
        f"inline void evaluate_gfn2_{tag}_block_{bra}{ket}(",
        "    double bra_alpha, double ket_alpha, const double* vector,",
        "    Gfn2SdqPrimitive* result) noexcept {",
    ]
    for pair, kernel in enumerate(kernels):
        start = len(emitter.lines)
        emitter.emit(_roots(kernel, value_count, gradient))
        lines.extend(emitter.lines[start:])
        for index, root in enumerate(kernel.values[:value_count]):
            lines.append(
                f"  result[{pair}].values[{index}] = {emitter.reference(root)};"
            )
        if gradient:
            for axis in range(3):
                for index, root in enumerate(kernel.gradients[1][axis][:value_count]):
                    lines.append(
                        f"  result[{pair}].ket_gradient[{axis}][{index}] = {emitter.reference(root)};"
                    )
    return "\n".join([*lines, "}"])


def _emit_block_dispatch(tag: str) -> str:
    """Keep checked admission; callers supply space for all Cartesian pairs.

    A valid block writes only the value/gradient components selected by its
    entry point. Unrequested components retain their previous contents.
    """
    lines = [
        f"inline bool evaluate_gfn2_{tag}_shell_block(",
        "    unsigned bra_angular, unsigned ket_angular, double bra_alpha, double ket_alpha,",
        "    const double* vector, Gfn2SdqPrimitive* result) noexcept {",
        "  if (bra_angular > 2U || ket_angular > 2U || !result || !vector) return false;",
        "  if (!std::isfinite(bra_alpha) || !std::isfinite(ket_alpha) || bra_alpha <= 0.0 || ket_alpha <= 0.0) return false;",
        "  for (unsigned axis = 0; axis < 3U; ++axis) if (!std::isfinite(vector[axis])) return false;",
        "  switch (bra_angular * 3U + ket_angular) {",
    ]
    for bra, ket in product(range(3), repeat=2):
        lines.append(
            f"    case {bra * 3 + ket}U: evaluate_gfn2_{tag}_block_{bra}{ket}(bra_alpha, ket_alpha, vector, result); return true;"
        )
    return "\n".join([*lines, "  }", "  return false;", "}"])


@cache
def emit_gfn2_sdq_cuda() -> str:
    """Emit device primitive DAGs and host selection of the admission width."""

    from .gfn2_force_schedule import emit_gfn2_force_preflight_schedule

    return (
        _emit_gfn2_sdq_header(shell_blocks=False)
        .replace(
            "// Generated by tools/generate_gfn2_sdq_native.py; do not edit.",
            "// Generated by tools/generate_gfn2_sdq_native.py for CUDA; do not edit.",
            1,
        )
        .replace("inline bool ", "__device__ inline bool ")
        .replace("std::isfinite", "isfinite")
        .replace("#include <cmath>", "#include <cmath>\n#include <cstdint>", 1)
        .replace(
            "namespace generativeqc::xtb::generated {",
            "namespace generativeqc::xtb::generated {\n"
            + emit_gfn2_force_preflight_schedule(),
            1,
        )
    )


@cache
def emit_gfn2_sdq_cpu() -> str:
    """Emit cost-bounded s/p/d primitive entry points for each runtime use."""
    return _emit_gfn2_sdq_header(shell_blocks=True)


def _emit_gfn2_sdq_header(*, shell_blocks: bool) -> str:
    """Share primitive math while keeping CPU block scheduling off CUDA lanes."""
    helpers = []
    dispatches = []
    for tag, value_count, gradient in _MODES:
        helpers.extend(
            _emit_shell_pair(tag, bra, ket, value_count, gradient)
            for bra, ket in product(range(3), repeat=2)
        )
        dispatches.append(_emit_dispatch(tag))
        if shell_blocks:
            helpers.extend(
                _emit_shell_block(tag, bra, ket, value_count, gradient)
                for bra, ket in product(range(3), repeat=2)
            )
            dispatches.append(_emit_block_dispatch(tag))
    return "\n".join(
        [
            "// Generated by tools/generate_gfn2_sdq_native.py; do not edit.",
            "#pragma once",
            "#include <cmath>",
            "namespace generativeqc::xtb::generated {",
            "struct Gfn2SdqPrimitive {",
            f"  double values[{len(GFN2_SDQ_COMPONENTS)}]{{}};",
            f"  double ket_gradient[3][{len(GFN2_SDQ_COMPONENTS)}]{{}};",
            "};",
            *helpers,
            *dispatches,
            "}  // namespace generativeqc::xtb::generated",
            "",
        ]
    )
