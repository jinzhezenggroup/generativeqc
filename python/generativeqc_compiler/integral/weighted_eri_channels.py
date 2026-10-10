"""Source-driven derivative reuse for bounded external-weight shell channels.

The weighted derivative DAG is linear in its fixed component cotangents. AD
with respect to those cotangents exposes the existing, unweighted component
derivatives without introducing another integral recurrence. One scalar CSE
state serves the complete shell subset; each component is immediately consumed
by every active channel rather than stored in a derivative tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .expr import AlgebraForm
from .scalar_c import ScalarCEmitter

if TYPE_CHECKING:
    from .weighted_eri import WeightedEriKernel


def weighted_eri_geometry_bindings(kernel: WeightedEriKernel) -> dict[str, str]:
    """Bind the common scalar geometry contract independently of cotangents."""
    variables = {
        name: f"geometry.{name}"
        for name in ("inverse_two_p", "inverse_two_q", "rho", "prefactor")
    }
    for axis, label in enumerate("xyz"):
        variables[f"difference_{label}"] = f"geometry.difference[{axis}]"
        for slot, prefix in enumerate(("pa", "pb", "qc", "qd")):
            variables[f"{prefix}_{label}"] = f"geometry.shifts[{slot}][{axis}]"
        for center, prefix in enumerate(("first", "second", "third", "fourth")):
            variables[f"decay_{prefix}_{label}"] = f"geometry.decay[{center}][{axis}]"
    for center, prefix in enumerate(("first", "second", "third", "fourth")):
        variables[f"{prefix}_product_scale"] = f"geometry.product_scales[{center}]"
    for order in range(kernel.integral.maximum_coulomb_order + 1):
        variables[f"boys_{order}"] = f"geometry.boys[{order}]"
    return variables


def emit_weighted_eri_channel_function(
    kernel: WeightedEriKernel,
    name: str,
    *,
    backend: str = "cuda",
) -> str:
    """Emit independent-center derivatives shared by all requested channels.

    Weights use the full shell's component order, including for bounded subsets.
    Inactive channels and exact-zero component weights are zero operators, not
    numerical screening. No output is published on a nonfinite active component
    or contraction; the native caller can replay the retained weight-first DAG
    to preserve its wider intermediate dynamic range. Translation recovery and
    physical atom scatter remain in the caller. Under strict FP64, a nonfinite
    contribution cannot restore a finite running sum, so the final transactional
    audit also covers active component derivatives without per-component checks.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("weighted channel emission supports cpu or cuda")
    if not name.isascii() or not name.isidentifier():
        raise ValueError("weighted channel helper requires an ASCII identifier")
    source_roots = tuple(
        kernel.graph.differentiate(
            derivative, kernel.graph.variable(f"component_weight_{component}")
        )
        for component in kernel.component_indices
        for center in range(3)
        for derivative in kernel.gradients[center]
    )
    graph, roots = kernel.graph.apply_algebra_form(
        source_roots, AlgebraForm.FACTORED_NARY
    )
    variables = weighted_eri_geometry_bindings(kernel)
    if any(
        graph.nodes[identifier].operation == "variable"
        and str(graph.nodes[identifier].payload).startswith("component_weight_")
        for identifier in graph.topological_order(roots)
    ):
        raise ValueError("shared shell derivatives must be independent of weights")
    emitter = ScalarCEmitter(graph, variables)
    qualifier = "__device__ __forceinline__" if backend == "cuda" else "inline"
    lines = [
        "/** One shared shell recurrence; fixed external cotangents, transactional publication. */",
        "template <unsigned ChannelCount>",
        f"{qualifier} bool {name}_channels(const Geometry& geometry,",
        f"    const double (&component_weights)[ChannelCount][{kernel.spec.component_count}],",
        "    const bool (&active)[ChannelCount], IndependentGradient (&output)[ChannelCount]) {",
        '  static_assert(ChannelCount > 0U, "at least one derivative channel is required");',
        "  IndependentGradient candidate[ChannelCount]{};",
    ]
    for packed, component in enumerate(kernel.component_indices):
        component_roots = roots[9 * packed : 9 * (packed + 1)]
        previous_lines = len(emitter.lines)
        emitter.emit(component_roots)
        lines.extend(emitter.lines[previous_lines:])
        references = tuple(emitter.reference(root) for root in component_roots)
        lines.extend(
            [
                "  for (unsigned channel = 0; channel < ChannelCount; ++channel) {",
                f"    if (!active[channel] || component_weights[channel][{component}] == 0.0) continue;",
            ]
        )
        for coordinate, reference in enumerate(references):
            lines.append(
                f"    candidate[channel].center[{coordinate // 3}][{coordinate % 3}] += "
                f"component_weights[channel][{component}] * {reference};"
            )
        lines.append("  }")
    lines.extend(
        [
            "  for (unsigned channel = 0; channel < ChannelCount; ++channel)",
            "    for (unsigned center = 0; center < 3; ++center)",
            "      for (unsigned axis = 0; axis < 3; ++axis)",
            "        if (!std::isfinite(candidate[channel].center[center][axis])) return false;",
            "  for (unsigned channel = 0; channel < ChannelCount; ++channel)",
            "    output[channel] = candidate[channel];",
            "  return true;",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"
