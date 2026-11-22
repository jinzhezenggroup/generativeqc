"""Energy-derived split-hybrid scalar programs for stationary CUDA consumers.

The new point domain is deliberately distinct from the incumbent Libxc work
coordinates. Select it before SCF; never attach its derivatives to an
incumbent final state. Exact exchange remains an independent Fock term.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from .libxc_maple import MapleImportError
from .split_hybrid_codegen import _component_work_policy, _selected_features
from .split_hybrid_program import build_split_global_hybrid
from .stationary_continuation import smooth_activation, smooth_floor

if TYPE_CHECKING:
    from generativeqc_compiler.integral.expr import Expr

    from .libxc_bulk import BulkProgram

_FEATURES = ("rho_a", "rho_b", "ax", "ay", "az", "bx", "by", "bz", "tau_a", "tau_b")
_POINT_SYMBOLS = {"M06-2X": ("M062X", "m06_2x"), "MN15": ("MN15", "mn15")}
# A relative density floor keeps the opposite spin fraction above the source's
# `1 ± zeta` guard even when the majority density is large. It is inactive for
# ordinary interior points and appears in the energy before differentiation.
_MINORITY_RELATIVE_FLOOR = Fraction(1, 1 << 46)


def split_hybrid_stationary_symbol(functional: str) -> str:
    """Return the prefix of the generated energy-derived entry points."""
    if functional not in _POINT_SYMBOLS:
        raise ValueError(f"no energy-derived split-hybrid point for {functional!r}")
    return f"{_POINT_SYMBOLS[functional][1]}_stationary"


def _correlation_flat_gradient_limit(program: BulkProgram) -> Expr:
    """Cancel the two squared roots in D_sigma *before* differentiating."""
    graph = program.graph
    reachable = set(graph.topological_order((program.energy,)))
    replacements: dict[Expr, Expr] = {}
    roots = []
    for density, sigma in zip(
        program.variables[:2], program.variables[2:5:2], strict=True
    ):
        density_scale = density.pow(-4.0 / 3.0)
        radical = sigma.pow(0.5)
        reduced_gradient = radical * density_scale
        squared = reduced_gradient * reduced_gradient
        if squared.identifier not in reachable:
            raise MapleImportError("M06-2X D_sigma square changed in the pinned graph")
        replacements[squared] = sigma * density_scale.pow(2)
        roots.append(radical.identifier)
    (energy,) = graph.replace_subexpressions((program.energy,), replacements)
    if set(roots).intersection(graph.topological_order((energy,))):
        raise MapleImportError("M06-2X has another sqrt(sigma) use at a flat gradient")
    return energy


def _component_energy(
    program: BulkProgram, *, functional: str, exchange: bool
) -> tuple[Expr, tuple[Expr, ...]]:
    """Bind the pinned component to physical inputs, then continue its energy."""
    _selected_features(program)  # Reject an unhandled Laplacian dependence.
    graph = program.graph
    policy = _component_work_policy(program)
    threshold = policy.density_threshold
    physical = tuple(graph.variable(f"point_{name}") for name in _FEATURES)
    density_a, density_b, *gradients, tau_a, tau_b = physical
    grad_a, grad_b = gradients[:3], gradients[3:]
    sigma_a, sigma_b = (
        graph.sum(value.pow(2) for value in grad) for grad in (grad_a, grad_b)
    )
    tau_limit = graph.approximate_constant(policy.tau_floor)
    work_tau = tuple(smooth_floor(tau, tau_limit) for tau in (tau_a, tau_b))
    zero = graph.constant(0)
    interior = program.energy
    if functional == "M06-2X" and not exchange:
        interior = _correlation_flat_gradient_limit(program)
    elif functional == "MN15":
        # MN15 has no D_sigma factor. Do not silently carry a new singular
        # gradient radical into the action if the pinned source changes.
        sigma_roots = (sigma.pow(0.5) for sigma in program.variables[2:5:2])
        reachable = set(graph.topological_order((interior,)))
        if any(root.identifier in reachable for root in sigma_roots):
            raise MapleImportError("MN15 introduces sqrt(sigma) at a flat gradient")

    def bind(densities: tuple[Expr, Expr], cross_sigma: Expr) -> Expr:
        work = (*densities, sigma_a, cross_sigma, sigma_b, zero, zero, *work_tau)
        (energy,) = graph.replace_subexpressions(
            (interior,), dict(zip(program.variables, work, strict=True))
        )
        return energy

    if exchange:
        # The source screens the inactive spin at the threshold. A populated
        # spin always uses its own physical gradient norm, even at sigma=0.
        inactive = graph.approximate_constant(threshold)
        energy = smooth_activation(
            density_a, threshold, bind((density_a, inactive), zero)
        )
        energy += smooth_activation(
            density_b, threshold, bind((inactive, density_b), zero)
        )
        return energy, physical

    total = density_a + density_b
    # The minimum exceeds the source's internal hard spin screen. A floor
    # placed at that screen would cross a discontinuity during its own join.
    limit = graph.approximate_constant(2 * threshold) + total * _MINORITY_RELATIVE_FLOOR
    work_a, work_b = (smooth_floor(rho, limit) for rho in (density_a, density_b))
    cross = graph.sum(alpha * beta for alpha, beta in zip(grad_a, grad_b, strict=True))
    energy = bind((work_a, work_b), cross) * total / (work_a + work_b)
    return smooth_activation(total, threshold, energy), physical


def emit_split_hybrid_stationary_point(
    functional: str, *, with_action: bool, qualifier: str = "inline"
) -> str:
    """Emit FP64 energy and physical potential, optionally with their HVP.

    Point coordinates are rho_a/rho_b, three alpha gradients, three beta
    gradients, and tau_a/tau_b (half the occupied-orbital gradient norm). The
    tangent follows the same layout. The caller owns physical nonnegative
    densities and kinetic densities, and finite point inputs.
    """
    prefix = split_hybrid_stationary_symbol(functional)
    type_prefix = _POINT_SYMBOLS[functional][0]
    method = build_split_global_hybrid(functional)
    count = len(_FEATURES)
    mode = "action" if with_action else "value"
    value_type = f"{type_prefix}Stationary{mode.title()}"
    fields = f"double energy; double potential[{count}];"
    parameters = "const double* point"
    arguments = "point"
    if with_action:
        fields += f" double action[{count}];"
        parameters += ", const double* direction"
        arguments += ", direction"
    lines = [f"struct {value_type} {{ {fields} }};"]

    for component, label in (
        (method.exchange, "exchange"),
        (method.correlation, "correlation"),
    ):
        energy, features = _component_energy(
            component, functional=functional, exchange=label == "exchange"
        )
        graph = energy.graph
        potential = tuple(graph.differentiate(energy, feature) for feature in features)
        roots = (energy, *potential)
        bindings = {
            f"point_{name}": f"point[{index}]" for index, name in enumerate(_FEATURES)
        }
        if with_action:
            leaf_directions = {
                f"point_{name}": graph.variable(f"dir_{name}") for name in _FEATURES
            }
            roots += tuple(
                graph.differentiate(root, features[0], leaf_derivatives=leaf_directions)
                for root in potential
            )
            bindings.update(
                {
                    f"dir_{name}": f"direction[{index}]"
                    for index, name in enumerate(_FEATURES)
                }
            )
        graph, roots = graph.lower_small_integer_powers(roots)
        emitter = ScalarCEmitter(graph, bindings, coalesce_branches=True)
        emitter.emit(roots)
        energy_ref, *partials = map(emitter.reference, roots)
        initializers = [energy_ref, "{" + ", ".join(partials[:count]) + "}"]
        if with_action:
            initializers.append("{" + ", ".join(partials[count:]) + "}")
        lines.extend(
            (
                f"{qualifier} {value_type} {prefix}_{label}_{mode}({parameters}) {{",
                *emitter.lines,
                "  return {" + ", ".join(initializers) + "};",
                "}",
            )
        )
    lines.extend(
        (
            f"{qualifier} {value_type} {prefix}_{mode}({parameters}) {{",
            f"  auto out = {prefix}_exchange_{mode}({arguments});",
            f"  const auto correlation = {prefix}_correlation_{mode}({arguments});",
            "  out.energy += correlation.energy;",
            f"  for (unsigned feature = 0; feature < {count}; ++feature) {{",
            "    out.potential[feature] += correlation.potential[feature];",
            *(
                ("    out.action[feature] += correlation.action[feature];",)
                if with_action
                else ()
            ),
            "  }",
            "  return out;",
            "}",
            "",
        )
    )
    return "\n".join(lines)
