"""Development lowering of a variational r2SCAN boundary continuation.

This is not the deployed Libxc-style work policy, and is not selected by
native SCF, force, or Hessian admission. It shares the pinned r2SCAN interior
expressions, but derives both the point potential and its directional action
from one energy through the continuation.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from .expression_dispatch import build_energy_expression
from .scan_maple import scan_runtime_policy
from .spec import FunctionalSpec
from .stationary_continuation import smooth_activation, smooth_floor

if TYPE_CHECKING:
    from generativeqc_compiler.integral.expr import Expr, Graph

# Native units: spin densities, Cartesian density gradients, and kinetic
# densities tau = (1/2) sum |grad(occupied orbital)|^2. Using gradients rather
# than the Gram matrix makes the correlation total gradient stable at exact
# and nearly antiparallel spin gradients.
_FEATURES = ("rho_a", "rho_b", "ax", "ay", "az", "bx", "by", "bz", "tau_a", "tau_b")


def _component_energy(component: str) -> tuple[Graph, Expr, tuple[Expr, ...]]:
    spec = FunctionalSpec(component, ((component, Fraction(1)),), spin="polarized")
    graph, raw_energy, raw_inputs = build_energy_expression(spec, production=True)
    # The replacement graph uses physical inputs, distinct from the raw
    # Libxc variables, and is differentiated only after conditioning.
    physical = tuple(graph.variable(f"p_{name}") for name in _FEATURES)
    rho_a, rho_b, *gradients, tau_a, tau_b = physical
    grad_a, grad_b = gradients[:3], gradients[3:]
    policy = scan_runtime_policy(component)
    rho_floor = policy.density_threshold
    physical_self_sigma = tuple(
        graph.sum(value.pow(2) for value in gradient) for gradient in (grad_a, grad_b)
    )
    tau_limit = graph.approximate_constant(policy.tau_threshold)
    work_tau = tuple(smooth_floor(tau, tau_limit) for tau in (tau_a, tau_b))

    if component == "MGGA_X_R2SCAN":
        # Exchange is spin additive. The raw graph screens a spin exactly at
        # its density threshold, so that spin contributes no energy when the
        # other spin's expression is extracted. The activation branch never
        # evaluates the active spin below its own threshold.
        inactive = graph.approximate_constant(rho_floor)
        spin_energies = []
        for spin, density in enumerate((rho_a, rho_b)):
            work_rho = [inactive, inactive]
            work_rho[spin] = density
            work = (
                *work_rho,
                physical_self_sigma[0],
                graph.constant(0),
                physical_self_sigma[1],
                *work_tau,
            )
            (energy,) = graph.replace_subexpressions(
                (raw_energy,), dict(zip(raw_inputs, work, strict=True))
            )
            spin_energies.append(smooth_activation(density, rho_floor, energy))
        return graph, graph.sum(spin_energies), physical

    rho_limit = graph.approximate_constant(rho_floor)
    work_rho_a, work_rho_b = (
        smooth_floor(value, rho_limit) for value in (rho_a, rho_b)
    )
    # Use the nonnegative physical gradient norm with no artificial floor on
    # sigma. A floor on the diagonal Gram entries would change the curvature
    # at exactly flat spin gradients. Correlation uses only total sigma.
    total_sigma = graph.sum((a + b).pow(2) for a, b in zip(grad_a, grad_b, strict=True))
    raw_total_sigma = raw_inputs[2] + 2 * raw_inputs[3] + raw_inputs[4]
    (energy,) = graph.replace_subexpressions(
        (raw_energy,),
        {
            raw_inputs[0]: work_rho_a,
            raw_inputs[1]: work_rho_b,
            raw_total_sigma: total_sigma,
            raw_inputs[5]: work_tau[0],
            raw_inputs[6]: work_tau[1],
        },
    )
    total = rho_a + rho_b
    # Keep the energy per physical volume, not the work-volume density.
    energy = energy * total / (work_rho_a + work_rho_b)
    return graph, smooth_activation(total, rho_floor, energy), physical


def emit_r2scan_stationary_point(
    *, with_action: bool, qualifier: str = "inline"
) -> str:
    """Emit FP64 energy/potential with an optional physical Hessian direction.

    A future SCF/force consumer needs only the value variant; the action
    variant takes the same ten features plus a ten-coordinate tangent. Both
    lower the same energy and differentiate the same physical inputs. The
    caller must establish finite gradients and nonnegative densities and
    kinetic densities. This development ABI does not qualify production use.
    """
    count = len(_FEATURES)
    mode = "action" if with_action else "value"
    value_type = f"R2scanStationary{mode.title()}"
    parameters = "const double* point"
    arguments = "point"
    fields = f"double energy; double potential[{count}];"
    if with_action:
        parameters += ", const double* direction"
        arguments += ", direction"
        fields += f" double action[{count}];"
    lines = [
        f"struct {value_type} {{ {fields} }};",
    ]
    for name, label in (("MGGA_X_R2SCAN", "x"), ("MGGA_C_R2SCAN", "c")):
        graph, energy, features = _component_energy(name)
        potential = tuple(graph.differentiate(energy, feature) for feature in features)
        roots = (energy, *potential)
        bindings = {
            f"p_{name}": f"point[{index}]" for index, name in enumerate(_FEATURES)
        }
        if with_action:
            leaf_directions = {
                f"p_{name}": graph.variable(f"d_{name}") for name in _FEATURES
            }
            roots += tuple(
                graph.differentiate(root, features[0], leaf_derivatives=leaf_directions)
                for root in potential
            )
            bindings.update(
                {
                    f"d_{name}": f"direction[{index}]"
                    for index, name in enumerate(_FEATURES)
                }
            )
        graph, roots = graph.lower_small_integer_powers(roots)
        unbound = {
            str(graph.nodes[index].payload)
            for index in graph.topological_order(roots)
            if graph.nodes[index].operation == "variable"
        } - bindings.keys()
        if unbound:
            raise ValueError(f"unbound r2SCAN stationary features: {sorted(unbound)}")
        # Share conditioning branches across energy, potential, and action.
        emitter = ScalarCEmitter(graph, bindings, coalesce_branches=True)
        emitter.emit(roots)
        energy_ref, *partials = map(emitter.reference, roots)
        initializers = [energy_ref, "{" + ", ".join(partials[:count]) + "}"]
        if with_action:
            initializers.append("{" + ", ".join(partials[count:]) + "}")
        lines.extend(
            (
                f"{qualifier} {value_type} r2scan_stationary_{label}_{mode}({parameters}) {{",
                *emitter.lines,
                "  return {" + ", ".join(initializers) + "};",
                "}",
            )
        )
    lines.extend(
        (
            f"{qualifier} {value_type} r2scan_stationary_{mode}({parameters}) {{",
            f"  auto out = r2scan_stationary_x_{mode}({arguments});",
            f"  const auto correlation = r2scan_stationary_c_{mode}({arguments});",
            "  out.energy += correlation.energy;",
            f"  for (unsigned index = 0; index < {count}; ++index) {{",
            "    out.potential[index] += correlation.potential[index];",
            *(
                ("    out.action[index] += correlation.action[index];",)
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
