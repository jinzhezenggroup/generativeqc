"""Common CPU/CUDA physical-spin point lowering for automatic Libxc Graphs.

Mathematics and the versioned Libxc work policy have one compiler owner. Native
runtime descriptors, scheduling, allocation, and public method admission remain
with their existing consumers; this module never imports the runtime or oracle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fractions import Fraction

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from .bulk_runtime import build_bulk_runtime_program
from .constant_parameters import lift_point_constants
from .density_work import DENSITY_WORK_DOMAIN, DENSITY_WORK_DOMAIN_VERSION
from .equal_spin import equal_spin_roots
from .libxc_work import (
    LIBXC_WORK_DOMAIN,
    LIBXC_WORK_DOMAIN_VERSION,
    automatic_work_policy,
    polarized_work_setup,
)
from .parameter_products import prepare_parameter_products
from .physical_kinetic_predicate import (
    PREDICATE_VARIABLE,
    bind_physical_kinetic_predicate,
    emit_physical_kinetic_side,
)

AUTOMATIC_POINT_LAYOUTS = {
    ("rho",): 1,
    ("rho_a", "rho_b"): 1,
    ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb"): 7,
    (
        "rho_a",
        "rho_b",
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "tau_a",
        "tau_b",
    ): 15,
}


@dataclass(frozen=True)
class AutomaticPointSource:
    """Backend-specific source with one backend-neutral E/vxc semantic identity.

    ``source_identity`` distinguishes emitted ABI/guard bytes, not cache slots or
    admission. Existing AOT/runtime owners must bind compiler/schedule/resource
    provenance when this source becomes an executable artifact.
    """

    name: str
    backend: Literal["cpu", "cuda"]
    function_name: str
    ingredient_mask: int
    features: tuple[str, ...]
    outputs: tuple[tuple[int, ...], ...]
    expression_identity: str
    source: str
    spin: Literal["polarized", "unpolarized"]
    domain: str
    domain_version: int
    parameters: tuple[tuple[str, float], ...] = ()

    @property
    def mathematical_contract(self) -> dict:
        """Bind derivatives to their actual layout, not an equal-input inference."""
        return {
            "expression_identity": self.expression_identity,
            "spin": self.spin,
            "domain": self.domain,
            "domain_version": self.domain_version,
            "features": self.features,
            "outputs": self.outputs,
            "ingredient_mask": self.ingredient_mask,
            "normalization": (
                "original-total-density-partial"
                if self.spin == "unpolarized"
                else "physical-spin-partials"
            ),
        }

    @property
    def source_identity(self) -> str:
        """Hash exact emitted bytes and layout without assigning runtime support."""
        return canonical_hash(
            {
                "schema": "generativeqc.automatic-libxc-point-source.v2",
                "backend": self.backend,
                "mathematics": self.mathematical_contract,
                "source": self.source,
                **({"parameters": self.parameters} if self.parameters else {}),
            }
        )


def emit_automatic_point(
    name: str,
    *,
    backend: Literal["cpu", "cuda"] = "cpu",
    equal_spin: bool = False,
    spin: Literal["polarized", "unpolarized"] = "polarized",
    parameter_names: tuple[str, ...] = (),
    precompute_parameters: bool = False,
    parameterize_constants: bool = False,
) -> AutomaticPointSource:
    """Lower one ordinary semilocal Graph into the native physical-spin ABI.

    Both backends evaluate the same symbolic E/vxc roots at Libxc work inputs.
    Work floors/clamps are deliberately not differentiated. Energy keeps its
    original-density weighting, Cartesian gradients apply the sigma chain rule,
    and kinetic coefficients include the native half-vtau convention. CUDA
    reports invalid physical inputs/outputs with ``DevicePointValue.valid``;
    CPU retains domain exceptions. Unchanged work inputs use exact original
    Cartesian coordinates for kinetic predicates, retaining branch payloads.
    The optional CUDA slice binds paired work
    inputs after differentiation, retaining every polarized partial. Only a
    producer proving equal physical spins may call that slice. This source
    grants no runtime admission. Explicit unpolarized density work instead uses
    the canonical U1 Graph: paired transport channels are restored to total
    density before any worker floor, and the scalar derivative is duplicated
    only for the existing native half-channel contraction. This is not the
    polarized equal-spin optimization, which retains P2 derivative roots.

    Explicit parameter_names add a final read-only coefficient pointer to the
    emitted point ABI. The caller supplies values in that exact order; fixed
    worker thresholds and physical-input guards remain outside the parameters.
    Parameters are held constant during feature-coordinate differentiation.
    With precompute_parameters, coefficient/literal products are prepared after
    AD. The returned parameters describe the complete ordered launch-data ABI.
    Alternatively, parameterize_constants lifts arithmetic literals only after
    ordinary specialization and AD, preserving the literal mathematical identity.
    The package planner must match both spin bodies and bind the returned rows.
    """
    if parameterize_constants and (parameter_names or precompute_parameters):
        raise ValueError("literal and symbolic point parameterization are exclusive")
    if backend not in ("cpu", "cuda"):
        raise ValueError("automatic point backend must be cpu or cuda")
    if equal_spin and backend != "cuda":
        raise ValueError("automatic equal-spin point lowering requires CUDA")
    if spin not in ("polarized", "unpolarized"):
        raise ValueError("automatic point spin must be polarized or unpolarized")
    if equal_spin and spin != "polarized":
        raise ValueError("equal-spin lowering requires polarized derivative roots")
    from ._generated_semilocal_compositions import COMPOSITIONS

    if name in COMPOSITIONS:
        if parameter_names or precompute_parameters or parameterize_constants:
            raise ValueError("composed point workers retain component-local constants")
        return _emit_composed_point(
            name, backend=backend, spin=spin, equal_spin=equal_spin
        )
    policy = automatic_work_policy(name)
    density_only = policy.family == "lda"
    if spin == "unpolarized" and not density_only:
        raise ValueError("automatic unpolarized point requires density-only work")
    program = build_bulk_runtime_program(
        name,
        spin=spin,
        order=1,
        parameter_names=parameter_names,
        **({"domain": DENSITY_WORK_DOMAIN} if density_only else {}),
    )
    features = program.spec.features
    try:
        ingredient_mask = AUTOMATIC_POINT_LAYOUTS[features]
    except KeyError as exc:
        raise ValueError(
            f"unsupported automatic Libxc point layout for {name}: {features!r}"
        ) from exc
    expected_outputs = (
        (),
        *((feature_index,) for feature_index in range(len(features))),
    )
    if program.outputs != expected_outputs:
        raise ValueError(f"automatic Libxc E/vxc output layout changed: {name}")

    if density_only:
        # The projected Graph owns floors and original-density energy weighting.
        # Passing floored inputs here would reweight energy a second time.
        work_lines = (
            "  const double total_density = rho_a + rho_b;",
            f"  if (total_density < {float(policy.density_threshold).hex()}) return {{}};",
        )
        work_arguments = ("total_density",) if spin == "unpolarized" else features
    else:
        work_lines, work_arguments = polarized_work_setup(policy, features)
    variables = dict(zip(features, work_arguments, strict=True))
    roots, physical_kinetic = bind_physical_kinetic_predicate(
        program.graph, program.roots
    )
    parameters = program.parameters
    if precompute_parameters:
        roots, parameters = prepare_parameter_products(program.graph, roots, parameters)
    if parameterize_constants:
        roots, parameters = lift_point_constants(program.graph, roots)
    variables.update(
        {name: f"parameters[{index}]" for index, (name, _) in enumerate(parameters)}
    )
    if equal_spin:
        roots = equal_spin_roots(program.graph, roots)
    kinetic_lines = emit_physical_kinetic_side() if physical_kinetic else ()
    if physical_kinetic:
        variables[PREDICATE_VARIABLE] = PREDICATE_VARIABLE
    emitter = ScalarCEmitter(program.graph, variables)
    emitter.emit(roots)
    refs = [emitter.reference(root) for root in roots]
    expression_identity = (
        program.expression_hash
        if density_only
        else canonical_hash(
            {
                "schema": "generativeqc.automatic-libxc-work-point.v1",
                "interior_expression": program.expression_hash,
                "work_policy": policy.to_payload(),
                "features": features,
                **(
                    {"physical_kinetic_predicate": kinetic_lines}
                    if physical_kinetic
                    else {}
                ),
            }
        )
    )
    stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    function_name = f"automatic_{stem}_point"
    if spin == "unpolarized":
        function_name += "_unpolarized"
    if equal_spin:
        function_name += "_equal_spin"
    value_type = "SemilocalPointValue" if backend == "cpu" else "DevicePointValue"
    qualifier = "" if backend == "cpu" else "__device__ inline "
    finite = "std::isfinite" if backend == "cpu" else "isfinite"

    def reject(reason: str, indent: str = "    ") -> str:
        if backend == "cpu":
            return f'{indent}throw std::domain_error("{name} {reason}");'
        return (
            f"{indent}{{ DevicePointValue invalid{{}}; "
            "invalid.valid = false; return invalid; }"
        )

    lines = [
        f"{qualifier}{value_type} {function_name}(",
        "    const double rho[2], const double (&gradient)[2][3], const double tau[2]"
        + (", const double* parameters" if parameter_names or parameters else "")
        + ") {",
        f"  if (!{finite}(rho[0]) || !{finite}(rho[1]) ||",
        "      rho[0] < 0.0 || rho[1] < 0.0)",
        reject("requires finite nonnegative density"),
        f"  if (!{finite}(rho[0] + rho[1]))",
        reject("total density is nonfinite"),
        "  const double rho_a = rho[0];",
        "  const double rho_b = rho[1];",
    ]
    if spin == "unpolarized":
        lines.extend(
            [
                "  if (rho_a != rho_b)",
                reject("unpolarized worker requires paired transport channels"),
            ]
        )
    if ingredient_mask != 1:
        lines.extend(
            [
                "  for (unsigned spin = 0; spin < 2; ++spin)",
                "    for (unsigned axis = 0; axis < 3; ++axis)",
                f"      if (!{finite}(gradient[spin][axis]))",
                reject("density gradient is nonfinite", "        "),
                "  const double sigma_aa = gradient[0][0] * gradient[0][0] +",
                "      gradient[0][1] * gradient[0][1] + gradient[0][2] * gradient[0][2];",
                "  const double sigma_ab = gradient[0][0] * gradient[1][0] +",
                "      gradient[0][1] * gradient[1][1] + gradient[0][2] * gradient[1][2];",
                "  const double sigma_bb = gradient[1][0] * gradient[1][0] +",
                "      gradient[1][1] * gradient[1][1] + gradient[1][2] * gradient[1][2];",
            ]
        )
    else:
        lines.extend(["  (void)gradient;", "  (void)tau;"])
    if ingredient_mask == 15:
        lines.extend(
            [
                f"  if (!{finite}(tau[0]) || !{finite}(tau[1]) ||",
                "      tau[0] < 0.0 || tau[1] < 0.0)",
                reject("requires finite nonnegative tau"),
                "  const double tau_a = tau[0];",
                "  const double tau_b = tau[1];",
            ]
        )
    elif ingredient_mask == 7:
        lines.append("  (void)tau;")

    lines.extend(work_lines)
    lines.extend(kinetic_lines)

    lines.extend(emitter.lines)
    energy = (
        refs[0]
        if density_only
        else f"{refs[0]} * total_density / (work_rho_a + work_rho_b)"
    )
    lines.extend(
        [
            f"  {value_type} out{{}};",
            f"  out.energy = {energy};",
            f"  out.rho[0] = {refs[1]};",
            f"  out.rho[1] = {refs[1] if spin == 'unpolarized' else refs[2]};",
        ]
    )
    if ingredient_mask != 1:
        lines.extend(
            [
                "  for (unsigned axis = 0; axis < 3; ++axis) {",
                f"    out.gradient[0][axis] = 2.0 * {refs[3]} * gradient[0][axis] +",
                f"        {refs[4]} * gradient[1][axis];",
                f"    out.gradient[1][axis] = {refs[4]} * gradient[0][axis] +",
                f"        2.0 * {refs[5]} * gradient[1][axis];",
                "  }",
            ]
        )
    if ingredient_mask == 15:
        lines.extend(
            [
                f"  out.kinetic[0] = 0.5 * {refs[6]};",
                f"  out.kinetic[1] = 0.5 * {refs[7]};",
            ]
        )
    lines.extend(
        [
            f"  if (!{finite}(out.energy) || !{finite}(out.rho[0]) ||",
            f"      !{finite}(out.rho[1]))",
            reject("produced nonfinite E/vrho"),
            "  for (unsigned spin = 0; spin < 2; ++spin) {",
            f"    if (!{finite}(out.kinetic[spin]))",
            reject("produced nonfinite vtau", "      "),
            "    for (double value : out.gradient[spin])",
            f"      if (!{finite}(value))",
            reject("produced nonfinite gradient derivative", "        "),
            "  }",
            "  return out;",
            "}",
        ]
    )
    return AutomaticPointSource(
        name=name,
        backend=backend,
        function_name=function_name,
        ingredient_mask=ingredient_mask,
        features=features,
        outputs=program.outputs,
        expression_identity=expression_identity,
        source="\n".join(lines),
        spin=spin,
        domain=DENSITY_WORK_DOMAIN if density_only else LIBXC_WORK_DOMAIN,
        domain_version=(
            DENSITY_WORK_DOMAIN_VERSION if density_only else LIBXC_WORK_DOMAIN_VERSION
        ),
        parameters=parameters,
    )


def _emit_composed_point(
    name: str,
    *,
    backend: Literal["cpu", "cuda"],
    spin: Literal["polarized", "unpolarized"],
    equal_spin: bool,
) -> AutomaticPointSource:
    """Fuse rational physical E/vxc sums inside the existing point traversal.

    Each component keeps its own density/sigma/tau work projection. Projecting
    one shared input first would change tails when component thresholds differ.
    Exact exchange never enters this sum; the prepared Fock owner supplies it.
    """
    from .automatic_semilocal import automatic_components, automatic_scf_domain

    terms = automatic_components(name)
    density_only = automatic_scf_domain(name) == DENSITY_WORK_DOMAIN
    if spin == "unpolarized" and not density_only:
        raise ValueError("unpolarized point ABI requires density-only composition")
    parts: list[tuple[AutomaticPointSource, Fraction]] = []
    for component, weight in terms:
        lda = automatic_work_policy(component).family == "lda"
        # RKS density components always use total-density U1, including when
        # nested in a gradient-dependent sum. UKS retains physical P2 roots.
        unpolarized = spin == "unpolarized" or (equal_spin and lda)
        parts.append(
            (
                emit_automatic_point(
                    component,
                    backend=backend,
                    spin="unpolarized" if unpolarized else "polarized",
                    equal_spin=equal_spin and not unpolarized,
                ),
                weight,
            )
        )
    features = max((part.features for part, _ in parts), key=lambda values: len(values))
    mask = AUTOMATIC_POINT_LAYOUTS[features]
    suffix = (
        "_unpolarized" if spin == "unpolarized" else "_equal_spin" if equal_spin else ""
    )
    function = f"automatic_{name.lower()}_point{suffix}"
    value_type = "SemilocalPointValue" if backend == "cpu" else "DevicePointValue"
    qualifier = "" if backend == "cpu" else "__device__ inline "
    source = []
    for index, (part, _) in enumerate(parts):
        source.append(f"namespace {function}_component_{index} {{\n{part.source}\n}}")
    source.extend(
        [
            f"{qualifier}{value_type} {function}(const double rho[2],",
            "    const double (&gradient)[2][3], const double tau[2]) {",
            f"  {value_type} out{{}};",
        ]
    )
    for index, (part, weight) in enumerate(parts):
        coefficient = float(weight).hex()
        source.append(
            f"  const auto term{index} = {function}_component_{index}::{part.function_name}(rho, gradient, tau);"
        )
        if backend == "cuda":
            source.append(f"  if (!term{index}.valid) return term{index};")
        source.extend(
            [
                f"  out.energy += {coefficient} * term{index}.energy;",
                "  for (unsigned spin = 0; spin < 2; ++spin) {",
                f"    out.rho[spin] += {coefficient} * term{index}.rho[spin];",
                f"    out.kinetic[spin] += {coefficient} * term{index}.kinetic[spin];",
                "    for (unsigned axis = 0; axis < 3; ++axis)",
                f"      out.gradient[spin][axis] += {coefficient} * term{index}.gradient[spin][axis];",
                "  }",
            ]
        )
    finite = "std::isfinite" if backend == "cpu" else "isfinite"
    rejection = (
        'throw std::domain_error("nonfinite composed XC output");'
        if backend == "cpu"
        else "out.valid = false; return out;"
    )
    source.extend(
        [
            f"  bool valid = {finite}(out.energy);",
            "  for (unsigned spin = 0; spin < 2; ++spin) {",
            f"    valid = valid && {finite}(out.rho[spin]) && {finite}(out.kinetic[spin]);",
            "    for (double value : out.gradient[spin])",
            f"      valid = valid && {finite}(value);",
            "  }",
            f"  if (!valid) {{ {rejection} }}",
            "  return out;",
            "}",
        ]
    )
    identity = canonical_hash(
        {
            "schema": "generativeqc.composed-physical-xc.v1",
            "components": [
                (part.mathematical_contract, str(weight)) for part, weight in parts
            ],
        }
    )
    return AutomaticPointSource(
        name=name,
        backend=backend,
        function_name=function,
        ingredient_mask=mask,
        features=features,
        outputs=((), *((index,) for index in range(len(features)))),
        expression_identity=identity,
        source="\n".join(source),
        spin=spin,
        domain=DENSITY_WORK_DOMAIN if density_only else LIBXC_WORK_DOMAIN,
        domain_version=DENSITY_WORK_DOMAIN_VERSION
        if density_only
        else LIBXC_WORK_DOMAIN_VERSION,
    )
