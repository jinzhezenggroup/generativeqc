"""Common CPU/CUDA physical-spin point lowering for automatic Libxc Graphs.

Mathematics and the versioned Libxc work policy have one compiler owner. Native
runtime descriptors, scheduling, allocation, and public method admission remain
with their existing consumers; this module never imports the runtime or oracle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral.scalar_c import ScalarCEmitter

from .bulk_runtime import build_bulk_runtime_program
from .libxc_work import automatic_work_policy, polarized_work_setup

AUTOMATIC_POINT_LAYOUTS = {
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

    @property
    def source_identity(self) -> str:
        """Hash exact emitted bytes and layout without assigning runtime support."""
        return canonical_hash(
            {
                "schema": "generativeqc.automatic-libxc-point-source.v1",
                "backend": self.backend,
                "expression_identity": self.expression_identity,
                "features": self.features,
                "outputs": self.outputs,
                "ingredient_mask": self.ingredient_mask,
                "source": self.source,
            }
        )


def emit_automatic_point(
    name: str, *, backend: Literal["cpu", "cuda"] = "cpu"
) -> AutomaticPointSource:
    """Lower one ordinary semilocal Graph into the native physical-spin ABI.

    Both backends evaluate the same symbolic E/vxc roots at Libxc work inputs.
    Work floors/clamps are deliberately not differentiated. Energy keeps its
    original-density weighting, Cartesian gradients apply the sigma chain rule,
    and kinetic coefficients include the native half-vtau convention. CUDA
    reports invalid physical inputs/outputs with ``DevicePointValue.valid``;
    CPU retains domain exceptions. This source grants no runtime admission.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("automatic point backend must be cpu or cuda")
    program = build_bulk_runtime_program(
        name,
        spin="polarized",
        order=1,
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

    policy = automatic_work_policy(name)
    work_lines, work_arguments = polarized_work_setup(policy, features)
    variables = dict(zip(features, work_arguments, strict=True))
    emitter = ScalarCEmitter(program.graph, variables)
    emitter.emit(program.roots)
    refs = [emitter.reference(root) for root in program.roots]
    expression_identity = canonical_hash(
        {
            "schema": "generativeqc.automatic-libxc-work-point.v1",
            "interior_expression": program.expression_hash,
            "work_policy": policy.to_payload(),
            "features": features,
        }
    )
    stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    function_name = f"automatic_{stem}_point"
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
        "    const double rho[2], const double (&gradient)[2][3], const double tau[2]) {",
        f"  if (!{finite}(rho[0]) || !{finite}(rho[1]) ||",
        "      rho[0] < 0.0 || rho[1] < 0.0)",
        reject("requires finite nonnegative density"),
        f"  if (!{finite}(rho[0] + rho[1]))",
        reject("total density is nonfinite"),
        "  const double rho_a = rho[0];",
        "  const double rho_b = rho[1];",
    ]
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

    lines.extend(emitter.lines)
    lines.extend(
        [
            f"  {value_type} out{{}};",
            f"  out.energy = {refs[0]} * total_density / (work_rho_a + work_rho_b);",
            f"  out.rho[0] = {refs[1]};",
            f"  out.rho[1] = {refs[2]};",
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
    )
