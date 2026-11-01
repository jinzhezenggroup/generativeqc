"""Scientific admission and scalar lowering for resident spatial XC consumers.

An optional grid artifact binds one exact semilocal composition. Its identity
is checked before lending device buffers; a functional name alone never admits
a modified expression. This is the prepared ``interior-v1`` contract, distinct
from the native SCF production continuations.
"""

import typing
from dataclasses import replace
from functools import lru_cache

from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.xc_kernel import (
    GridXcConsumer,
    GridXcKernel,
    grid_xc_consumer,
)

from .semilocal_codegen import emit_polarized_semilocal
from .spec import PUBLIC_COMPONENTS, FunctionalSpec, UnsupportedXC, functional


def _spec(value: FunctionalSpec | str) -> FunctionalSpec:
    return functional(value) if isinstance(value, str) else value


@lru_cache(maxsize=128)
def compiled_spatial_xc(spec: FunctionalSpec) -> GridXcConsumer:
    """Admit compositions whose complete device domain check is implemented.

    The audited LDA/PBE/SCAN component family shares the same interior constraints.
    Other families have extra domain requirements and retain their existing route
    until those checks and independent device gates are supplied. Exchange operator
    metadata is never interpreted as part of a semilocal matrix contraction.
    """
    if not isinstance(spec, FunctionalSpec):
        raise TypeError("spatial XC compilation requires a FunctionalSpec")
    active = {name for name, coefficient in spec.components if coefficient}
    if (
        not active <= set(PUBLIC_COMPONENTS)
        or spec.exact_exchange
        or spec.long_range_exchange
        or spec.range_omega
    ):
        raise UnsupportedXC("spatial CUDA XC component/domain is not qualified")
    ingredients = ("rho",)
    if "sigma" in spec.ingredients:
        ingredients += ("gradient",)
    if "tau" in spec.ingredients:
        ingredients += ("tau",)
    # A polarized evaluator serves both layouts. Restricted consumers verify
    # equal spin features before taking either identical spin potential.
    identity = canonical_hash(
        {
            "schema": "generativeqc.spatial-xc-interior.v1",
            "functional": replace(spec, spin="polarized").to_payload(),
        }
    )
    return GridXcConsumer(2, ingredients, identity)


def spatial_xc_capability(
    value: FunctionalSpec | str, compiled_identity: str = ""
) -> GridXcConsumer | None:
    """Resolve an exact expression against built-in or artifact-owned science."""
    spec = _spec(value)
    if not isinstance(spec, FunctionalSpec):
        raise TypeError("expected a functional name or FunctionalSpec")
    builtin = grid_xc_consumer(spec)
    if builtin is not None:
        return builtin
    if compiled_identity:
        try:
            capability = compiled_spatial_xc(spec)
        except UnsupportedXC:
            return None
        if capability.identity == compiled_identity:
            return capability
    return None


def emit_spatial_xc(spec: FunctionalSpec) -> GridXcKernel:
    """Emit shared scalar mathematics and its explicit interior-domain adapter.

    No GPU/native library is loaded. The adapter maps sigma derivatives to Cartesian
    gradient coefficients and applies tau's one-half weak-form factor exactly once.
    """
    capability = compiled_spatial_xc(spec)
    spec = replace(spec, spin="polarized")
    gradient = "gradient" in capability.ingredients
    tau = "tau" in capability.ingredients
    mask = 1 | (2 if gradient else 0) | (8 if tau else 0)
    scalar = emit_polarized_semilocal(
        spec,
        function_name="grid_custom_xc_scalar",
        value_type="GridXcScalar",
        identity_constant="grid_custom_xc_expression",
        function_qualifier="__device__ inline",
    )
    arguments = "rho[0], rho[1]"
    if gradient:
        arguments += ", sigma[0], sigma[1], sigma[2]"
    if tau:
        arguments += ", kinetic[0], kinetic[1]"
    lines = [
        f"constexpr unsigned grid_custom_xc_mask = {mask};",
        f'constexpr const char* grid_custom_xc_identity = "{capability.identity}";',
        scalar,
        "__device__ inline GridXcPoint grid_custom_xc_point(bool restricted, const double* features, I npoint, I point) {",
        "  GridXcPoint out;",
        "  const double rho[2]{features[point], features[5 * npoint + point]};",
        "  double gradient[2][3]{}, kinetic[2]{}, sigma[3]{};",
        "  const double total = rho[0] + rho[1];",
        "  if (!isfinite(total) || total < 1e-12 || total > 1e12 ||",
        "      !isfinite(rho[0]) || !isfinite(rho[1]) || fmin(rho[0], rho[1]) / total < 1e-10) {",
        "    out.valid = false; return out;",
        "  }",
        "  if (restricted && rho[0] != rho[1]) { out.valid = false; return out; }",
    ]
    if gradient:
        lines += [
            "  for (int spin = 0; spin < 2; ++spin) {",
            "    for (int axis = 0; axis < 3; ++axis) {",
            "      gradient[spin][axis] = features[(5 * spin + axis + 1) * npoint + point];",
            "      if (!isfinite(gradient[spin][axis])) { out.valid = false; return out; }",
            "    }",
            "  }",
            "  for (int axis = 0; axis < 3; ++axis) {",
            "    if (restricted && gradient[0][axis] != gradient[1][axis]) { out.valid = false; return out; }",
            "    sigma[0] += gradient[0][axis] * gradient[0][axis];",
            "    sigma[1] += gradient[0][axis] * gradient[1][axis];",
            "    sigma[2] += gradient[1][axis] * gradient[1][axis];",
            "  }",
            "  for (int spin = 0; spin < 2; ++spin)",
            "    if (!isfinite(sigma[2 * spin]) || sqrt(sigma[2 * spin]) / pow(rho[spin], 4.0 / 3.0) > 1e6) {",
            "      out.valid = false; return out;",
            "    }",
            "  if (!isfinite(sigma[1]) || fabs(sigma[1]) > sqrt(sigma[0]) * sqrt(sigma[2]) * (1.0 + 16.0 * 0x1p-52)) {",
            "    out.valid = false; return out;",
            "  }",
        ]
    if tau:
        lines += [
            "  for (int spin = 0; spin < 2; ++spin) {",
            "    kinetic[spin] = features[(5 * spin + 4) * npoint + point];",
            "    if (!isfinite(kinetic[spin]) || kinetic[spin] < 0.0) { out.valid = false; return out; }",
            "  }",
            "  if (restricted && kinetic[0] != kinetic[1]) { out.valid = false; return out; }",
        ]
    lines += [
        f"  const auto raw = grid_custom_xc_scalar({arguments});",
        "  out.energy = raw.energy_density;",
        "  out.rho[0] = raw.feature_derivative[0];",
        "  out.rho[1] = raw.feature_derivative[1];",
    ]
    if gradient:
        lines += [
            "  for (int axis = 0; axis < 3; ++axis) {",
            "    out.gradient[0][axis] = 2.0 * raw.feature_derivative[2] * gradient[0][axis] + raw.feature_derivative[3] * gradient[1][axis];",
            "    out.gradient[1][axis] = 2.0 * raw.feature_derivative[4] * gradient[1][axis] + raw.feature_derivative[3] * gradient[0][axis];",
            "  }",
        ]
    if tau:
        lines += [
            "  out.kinetic[0] = 0.5 * raw.feature_derivative[5];",
            "  out.kinetic[1] = 0.5 * raw.feature_derivative[6];",
        ]
    lines += ["  return out;", "}", ""]
    return GridXcKernel("\n".join(lines), capability)


def compile_spatial_cuda(
    spec: FunctionalSpec, compiler: typing.Any, cache: typing.Any
) -> typing.Any:
    """Compile a resident grid artifact for this exact interior XC expression."""
    from generativeqc_compiler.dft.cuda import compile_cuda

    return compile_cuda(compiler, cache, xc_kernel=emit_spatial_xc(spec))
