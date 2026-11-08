"""Compiler-owned scalar VV10/rVV10 pair and local-scale expressions.

Runtime owners retain traversal, screening, reduction, and failure publication.
CPU and CUDA keep their qualified floating-point operation orders as separate
compiler-emitted local-scale policies; changing representation must not silently
change the established exceptional-arithmetic domain.
"""

from __future__ import annotations

import math
import typing
from fractions import Fraction

from generativeqc_compiler.common.nonlocal_correlation import (
    NONLOCAL_CORRELATION_VERSION,
    RVV10,
    VV10,
    UnsupportedNonlocalCorrelation,
)
from generativeqc_compiler.tensor.ir import (
    add,
    constant,
    divide,
    input_tensor,
    multiply,
    power,
)
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import TensorSpec

from .nonlocal_pair_ad import (
    local_scale_primal_cpp,
    local_scale_response_cpp,
    row_feature_responses,
    rvv10_pair_responses,
    vv10_pair_responses,
)

NONLOCAL_PAIR_LOWERING_VERSION = "nonlocal-pair-ordered-native-fp64-v1"
NONLOCAL_PAIR_RATIONAL_VERSION = "nonlocal-pair-vv10-bounded-energy-denominator-fp64-v1"
NONLOCAL_PAIR_RECIPROCAL_ENERGY_VERSION = (
    "nonlocal-pair-vv10-bounded-unit-reciprocal-fp64-v1"
)
NONLOCAL_PAIR_SHARED_RECIPROCAL_VERSION = (
    "nonlocal-pair-vv10-shared-reciprocal-partials-fp64-v1"
)
NONLOCAL_PAIR_MIXED_VERSION = "nonlocal-pair-bounded-fp32-rational-fp64-sums-v2"
PAIR_INPUT_ORDER = ("r2", "wi", "wj", "ki", "kj", "row_inverse_kappa")
PAIR_OUTPUT_ORDER = ("phi", "dphi_domega", "dphi_dkappa", "dphi_dr2")


def _vv10_pair_factors(
    radius: typing.Any,
    omega_a: typing.Any,
    omega_b: typing.Any,
    kappa_a: typing.Any,
    kappa_b: typing.Any,
    *,
    addition: typing.Any = add,
    product: typing.Any = multiply,
) -> tuple:
    """Share the ordered VV10 primal between TensorIR emission and scalar AD."""
    first = addition(product(omega_a, radius), kappa_a)
    second = addition(product(omega_b, radius), kappa_b)
    combined = addition(first, second)
    return first, second, combined, product(product(first, second), combined)


def vv10_grid_beta(spec: typing.Any) -> float:
    """Canonical VV10 local energy coefficient, before the method coefficient."""
    if spec.variant != VV10:
        raise UnsupportedNonlocalCorrelation("grid local AD currently requires VV10")
    return (3 / float(spec.b) ** 2) ** 0.75 / 32


def vv10_grid_local_expression(
    graph: typing.Any, leg: list, spec: typing.Any
) -> typing.Any:
    """The local beta energy uses the same total-density/measure field ABI."""
    return vv10_grid_beta(spec) * leg[0] * leg[5]


def vv10_grid_pair_expression(
    graph: typing.Any, first: list, second: list, spec: typing.Any
) -> typing.Any:
    """Weighted ordered pair primal for total-density Cartesian grid AD.

    Each leg is [rho,gx,gy,gz,tau,weight,x,y,z]. There is no one-half here:
    differentiating only the first leg accounts for both symmetric energy
    legs. Energy integration applies its one-half separately. Local scales
    follow the raw CPU policy; runtime masks and admission remain external.
    """
    if spec.variant != VV10:
        raise UnsupportedNonlocalCorrelation("grid pair AD currently requires VV10")

    def scales(leg: list) -> tuple:
        rho = leg[0]
        sigma = graph.sum(value * value for value in leg[1:4])
        omega = (
            float(spec.c) * sigma * sigma / rho.pow(4) + (4 * math.pi / 3) * rho
        ).pow(0.5)
        kappa = float(spec.b) * 1.5 * math.pi * (rho / (9 * math.pi)).pow(1 / 6)
        return omega, kappa

    omega_a, kappa_a = scales(first)
    omega_b, kappa_b = scales(second)
    radius = graph.sum((first[index] - second[index]).pow(2) for index in range(6, 9))
    *_, denominator = _vv10_pair_factors(
        radius,
        omega_a,
        omega_b,
        kappa_a,
        kappa_b,
        addition=lambda left, right: left + right,
        product=lambda left, right: left * right,
    )
    return Fraction(-3, 2) * first[0] * first[5] * second[0] * second[5] / denominator


def native_local_scale_cpp() -> str:
    """Emit the qualified CPU/CUDA local-scale policies from one compiler owner.

    The value policies keep their qualified operation orders. CPU forms rho^4 and
    sigma^2 explicitly; CUDA forms sigma/rho^2 before squaring and optionally
    emits rVV10 preconditioning separately. Runtime callers validate the raw
    scales before converting representation, preserving the prior failure domain.
    """
    return (
        r"""
struct LocalScaleValues {
  double omega{};
  double kappa{};
  double domega_drho{};
  double domega_dsigma{};
  double dkappa_drho{};
};

template <bool Features>
GENERATIVEQC_NONLOCAL_PAIR_HD inline LocalScaleValues local_scales_cpu(
    double rho, double sigma, double b, double c) noexcept {
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double four_pi_over_three = 4.0 * pi / 3.0;
  const double rho2 = rho * rho;
  const double rho4 = rho2 * rho2;
  const double sigma2 = sigma * sigma;
  LocalScaleValues out{};
@CPU_PRIMAL@
  if constexpr (Features) {
    const double rho5 = rho4 * rho;
@CPU_RESPONSE@
  }
  return out;
}

template <Vv10Variant Variant, bool Features>
GENERATIVEQC_NONLOCAL_PAIR_HD inline LocalScaleValues local_scales_cuda(
    double rho, double sigma, double b, double c) noexcept {
  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double four_pi_over_three = 4.0 * pi / 3.0;
  const double ratio = sigma / (rho * rho);
  LocalScaleValues out{};
@CUDA_PRIMAL@
  if constexpr (Features) {
@CUDA_RESPONSE@
  }
  return out;
}

// The runtime validates raw scales before this representation change. In
// particular, a positive raw kappa may underflow to zero after preconditioning.
template <Vv10Variant Variant>
GENERATIVEQC_NONLOCAL_PAIR_HD inline void precondition_local_scales_cuda(
    double& omega, double& kappa) noexcept {
  if constexpr (Variant == Vv10Variant::rvv10) {
    omega /= kappa;
    kappa *= ::sqrt(kappa);
  }
}
""".replace("@CPU_RESPONSE@", local_scale_response_cpp(cuda=False))
        .replace("@CUDA_RESPONSE@", local_scale_response_cpp(cuda=True))
        .replace("@CPU_PRIMAL@", local_scale_primal_cpp(cuda=False))
        .replace("@CUDA_PRIMAL@", local_scale_primal_cpp(cuda=True))
    )


def build_nonlocal_pair_program(
    variant: str,
    *,
    preconditioned: bool = False,
    features: bool = True,
    geometry: bool = True,
    rational_derivatives: bool = False,
    reciprocal_energy: bool = False,
    partner_features: bool = False,
    shared_reciprocal_derivatives: bool = False,
) -> Program:
    """Select live pair outputs without changing runtime admission semantics.

    Raw inputs are omega and kappa. Preconditioned rVV10 inputs are
    omega/kappa and kappa**(3/2), with the row's inverse *raw* kappa supplied
    separately for feature outputs. CPU raw callers retain all legacy output
    roots for failure observation even when an output is not physically used.
    Only their existing bounded admission may omit the radial root.

    Rational derivatives reuse the ordered VV10 energy denominator.
    The generated caller admits a bounded positive domain and retains the
    ordered closure elsewhere. Energy is unchanged; derivative roots deliberately
    use a separately versioned FP64 operation order. The optional unit-reciprocal
    energy policy changes energy rounding too, under its own lowering identity.
    Partner features expose the same second-leg derivative nodes used by the
    radial root, allowing a schedule to consume both ends of an unordered pair.
    Shared-reciprocal derivatives reuse -phi/D instead of (2/3)*phi^2.
    This saves one multiplication in the admitted FP64 feature closure without
    approximating the reciprocal; its changed rounding has a separate identity.
    """
    if variant not in (VV10, RVV10):
        raise UnsupportedNonlocalCorrelation(f"unsupported pair variant {variant!r}")
    if any(
        type(flag) is not bool
        for flag in (
            preconditioned,
            features,
            geometry,
            rational_derivatives,
            reciprocal_energy,
            partner_features,
            shared_reciprocal_derivatives,
        )
    ):
        raise TypeError("pair representation and output demands must be Boolean")
    if rational_derivatives and not (variant == VV10 and features):
        raise ValueError("rational derivatives require VV10 features")
    if partner_features and not (variant == VV10 and features):
        raise ValueError("partner features require VV10 features")
    if reciprocal_energy and not rational_derivatives:
        raise ValueError("reciprocal energy requires bounded rational derivatives")
    if shared_reciprocal_derivatives and not reciprocal_energy:
        raise ValueError("shared reciprocal derivatives require reciprocal energy")
    scalar = TensorSpec((), role="input", differentiable=True)
    r2, wi, wj, ki, kj, row_inverse = (
        input_tensor(name, scalar) for name in PAIR_INPUT_ORDER
    )
    one, minus_three_halves = constant(1), constant("-3/2")
    outputs = {}
    if variant == RVV10:
        ai, aj = (wi, wj) if preconditioned else (divide(wi, ki), divide(wj, kj))
        if preconditioned:
            zi, zj = add(multiply(ai, r2), one), add(multiply(aj, r2), one)
            kappa_factor = multiply(ki, kj)
        else:
            zi, zj = add(one, multiply(ai, r2)), add(one, multiply(aj, r2))
            kappa_factor = power(multiply(ki, kj), "3/2")
        zsum = add(zi, zj)
        denominator = multiply(multiply(multiply(kappa_factor, zi), zj), zsum)
        phi = divide(minus_three_halves, denominator)
        outputs["phi"] = phi
        if features or geometry:
            responses = rvv10_pair_responses(
                zi,
                zj,
                ai,
                aj,
                r2,
                ki,
                row_inverse,
                phi,
                preconditioned=preconditioned,
            )
            names = ["dphi_domega", "dphi_dkappa"] if features else []
            if geometry:
                names += ["dphi_dr2"]
            outputs.update((name, responses[name]) for name in names)
    else:
        gi, gj, _, denominator = _vv10_pair_factors(r2, wi, wj, ki, kj)
        # A unit numerator selects the compiler's FP64 reciprocal lowering.
        # The admitted domain keeps both reciprocal and product normal/finite;
        # the extra rounding is deliberate and independently qualified.
        energy_reciprocal = divide(one, denominator)
        phi = (
            multiply(minus_three_halves, energy_reciprocal)
            if reciprocal_energy
            else divide(minus_three_halves, denominator)
        )
        outputs["phi"] = phi
        if features or geometry:
            responses = vv10_pair_responses(
                gi,
                gj,
                wi,
                wj,
                r2,
                phi,
                energy_reciprocal,
                rational=rational_derivatives,
                shared_reciprocal=shared_reciprocal_derivatives,
            )
            names = ["dphi_domega", "dphi_dkappa"] if features else []
            if partner_features:
                names += ["dphi_domega_j", "dphi_dkappa_j"]
            if geometry:
                names += ["dphi_dr2"]
            outputs.update((name, responses[name]) for name in names)
    return Program(
        outputs,
        provenance={
            "kind": "nonlocal-correlation-pair",
            "scientific_version": NONLOCAL_CORRELATION_VERSION,
            "lowering_version": (
                NONLOCAL_PAIR_SHARED_RECIPROCAL_VERSION
                if shared_reciprocal_derivatives
                else NONLOCAL_PAIR_RECIPROCAL_ENERGY_VERSION
                if reciprocal_energy
                else NONLOCAL_PAIR_RATIONAL_VERSION
                if rational_derivatives
                else NONLOCAL_PAIR_LOWERING_VERSION
            ),
            "variant": variant,
            "representation": "preconditioned"
            if preconditioned and variant == RVV10
            else "raw",
            "arithmetic": (
                "bounded-shared-reciprocal-partials-fp64; caller-owned-failure-checks"
                if shared_reciprocal_derivatives
                else "bounded-unit-reciprocal-energy-fp64; caller-owned-failure-checks"
                if reciprocal_energy
                else "bounded-rational-derivatives-fp64; caller-owned-failure-checks"
                if rational_derivatives
                else "ordered-native-fp64; caller-owned-failure-checks"
            ),
            "features": features,
            "geometry": geometry,
            **({"partner_features": True} if partner_features else {}),
        },
    )


ROW_FEATURE_INPUT_ORDER = (
    "sum_phi",
    "sum_domega",
    "sum_dkappa",
    "rho",
    "domega_drho",
    "domega_dsigma",
    "dkappa_drho",
    "beta",
    "coefficient",
)


def build_nonlocal_row_feature_program() -> Program:
    """Contract ordered parameter-partial sums with row-local scale derivatives.

    The CUDA caller bounds the entire row before publishing these reassociated
    feature sums. This saves row-invariant chain factors inside the pair loop;
    its rounding policy is distinct from summing fully chained pair features.
    """
    scalar = TensorSpec((), role="input", differentiable=True)
    bindings = {name: input_tensor(name, scalar) for name in ROW_FEATURE_INPUT_ORDER}
    return Program(
        row_feature_responses(bindings),
        provenance={
            "kind": "nonlocal-correlation-row-feature-contraction",
            "scientific_version": NONLOCAL_CORRELATION_VERSION,
            "lowering_version": "vv10-bounded-row-chain-fp64-v1",
            "arithmetic": "ordered-parameter-sums-then-row-chain-fp64",
        },
    )
