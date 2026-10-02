"""Compiler-owned scalar VV10/rVV10 pair and local-scale expressions.

Runtime owners retain traversal, screening, reduction, and failure publication.
CPU and CUDA keep their qualified floating-point operation orders as separate
compiler-emitted local-scale policies; changing representation must not silently
change the established exceptional-arithmetic domain.
"""

from __future__ import annotations

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

NONLOCAL_PAIR_LOWERING_VERSION = "nonlocal-pair-ordered-native-fp64-v1"
NONLOCAL_PAIR_RATIONAL_VERSION = "nonlocal-pair-vv10-bounded-energy-denominator-fp64-v1"
NONLOCAL_PAIR_RECIPROCAL_ENERGY_VERSION = (
    "nonlocal-pair-vv10-bounded-unit-reciprocal-fp64-v1"
)
PAIR_INPUT_ORDER = ("r2", "wi", "wj", "ki", "kj", "row_inverse_kappa")
PAIR_OUTPUT_ORDER = ("phi", "dphi_domega", "dphi_dkappa", "dphi_dr2")


def native_local_scale_cpp() -> str:
    """Emit the qualified CPU/CUDA local-scale policies from one compiler owner.

    These spell the existing operation orders verbatim. CPU forms rho^4 and
    sigma^2 explicitly; CUDA forms sigma/rho^2 before squaring and optionally
    emits rVV10 preconditioning separately. Runtime callers validate the raw
    scales before converting representation, preserving the prior failure domain.
    """
    return r"""
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
  const double omega2 = c * sigma2 / rho4 + four_pi_over_three * rho;
  LocalScaleValues out{};
  out.omega = ::sqrt(omega2);
  out.kappa = b * 1.5 * pi * ::pow(rho / (9.0 * pi), 1.0 / 6.0);
  if constexpr (Features) {
    const double rho5 = rho4 * rho;
    out.domega_drho =
        (four_pi_over_three - 4.0 * c * sigma2 / rho5) / (2.0 * out.omega);
    out.domega_dsigma = c * sigma / (out.omega * rho4);
    out.dkappa_drho = out.kappa / (6.0 * rho);
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
  out.omega = ::sqrt(c * ratio * ratio + four_pi_over_three * rho);
  out.kappa = b * 1.5 * pi * ::pow(rho / (9.0 * pi), 1.0 / 6.0);
  if constexpr (Features) {
    out.domega_drho =
        (four_pi_over_three - 4.0 * c * sigma * sigma / ::pow(rho, 5.0)) /
        (2.0 * out.omega);
    out.domega_dsigma = c * sigma / (out.omega * ::pow(rho, 4.0));
    out.dkappa_drho = out.kappa / (6.0 * rho);
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
"""


def build_nonlocal_pair_program(
    variant: str,
    *,
    preconditioned: bool = False,
    features: bool = True,
    geometry: bool = True,
    rational_derivatives: bool = False,
    reciprocal_energy: bool = False,
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
        )
    ):
        raise TypeError("pair representation and output demands must be Boolean")
    if rational_derivatives and not (variant == VV10 and features):
        raise ValueError("rational derivatives require VV10 features")
    if reciprocal_energy and not rational_derivatives:
        raise ValueError("reciprocal energy requires bounded rational derivatives")
    scalar = TensorSpec((), role="input", differentiable=True)
    r2, wi, wj, ki, kj, row_inverse = (
        input_tensor(name, scalar) for name in PAIR_INPUT_ORDER
    )
    one, minus_one, minus_three_halves = constant(1), constant(-1), constant("-3/2")
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
        negative_phi = multiply(minus_one, phi)
        outputs["phi"] = phi
        if features:
            factor_z = add(divide(one, zi), divide(one, zsum))
            omega_numerator = multiply(negative_phi, r2)
            if preconditioned:
                omega_factor = multiply(omega_numerator, row_inverse)
                kappa_prefactor = multiply(phi, row_inverse)
            else:
                omega_factor = divide(omega_numerator, ki)
                kappa_prefactor = divide(phi, ki)
            outputs["dphi_domega"] = multiply(omega_factor, factor_z)
            outputs["dphi_dkappa"] = multiply(
                kappa_prefactor,
                add(
                    minus_three_halves,
                    multiply(add(zi, one, coefficients=(1, -1)), factor_z),
                ),
            )
        if geometry:
            logarithmic = add(
                add(divide(ai, zi), divide(aj, zj)), divide(add(ai, aj), zsum)
            )
            outputs["dphi_dr2"] = multiply(negative_phi, logarithmic)
    else:
        gi, gj = add(multiply(wi, r2), ki), add(multiply(wj, r2), kj)
        gsum = add(gi, gj)
        denominator = multiply(multiply(gi, gj), gsum)
        # A unit numerator selects the compiler's FP64 reciprocal lowering.
        # The admitted domain keeps both reciprocal and product normal/finite;
        # the extra rounding is deliberate and independently qualified.
        phi = (
            multiply(minus_three_halves, divide(one, denominator))
            if reciprocal_energy
            else divide(minus_three_halves, denominator)
        )
        negative_phi = multiply(minus_one, phi)
        outputs["phi"] = phi
        if features:
            if rational_derivatives:
                # phi = -3/(2*D), so (2/3)*phi^2 = 3/(2*D^2).
                # Reuse the ordered energy's denominator without another divide.
                # The generated caller bounds all positive factors before this
                # reassociation; the original closure owns extreme inputs.
                denominator_partial = multiply(
                    constant("2/3"), multiply(negative_phi, negative_phi)
                )
                dphi_dgi = multiply(multiply(denominator_partial, gj), add(gi, gsum))
            else:
                dphi_dgi = multiply(
                    negative_phi, add(divide(one, gi), divide(one, gsum))
                )
            outputs["dphi_domega"] = multiply(dphi_dgi, r2)
            outputs["dphi_dkappa"] = dphi_dgi
        if geometry:
            if rational_derivatives:
                # Positive denominator factors avoid cancellation in each
                # logarithmic partial. This shares dphi/dg_i with SCF features.
                dphi_dgj = multiply(multiply(denominator_partial, gi), add(gj, gsum))
                outputs["dphi_dr2"] = add(
                    multiply(wi, dphi_dgi), multiply(wj, dphi_dgj)
                )
            else:
                logarithmic = add(
                    add(divide(wi, gi), divide(wj, gj)), divide(add(wi, wj), gsum)
                )
                outputs["dphi_dr2"] = multiply(negative_phi, logarithmic)
    return Program(
        outputs,
        provenance={
            "kind": "nonlocal-correlation-pair",
            "scientific_version": NONLOCAL_CORRELATION_VERSION,
            "lowering_version": (
                NONLOCAL_PAIR_RECIPROCAL_ENERGY_VERSION
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
                "bounded-unit-reciprocal-energy-fp64; caller-owned-failure-checks"
                if reciprocal_energy
                else "bounded-rational-derivatives-fp64; caller-owned-failure-checks"
                if rational_derivatives
                else "ordered-native-fp64; caller-owned-failure-checks"
            ),
            "features": features,
            "geometry": geometry,
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
    phi, dw, dk, rho, wrho, wsigma, krho, beta, coefficient = (
        input_tensor(name, scalar) for name in ROW_FEATURE_INPUT_ORDER
    )
    return Program(
        {
            "vrho": multiply(
                coefficient,
                add(
                    add(beta, phi),
                    multiply(rho, add(multiply(dw, wrho), multiply(dk, krho))),
                ),
            ),
            "vsigma": multiply(multiply(coefficient, rho), multiply(dw, wsigma)),
        },
        provenance={
            "kind": "nonlocal-correlation-row-feature-contraction",
            "scientific_version": NONLOCAL_CORRELATION_VERSION,
            "lowering_version": "vv10-bounded-row-chain-fp64-v1",
            "arithmetic": "ordered-parameter-sums-then-row-chain-fp64",
        },
    )
