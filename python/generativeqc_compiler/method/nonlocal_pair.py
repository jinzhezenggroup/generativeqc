"""Compiler-owned scalar VV10/rVV10 pair expressions for both native backends.

Runtime owners retain local-scale construction, traversal, screening, reduction,
and failure publication.  The raw and preconditioned rVV10 representations have
separate ordered FP64 lowerings of the same equations; changing representation
must not silently change the established exceptional-arithmetic domain.
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
PAIR_INPUT_ORDER = ("r2", "wi", "wj", "ki", "kj", "row_inverse_kappa")
PAIR_OUTPUT_ORDER = ("phi", "dphi_domega", "dphi_dkappa", "dphi_dr2")


def build_nonlocal_pair_program(
    variant: str,
    *,
    preconditioned: bool = False,
    features: bool = True,
    geometry: bool = True,
) -> Program:
    """Select live pair outputs without changing runtime admission semantics.

    Raw inputs are omega and kappa. Preconditioned rVV10 inputs are
    omega/kappa and kappa**(3/2), with the row's inverse *raw* kappa supplied
    separately for feature outputs. CPU raw callers retain all legacy output
    roots for failure observation even when an output is not physically used.
    Only their existing bounded admission may omit the radial root.
    """
    if variant not in (VV10, RVV10):
        raise UnsupportedNonlocalCorrelation(f"unsupported pair variant {variant!r}")
    if any(type(flag) is not bool for flag in (preconditioned, features, geometry)):
        raise TypeError("pair representation and output demands must be Boolean")
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
        phi = divide(minus_three_halves, multiply(multiply(gi, gj), gsum))
        negative_phi = multiply(minus_one, phi)
        outputs["phi"] = phi
        if features:
            dphi_dgi = multiply(negative_phi, add(divide(one, gi), divide(one, gsum)))
            outputs["dphi_domega"] = multiply(dphi_dgi, r2)
            outputs["dphi_dkappa"] = dphi_dgi
        if geometry:
            logarithmic = add(
                add(divide(wi, gi), divide(wj, gj)), divide(add(wi, wj), gsum)
            )
            outputs["dphi_dr2"] = multiply(negative_phi, logarithmic)
    return Program(
        outputs,
        provenance={
            "kind": "nonlocal-correlation-pair",
            "scientific_version": NONLOCAL_CORRELATION_VERSION,
            "lowering_version": NONLOCAL_PAIR_LOWERING_VERSION,
            "variant": variant,
            "representation": "preconditioned"
            if preconditioned and variant == RVV10
            else "raw",
            "arithmetic": "ordered-native-fp64; caller-owned-failure-checks",
            "features": features,
            "geometry": geometry,
        },
    )
