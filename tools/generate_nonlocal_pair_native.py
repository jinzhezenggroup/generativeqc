"""Generate the shared host/device VV10/rVV10 scalar pair family from TensorIR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.method.nonlocal_pair import (
    NONLOCAL_PAIR_LOWERING_VERSION,
    NONLOCAL_PAIR_MIXED_VERSION,
    NONLOCAL_PAIR_RATIONAL_VERSION,
    NONLOCAL_PAIR_RECIPROCAL_ENERGY_VERSION,
    PAIR_INPUT_ORDER,
    PAIR_OUTPUT_ORDER,
    ROW_FEATURE_INPUT_ORDER,
    build_nonlocal_pair_program,
    build_nonlocal_row_feature_program,
    native_local_scale_cpp,
)
from generativeqc_compiler.method.nonlocal_pair_cuda import native_parallel_pair_cuda
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp


def native_header() -> str:
    """Emit each output closure once, shared unchanged by CPU and CUDA."""
    bodies, branches, mixed_branches = [], [], []
    for variant, preconditioned in (("vv10", False), ("rvv10", False), ("rvv10", True)):
        for features in (False, True):
            for geometry in (False, True):
                program = build_nonlocal_pair_program(
                    variant,
                    preconditioned=preconditioned,
                    features=features,
                    geometry=geometry,
                )
                inputs = {
                    node.attrs["name"]
                    for node in program.live_nodes
                    if node.op == "input"
                }
                input_order = tuple(name for name in PAIR_INPUT_ORDER if name in inputs)
                output_order = tuple(
                    name for name in PAIR_OUTPUT_ORDER if name in program.outputs
                )
                name = f"pair_{variant}_{'prepared' if preconditioned else 'raw'}_f{int(features)}_g{int(geometry)}"
                body = emit_scalar_cpp(
                    program,
                    function_name=name,
                    input_order=input_order,
                    output_order=output_order,
                    caller_owned_checks=True,
                    ordered_native_sums=True,
                    output_dependency_order=True,
                )
                body = body.replace("std::sqrt(", "::sqrt(").replace(
                    "std::pow(", "::pow("
                )
                bodies.append(
                    f"// TensorIR logical hash: {program.logical_hash}\n"
                    + body.replace(
                        "inline bool ", "GENERATIVEQC_NONLOCAL_PAIR_HD inline bool ", 1
                    )
                )
                condition = f"Variant == Vv10Variant::{variant}"
                if variant == "rvv10":
                    condition += " && " + (
                        "Preconditioned" if preconditioned else "!Preconditioned"
                    )
                condition += " && " + ("Features" if features else "!Features")
                condition += " && " + ("Geometry" if geometry else "!Geometry")
                arguments = (
                    *input_order,
                    *(f"result.{output}" for output in output_order),
                )
                branches.append(
                    f"  if constexpr ({condition}) {{\n    {name}({', '.join(arguments)});\n  }}"
                )
                if variant == "vv10" or preconditioned:
                    # The bounded FP32 domain also keeps the shared rational
                    # derivative intermediates representable. Use the same
                    # denominator reuse as FP64 instead of restoring divides
                    # merely because the scalar lowering selects float.
                    mixed_program = build_nonlocal_pair_program(
                        variant,
                        preconditioned=preconditioned,
                        features=features,
                        geometry=geometry,
                        rational_derivatives=variant == "vv10" and features,
                        reciprocal_energy=variant == "vv10" and features,
                    )
                    mixed_body = emit_scalar_cpp(
                        mixed_program,
                        function_name=name + "_fp32",
                        input_order=input_order,
                        output_order=output_order,
                        caller_owned_checks=True,
                        ordered_native_sums=True,
                        output_dependency_order=True,
                        scalar_type="float",
                    )
                    bodies.append(
                        mixed_body.replace(
                            "inline bool ",
                            "GENERATIVEQC_NONLOCAL_PAIR_HD inline bool ",
                            1,
                        )
                    )
                    mixed_condition = condition.replace(" && Preconditioned", "")
                    mixed_arguments = (
                        *(f"static_cast<float>({item})" for item in input_order),
                        *(f"mixed_{item}" for item in output_order),
                    )
                    mixed_branches.append(
                        f"  if constexpr ({mixed_condition}) {{\n"
                        f"    {name}_fp32({', '.join(mixed_arguments)});\n  }}"
                    )
    for geometry, unit in ((g, u) for g in (False, True) for u in (False, True)):
        label = "unit_reciprocal" if unit else "rational"
        rational = build_nonlocal_pair_program(
            "vv10", geometry=geometry, rational_derivatives=True, reciprocal_energy=unit
        )
        rational_body = emit_scalar_cpp(
            rational,
            function_name=f"pair_vv10_{label}_g{int(geometry)}",
            input_order=PAIR_INPUT_ORDER[:5],
            output_order=tuple(
                name for name in PAIR_OUTPUT_ORDER if name in rational.outputs
            ),
            caller_owned_checks=True,
            ordered_native_sums=True,
            output_dependency_order=True,
        )
        bodies.append(
            f"// TensorIR logical hash: {rational.logical_hash}\n"
            + rational_body.replace(
                "inline bool ", "GENERATIVEQC_NONLOCAL_PAIR_HD inline bool ", 1
            )
        )
    # Both feature legs share the same admitted denominator and canonical
    # TensorIR roots. Generated traversal templates keep stream/output ownership
    # with the CUDA runtime.
    for geometry in (False, True):
        bilateral = build_nonlocal_pair_program(
            "vv10",
            geometry=geometry,
            rational_derivatives=True,
            reciprocal_energy=True,
            partner_features=True,
        )
        output_order = (
            "phi",
            "dphi_domega",
            "dphi_dkappa",
            "dphi_domega_j",
            "dphi_dkappa_j",
        )
        if geometry:
            output_order += ("dphi_dr2",)
        body = emit_scalar_cpp(
            bilateral,
            function_name=f"pair_vv10_bilateral_g{int(geometry)}",
            input_order=PAIR_INPUT_ORDER[:5],
            output_order=output_order,
            caller_owned_checks=True,
            ordered_native_sums=True,
            output_dependency_order=True,
        )
        bodies.append(
            f"// TensorIR logical hash: {bilateral.logical_hash}\n"
            + body.replace(
                "inline bool ", "GENERATIVEQC_NONLOCAL_PAIR_HD inline bool ", 1
            )
        )
        mixed_bilateral = build_nonlocal_pair_program(
            "vv10",
            geometry=geometry,
            rational_derivatives=True,
            reciprocal_energy=True,
            partner_features=True,
        )
        bodies.append(
            emit_scalar_cpp(
                mixed_bilateral,
                function_name=f"pair_vv10_bilateral_g{int(geometry)}_fp32",
                input_order=PAIR_INPUT_ORDER[:5],
                output_order=output_order,
                caller_owned_checks=True,
                ordered_native_sums=True,
                output_dependency_order=True,
                scalar_type="float",
            ).replace("inline bool ", "GENERATIVEQC_NONLOCAL_PAIR_HD inline bool ", 1)
        )
    row = build_nonlocal_row_feature_program()
    row_body = emit_scalar_cpp(
        row,
        function_name="row_feature_values",
        input_order=ROW_FEATURE_INPUT_ORDER,
        output_order=("vrho", "vsigma"),
        caller_owned_checks=True,
        ordered_native_sums=True,
        output_dependency_order=True,
    )
    bodies.append(
        f"// TensorIR logical hash: {row.logical_hash}\n"
        + row_body.replace(
            "inline bool ", "GENERATIVEQC_NONLOCAL_PAIR_HD inline bool ", 1
        )
    )
    return (
        """// Generated by tools/generate_nonlocal_pair_native.py from TensorIR; do not edit.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define GENERATIVEQC_NONLOCAL_PAIR_HD __host__ __device__
#else
#define GENERATIVEQC_NONLOCAL_PAIR_HD
#endif

// The runtime declares Vv10Variant before including this generated header.
namespace generativeqc::dft::nlc::generated {

"""
        + f'inline constexpr const char* nonlocal_pair_lowering_version = "{NONLOCAL_PAIR_LOWERING_VERSION}";\n'
        + f'inline constexpr const char* nonlocal_pair_rational_version = "{NONLOCAL_PAIR_RATIONAL_VERSION}";\n'
        + f'inline constexpr const char* nonlocal_pair_admitted_version = "{NONLOCAL_PAIR_RECIPROCAL_ENERGY_VERSION}";\n'
        + f'inline constexpr const char* nonlocal_pair_mixed_version = "{NONLOCAL_PAIR_MIXED_VERSION}";\n'
        + """
struct PairValues {
  double phi{};
  double dphi_domega{};
  double dphi_dkappa{};
  double dphi_dr2{};
};

"""
        + native_local_scale_cpp()
        + "\n"
        + "\n".join(bodies)
        + """
// Keep scale construction, coordinate subtraction and all weighted sums in
// FP64. This sufficient domain bounds the FP32 pair's positive denominators
// and positive factor products away from overflow/underflow. Extreme inputs retain
// the established strict closure, rather than being clipped or screened out.
GENERATIVEQC_NONLOCAL_PAIR_HD inline bool fp32_pair_domain(
    double r2, double wi, double wj, double ki, double kj) noexcept {
  constexpr double lower = 0x1p-8, upper = 0x1p8;
  return (r2 == 0.0 || r2 >= 0x1p-16) && r2 <= 0x1p12 && wi >= lower && wi <= upper &&
         wj >= lower && wj <= upper && ki >= lower && ki <= upper &&
         kj >= lower && kj <= upper;
}

template <Vv10Variant Variant, bool Features, bool Geometry>
GENERATIVEQC_NONLOCAL_PAIR_HD inline bool try_pair_values_fp32(
    double r2, double wi, double wj, double ki, double kj,
    double row_inverse_kappa, PairValues& result) noexcept {
  if (!fp32_pair_domain(r2, wi, wj, ki, kj)) return false;
  if constexpr (Variant == Vv10Variant::rvv10 && Features)
    if (!(row_inverse_kappa >= 0x1p-8 && row_inverse_kappa <= 0x1p8)) return false;
  float mixed_phi{}, mixed_dphi_domega{}, mixed_dphi_dkappa{}, mixed_dphi_dr2{};
"""
        + "\n".join(mixed_branches)
        + """
  result = {mixed_phi, mixed_dphi_domega, mixed_dphi_dkappa, mixed_dphi_dr2};
  return true;
}

template <Vv10Variant Variant, bool Features, bool Geometry, bool Preconditioned = false>
GENERATIVEQC_NONLOCAL_PAIR_HD inline PairValues pair_values(
    double r2, double wi, double wj, double ki, double kj,
    double row_inverse_kappa = 0.0) noexcept {
  static_assert(Variant == Vv10Variant::vv10 || Variant == Vv10Variant::rvv10);
  PairValues result{};
"""
        + "\n".join(branches)
        + """
  return result;
}

// Caller proves the same domain as pair_values_vv10_rational. Keeping this
// entry point in generated code lets a linear device preflight discharge the
// pairwise checks without duplicating or changing scientific arithmetic.
template <bool Geometry>
GENERATIVEQC_NONLOCAL_PAIR_HD inline PairValues pair_values_vv10_admitted(
    double r2, double wi, double wj, double ki, double kj) noexcept {
  PairValues result{};
  if constexpr (Geometry)
    pair_vv10_unit_reciprocal_g1(r2, wi, wj, ki, kj, result.phi,
                         result.dphi_domega, result.dphi_dkappa, result.dphi_dr2);
  else
    pair_vv10_unit_reciprocal_g0(r2, wi, wj, ki, kj, result.phi,
                         result.dphi_domega, result.dphi_dkappa);
  return result;
}

// Exact scalar pair outputs for two live legs; the caller owns the separately
// qualified FP64 accumulation schedule and the admitted numerical domain.
struct BilateralPairValues : PairValues {
  double dphi_domega_j{};
  double dphi_dkappa_j{};
};

template <bool Geometry>
GENERATIVEQC_NONLOCAL_PAIR_HD inline BilateralPairValues pair_values_vv10_bilateral(
    double r2, double wi, double wj, double ki, double kj) noexcept {
  BilateralPairValues result{};
  if constexpr (Geometry)
    pair_vv10_bilateral_g1(r2, wi, wj, ki, kj, result.phi, result.dphi_domega,
                          result.dphi_dkappa, result.dphi_domega_j,
                          result.dphi_dkappa_j, result.dphi_dr2);
  else
    pair_vv10_bilateral_g0(r2, wi, wj, ki, kj, result.phi, result.dphi_domega,
                          result.dphi_dkappa, result.dphi_domega_j,
                          result.dphi_dkappa_j);
  return result;
}

template <bool Geometry>
GENERATIVEQC_NONLOCAL_PAIR_HD inline bool try_pair_values_vv10_bilateral_fp32(
    double r2, double wi, double wj, double ki, double kj,
    BilateralPairValues& result) noexcept {
  if (!fp32_pair_domain(r2, wi, wj, ki, kj)) return false;
  float phi{}, omega{}, kappa{}, omega_j{}, kappa_j{}, radial{};
  if constexpr (Geometry)
    pair_vv10_bilateral_g1_fp32(static_cast<float>(r2), static_cast<float>(wi),
        static_cast<float>(wj), static_cast<float>(ki), static_cast<float>(kj),
        phi, omega, kappa, omega_j, kappa_j, radial);
  else
    pair_vv10_bilateral_g0_fp32(static_cast<float>(r2), static_cast<float>(wi),
        static_cast<float>(wj), static_cast<float>(ki), static_cast<float>(kj),
        phi, omega, kappa, omega_j, kappa_j);
  result = {{phi, omega, kappa, radial}, omega_j, kappa_j};
  return true;
}

// The general bounded closure keeps the original energy root. Only the
// prevalidated molecular entry point above selects unit-reciprocal rounding.
// Positive g_i/g_j and their sums/products stay finite and normal in this
// domain. Outside it, preserve the original ordered exception behavior.
template <bool Geometry>
GENERATIVEQC_NONLOCAL_PAIR_HD inline PairValues pair_values_vv10_rational(
    double r2, double wi, double wj, double ki, double kj,
    double row_inverse_kappa) noexcept {
  constexpr double lower = 0x1p-32, upper = 0x1p32;
  if (r2 >= 0.0 && r2 <= upper && wi >= lower && wi <= upper &&
      wj >= lower && wj <= upper && ki >= lower && ki <= upper &&
      kj >= lower && kj <= upper) {
    PairValues result{};
    if constexpr (Geometry)
      pair_vv10_rational_g1(r2, wi, wj, ki, kj, result.phi,
                           result.dphi_domega, result.dphi_dkappa, result.dphi_dr2);
    else
      pair_vv10_rational_g0(r2, wi, wj, ki, kj, result.phi,
                           result.dphi_domega, result.dphi_dkappa);
    return result;
  }
  return pair_values<Vv10Variant::vv10, true, Geometry, true>(
      r2, wi, wj, ki, kj, row_inverse_kappa);
}

"""
        + native_parallel_pair_cuda()
        + """
}  // namespace generativeqc::dft::nlc::generated

#undef GENERATIVEQC_NONLOCAL_PAIR_HD
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(native_header(), encoding="utf-8")


if __name__ == "__main__":
    main()
