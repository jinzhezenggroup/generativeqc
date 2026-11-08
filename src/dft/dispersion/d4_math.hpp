
#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dft/dispersion/d4_types.hpp"
#include "generated_dispersion_scalar.hpp"

#if defined(__CUDACC__)
#define GENERATIVEQC_D4_MATH_HD __host__ __device__
#else
#define GENERATIVEQC_D4_MATH_HD
#endif

namespace generativeqc::dft::dispersion::math {

inline constexpr int kMaximumReferences = 7;
inline constexpr double kReferenceWeightFactor = 6.0;
inline constexpr double kMinimumWeightNorm = 1.4916681462400413e-154;
inline constexpr double kElectronegativityScale = 4.10451;
inline constexpr double kElectronegativityShift = 19.08857;
inline constexpr double kElectronegativityWidth = 11.28174;

struct ChargeScale {
  double value;
  double derivative;
};

template <bool NeedDerivative>
GENERATIVEQC_D4_MATH_HD inline ChargeScale charge_scale_impl(double a, double c, double qref,
                                                             double qmod) {
  ChargeScale result{exp(a), 0.0};
  if (qmod <= 0.0) return result;
  double values[2]{};
  ::generativeqc::generated::dispersion::d4_charge<NeedDerivative>(a, c, qref, qmod, values);
  result.value = values[0];
  if constexpr (NeedDerivative) {
    result.derivative = values[1];
  }
  return result;
}

GENERATIVEQC_D4_MATH_HD inline ChargeScale charge_scale(double a, double c, double qref,
                                                        double qmod) {
  return charge_scale_impl<true>(a, c, qref, qmod);
}

GENERATIVEQC_D4_MATH_HD inline double charge_scale_value(double a, double c, double qref,
                                                         double qmod) {
  return charge_scale_impl<false>(a, c, qref, qmod).value;
}

struct CoordinationPair {
  double value;
  double derivative;
};

struct CoordinationParameters {
  double radius;
  double electronegativity_factor;
};

template <class FirstElement, class SecondElement>
GENERATIVEQC_D4_MATH_HD inline CoordinationParameters coordination_parameters(
    const FirstElement& first, const SecondElement& second) {
  const double radius = first.covalent_radius + second.covalent_radius;
  const double den =
      fabs(first.electronegativity - second.electronegativity) + kElectronegativityShift;
  const double en = kElectronegativityScale *
                    exp(-(den * den) / (2.0 * kElectronegativityWidth * kElectronegativityWidth));
  return {radius, en};
}

GENERATIVEQC_D4_MATH_HD inline CoordinationPair coordination_pair(CoordinationParameters parameters,
                                                                  double distance) {
  double values[2];
  ::generativeqc::generated::dispersion::d4_coordination(
      distance, parameters.radius, parameters.electronegativity_factor, values);
  return {values[0], values[1]};
}

template <class FirstElement, class SecondElement>
GENERATIVEQC_D4_MATH_HD inline CoordinationPair coordination_pair(const FirstElement& first,
                                                                  const SecondElement& second,
                                                                  double distance) {
  return coordination_pair(coordination_parameters(first, second), distance);
}

template <class FirstElement, class SecondElement>
GENERATIVEQC_D4_MATH_HD inline double pair_rr(const FirstElement& first,
                                              const SecondElement& second) {
  return 3.0 * first.r4r2 * second.r4r2;
}

template <class FirstElement, class SecondElement>
GENERATIVEQC_D4_MATH_HD inline double damping_radius(const FirstElement& first,
                                                     const SecondElement& second, double a1,
                                                     double a2) {
  return a1 * sqrt(pair_rr(first, second)) + a2;
}

/** BJ damping value and Cartesian scatter coefficient (damping radial derivative / r). */
struct PairDamping {
  double value;
  double derivative;
};

GENERATIVEQC_D4_MATH_HD inline PairDamping pair_damping(double distance_squared, double rr,
                                                        double r0, double s6, double s8) {
  double values[2];
  ::generativeqc::generated::dispersion::bj_damping(distance_squared, rr, r0, s6, s8, values);
  return {values[0], values[1]};
}

template <class FirstElement, class SecondElement>
GENERATIVEQC_D4_MATH_HD inline PairDamping pair_damping(const FirstElement& first,
                                                        const SecondElement& second,
                                                        double distance_squared, double s6,
                                                        double s8, double a1, double a2) {
  const double rr = pair_rr(first, second);
  return pair_damping(distance_squared, rr, damping_radius(first, second, a1, a2), s6, s8);
}

template <class ElementData, class ReferenceData>
GENERATIVEQC_D4_MATH_HD inline void atom_weights(const ElementData& element,
                                                 const ReferenceData* references,
                                                 double coordination, double charge,
                                                 bool zero_charge, double ga, double gc,
                                                 double* weights, double* cn_derivatives,
                                                 double* charge_derivatives) {
  for (int local = 0; local < kMaximumReferences; ++local) {
    weights[local] = 0.0;
    if (cn_derivatives != nullptr) cn_derivatives[local] = 0.0;
    if (charge_derivatives != nullptr) charge_derivatives[local] = 0.0;
  }

  double normalization = 0.0;
  double normalization_derivative = 0.0;
  double maximum_reference_cn = -DBL_MAX;
  for (int local = 0; local < element.reference_count; ++local) {
    const auto reference = references[element.reference_offset + local];
    maximum_reference_cn = fmax(maximum_reference_cn, reference.coordination_number);
    for (int gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
      double values[2];
      if (cn_derivatives != nullptr)
        ::generativeqc::generated::dispersion::gaussian(coordination, reference.coordination_number,
                                                        factor, values);
      else
        ::generativeqc::generated::dispersion::gaussian<false>(
            coordination, reference.coordination_number, factor, values);
      normalization += values[0];
      if (cn_derivatives != nullptr) normalization_derivative += values[1];
    }
  }

  const double inverse_normalization =
      normalization > kMinimumWeightNorm ? 1.0 / normalization : 0.0;
  const double qmod = (zero_charge ? 0.0 : charge) + element.effective_charge;
  for (int local = 0; local < element.reference_count; ++local) {
    const auto reference = references[element.reference_offset + local];
    double numerator = 0.0;
    double numerator_derivative = 0.0;
    for (int gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
      double values[2];
      if (cn_derivatives != nullptr)
        ::generativeqc::generated::dispersion::gaussian(coordination, reference.coordination_number,
                                                        factor, values);
      else
        ::generativeqc::generated::dispersion::gaussian<false>(
            coordination, reference.coordination_number, factor, values);
      numerator += values[0];
      if (cn_derivatives != nullptr) numerator_derivative += values[1];
    }
    const double cn_weight =
        inverse_normalization != 0.0
            ? numerator * inverse_normalization
            : (fabs(maximum_reference_cn - reference.coordination_number) < 1.0e-12 ? 1.0 : 0.0);
    double cn_derivative = 0.0;
    if (cn_derivatives != nullptr && inverse_normalization != 0.0) {
      double normalized[2];
      ::generativeqc::generated::dispersion::normalize(
          numerator, normalization, numerator_derivative, normalization_derivative, normalized);
      cn_derivative = normalized[1];
    }
    const double charge_steepness = gc * element.hardness;
    const double reference_charge = reference.charge + element.effective_charge;
    if (charge_derivatives != nullptr) {
      const auto scale = charge_scale(ga, charge_steepness, reference_charge, qmod);
      weights[local] = cn_weight * scale.value;
      if (cn_derivatives != nullptr) cn_derivatives[local] = cn_derivative * scale.value;
      charge_derivatives[local] = zero_charge ? 0.0 : cn_weight * scale.derivative;
    } else {
      const double scale = charge_scale_value(ga, charge_steepness, reference_charge, qmod);
      weights[local] = cn_weight * scale;
      if (cn_derivatives != nullptr) cn_derivatives[local] = cn_derivative * scale;
    }
  }
}

struct Coefficient {
  double c6;
  double first_cn;
  double second_cn;
  double first_charge;
  double second_charge;
};

struct PackedReferenceC6 {
  const double* values;
  GENERATIVEQC_D4_MATH_HD inline double operator()(int first, int second) const {
    const int high = first > second ? first : second;
    const int low = first > second ? second : first;
    return values[high * (high + 1) / 2 + low];
  }
};

struct DenseReferenceC6 {
  const double* values;
  std::size_t stride;
  GENERATIVEQC_D4_MATH_HD inline double operator()(int first, int second) const {
    return values[static_cast<std::size_t>(first) * stride + static_cast<std::size_t>(second)];
  }
};

template <bool NeedC6, bool NeedCn, bool NeedCharge, class FirstElement, class SecondElement,
          class ReferenceC6>
GENERATIVEQC_D4_MATH_HD inline Coefficient coefficient_selected(
    const FirstElement& first_element, const SecondElement& second_element,
    ReferenceC6 reference_c6, const double* first_weights, const double* first_cn_derivatives,
    const double* first_charge_derivatives, const double* second_weights,
    const double* second_cn_derivatives, const double* second_charge_derivatives) {
  Coefficient result{};
  for (int first_ref = 0; first_ref < first_element.reference_count; ++first_ref) {
    const int global_first = first_element.reference_offset + first_ref;
    const double first_value = first_weights[first_ref];
    const double first_cn =
        NeedCn && first_cn_derivatives != nullptr ? first_cn_derivatives[first_ref] : 0.0;
    const double first_charge = NeedCharge && first_charge_derivatives != nullptr
                                    ? first_charge_derivatives[first_ref]
                                    : 0.0;
    for (int second_ref = 0; second_ref < second_element.reference_count; ++second_ref) {
      const int global_second = second_element.reference_offset + second_ref;
      const double reference = reference_c6(global_first, global_second);
      const double second_value = second_weights[second_ref];
      if constexpr (NeedC6) result.c6 += first_value * second_value * reference;
      if constexpr (NeedCn) {
        const double second_cn =
            second_cn_derivatives == nullptr ? 0.0 : second_cn_derivatives[second_ref];
        result.first_cn += first_cn * second_value * reference;
        result.second_cn += first_value * second_cn * reference;
      }
      if constexpr (NeedCharge) {
        const double second_charge =
            second_charge_derivatives == nullptr ? 0.0 : second_charge_derivatives[second_ref];
        result.first_charge += first_charge * second_value * reference;
        result.second_charge += first_value * second_charge * reference;
      }
    }
  }
  return result;
}

template <class FirstElement, class SecondElement, class ReferenceC6>
GENERATIVEQC_D4_MATH_HD inline Coefficient coefficient(
    const FirstElement& first_element, const SecondElement& second_element,
    ReferenceC6 reference_c6, const double* first_weights, const double* first_cn_derivatives,
    const double* first_charge_derivatives, const double* second_weights,
    const double* second_cn_derivatives, const double* second_charge_derivatives) {
  return coefficient_selected<true, true, true>(
      first_element, second_element, reference_c6, first_weights, first_cn_derivatives,
      first_charge_derivatives, second_weights, second_cn_derivatives, second_charge_derivatives);
}

struct AtmResponse {
  double energy;
  double radial[3];
  double coefficient[3];
};

// Edge responses are Cartesian scatter coefficients, in ij, ik, jk order.
// Coefficient responses are dE/dC6 in the same order; energy has the physical sign.
template <bool AbsoluteC6Product = false, bool NeedDerivative = true>
GENERATIVEQC_D4_MATH_HD inline AtmResponse atm_response(
    double first_second_r2, double first_third_r2, double second_third_r2, double first_second_r0,
    double first_third_r0, double second_third_r0, double first_second_c6, double first_third_c6,
    double second_third_c6, double s9, double exponent = 16.0) {
  double output[7]{};
  if constexpr (AbsoluteC6Product) {
    ::generativeqc::generated::dispersion::atm_absolute<NeedDerivative>(
        first_second_r2, first_third_r2, second_third_r2,
        first_second_r0 * first_third_r0 * second_third_r0, first_second_c6, first_third_c6,
        second_third_c6, s9, exponent, output);
  } else {
    ::generativeqc::generated::dispersion::atm<NeedDerivative>(
        first_second_r2, first_third_r2, second_third_r2,
        first_second_r0 * first_third_r0 * second_third_r0, first_second_c6, first_third_c6,
        second_third_c6, s9, exponent, output);
  }
  return {output[0], {output[1], output[2], output[3]}, {output[4], output[5], output[6]}};
}

}  // namespace generativeqc::dft::dispersion::math

#undef GENERATIVEQC_D4_MATH_HD
