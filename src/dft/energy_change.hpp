#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#define GENERATIVEQC_DFT_ENERGY_HD __host__ __device__
#else
#define GENERATIVEQC_DFT_ENERGY_HD
#endif

namespace generativeqc::dft::detail {

/** Difference of compensated electronic energies at one fixed geometry.
 *
 * Nuclear repulsion cancels exactly. Retaining the diagnostic reduction's
 * low words avoids quantizing each large trace and total energy before a
 * sub-ulp convergence test. This does not relax any convergence tolerance.
 * The first evaluation has no finite baseline and remains ineligible.
 */
GENERATIVEQC_DFT_ENERGY_HD inline double electronic_energy_change(double current,
                                                                  double current_correction,
                                                                  double previous,
                                                                  double previous_correction) {
  if (previous == std::numeric_limits<double>::infinity())
    return std::numeric_limits<double>::infinity();
#if defined(__CUDA_ARCH__)
  const double difference = __dsub_rn(current, previous);
  const double virtual_previous = __dsub_rn(difference, current);
  const double roundoff = __dadd_rn(__dsub_rn(current, __dsub_rn(difference, virtual_previous)),
                                    __dsub_rn(-previous, virtual_previous));
  return fabs(__dadd_rn(difference,
                        __dadd_rn(roundoff, __dsub_rn(current_correction, previous_correction))));
#else
  const volatile double difference = current - previous;
  const volatile double virtual_previous = difference - current;
  const volatile double recovered_current = difference - virtual_previous;
  const volatile double current_roundoff = current - recovered_current;
  const volatile double previous_roundoff = -previous - virtual_previous;
  const volatile double roundoff = current_roundoff + previous_roundoff;
  const volatile double correction = current_correction - previous_correction;
  const volatile double low = roundoff + correction;
  return std::abs(difference + low);
#endif
}

}  // namespace generativeqc::dft::detail

#undef GENERATIVEQC_DFT_ENERGY_HD
