#pragma once

#include <cmath>
#include <cstddef>
#include <span>

namespace generativeqc::scf {

/** Shared FP64 proof limits for algebraic cold-density exchange factors.
 * These are reconstruction gates, not SCF tolerances. A factor never grants
 * canonical-orbital, warm-state or stationary-response provenance by itself.
 */
inline constexpr double kDensitySeedSpectralThreshold = 1e-13;
inline constexpr double kDensitySeedDiscardedFrobeniusTolerance = 1e-12;
inline constexpr double kDensitySeedMaximumTolerance = 1e-12;
inline constexpr double kDensitySeedRmsTolerance = 1e-13;

struct DensitySeedSpectrum {
  bool valid{};
  std::size_t rank{};
  double discarded_square{};

  bool within_discarded_budget() const noexcept {
    return valid && discarded_square <= kDensitySeedDiscardedFrobeniusTolerance *
                                            kDensitySeedDiscardedFrobeniusTolerance;
  }
};

/** Inspect the ascending Euclidean spectrum of an AO density. AO metric
 * idempotency is deliberately not inferred here: both triangles of the
 * original density must still pass the independent full reconstruction.
 */
inline DensitySeedSpectrum density_seed_spectrum(std::span<const double> values) noexcept {
  DensitySeedSpectrum result;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double value = values[i];
    if (!std::isfinite(value) || value < -kDensitySeedSpectralThreshold ||
        (i && value < values[i - 1]))
      return result;
    if (value > kDensitySeedSpectralThreshold)
      ++result.rank;
    else
      result.discarded_square += value * value;
  }
  result.valid = true;
  return result;
}

/** The error reduction covers every original matrix entry, including the
 * triangle ignored by the symmetric eigensolver and any nonfinite input. */
inline bool density_seed_reconstruction_accepted(double maximum, double rms) noexcept {
  return std::isfinite(maximum) && std::isfinite(rms) && maximum >= 0.0 && rms >= 0.0 &&
         maximum <= kDensitySeedMaximumTolerance && rms <= kDensitySeedRmsTolerance;
}

}  // namespace generativeqc::scf
