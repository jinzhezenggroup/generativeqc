// SPDX-License-Identifier: LGPL-3.0-or-later
// r2SCAN-3c gCP qualification/runtime kernel.
// Equations and parameters audited against simple-dftd3 commit
// 41d5a07b98ce15e97bec7a1815869725f6c7b0c2 (LGPL-3.0-or-later).
#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "dft/dispersion/gcp_r2scan3c_data.hpp"
#include "generated_dispersion_scalar.hpp"
#include "generated_gcp_scalar.hpp"
#include "generated_method_parameters.hpp"

#if defined(__CUDACC__)
#define GENERATIVEQC_GCP_HD __host__ __device__
#else
#define GENERATIVEQC_GCP_HD
#endif

namespace generativeqc::dft::dispersion {

enum class GCPStatus : std::int32_t {
  success = 0,
  invalid_argument = 1,
  unsupported_element = 2,
  coincident_atoms = 3,
  numerical_failure = 4,
};

struct GCPParameters {
  double sigma = ::generativeqc::generated::method_parameters::r2scan3cGcp().sigma;
  double alpha = ::generativeqc::generated::method_parameters::r2scan3cGcp().alpha;
  double beta = ::generativeqc::generated::method_parameters::r2scan3cGcp().beta;
  double damping_scale = ::generativeqc::generated::method_parameters::r2scan3cGcp().damping_scale;
  double damping_exponent =
      ::generativeqc::generated::method_parameters::r2scan3cGcp().damping_exponent;
};

inline constexpr GCPParameters r2scan3c_gcp_parameters() { return {}; }

namespace gcp_detail {
GENERATIVEQC_GCP_HD inline bool finite(double x) { return x == x && x <= DBL_MAX && x >= -DBL_MAX; }

inline constexpr double kAngstromToBohr = 1.8897261254578281;

GENERATIVEQC_GCP_HD inline bool overlap(double r, int shell_a, int shell_b, double za, double zb,
                                        double* s, double* ds) {
  const bool same = std::abs(za - zb) < 0.1;
  const int key = shell_a * shell_b;
  if (shell_a > shell_b) {
    const double old = za;
    za = zb;
    zb = old;
  }
  double output[2];
  const bool valid =
      ds ? ::generativeqc::generated::dispersion::gcp_overlap<true>(key, same, r, za, zb, output)
         : ::generativeqc::generated::dispersion::gcp_overlap<false>(key, same, r, za, zb, output);
  if (!valid) return false;
  *s = output[0];
  if (ds) {
    *ds = output[1];
  }
  return finite(*s) && (!ds || finite(*ds)) && *s > 0.0;
}
/** Shared audited pair equation. r and r0 are in Bohr; derivative is dE/dr.
 * A null derivative performs energy work only, on either backend. */
GENERATIVEQC_GCP_HD inline GCPStatus pair(const gcp_data::Element& ei, const gcp_data::Element& ej,
                                          double r0, double r, const GCPParameters& p,
                                          double* energy, double* derivative) {
  double s = 0.0, ds = 0.0;
  if (!overlap(r, ei.shell, ej.shell, ei.slater, ej.slater, &s, derivative ? &ds : nullptr)) {
    return GCPStatus::numerical_failure;
  }
  const double xi = ei.xv >= 0.5 ? 1.0 / std::sqrt(ei.xv) : 0.0;
  const double xj = ej.xv >= 0.5 ? 1.0 / std::sqrt(ej.xv) : 0.0;
  const double pref = p.sigma * (ei.emiss * xj + ej.emiss * xi);
  double output[2];
  if (derivative) {
    ::generativeqc::generated::dispersion::gcp_pair<true>(r, r0, p.alpha, p.beta, p.damping_scale,
                                                          p.damping_exponent, pref, s, ds, output);
  } else {
    ::generativeqc::generated::dispersion::gcp_pair<false>(
        r, r0, p.alpha, p.beta, p.damping_scale, p.damping_exponent, pref, s, 0.0, output);
  }
  if (!finite(output[0])) return GCPStatus::numerical_failure;
  *energy = output[0];
  if (derivative) {
    if (!finite(output[1])) return GCPStatus::numerical_failure;
    *derivative = output[1];
  }
  return GCPStatus::success;
}
}  // namespace gcp_detail

/** A null gradient selects the same energy without derivative arithmetic or
 * scratch gradient storage. Gradient calls consume compiler-AD scalar programs. */
inline GCPStatus evaluate_r2scan3c_gcp(int n, const std::int32_t* z, const double* xyz,
                                       const GCPParameters& p, double* energy, double* gradient) {
  if (n < 0 || n > std::numeric_limits<int>::max() / 3 || (n > 0 && (!z || !xyz)) || !energy ||
      (gradient && gradient == xyz)) {
    return GCPStatus::invalid_argument;
  }
  if (!(std::isfinite(p.sigma) && p.sigma > 0.0 && std::isfinite(p.alpha) && p.alpha > 0.0 &&
        std::isfinite(p.beta) && p.beta > 0.0 && std::isfinite(p.damping_scale) &&
        p.damping_scale > 0.0 && std::isfinite(p.damping_exponent) && p.damping_exponent > 0.0)) {
    return GCPStatus::invalid_argument;
  }
  *energy = 0.0;
  if (gradient) std::fill(gradient, gradient + 3 * n, 0.0);
  for (int i = 0; i < n; ++i) {
    if (!::generativeqc::generated::method_parameters::r2scan3cGcpSupportsAtomicNumber(z[i]))
      return GCPStatus::unsupported_element;
    for (int k = 0; k < 3; ++k) {
      if (!std::isfinite(xyz[3 * i + k])) return GCPStatus::invalid_argument;
    }
  }
  for (int i = 0; i < n; ++i) {
    const auto& ei = gcp_data::kElements[static_cast<std::size_t>(z[i] - 1)];
    for (int j = 0; j < i; ++j) {
      const auto& ej = gcp_data::kElements[static_cast<std::size_t>(z[j] - 1)];
      std::array<double, 3> v{};
      double r2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        v[k] = xyz[3 * i + k] - xyz[3 * j + k];
        r2 += v[k] * v[k];
      }
      if (!(r2 > 0.0) || !std::isfinite(r2)) return GCPStatus::coincident_atoms;
      const double r = std::sqrt(r2);
      const int hi = std::max(z[i], z[j]), lo = std::min(z[i], z[j]);
      const auto ridx = static_cast<std::size_t>(hi * (hi - 1) / 2 + lo - 1);
      double pair = 0.0, dedr = 0.0;
      const auto status =
          gcp_detail::pair(ei, ej, gcp_data::kVdwAngstrom[ridx] * gcp_detail::kAngstromToBohr, r, p,
                           &pair, gradient ? &dedr : nullptr);
      if (status != GCPStatus::success) return status;
      *energy += pair;
      if (!gradient) continue;
      for (int k = 0; k < 3; ++k) {
        double projected[1];
        ::generativeqc::generated::dispersion::radial_projection(r, dedr, v[k], projected);
        const double g = projected[0];
        gradient[3 * i + k] += g;
        gradient[3 * j + k] -= g;
      }
    }
  }
  return std::isfinite(*energy) ? GCPStatus::success : GCPStatus::numerical_failure;
}

}  // namespace generativeqc::dft::dispersion

#undef GENERATIVEQC_GCP_HD
