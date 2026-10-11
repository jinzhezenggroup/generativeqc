#pragma once

#include "integrals/execution_qualifiers.hpp"
#include "integrals/range_moments.hpp"
#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/gaussian_geometry.cuh"

// Retained Coulomb auxiliary recurrence, bounded by the instantiated
// angular order. Layout assertions guard the existing workspace contract.
namespace generativeqc::scf::cuda_execution {

/** Clip response bounds without host-only standard-library calls in CUDA. */
GENERATIVEQC_ERI_SHARED_HD inline unsigned coulomb_domain_bound(unsigned requested,
                                                                unsigned capacity) {
  return requested < capacity ? requested : capacity;
}

/** Optional exact dependency domain of one Cartesian component contraction.
 *
 * The consumer reads R(0,t,u,v) only through the component's summed x/y/z
 * powers. Its x-first dependency graph needs n+t<=x, then n+u<=x+y
 * on the t=0 plane, and n+v<=L on the t=u=0 line. This prunes recurrence
 * states, never integrals or small numerical values. Unselected or invalid
 * domains retain the complete simplex. Its z/y/x traversal overwrites every
 * state before consumption, including each dependency of the next plane.
 * A first-response domain includes one raised Cartesian root per axis. Its
 * bounds enclose all three response faces and their dependencies, clipped to
 * the L+1 simplex; the AO/component domain itself is unchanged.
 */
struct CoulombComponentDomain {
  unsigned x{};
  unsigned y{};
  unsigned z{};
  bool selected{};
  bool first_response{};

  template <unsigned MaximumAngular>
  GENERATIVEQC_ERI_SHARED_HD bool valid() const {
    return selected && x <= MaximumAngular && y <= MaximumAngular && z <= MaximumAngular &&
           x + y + z + static_cast<unsigned>(first_response) == MaximumAngular;
  }
};

template <typename Scalar, unsigned MaximumAngular>
struct CoulombAuxiliary {
  static_assert(MaximumAngular <= kMaximumCoulombResponseOrder);

  GENERATIVEQC_ERI_SHARED_HD static constexpr unsigned choose3(unsigned value) {
    return value < 3 ? 0 : value * (value - 1) * (value - 2) / 6;
  }

  GENERATIVEQC_ERI_SHARED_HD static constexpr unsigned choose4(unsigned value) {
    return value < 4 ? 0 : value * (value - 1) * (value - 2) * (value - 3) / 24;
  }

  static constexpr unsigned kStateCount = choose4(MaximumAngular + 4);
  Scalar data[kStateCount];

  GENERATIVEQC_ERI_DEVICE inline unsigned index(unsigned n, unsigned t, unsigned u,
                                                unsigned v) const {
    const unsigned total = choose4(MaximumAngular + 4);
    const unsigned n_offset = total - choose4(MaximumAngular - n + 4);
    const unsigned remaining_after_n = MaximumAngular - n;
    const unsigned t_offset = choose3(remaining_after_n + 3) - choose3(remaining_after_n - t + 3);
    const unsigned remaining_after_t = remaining_after_n - t;
    const unsigned u_offset = u * (remaining_after_t + 1) - u * (u - 1) / 2;
    return n_offset + t_offset + u_offset + v;
  }

  GENERATIVEQC_ERI_DEVICE inline Scalar& at(unsigned n, unsigned t, unsigned u, unsigned v) {
    return data[index(n, t, u, v)];
  }
  GENERATIVEQC_ERI_DEVICE inline const Scalar& at(unsigned n, unsigned t, unsigned u,
                                                  unsigned v) const {
    return data[index(n, t, u, v)];
  }
};

// Keep each angular specialization at the exact simplex size so lower-order
// work initializes and indexes only the states its recurrence can reach.
static_assert(CoulombAuxiliary<double, 0>::kStateCount == 1);
static_assert(CoulombAuxiliary<double, 1>::kStateCount == 5);
static_assert(CoulombAuxiliary<double, 2>::kStateCount == 15);
static_assert(CoulombAuxiliary<double, 6>::kStateCount == 210);
static_assert(CoulombAuxiliary<double, 12>::kStateCount == 1820);
static_assert(CoulombAuxiliary<double, 13>::kStateCount == 2380);

template <unsigned MaximumAngular, typename Scalar>
GENERATIVEQC_ERI_DEVICE inline void fill_coulomb(
    EvaluationReal<Scalar> exponent, const Vec3<Scalar>& product, const Vec3<Scalar>& center,
    CoulombAuxiliary<Scalar, MaximumAngular>& auxiliary, CoulombComponentDomain domain = {}) {
  const bool reachable = domain.valid<MaximumAngular>();
  const unsigned halo = reachable && domain.first_response ? 1U : 0U;
  // Roots and the successive z/y/x planes write the entire full-domain
  // simplex exactly once. Clearing it first only adds per-primitive stores;
  // neither the complete nor the reachable recurrence reads an unwritten cell.
  const Vec3<Scalar> pc{product.x - center.x, product.y - center.y, product.z - center.z};
  Scalar boys[MaximumAngular + 1];
  boys_values<MaximumAngular>(exponent * distance_squared(product, center), boys);
  EvaluationReal<Scalar> factor{1.0};
  for (unsigned n = 0; n <= MaximumAngular; ++n) {
    auxiliary.at(n, 0, 0, 0) = factor * boys[n];
    factor = factor * (-2.0 * exponent);
  }

  // Every reachable read has an earlier write in this unchanged z/y/x
  // recurrence order. Unused simplex cells need neither evaluation nor zeroing.
  const unsigned maximum_v = reachable ? domain.z + halo : MaximumAngular;
  for (unsigned v = 1; v <= maximum_v; ++v) {
    for (unsigned n = 0; n + v <= MaximumAngular; ++n) {
      Scalar value = pc.z * auxiliary.at(n + 1, 0, 0, v - 1);
      if (v > 1) {
        value = value + static_cast<double>(v - 1) * auxiliary.at(n + 1, 0, 0, v - 2);
      }
      auxiliary.at(n, 0, 0, v) = value;
    }
  }
  for (unsigned v = 0; v <= maximum_v; ++v) {
    const unsigned maximum_u =
        reachable ? coulomb_domain_bound(domain.y + halo, MaximumAngular - v) : MaximumAngular - v;
    for (unsigned u = 1; u <= maximum_u; ++u) {
      const unsigned maximum_n =
          reachable ? coulomb_domain_bound(domain.x + domain.y + halo - u, MaximumAngular - u - v)
                    : MaximumAngular - u - v;
      for (unsigned n = 0; n <= maximum_n; ++n) {
        Scalar value = pc.y * auxiliary.at(n + 1, 0, u - 1, v);
        if (u > 1) {
          value = value + static_cast<double>(u - 1) * auxiliary.at(n + 1, 0, u - 2, v);
        }
        auxiliary.at(n, 0, u, v) = value;
      }
    }
  }
  for (unsigned v = 0; v <= maximum_v; ++v) {
    const unsigned maximum_u =
        reachable ? coulomb_domain_bound(domain.y + halo, MaximumAngular - v) : MaximumAngular - v;
    for (unsigned u = 0; u <= maximum_u; ++u) {
      const unsigned maximum_t = reachable
                                     ? coulomb_domain_bound(domain.x + halo, MaximumAngular - u - v)
                                     : MaximumAngular - u - v;
      for (unsigned t = 1; t <= maximum_t; ++t) {
        const unsigned maximum_n =
            reachable ? coulomb_domain_bound(domain.x + halo - t, MaximumAngular - t - u - v)
                      : MaximumAngular - t - u - v;
        for (unsigned n = 0; n <= maximum_n; ++n) {
          Scalar value = pc.x * auxiliary.at(n + 1, t - 1, u, v);
          if (t > 1) {
            value = value + static_cast<double>(t - 1) * auxiliary.at(n + 1, t - 2, u, v);
          }
          auxiliary.at(n, t, u, v) = value;
        }
      }
    }
  }
}

/** SR/LR radial owner for exact CUDA exchange and nuclear derivatives.
 *
 * Range moments are shared with the CPU/generated range-ERI path. Keep this
 * separate from fill_coulomb so the established full-range Boys path and its
 * rounding remain unchanged.
 */
template <unsigned MaximumAngular, typename Scalar>
GENERATIVEQC_ERI_DEVICE inline bool fill_range_coulomb(
    double exponent, const Vec3<Scalar>& product, const Vec3<Scalar>& center,
    generativeqc::integrals::CoulombRange range, double omega,
    CoulombAuxiliary<Scalar, MaximumAngular>& auxiliary, CoulombComponentDomain domain = {}) {
  static_assert(MaximumAngular <= kMaximumCoulombResponseOrder);
  if (range == generativeqc::integrals::CoulombRange::Full) return false;
  const bool reachable = domain.valid<MaximumAngular>();
  const unsigned halo = reachable && domain.first_response ? 1U : 0U;
  if (!reachable) {
    for (unsigned item = 0; item < CoulombAuxiliary<Scalar, MaximumAngular>::kStateCount; ++item)
      auxiliary.data[item] = scalar<Scalar>(0.0);
  }

  const Vec3<Scalar> pc{product.x - center.x, product.y - center.y, product.z - center.z};
  const Scalar argument = exponent * distance_squared(product, center);
  constexpr unsigned extra = std::is_same_v<Scalar, double> ? 0U : 1U;
  double moments[MaximumAngular + extra + 1];
  if (!generativeqc::integrals::bounded_range_moments<MaximumAngular + extra>(
          MaximumAngular + extra, scalar_value(argument), exponent, range, omega, moments))
    return false;

  double factor = 1.0;
  for (unsigned n = 0; n <= MaximumAngular; ++n) {
    // At fixed exponents and omega, d M_n(T)/dT = -M_(n+1)(T)
    // for both radial domains. Carry that seed through the SAME recurrence
    // as values; no displaced integrals or subtraction of full/LR forces.
    Scalar moment = scalar<Scalar>(moments[n]);
    if constexpr (std::is_same_v<Scalar, Dual>)
      moment.derivative = -moments[n + 1] * argument.derivative;
    else if constexpr (std::is_same_v<Scalar, Dual3>) {
      moment.derivative_x = -moments[n + 1] * argument.derivative_x;
      moment.derivative_y = -moments[n + 1] * argument.derivative_y;
      moment.derivative_z = -moments[n + 1] * argument.derivative_z;
    }
    auxiliary.at(n, 0, 0, 0) = factor * moment;
    factor *= -2.0 * exponent;
  }
  const unsigned maximum_v = reachable ? domain.z + halo : MaximumAngular;
  for (unsigned v = 1; v <= maximum_v; ++v) {
    for (unsigned n = 0; n + v <= MaximumAngular; ++n) {
      Scalar value = pc.z * auxiliary.at(n + 1, 0, 0, v - 1);
      if (v > 1) value = value + static_cast<double>(v - 1) * auxiliary.at(n + 1, 0, 0, v - 2);
      auxiliary.at(n, 0, 0, v) = value;
    }
  }
  for (unsigned v = 0; v <= maximum_v; ++v) {
    const unsigned maximum_u =
        reachable ? coulomb_domain_bound(domain.y + halo, MaximumAngular - v) : MaximumAngular - v;
    for (unsigned u = 1; u <= maximum_u; ++u) {
      const unsigned maximum_n =
          reachable ? coulomb_domain_bound(domain.x + domain.y + halo - u, MaximumAngular - u - v)
                    : MaximumAngular - u - v;
      for (unsigned n = 0; n <= maximum_n; ++n) {
        Scalar value = pc.y * auxiliary.at(n + 1, 0, u - 1, v);
        if (u > 1) value = value + static_cast<double>(u - 1) * auxiliary.at(n + 1, 0, u - 2, v);
        auxiliary.at(n, 0, u, v) = value;
      }
    }
  }
  for (unsigned v = 0; v <= maximum_v; ++v) {
    const unsigned maximum_u =
        reachable ? coulomb_domain_bound(domain.y + halo, MaximumAngular - v) : MaximumAngular - v;
    for (unsigned u = 0; u <= maximum_u; ++u) {
      const unsigned maximum_t = reachable
                                     ? coulomb_domain_bound(domain.x + halo, MaximumAngular - u - v)
                                     : MaximumAngular - u - v;
      for (unsigned t = 1; t <= maximum_t; ++t) {
        const unsigned maximum_n =
            reachable ? coulomb_domain_bound(domain.x + halo - t, MaximumAngular - t - u - v)
                      : MaximumAngular - t - u - v;
        for (unsigned n = 0; n <= maximum_n; ++n) {
          Scalar value = pc.x * auxiliary.at(n + 1, t - 1, u, v);
          if (t > 1) value = value + static_cast<double>(t - 1) * auxiliary.at(n + 1, t - 2, u, v);
          auxiliary.at(n, t, u, v) = value;
        }
      }
    }
  }
  return true;
}

}  // namespace generativeqc::scf::cuda_execution
