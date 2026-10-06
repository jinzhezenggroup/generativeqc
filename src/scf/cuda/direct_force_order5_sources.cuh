#pragma once

#include "order5_weighted_eri.cuh"
#include "scf/cuda/direct_force_low_order_sources.cuh"

namespace generativeqc::scf::cuda_execution {

/** Closed s/p/d order-five domain. f-containing and higher classes retain
 * their existing bounded Cartesian derivative consumer. */
__host__ __device__ inline constexpr bool weighted_order5_source_class(unsigned shell_class) {
  return shell_class == kDpppShellClass || shell_class == kDpdsShellClass ||
         shell_class == kDdpsShellClass;
}

template <unsigned ShellClass>
struct Order5SourceRoots {
  static_assert(weighted_order5_source_class(ShellClass));
  static constexpr unsigned angular_order = 5U;
  static constexpr unsigned component_count = ShellClass == kDpppShellClass ? 162U : 108U;

  __device__ static __forceinline__ generated_weighted_eri::IndependentGradient evaluate(
      const generated_weighted_eri::Geometry& geometry, const double* weights) {
    if constexpr (ShellClass == kDpppShellClass) {
      return generated_weighted_eri::dppp_force(geometry, weights);
    } else if constexpr (ShellClass == kDpdsShellClass) {
      return generated_weighted_eri::dpds_force(geometry, weights);
    } else {
      return generated_weighted_eri::ddps_force(geometry, weights);
    }
  }
};

/** Contract separate or combined J/K sources with one shell-pair geometry and
 * radial ladder per primitive. The existing task adapter owns screening,
 * canonical component weights, source-zero guards and atom scatter. This
 * experimental owner changes no shell/AO admission predicate or queue size. */
template <bool Unrestricted, DirectForceOutputMode Mode = DirectForceOutputMode::Separate>
__device__ inline void contract_two_electron_force_order5_sources(
    unsigned shell_class, const DeviceBatch& batch, ActiveShellQuartetTile task,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, double coulomb_coefficient,
    double exchange_coefficient) {
#define GENERATIVEQC_ORDER5_SOURCE_CASE(ShellClass)                                          \
  case ShellClass:                                                                           \
    contract_two_electron_force_low_order_sources_task<Unrestricted, ShellClass, false,      \
                                                       Order5SourceRoots<ShellClass>, Mode>( \
        batch, task, screening_tolerance, schwarz_bounds, density, active, forces,           \
        coulomb_coefficient, exchange_coefficient);                                          \
    break
  switch (shell_class) {
    GENERATIVEQC_ORDER5_SOURCE_CASE(kDpppShellClass);
    GENERATIVEQC_ORDER5_SOURCE_CASE(kDpdsShellClass);
    GENERATIVEQC_ORDER5_SOURCE_CASE(kDdpsShellClass);
    default:
      break;
  }
#undef GENERATIVEQC_ORDER5_SOURCE_CASE
}

}  // namespace generativeqc::scf::cuda_execution
