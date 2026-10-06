#pragma once

#include "order4_weighted_eri.cuh"
#include "scf/cuda/direct_force_low_order_sources.cuh"

namespace generativeqc::scf::cuda_execution {

/** Closed s/p/d order-four domain. f-containing and higher classes retain
 * their existing bounded Cartesian derivative consumer. */
__host__ __device__ inline constexpr bool weighted_order4_source_class(unsigned shell_class) {
  return shell_class == kPpppShellClass || shell_class == kDsppShellClass ||
         shell_class == kDsdsShellClass || shell_class == kDppsShellClass ||
         shell_class == kDdssShellClass;
}

template <unsigned ShellClass>
struct Order4SourceRoots {
  static_assert(weighted_order4_source_class(ShellClass));
  static constexpr unsigned angular_order = 4U;
  static constexpr unsigned component_count =
      ShellClass == kPpppShellClass                                    ? 81U
      : ShellClass == kDsppShellClass || ShellClass == kDppsShellClass ? 54U
                                                                       : 36U;

  __device__ static __forceinline__ generated_weighted_eri::IndependentGradient evaluate(
      const generated_weighted_eri::Geometry& geometry, const double* weights) {
    if constexpr (ShellClass == kPpppShellClass) {
      return generated_weighted_eri::pppp_force(geometry, weights);
    } else if constexpr (ShellClass == kDsppShellClass) {
      return generated_weighted_eri::dspp_force(geometry, weights);
    } else if constexpr (ShellClass == kDsdsShellClass) {
      return generated_weighted_eri::dsds_force(geometry, weights);
    } else if constexpr (ShellClass == kDppsShellClass) {
      return generated_weighted_eri::dpps_force(geometry, weights);
    } else {
      return generated_weighted_eri::ddss_force(geometry, weights);
    }
  }
};

/** Contract separate or combined J/K sources with one shell-pair geometry and
 * radial ladder per primitive. The existing task adapter owns screening,
 * canonical component weights, source-zero guards and atom scatter. This
 * experimental owner changes no shell/AO admission predicate or queue size. */
template <bool Unrestricted, DirectForceOutputMode Mode = DirectForceOutputMode::Separate>
__device__ inline void contract_two_electron_force_order4_sources(
    unsigned shell_class, const DeviceBatch& batch, ActiveShellQuartetTile task,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, double coulomb_coefficient,
    double exchange_coefficient) {
#define GENERATIVEQC_ORDER4_SOURCE_CASE(ShellClass)                                          \
  case ShellClass:                                                                           \
    contract_two_electron_force_low_order_sources_task<Unrestricted, ShellClass, false,      \
                                                       Order4SourceRoots<ShellClass>, Mode>( \
        batch, task, screening_tolerance, schwarz_bounds, density, active, forces,           \
        coulomb_coefficient, exchange_coefficient);                                          \
    break
  switch (shell_class) {
    GENERATIVEQC_ORDER4_SOURCE_CASE(kPpppShellClass);
    GENERATIVEQC_ORDER4_SOURCE_CASE(kDsppShellClass);
    GENERATIVEQC_ORDER4_SOURCE_CASE(kDsdsShellClass);
    GENERATIVEQC_ORDER4_SOURCE_CASE(kDppsShellClass);
    GENERATIVEQC_ORDER4_SOURCE_CASE(kDdssShellClass);
    default:
      break;
  }
#undef GENERATIVEQC_ORDER4_SOURCE_CASE
}

}  // namespace generativeqc::scf::cuda_execution
