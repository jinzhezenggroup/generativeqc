#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define GENERATIVEQC_DIRECT_FORCE_HD __host__ __device__
#else
#define GENERATIVEQC_DIRECT_FORCE_HD
#endif

namespace generativeqc::scf::cuda_execution {

/** Output layout for one shared Direct force execution, independent of method.
 * Combined publishes one signed force; Separate publishes contiguous J'/K'
 * force arrays. Coefficients already include the caller's exchange convention.
 * Selecting a layout must never introduce another RHF/UHF or hybrid factor. */
enum class DirectForceOutputMode : std::uint8_t { Combined, Separate };

struct DirectForceSourceCoefficients {
  double coulomb;
  double exchange;
};

/** Compile-time source layout used by shell, resident and precontracted tasks.
 * Every source uses the same screened component/primitive domain. Separate
 * weights remain independent, including when their physical sum cancels. The
 * caller owns count * total_atoms * 3 output doubles and the execution lifetime. */
template <DirectForceOutputMode Mode>
struct DirectForceSources {
  static constexpr unsigned count = Mode == DirectForceOutputMode::Separate ? 2U : 1U;

  GENERATIVEQC_DIRECT_FORCE_HD static constexpr DirectForceSourceCoefficients coefficients(
      unsigned source, double coulomb, double exchange) noexcept {
    if constexpr (Mode == DirectForceOutputMode::Separate) {
      return {source == 0U ? coulomb : 0.0, source == 1U ? exchange : 0.0};
    } else {
      return {coulomb, exchange};
    }
  }

  /** source is in [0, count); each array keeps the native -dE/dR convention. */
  GENERATIVEQC_DIRECT_FORCE_HD static constexpr double* output(
      double* forces, std::size_t total_atoms, unsigned source) noexcept {
    return forces + source * total_atoms * 3U;
  }
};

}  // namespace generativeqc::scf::cuda_execution

#undef GENERATIVEQC_DIRECT_FORCE_HD
