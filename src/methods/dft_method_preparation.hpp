#pragma once

#include <cstdint>

#include "dft/global_hybrid_fock.hpp"
#include "dft/grid.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "dft/semilocal_family.hpp"
#include "generativeqc/generativeqc.h"
#include "scf/types.hpp"

namespace generativeqc::core {
struct System;
}

namespace generativeqc::dft {
struct SemilocalPointProgram;
struct RestrictedSemilocalPointProgram;
}  // namespace generativeqc::dft

namespace generativeqc::methods::detail {

// Internal KS admission result, not a public method/ABI descriptor. Only
// preparation may construct it; prepared owners consume a frozen copy.
struct NativeKsExecutionPlan {
  std::uint32_t spin_channels{1};
  dft::SemilocalFamily semilocal_family{dft::SemilocalFamily::Lda};
  bool compiler_resolved{};
  bool d4_correction{};
  bool nonlocal_correlation{};
  dft::nlc::Vv10Parameters nonlocal_parameters{};
  std::uint64_t nonlocal_maximum_bytes{};
  bool range_exchange{};
  double short_range_exchange{};
  double long_range_exchange{};
  double range_omega{};
  std::uint32_t functional{};
  bool generated_split_hybrid{};
  const dft::SemilocalPointProgram* automatic_program{};
  const dft::RestrictedSemilocalPointProgram* automatic_restricted_program{};
};

/** Qualify the global-hybrid composition before preparing any scientific owner.
 * Direct AUTO precision is selected per component by CudaKsPlan. Fitted
 * hybrids retain their separate FP64 domain; range/nonlocal compositions use
 * their own admission and cannot enter this full-range-only route.
 */
inline bool cuda_global_hybrid_admitted(generativeqc_backend backend,
                                        const NativeKsExecutionPlan& plan,
                                        const scf::ScfOptions& options,
                                        const scf::FockBuildSpec& fock,
                                        std::uint32_t functional) noexcept {
  return backend == GENERATIVEQC_BACKEND_CUDA && !plan.range_exchange &&
         !plan.nonlocal_correlation &&
         (options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE ||
          options.precision_mode != GENERATIVEQC_PRECISION_AUTO) &&
         dft::global_hybrid_coefficients_match(fock, functional, plan.spin_channels,
                                               options.semilocal_exchange_scale,
                                               options.semilocal_correlation_scale) &&
         (options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE ||
          dft::cuda_global_hybrid_df_admitted(functional));
}

bool ks_is_unrestricted(const NativeKsExecutionPlan& plan) noexcept;
std::uint32_t ks_scf_domain_version(const NativeKsExecutionPlan& plan) noexcept;
std::uint32_t ks_xc_functional_code(const NativeKsExecutionPlan& plan) noexcept;
const char* ks_semilocal_family_name(const NativeKsExecutionPlan& plan) noexcept;

scf::ScfOptions dft_options(const generativeqc_method_descriptor& descriptor,
                            generativeqc_backend backend, NativeKsExecutionPlan& plan);
dft::GridSpec ks_grid_options(const generativeqc_method_descriptor& descriptor,
                              scf::ScfOptions& options, const NativeKsExecutionPlan& plan);
void validate_ks_spin_state(const NativeKsExecutionPlan& plan, const core::System& system);

}  // namespace generativeqc::methods::detail
