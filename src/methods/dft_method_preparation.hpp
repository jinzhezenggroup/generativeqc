#pragma once

#include <cstdint>

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
  /** Frozen at preparation: explicit legacy Direct LR versus operator-matched RI-K. */
  bool density_fitted_range_exchange{};
  double short_range_exchange{};
  double long_range_exchange{};
  double range_omega{};
  std::uint32_t functional{};
  bool generated_split_hybrid{};
  const dft::SemilocalPointProgram* automatic_program{};
  const dft::RestrictedSemilocalPointProgram* automatic_restricted_program{};
};

bool ks_is_unrestricted(const NativeKsExecutionPlan& plan) noexcept;
std::uint32_t ks_scf_domain_version(const NativeKsExecutionPlan& plan) noexcept;
std::uint32_t ks_xc_functional_code(const NativeKsExecutionPlan& plan) noexcept;
const char* ks_semilocal_family_name(const NativeKsExecutionPlan& plan) noexcept;

scf::ScfOptions dft_options(const generativeqc_method_descriptor& descriptor,
                            generativeqc_backend backend, NativeKsExecutionPlan& plan);
dft::GridSpec ks_grid_options(const generativeqc_method_descriptor& descriptor,
                              scf::ScfOptions& options, const NativeKsExecutionPlan& plan);
void validate_ks_spin_state(const NativeKsExecutionPlan& plan, const core::System& system,
                            generativeqc_backend backend);

}  // namespace generativeqc::methods::detail
