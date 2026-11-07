#pragma once

#include "scf/cuda/rhf_policy.hpp"

namespace generativeqc::scf::cuda_policy {

/** Actual owner facts for bounded HF scratch, independent of proposal history.
 * Imported and warm owners may retain the same exact page; this policy never
 * selects an approximate operator or authorizes a screened proposal. */
struct HfBoundedPageDomain {
  std::size_t nbf{};
  std::size_t systems{};
  unsigned maximum_shell_angular{};
  unsigned maximum_iterations{};
  bool unrestricted{};
  bool physical_reference{};
  bool mixed_precision{};
  bool incremental_jk{};
};

/** Resolve an already-admitted bounded HF owner's reusable task-page limits.
 * Automatic expansion is restricted to qualified large FP64 RHF consumers.
 * Explicit diagnostic selection keeps its existing semantics; allocation
 * refusal must still retry the incumbent page before upload or graph capture. */
DirectJkSchedulePolicy resolve_hf_bounded_page_policy(const runtime::CudaTargetInfo& target,
                                                      const HfBoundedPageDomain& domain) noexcept;

}  // namespace generativeqc::scf::cuda_policy
