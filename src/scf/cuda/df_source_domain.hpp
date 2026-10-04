#pragma once

#include <string>

#include "core/types.hpp"

namespace generativeqc::scf::cuda_execution {

/** Diagnostic choices frozen before expensive reference work. Source construction
 * consumes this snapshot rather than reading a possibly changed environment.
 */
struct CudaDfSourcePolicy {
  unsigned requested_value_mapping{};
  unsigned value_math{};
};

/** Host-only policy resolution; leaves policy unchanged on failure. */
bool resolve_cuda_df_source_policy(CudaDfSourcePolicy& policy, std::string& detail);

/** Host-only capability preflight shared by source factories and their callers.
 * This query creates no CUDA context or numerical state.
 */
bool cuda_df_shell_domain(const core::System& system, const char* role, std::string& detail);

/** Source value domain: orbital through f, auxiliary through g. Auxiliary g
 * requires generic generated value math. Raw source derivatives also have
 * explicit auxiliary-g polynomial lowering. Legacy exporters and method
 * force-capability queries retain their own stricter admission; creating this
 * source does not enable a molecular force.
 */
bool cuda_df_value_domain(const core::System& orbital, const core::System& auxiliary,
                          const CudaDfSourcePolicy& policy, std::string& detail);

}  // namespace generativeqc::scf::cuda_execution
