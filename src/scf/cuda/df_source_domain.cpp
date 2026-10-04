#include "scf/cuda/df_source_domain.hpp"

#include "scf/cuda/rhf_policy.hpp"

namespace generativeqc::scf::cuda_execution {

bool resolve_cuda_df_source_policy(CudaDfSourcePolicy& policy, std::string& detail) {
  CudaDfSourcePolicy candidate;
  if (!cuda_policy::df_value_math_requested(candidate.value_math)) {
    detail = "GENERATIVEQC_DF_VALUE_MATH must be auto, generic, polynomial, rys or candidate";
    return false;
  }
  candidate.requested_value_mapping = cuda_policy::df_value_mapping_requested();
  policy = candidate;
  return true;
}

bool cuda_df_shell_domain(const core::System& system, const char* role, std::string& detail) {
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 3U) {
      detail = std::string("CUDA DF ") + role + " shells beyond f (l > 3) are unsupported";
      return false;
    }
  }
  return true;
}

bool cuda_df_value_domain(const core::System& orbital, const core::System& auxiliary,
                          const CudaDfSourcePolicy& policy, std::string& detail) {
  if (policy.value_math > 3U) {
    detail = "invalid CUDA DF source value math policy";
    return false;
  }
  if (!cuda_df_shell_domain(orbital, "orbital", detail)) return false;
  for (const auto& shell : auxiliary.shells) {
    if (shell.angular_momentum == 4U && policy.value_math != 0U) {
      detail = "g auxiliary DF values require the generic generated math policy";
      return false;
    }
    if (shell.angular_momentum > 4U) {
      detail = "CUDA DF auxiliary value shells beyond g (l > 4) are unsupported";
      return false;
    }
  }
  return true;
}

}  // namespace generativeqc::scf::cuda_execution
