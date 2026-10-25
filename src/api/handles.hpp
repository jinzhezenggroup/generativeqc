#ifndef GENERATIVEQC_API_HANDLES_HPP
#define GENERATIVEQC_API_HANDLES_HPP

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "methods/method.hpp"
#include "scf/types.hpp"

struct generativeqc_context {
  mutable std::recursive_mutex mutex;
  generativeqc::core::ContextState state;
  // Borrowed by the public error getter: mutate only when recording a failure.
  std::string last_detail;
};

struct generativeqc_system {
  generativeqc::core::System data;
  /** A source-only fit handle is not an orbital or a generic DF-HF export.
   * Full correlated RI-JK admits it through the bounded source/gradient owner. */
  bool correlation_auxiliary{false};
  /** Values-only KS DF source handle, never an orbital or generic DF-HF export.
   * Second-order consumers must independently authenticate their model/closure. */
  bool ks_auxiliary{false};
  /** Complete CUDA HF/KS fitting values and first nuclear derivatives.
   * The auxiliary-g sink owns six-term expansions; orbital admission is separate. */
  bool scf_auxiliary{false};
};

namespace generativeqc::api {

inline bool valid_system_roles(const generativeqc_system& orbital,
                               const generativeqc_method_descriptor& descriptor,
                               core::Backend backend) {
  if (orbital.correlation_auxiliary || orbital.ks_auxiliary || orbital.scf_auxiliary) return false;
  const auto* auxiliary = descriptor.density_fitting_auxiliary_basis;
  // AUTO follows the prepared context, including typed CUDA-only auxiliary
  // owners. Do not let an AUTO request reuse those owners on the CPU backend.
  const bool cuda_df = backend == core::kCudaBackend &&
                       (descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_CUDA ||
                        descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_AUTO);
  if (auxiliary && auxiliary->scf_auxiliary) {
    const auto* capabilities = methods::find_capabilities(descriptor.method);
    return capabilities &&
           (capabilities->family == GENERATIVEQC_METHOD_FAMILY_HARTREE_FOCK ||
            (capabilities->family == GENERATIVEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
             descriptor.ks_options)) &&
           cuda_df;
  }
  if (auxiliary && auxiliary->ks_auxiliary) {
    const auto* capabilities = methods::find_capabilities(descriptor.method);
    return capabilities && capabilities->family == GENERATIVEQC_METHOD_FAMILY_DENSITY_FUNCTIONAL &&
           descriptor.ks_options && cuda_df;
  }
  if (!auxiliary || !auxiliary->correlation_auxiliary) return true;
  return descriptor.method == GENERATIVEQC_METHOD_DF_RCCSD_T ||
         (descriptor.method == GENERATIVEQC_METHOD_MP2 &&
          (descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_CORRELATION_CUDA ||
           descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_FULL_CUDA));
}

}  // namespace generativeqc::api

struct generativeqc_calculation {
  generativeqc_context* context{};
  std::unique_ptr<generativeqc::methods::PreparedCalculation> plan;
  /** How the precision policy resolved for the most recent successful run. */
  generativeqc::scf::PrecisionProvenance precision{};
  /** True only after a run completes and populates \p precision. */
  bool precision_available{false};
  generativeqc::scf::IncrementalDirectJkDiagnostic incremental_direct_jk{};
  /** Versioned detailed record for the same completed run. */
  std::optional<generativeqc::scf::PrecisionWork> precision_work;
  /** Completed-run SCF measures; cleared before a new backend execution. */
  std::optional<generativeqc_scf_diagnostic> scf_diagnostic;
  std::optional<generativeqc::dft::ScfDiagnostic> ks_diagnostic;
  std::optional<generativeqc::scf::initial_guess::PreliminaryDiagnostic> initial_guess;
  /** Same-transaction MP2 companion; absent on legacy/rejected/failed replay. */
  std::optional<generativeqc_correlation_diagnostic> mp2_component_diagnostic;
};

struct generativeqc_batch {
  generativeqc_context* context{};
  std::unique_ptr<generativeqc::methods::PreparedBatch> plan;
  generativeqc_batch_flags flags{};
  std::vector<std::uint32_t> atom_counts;
  std::vector<std::uint64_t> last_fock_builds;
  /** Input-ordered completed-run records; invalid/throwing items stay unavailable. */
  std::vector<std::optional<generativeqc::scf::PrecisionProvenance>> precision;
  std::vector<std::optional<generativeqc::scf::IncrementalDirectJkDiagnostic>>
      incremental_direct_jk;
  std::vector<std::optional<generativeqc::scf::PrecisionWork>> precision_work;
  /** Separate from the fixed-stride legacy batch output array. */
  std::vector<std::optional<generativeqc_scf_diagnostic>> scf_diagnostics;
  std::vector<std::optional<generativeqc::dft::ScfDiagnostic>> ks_diagnostics;
  std::vector<std::optional<generativeqc::scf::initial_guess::PreliminaryDiagnostic>>
      initial_guesses;
  std::vector<std::optional<generativeqc::methods::DispersionResult>> dispersion;
};

#endif
