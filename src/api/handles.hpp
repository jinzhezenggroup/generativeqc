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
};

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
};

#endif
