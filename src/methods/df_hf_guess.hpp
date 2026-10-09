#pragma once

#include <cstddef>
#include <vector>

#include "core/types.hpp"

namespace generativeqc::methods::detail {
enum class DFHFGuessOutcome {
  Disabled,
  Ineligible,
  CacheSkipped,
  WorkSkipped,
  BudgetSkipped,
  Failed,
  Used
};

/** A provisional density only. No fitted reference, Fock or DIIS state survives. */
struct DFHFGuess {
  std::vector<double> density;
  DFHFGuessOutcome outcome{DFHFGuessOutcome::Disabled};
  unsigned iterations{};
  std::size_t cartesian_functions{};
  std::size_t auxiliary_functions{};
  std::size_t preparation_peak_bytes{};
  std::size_t value_budget_bytes{};
  /** Dense four-center / bounded three-center sweep volume; not a timing estimate. */
  double work_amortization_ratio{};
  double seconds{};
  bool work_counters_complete{true};
};

/** Bounded native JK-fit preparation with compatibility, source-path and work guards.
 * Skip cache-eligible exact references and unamortized dense DF work, not AO intervals.
 * Live correlation metadata and optional response caches are charged before admission.
 * Unsupported topology, a tight budget or a refused guess retains cold Direct RHF.
 * GENERATIVEQC_DF_CCSDT_REFERENCE_GUESS=direct explicitly disables this policy;
 * auto (or an unset variable) enables it. Invalid selectors are errors. */
DFHFGuess prepare_df_hf_guess(const core::System&, const core::System& correlation_auxiliary,
                              const generativeqc_method_descriptor&, std::size_t retained_bytes,
                              int device_id, bool enabled = true);
}  // namespace generativeqc::methods::detail
