#pragma once

#include <cstddef>
#include <optional>

namespace generativeqc::cc {

/** Prepared compiler-region layouts, never an observation-driven policy.
 * Each admitted region retains its Q extent. Only preparation populates or
 * drops a layout; actions consume its already-bound Tensor descriptors.
 * Reporting is one-way so diagnostic edits cannot alter scientific execution.
 */
struct DFPreparedRegions {
  bool reduced{}, retained_core{}, mixed{};
  std::optional<std::size_t> response, audit, primal;
  std::size_t provider_allowance{};
  std::size_t batch() const noexcept { return response.value_or(1); }

  template <class Diagnostic>
  void report(Diagnostic& snapshot) const {
    snapshot.df_auxiliary_reduction = reduced;
    snapshot.df_matrix_gemm = response.has_value();
    snapshot.df_audit_matrix_gemm = audit.has_value();
    snapshot.df_primal_matrix_gemm = primal.has_value();
    snapshot.df_auxiliary_batch_size = batch();
    snapshot.df_core_reuse = retained_core;
    snapshot.df_mixed_precision = mixed;
    snapshot.df_provider_allowance_bytes = provider_allowance;
  }
};

}  // namespace generativeqc::cc
