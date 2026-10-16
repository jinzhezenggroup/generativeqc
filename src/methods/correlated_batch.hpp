#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "methods/method.hpp"
#include "scf/warm_state.hpp"

namespace generativeqc::methods {

/** Optional method-owned reference port, not a new solver or state owner.
 * A candidate seed is published only after the complete correlation endpoint
 * succeeds. Executable transfer never claims that a physical reference is valid. */
class ReferenceSeededCalculation {
 public:
  virtual ~ReferenceSeededCalculation() = default;
  virtual Result execute_seeded_reference(bool compute_forces, const scf::HfWarmState* seed,
                                          bool* fallback,
                                          std::optional<scf::HfWarmState>* candidate) = 0;
  virtual void transfer_reference_execution_to(ReferenceSeededCalculation&) {}
};

using CorrelatedCalculationFactory = std::unique_ptr<PreparedCalculation> (*)(
    const Capabilities&, core::ContextState&, const core::System&,
    const generativeqc_method_descriptor&, const core::System*);

/** One production ragged scheduler for existing conventional MP2/CC owners.
 * Scientific admission, reference/provider binding, force and amplitude work
 * remain with the supplied registered factory and typed calculation port. */
std::unique_ptr<PreparedBatch> prepare_correlated_batch(
    const Capabilities& capabilities, core::ContextState& context,
    std::vector<core::System> systems, const generativeqc_method_descriptor& descriptor,
    bool warm_starts_enabled, CorrelatedCalculationFactory factory, std::string_view name,
    generativeqc_method reference_method, bool indexed_buckets, bool cc_diagnostics);

}  // namespace generativeqc::methods
