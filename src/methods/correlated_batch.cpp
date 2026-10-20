#include "methods/correlated_batch.hpp"

#include <string>
#include <utility>

#include "methods/application_precision.hpp"
#include "methods/batch_capabilities.hpp"
#include "methods/correlated_warm_reference.hpp"
#include "methods/execution_failure.hpp"
#include "methods/geometry.hpp"

namespace generativeqc::methods {
namespace {

ReferenceSeededCalculation& reference_port(PreparedCalculation& calculation) {
  auto* port = dynamic_cast<ReferenceSeededCalculation*>(&calculation);
  if (!port) throw std::logic_error("correlated calculation has no reference-seed port");
  return *port;
}

class CorrelatedPreparedBatch : public PreparedBatch,
                                public BatchWarmStarts,
                                public BatchCorrelationDiagnostics {
 public:
  CorrelatedPreparedBatch(const Capabilities& capabilities, core::ContextState& context,
                          std::vector<core::System> systems,
                          const generativeqc_method_descriptor& descriptor, bool warm_enabled,
                          CorrelatedCalculationFactory factory, std::string_view name,
                          generativeqc_method reference_method, bool indexed_buckets)
      : PreparedBatch(context.requested_backend, context.device_id),
        capabilities_(capabilities),
        context_(context),
        systems_(std::move(systems)),
        descriptor_(descriptor),
        warm_enabled_(warm_enabled),
        factory_(factory),
        name_(name),
        reference_method_(reference_method),
        indexed_buckets_(indexed_buckets),
        warm_states_(systems_.size()) {
    if (!factory_ || systems_.empty())
      throw std::invalid_argument("correlated batch requires a factory and systems");
    descriptor_.struct_size = sizeof(descriptor_);
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.ks_options = nullptr;
    owners_.reserve(size());
    owner_coordinates_.reserve(size());
    for (const auto& system : systems_) {
      owners_.push_back(make_owner(system));
      owner_coordinates_.push_back(geometry::coordinates(system));
    }
  }

  std::size_t size() const noexcept override { return systems_.size(); }

  void invalidate_result() override {
    for (auto& owner : owners_) owner->invalidate_result();
  }

  std::vector<BatchItemResult> execute_impl(const Coordinates& coordinates,
                                            bool compute_forces) override {
    if (!coordinates.empty() && coordinates.size() != size())
      throw std::invalid_argument(name_ + " batch coordinates do not match system count");
    std::vector<BatchItemResult> results(size());
    for (std::size_t index = 0; index < size(); ++index) {
      auto& result = results[index];
      try {
        auto target =
            geometry::bind_batch_item(systems_[index], coordinates, index, result,
                                      context_.requested_backend, indexed_buckets_ ? index : 0);
        auto target_coordinates = geometry::coordinates(target);
        if (target_coordinates != owner_coordinates_[index]) {
          auto candidate = make_owner(target);
          reference_port(*owners_[index])
              .transfer_reference_execution_to(reference_port(*candidate));
          owners_[index] = std::move(candidate);
          owner_coordinates_[index].swap(target_coordinates);
        }
        const auto* seed = warm_enabled_ && warm_states_[index] ? &*warm_states_[index] : nullptr;
        result.warm_start_used = seed != nullptr;
        std::optional<scf::HfWarmState> candidate;
        result.calculation =
            reference_port(*owners_[index])
                .execute_seeded_reference(compute_forces, seed, &result.warm_start_fallback,
                                          warm_enabled_ && warm_updates_ ? &candidate : nullptr);
        result.status = result.calculation.convergence.converged
                            ? GENERATIVEQC_STATUS_SUCCESS
                            : GENERATIVEQC_STATUS_NOT_CONVERGED;
        if (result.status == GENERATIVEQC_STATUS_SUCCESS && candidate)
          warm_states_[index].swap(candidate);
      } catch (...) {
        result.status = item_exception_status();
      }
    }
    return results;
  }

  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic(
      std::size_t index) const override {
    return owner(index).correlation_diagnostic();
  }

  runtime::ExecutionResourceSnapshot execution_resources(
      std::size_t index) const noexcept override {
    return index < owners_.size() ? owners_[index]->execution_resources()
                                  : runtime::ExecutionResourceSnapshot{};
  }

  void clear_warm_starts() override {
    for (auto& state : warm_states_) state.reset();
  }

  std::size_t warm_density_size(std::size_t index) const override {
    const auto n = molecule::ao_count(systems_.at(index));
    if (!n) throw std::invalid_argument(name_ + " warm density has no orbitals");
    const auto spins = reference_method_ == GENERATIVEQC_METHOD_UHF ? 2U : 1U;
    return posthf::checked_mul(spins, posthf::checked_mul(n, n));
  }

  const std::optional<scf::HfWarmState>& warm_state(std::size_t index) const override {
    return warm_states_.at(index);
  }

  void restore_warm_states(std::vector<std::optional<scf::HfWarmState>> states) override {
    if (!warm_enabled_ || states.size() != size())
      throw std::invalid_argument("checkpoint restore requires a matching warm-enabled " + name_ +
                                  " batch");
    for (std::size_t index = 0; index < size(); ++index)
      if (states[index])
        warm_reference::validate_checkpoint(systems_[index], *states[index], name_,
                                            reference_method_);
    for (std::size_t index = 0; index < size(); ++index)
      if (states[index]) warm_states_[index].swap(states[index]);
  }

  void set_warm_start_updates(bool enabled) override { warm_updates_ = enabled; }

 protected:
  const PreparedCalculation& owner(std::size_t index) const {
    if (index >= owners_.size())
      throw std::invalid_argument("correlation diagnostic batch index is out of range");
    return *owners_[index];
  }

 private:
  std::unique_ptr<PreparedCalculation> make_owner(const core::System& system) const {
    auto owner = factory_(capabilities_, context_, system, descriptor_, nullptr);
    if (!owner) throw std::logic_error("correlated factory returned no calculation");
    reference_port(*owner);
    return owner;
  }

  Capabilities capabilities_;
  core::ContextState& context_;
  std::vector<core::System> systems_;
  generativeqc_method_descriptor descriptor_;
  bool warm_enabled_{};
  bool warm_updates_{true};
  CorrelatedCalculationFactory factory_;
  std::string name_;
  generativeqc_method reference_method_;
  bool indexed_buckets_{};
  std::vector<std::optional<scf::HfWarmState>> warm_states_;
  std::vector<std::unique_ptr<PreparedCalculation>> owners_;
  std::vector<std::vector<double>> owner_coordinates_;
};

class CcPreparedBatch final : public CorrelatedPreparedBatch, public BatchCcPerformanceDiagnostics {
 public:
  using CorrelatedPreparedBatch::CorrelatedPreparedBatch;
  std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic(
      std::size_t index) const override {
    return owner(index).cc_performance_diagnostic();
  }
};

}  // namespace

std::unique_ptr<PreparedBatch> prepare_correlated_batch(
    const Capabilities& capabilities, core::ContextState& context,
    std::vector<core::System> systems, const generativeqc_method_descriptor& descriptor,
    bool warm_starts_enabled, CorrelatedCalculationFactory factory, std::string_view name,
    generativeqc_method reference_method, bool indexed_buckets, bool cc_diagnostics) {
  // Batch owners currently copy only the legacy descriptor prefix. Reject an
  // allowance before copying rather than losing it or reading past that copy.
  if (detail::application_precision_budget(descriptor) != std::array<double, 3>{})
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "correlated convergence budgets require single-point groups");
  if (cc_diagnostics)
    return std::make_unique<CcPreparedBatch>(capabilities, context, std::move(systems), descriptor,
                                             warm_starts_enabled, factory, name, reference_method,
                                             indexed_buckets);
  return std::make_unique<CorrelatedPreparedBatch>(capabilities, context, std::move(systems),
                                                   descriptor, warm_starts_enabled, factory, name,
                                                   reference_method, indexed_buckets);
}

}  // namespace generativeqc::methods
