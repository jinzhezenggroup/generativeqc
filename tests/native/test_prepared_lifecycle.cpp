#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <utility>

#include "methods/batch_capabilities.hpp"
#include "methods/method.hpp"
#include "molecule/basis.hpp"

// Inject only inside reset. Ordinary numerical tests keep allocation enabled.
// The sweep does not assume a particular std::string small-buffer capacity.
namespace {
thread_local int reset_allocations_before_failure = -1;
}

void* operator new(std::size_t bytes) {
  if (reset_allocations_before_failure >= 0 && reset_allocations_before_failure-- == 0)
    throw std::bad_alloc();
  if (void* pointer = std::malloc(bytes ? bytes : 1)) return pointer;
  throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Operation>
void rejected(Operation operation) {
  bool failed = false;
  try {
    operation();
  } catch (const std::logic_error&) {
    failed = true;
  }
  require(failed, "illegal lifecycle transition was accepted");
}

void transitions() {
  using namespace generativeqc::runtime;
  CompiledExecutionRegion region;
  require(region.state() == CompiledExecutionState::Unbound, "initial state");
  rejected([&] { region.begin(); });
  CompiledExecutionBinding binding{"qualified", 0, nullptr, nullptr, nullptr, 0};
  region.bind(binding);
  region.begin();
  require(region.state() == CompiledExecutionState::Running, "running state");
  rejected([&] { region.begin(); });
  rejected([&] { region.bind(binding); });
  rejected([&] { region.recover(); });
  rejected([&] { region.close(); });
  require(!region.invalidate(), "running invalidation must be refused");
  region.mark_success();
  region.begin();
  region.abort();
  require(region.failed() && !region.warmed(), "failed output remained warmed");
  require(!region.bind(binding), "compatible bind must not erase sticky failure");
  rejected([&] { region.begin(); });
  rejected([&] { region.mark_success(); });
  region.recover();
  region.begin();
  region.mark_success();
  binding.backend = 1;
  require(region.bind(binding), "changing target must rebind native state");
  require(!region.warmed(), "target rebind retained old replay eligibility");
  region.close();
  region.close();
  rejected([&] { region.begin(); });
  rejected([&] { region.bind(binding); });
  rejected([&] { region.recover(); });
}

void cpu_consumers() {
  using namespace generativeqc;
  core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "H2 setup");
  core::ContextState context;
  for (const auto method : {GENERATIVEQC_METHOD_RHF, GENERATIVEQC_METHOD_PBE_RKS}) {
    generativeqc_method_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.method = method;
    descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
    descriptor.max_iterations = 200;
    descriptor.energy_tolerance = 1e-12;
    descriptor.density_tolerance = 1e-10;
    auto owner = methods::prepare_calculation(context, system, descriptor);
    require(owner->execution_lifecycle().bound(), "real owner was not bound");
    const auto cold = owner->execute(false);
    const auto replay = owner->execute(false);
    require(cold.convergence.converged && replay.convergence.converged &&
                std::abs(cold.energy - replay.energy) < 1e-10,
            "real CPU replay changed the result");
    owner->reject_execution();
    require(owner->execution_lifecycle().failed(), "publication failure not retained");
    const auto recovery = owner->execute(false);
    require(recovery.convergence.converged && std::abs(cold.energy - recovery.energy) < 1e-10,
            "real CPU recovery changed the result");
    // An allocation failure before or during reset must leave the original
    // identity usable. In particular, do not copy it after invalidating it.
    for (int fail_after = 0; fail_after != 8; ++fail_after) {
      reset_allocations_before_failure = fail_after;
      try {
        owner->reset_execution();
      } catch (const std::bad_alloc&) {
      }
      reset_allocations_before_failure = -1;
      require(owner->execution_lifecycle().bound(), "reset allocation failure lost owner binding");
      const auto recovered = owner->execute(false);
      require(recovered.convergence.converged && std::abs(cold.energy - recovered.energy) < 1e-10,
              "allocation-failed reset prevented real CPU recovery");
    }
    owner->reset_execution();
    require(!owner->execution_lifecycle().warmed(), "reset retained warm eligibility");
    owner->close_execution();
    rejected([&] { owner->execute(false); });
    rejected([&] { owner->reset_execution(); });
    auto fleet = methods::prepare_batch(context, {system}, descriptor,
                                        GENERATIVEQC_BATCH_ENABLE_WARM_STARTS);
    require(fleet->execute({}, false).front().status == GENERATIVEQC_STATUS_SUCCESS,
            "real CPU fleet setup");
    for (int fail_after = 0; fail_after != 8; ++fail_after) {
      reset_allocations_before_failure = fail_after;
      try {
        fleet->reset_execution();
      } catch (const std::bad_alloc&) {
      }
      reset_allocations_before_failure = -1;
      require(fleet->execution_lifecycle().bound(), "reset allocation failure lost fleet binding");
      const auto recovered = fleet->execute({}, false);
      require(recovered.front().status == GENERATIVEQC_STATUS_SUCCESS &&
                  std::abs(cold.energy - recovered.front().calculation.energy) < 1e-10,
              "allocation-failed reset prevented real CPU fleet recovery");
    }
  }
}

void optional_batch_capabilities() {
  using namespace generativeqc::methods;
  // A new batch method only implements the execution contract; it need not
  // carry any CUDA or warm-start diagnostics to be instantiated and reset.
  class PlainBatch final : public PreparedBatch {
   public:
    std::size_t size() const noexcept override { return 1; }

   protected:
    std::vector<BatchItemResult> execute_impl(const Coordinates&, bool) override {
      return std::vector<BatchItemResult>(1);
    }
  };
  PlainBatch plain;
  PreparedBatch* erased = &plain;
  require(dynamic_cast<BatchWarmStarts*>(erased) == nullptr, "seed capability is mandatory");
  require(dynamic_cast<BatchDirectQueueProfiles*>(erased) == nullptr,
          "direct-profile capability is mandatory");
  require(dynamic_cast<BatchEigensolverProfiles*>(erased) == nullptr,
          "eigensolver-profile capability is mandatory");
  require(dynamic_cast<BatchDfMetricDiagnostics*>(erased) == nullptr,
          "DF metric capability is mandatory");
  require(dynamic_cast<BatchKsTransportDiagnostics*>(erased) == nullptr,
          "KS transport capability is mandatory");
  require(dynamic_cast<BatchCorrelationDiagnostics*>(erased) == nullptr,
          "correlation capability is mandatory");
  require(dynamic_cast<BatchCcPerformanceDiagnostics*>(erased) == nullptr,
          "CC performance capability is mandatory");
  require(plain.execute({}, false).size() == 1, "plain batch execution failed");
  for (const auto selection : {std::vector<std::uint8_t>{}, std::vector<std::uint8_t>{2}}) {
    bool invalid = false;
    try {
      plain.execute_selected_forces({}, selection);
    } catch (const std::invalid_argument&) {
      invalid = true;
    }
    require(invalid && plain.execution_lifecycle().failed(),
            "malformed force selection did not fail the shared lifecycle");
    require(plain.execute({}, false).size() == 1, "force selection failure prevented recovery");
  }
  bool unsupported = false;
  try {
    plain.execute_selected_forces({}, std::array<std::uint8_t, 1>{1});
  } catch (const MethodError& error) {
    unsupported = error.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
  }
  require(unsupported && plain.execution_lifecycle().failed(),
          "optional per-item execution was silently emulated by an unsupported owner");
  require(plain.execute({}, false).size() == 1, "optional execution rejection prevented recovery");
  plain.reset_execution();
  require(!plain.execution_lifecycle().warmed(), "plain reset retained replay eligibility");

  class WarmBatch final : public PreparedBatch, public BatchWarmStarts {
   public:
    std::size_t size() const noexcept override { return 1; }
    void clear_warm_starts() override {
      ++clears;
      state.reset();
    }
    std::size_t warm_density_size(std::size_t) const override { return 0; }
    const std::optional<generativeqc::scf::HfWarmState>& warm_state(std::size_t) const override {
      return state;
    }
    void restore_warm_states(
        std::vector<std::optional<generativeqc::scf::HfWarmState>> values) override {
      if (values.size() != 1) throw std::invalid_argument("seed fleet size");
      state = std::move(values[0]);
    }
    void set_warm_start_updates(bool enabled) override { updates = enabled; }
    int clears{};
    bool updates{true};
    std::optional<generativeqc::scf::HfWarmState> state;

   protected:
    std::vector<BatchItemResult> execute_impl(const Coordinates&, bool) override {
      return std::vector<BatchItemResult>(1);
    }
  };
  WarmBatch warm;
  require(dynamic_cast<BatchWarmStarts*>(&warm) != nullptr, "missing seed capability");
  warm.set_warm_start_updates(false);
  require(!warm.updates, "seed update control was ignored");
  warm.restore_warm_states({generativeqc::scf::HfWarmState{}});
  require(warm.warm_state(0).has_value(), "seed was not retained");
  warm.reset_execution();
  require(warm.clears == 1 && !warm.warm_state(0), "reset did not clear opt-in seeds");
}
}  // namespace

int main() {
  try {
    transitions();
    optional_batch_capabilities();
    cpu_consumers();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
