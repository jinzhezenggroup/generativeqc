#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#if __has_include("scf/provider_initialization_generation.hpp")
#include "scf/provider_initialization_generation.hpp"
#if __has_include("methods/provider_initialization_intent.hpp")
#include "methods/provider_initialization_intent.hpp"
#endif

using namespace generativeqc;
using namespace generativeqc::scf;

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class Error, class Function>
void refuses(Function&& function) {
  try {
    function();
  } catch (const Error&) {
    return;
  }
  throw std::runtime_error("generation operation unexpectedly accepted");
}
runtime::CudaProviderIdentityObservation witness() {
  runtime::CudaProviderIdentity identity;
  identity.provider = runtime::CudaProviderKind::Nvidia;
  identity.providers = 3;
  identity.context = {3, 41, {}, 13020, 12090};
  identity.context.device_uuid[7] = 19;
  identity.build_ids = {"aa", "bb", "cc", "dd"};
  runtime::CudaProviderIdentityObservation value;
  value.status = runtime::ContextIdentityStatus::available;
  value.loaded_status = runtime::LoadedIdentityStatus::available;
  value.identity = identity;
  for (auto& pin : value.library_pins) pin = std::make_shared<int>(1);
  return value;
}
ProviderInitializationTeardownReport completed() {
  return {{{"drain", ProviderTeardownStepState::returned, 0},
           {"vendor", ProviderTeardownStepState::returned, 0},
           {"stream", ProviderTeardownStepState::returned, 0}},
          {}};
}

int run(const std::string& mode) {
  int observations = 0;
  auto current = witness();
  const auto observe = [&] {
    ++observations;
    return current;
  };
  PendingProviderInitializationGeneration generation(ProviderInitializationIntent(512ULL << 20),
                                                     {true, true},
                                                     ProviderInitializationEigenFamily::xsyevd, 3);
  if (mode == "domain") {
#if __has_include("methods/provider_initialization_intent.hpp")
    core::ContextState context;
    context.requested_backend = context.executed_backend = core::kCudaBackend;
    core::System system;
    generativeqc_method_descriptor method{};
    method.method = GENERATIVEQC_METHOD_MP2;
    methods::require_provider_initialization_intent_scope(context, system, method);
    method.method = GENERATIVEQC_METHOD_RCCSD;
    methods::require_provider_initialization_intent_scope(context, system, method);
    context.executed_backend = core::kCpuReferenceBackend;
    refuses<methods::MethodError>(
        [&] { methods::require_provider_initialization_intent_scope(context, system, method); });
    context.executed_backend = core::kCudaBackend;
    system.multiplicity = 2;
    refuses<methods::MethodError>(
        [&] { methods::require_provider_initialization_intent_scope(context, system, method); });
    system.multiplicity = 1;
    method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_AUTO;
    refuses<methods::MethodError>(
        [&] { methods::require_provider_initialization_intent_scope(context, system, method); });
    method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    method.method = GENERATIVEQC_METHOD_RCCSD_T;
    refuses<methods::MethodError>(
        [&] { methods::require_provider_initialization_intent_scope(context, system, method); });
#else
    std::cerr << "native intent method scope is unavailable\n";
    return 7;
#endif
  } else if (mode == "bind") {
    require(generation.declared_allowance() == 512ULL << 20, "declared reservation");
    require(observations == 0, "intent queried metadata before owner binding");
    generation.bind_current(observe);
    require(generation.has_bound_identity() && !generation.retired(), "actual binding absent");
    require(generation.resolved().identity->runtime.context.context_id == 41, "not actual key");
    require(generation.resolved().retained_capacity(9) == 512ULL << 20,
            "observed delta replaced selected reservation");
    generation.require_reuse(observe);
    require(!generation.retired(), "same actual owner was refused");
  } else if (mode == "legacy") {
    PendingProviderInitializationGeneration legacy({}, {true, true}, {}, -1);
    require(legacy.declared_allowance() == 192ULL << 20, "legacy allowance changed");
    legacy.bind_current(observe);
    legacy.require_reuse(observe);
    require(observations == 0 && !legacy.has_bound_identity(), "legacy metadata observation");
  } else if (mode == "context") {
    generation.bind_current(observe);
    current.identity->context.context_id = 42;
    refuses<ProviderInitializationUnavailable>([&] { generation.require_reuse(observe); });
    require(generation.retired(), "changed context did not retire replay");
    require(generation.resolved().identity->runtime.context.context_id == 41,
            "changed context overwrote the original owner key");
    int teardowns = 0;
    const auto report = generation.retire_current(observe, [&] {
      ++teardowns;
      return completed();
    });
    require(!report.confirmed() && report.steps.empty() && teardowns == 0,
            "foreign context was cleaned or reported successful");
    current.identity->context.context_id = 41;
    refuses<ProviderInitializationUnavailable>([&] { generation.require_reuse(observe); });
  } else if (mode == "loaded-owner") {
    generation.bind_current(observe);
    current.library_pins[1] = std::make_shared<int>(2);
    refuses<ProviderInitializationUnavailable>([&] { generation.require_reuse(observe); });
    require(generation.retired(), "same build ID permitted a different loaded owner");
  } else if (mode == "ledger") {
    auto previous = runtime::active_device_resource_ledger;
    auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
    ledger->device = 3;
    ledger->limit = 1024;
    runtime::active_device_resource_ledger = ledger;
    refuses<ProviderInitializationUnavailable>([&] { generation.bind_current(observe); });
    runtime::active_device_resource_ledger = previous;
    require(observations == 0 && !generation.has_bound_identity(),
            "unknown basis reached observer");
    require(ledger->live == 0 && ledger->provider_reserved == 0, "refusal charged ledger");
  } else if (mode == "query-error") {
    current.status = runtime::ContextIdentityStatus::query_error;
    current.error_domain = runtime::IdentityErrorDomain::driver;
    current.error_status = 715;
    try {
      generation.bind_current(observe);
    } catch (const ProviderInitializationObservationFailure& error) {
      require(error.domain() == runtime::IdentityErrorDomain::driver && error.status() == 715,
              "raw observation failure changed");
      require(!generation.has_bound_identity(), "failed first binding was published");
      return 0;
    }
    throw std::runtime_error("failed observation was accepted");
  } else if (mode == "retire-query-error") {
    generation.bind_current(observe);
    current.status = runtime::ContextIdentityStatus::query_error;
    current.error_domain = runtime::IdentityErrorDomain::cuda;
    current.error_status = 700;
    bool attempted = false, caught = false;
    try {
      (void)generation.retire_current(observe, [&] {
        attempted = true;
        return completed();
      });
    } catch (const ProviderInitializationObservationFailure& error) {
      caught = error.domain() == runtime::IdentityErrorDomain::cuda && error.status() == 700;
    }
    require(caught, "retirement observation error was swallowed");
    require(generation.retired() && !attempted, "unknown context reached teardown");
  } else if (mode == "retirement") {
    std::array<std::weak_ptr<void>, 4> weak;
    for (unsigned i = 0; i < 4; ++i) weak[i] = current.library_pins[i];
    generation.bind_current(observe);
    current.library_pins = {};
    const auto original_owner = [&] {
      auto value = current;
      for (unsigned i = 0; i < 4; ++i) value.library_pins[i] = weak[i].lock();
      return value;
    };
    auto failed = generation.retire_current(original_owner, [&] {
      require(!weak[1].expired(), "library lease released before teardown");
      return ProviderInitializationTeardownReport{
          {{"drain", ProviderTeardownStepState::returned, 700},
           {"vendor", ProviderTeardownStepState::unavailable, {}}},
          {}};
    });
    require(generation.retired() && !failed.confirmed() && failed.steps[0].status == 700 &&
                !failed.steps[1].status && !weak[1].expired(),
            "failed teardown was repaired");
    auto result = generation.retire_current(original_owner, [&] {
      require(!weak[1].expired(), "library lease released during teardown");
      return completed();
    });
    require(result.reported_steps_succeeded() && !result.confirmed() && !weak[1].expired(),
            "raw successful steps falsely released whole-owner library leases");
    refuses<ProviderInitializationUnavailable>([&] { generation.require_reuse(original_owner); });
  } else {
    throw std::runtime_error("unknown generation probe mode");
  }
  return 0;
}
}  // namespace
#endif

int main(int argc, char** argv) {
#if __has_include("scf/provider_initialization_generation.hpp")
  try {
    if (argc != 2) throw std::runtime_error("one generation probe mode required");
    return run(argv[1]);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
#else
  (void)argc;
  (void)argv;
  std::cerr << "native pending generation contract is unavailable\n";
  return 7;
#endif
}
