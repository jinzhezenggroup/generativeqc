// Exercise the actual C API publisher and resource owner with metadata-only factories.
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "api/device_resources.hpp"
#include "api/handles.hpp"
#include "api/provider_initialization_intent.hpp"
#include "methods/named_ks_catalog.hpp"
#include "methods/provider_initialization_intent.hpp"
#include "methods/provider_initialization_lifetime.hpp"
#include "runtime/resource_ledger.hpp"

namespace {
std::weak_ptr<generativeqc::runtime::DeviceResourceLedger> ledger_witness;
unsigned created{}, destroyed{};
bool destructor_had_live_ledger{};
constexpr std::size_t reservation = 64;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct MetadataPlan : generativeqc::methods::PreparedCalculation,
                      generativeqc::methods::ProviderInitializationCloseable {
  generativeqc::runtime::DeviceProviderAllowance allowance{0, reservation};
  MetadataPlan() {
    ++created;
    require(generativeqc::runtime::active_device_resource_ledger == ledger_witness.lock(),
            "factory did not borrow the selected native owner");
  }
  ~MetadataPlan() override {
    ++destroyed;
    auto ledger = ledger_witness.lock();
    destructor_had_live_ledger = ledger && ledger->provider_reserved == reservation;
  }
  std::size_t atom_count() const noexcept override { return 0; }
  const generativeqc::methods::Capabilities& capabilities() const noexcept override {
    static const generativeqc::methods::Capabilities capabilities{};
    return capabilities;
  }
  generativeqc::methods::Result execute_impl(bool) override {
    throw std::runtime_error("metadata probe must never execute a method");
  }
  generativeqc_status close_provider_initialization_current(std::string&) override {
    return GENERATIVEQC_STATUS_SUCCESS;
  }
};

generativeqc_device_resources* make_resources() {
  const generativeqc_device_resource_buffer_v1 row{4096, 4, 0, 0, 0, 0};
  const generativeqc_device_resource_descriptor descriptor{
      sizeof(descriptor), GENERATIVEQC_ABI_VERSION, &row, 1,
      GENERATIVEQC_DEVICE_RESOURCE_NUMERIC_CAP};
  generativeqc_device_resources* result{};
  require(generativeqc_device_resources_create(&descriptor, &result) == GENERATIVEQC_STATUS_SUCCESS,
          "public metadata owner creation failed");
  ledger_witness = generativeqc::api::device_resource_owner(result);
  return result;
}
}  // namespace

namespace generativeqc::methods {
std::unique_ptr<PreparedCalculation> prepare_calculation(core::ContextState&, const core::System&,
                                                         const generativeqc_method_descriptor&,
                                                         const core::System*,
                                                         const named_ks::Dispersion* dispersion,
                                                         std::uint64_t dispersion_bytes) {
  require(!dispersion && !dispersion_bytes, "legacy metadata factory received named dispersion");
  return std::make_unique<MetadataPlan>();
}
void prepare_calculation_with_provider_initialization_intent_into(
    core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor, const core::System*,
    scf::ProviderInitializationIntent, std::unique_ptr<PreparedCalculation>& destination) {
  require_provider_initialization_intent_scope(context, system, descriptor);
  destination = std::make_unique<MetadataPlan>();
}
const Capabilities* find_capabilities(generativeqc_method) noexcept { return nullptr; }
namespace named_ks {
// This fixture covers legacy/pending publisher ownership; actual named models
// have their own native catalog and SDK qualification tests.
const Method* find_method(std::string_view) noexcept { std::abort(); }
void electronic_options(const Method&, generativeqc_ks_options&) noexcept { std::abort(); }
void validate_bound_basis(const Method&, const core::System&) { std::abort(); }
}  // namespace named_ks
}  // namespace generativeqc::methods

int main(int argc, char** argv) {
  using namespace generativeqc;
  try {
    require(argc == 2, "integration mode required");
    const std::string mode = argv[1];
    generativeqc_context context;
    context.state.requested_backend = context.state.executed_backend = core::kCudaBackend;
    context.state.device_id = mode == "resource-device-refusal" ? 1 : 0;
    generativeqc_system system;
    system.data.multiplicity = 1;
    generativeqc_method_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.method = GENERATIVEQC_METHOD_MP2;
    auto* resources = make_resources();
    generativeqc_calculation* calculation{};
    if (mode == "resource-retention") {
      require(generativeqc_calculation_prepare_with_device_resources(&context, &system, &descriptor,
                                                                     resources, &calculation) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              "shared resource preparation failed");
      require(calculation && calculation->resource_ledger == ledger_witness.lock() &&
                  !runtime::active_device_resource_ledger &&
                  !context.provider_initialization_calculation,
              "shared publisher or native scope ownership changed");
      generativeqc_device_resources_destroy(resources);
      generativeqc_device_resources* reopened{};
      require(generativeqc_calculation_get_device_resources(calculation, &reopened) ==
                  GENERATIVEQC_STATUS_SUCCESS,
              "prepared owner did not survive initial metadata close");
      generativeqc_device_resource_diagnostic diagnostic{};
      diagnostic.struct_size = sizeof(diagnostic);
      diagnostic.abi_version = GENERATIVEQC_ABI_VERSION;
      require(generativeqc_device_resources_read(reopened, &diagnostic) ==
                      GENERATIVEQC_STATUS_SUCCESS &&
                  diagnostic.provider_reserved_bytes == reservation,
              "actual provider allowance was lost after metadata close");
      generativeqc_device_resources_destroy(reopened);
      require(!ledger_witness.expired(), "calculation lost its native owner");
      generativeqc_calculation_destroy(calculation);
      require(destructor_had_live_ledger && ledger_witness.expired(),
              "plan/allowance did not retire before the retained native ledger");
    } else if (mode == "pending-unknown-ledger") {
      {
        runtime::DeviceResourceScope outer(api::device_resource_owner(resources));
        const auto status = api::prepare_calculation_with_provider_initialization_intent(
            &context, &system, &descriptor, scf::ProviderInitializationIntent(512ULL << 20),
            &calculation);
        require(status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED && !calculation &&
                    !context.provider_initialization_calculation && created == 0 &&
                    runtime::active_device_resource_ledger == ledger_witness.lock() &&
                    ledger_witness.lock()->provider_reserved == 0 &&
                    context.last_detail.find("unknown active ledger accounting basis") !=
                        std::string::npos,
                "unknown ledger basis was accepted, reclassified or silently unbound");
      }
      require(!runtime::active_device_resource_ledger, "outer native binding leaked");
      generativeqc_device_resources_destroy(resources);
      require(ledger_witness.expired(), "rejected candidate retained an empty ledger");
    } else if (mode == "resource-device-refusal") {
      require(generativeqc_calculation_prepare_with_device_resources(&context, &system, &descriptor,
                                                                     resources, &calculation) ==
                      GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                  !calculation && !runtime::active_device_resource_ledger,
              "device mismatch reached a factory or leaked a binding");
      generativeqc_named_ks_descriptor named{};
      named.struct_size = sizeof(named);
      named.abi_version = GENERATIVEQC_ABI_VERSION;
      require(generativeqc_calculation_prepare_named_ks_with_device_resources_v1(
                  &context, &system, &descriptor, &named, resources, &calculation) ==
                      GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                  !calculation && !runtime::active_device_resource_ledger && created == 0,
              "named preparation bypassed resource admission or leaked a binding");
      generativeqc_device_resources_destroy(resources);
      require(ledger_witness.expired(), "refused resource metadata was retained");
    } else {
      throw std::runtime_error("unknown integration mode");
    }
    require(api::provider_initialization_intent_capabilities() == 0,
            "optional capability was enabled");
    std::cout << "RECONCILED_PREPARED_RESOURCE_HOST_PASS " << mode << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
