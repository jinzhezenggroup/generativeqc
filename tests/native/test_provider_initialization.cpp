#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "runtime/resource_ledger.hpp"
#include "scf/cuda/provider_initialization.cuh"
#include "scf/cuda/reference_export.cuh"
#include "scf/provider_initialization.hpp"

namespace {
using namespace generativeqc::scf;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

ProviderInitializationIdentity fixture_identity() {
  generativeqc::runtime::CudaProviderIdentity runtime;
  runtime.provider = generativeqc::runtime::CudaProviderKind::Nvidia;
  runtime.context.device = 3;
  runtime.context.context_id = 17;
  runtime.context.device_uuid[7] = 19;
  runtime.context.runtime_version = 12090;
  runtime.context.driver_version = 13020;
  runtime.providers = 3;
  runtime.build_ids = {"0102", "0304", "0506", "0708"};
  return {runtime, ProviderInitializationEigenFamily::xsyevd, 1};
}

ProviderInitializationRequest fixture_request() {
  return {ProviderInitializationMode::explicit_qualified, 128, fixture_identity()};
}

void legacy() {
  for (bool blas : {false, true})
    for (bool solver : {false, true}) {
      const auto policy = resolve_reference_provider_initialization({}, {blas, solver}, {});
      const auto expected = (std::size_t(blas) + std::size_t(solver)) * (96ULL << 20);
      require(policy.allowance_bytes == expected, "legacy provider allowance changed");
      require(policy.retained_capacity(19) == 19, "legacy retained accounting changed");
    }
}

void unknown_ledger_basis() {
  auto previous = generativeqc::runtime::active_device_resource_ledger;
  auto ledger = std::make_shared<generativeqc::runtime::DeviceResourceLedger>();
  ledger->limit = 1024;
  ledger->device = 3;
  generativeqc::runtime::active_device_resource_ledger = ledger;
  bool refused = false;
  try {
    preflight_reference_provider_initialization(fixture_request(), true);
  } catch (const ProviderInitializationUnavailable& error) {
    refused = std::string(error.what()).find("ledger") != std::string::npos;
  }
  generativeqc::runtime::active_device_resource_ledger = previous;
  require(refused, "custom provider allowance accepted an active ledger with unknown basis");
  require(ledger->provider_reserved == 0 && ledger->live == 0,
          "rejected override changed numeric or provider ledger state");
  bool wrong_route = false;
  try {
    preflight_reference_provider_initialization(fixture_request(), false);
  } catch (const std::invalid_argument&) {
    wrong_route = true;
  }
  require(wrong_route, "explicit request reached an ordinary route's provider probe");
}

void scalar_identity_and_capacity() {
  const auto request = fixture_request();
  const auto resolved =
      resolve_reference_provider_initialization(request, {true, true}, fixture_identity());
  require(resolved.allowance_bytes == 128 && resolved.retained_capacity(16) == 128,
          "scalar override did not retain its selected reservation exactly once");
  const auto base =
      generativeqc::scf::reference_detail::check_capacity(768, resolved.allowance_bytes, 1024);
  require(generativeqc::scf::reference_detail::check_capacity(base, 128, 1024) == 1024,
          "workspace/allowance exact budget boundary changed");
  bool short_budget = false;
  try {
    (void)generativeqc::scf::reference_detail::check_capacity(base, 128, 1023);
  } catch (const std::length_error&) {
    short_budget = true;
  }
  require(short_budget, "one byte short whole budget was accepted");
  bool overflow = false;
  try {
    (void)generativeqc::scf::reference_detail::check_capacity(static_cast<std::size_t>(INT64_MAX),
                                                              resolved.allowance_bytes,
                                                              static_cast<std::size_t>(INT64_MAX));
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  require(overflow, "allowance capacity overflow was hidden");
  bool unavailable = false;
  try {
    (void)resolve_reference_provider_initialization(request, {true, true}, {});
  } catch (const ProviderInitializationUnavailable&) {
    unavailable = true;
  }
  require(unavailable, "request identity was accepted without a native observation");
  auto mismatched = fixture_identity();
  mismatched.runtime.context.context_id++;
  bool mismatch = false;
  try {
    (void)resolve_reference_provider_initialization(request, {true, true}, mismatched);
  } catch (const std::invalid_argument&) {
    mismatch = true;
  }
  require(mismatch, "another context generation reused the qualified request");
  for (auto cap : {std::size_t(0), std::numeric_limits<std::size_t>::max()}) {
    auto invalid = request;
    invalid.allowance_bytes = cap;
    bool rejected = false;
    try {
      (void)resolve_reference_provider_initialization(invalid, {true, true}, fixture_identity());
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "invalid allowance accepted or normalized");
  }
  ScfOptions original;
  original.reference_provider_initialization = request;
  const auto captured = original;
  original.reference_provider_initialization->expected_identity->runtime.context.context_id++;
  require(!same_provider_initialization_request(original.reference_provider_initialization,
                                                captured.reference_provider_initialization),
          "prepared request borrowed mutable caller identity");
  auto changed = request;
  changed.allowance_bytes--;
  require(!same_provider_initialization_request(changed, request),
          "changed selected allowance reused the old prepared request");
  require(same_provider_initialization_request({}, ProviderInitializationRequest{}),
          "legacy absent/present identity changed");
}

void native_observation_adapter() {
  namespace runtime = generativeqc::runtime;
  auto request = fixture_request();
  runtime::CudaProviderIdentityObservation observation;
  observation.status = runtime::ContextIdentityStatus::available;
  observation.identity = fixture_identity().runtime;
  for (auto& pin : observation.library_pins)
    pin = std::shared_ptr<void>(reinterpret_cast<void*>(1), [](void*) {});
  const auto bound = bind_reference_provider_initialization(
      request, {true, true}, ProviderInitializationEigenFamily::xsyevd, 3, observation);
  require(bound.allowance_bytes == 128 && bound.retained_capacity(0) == 128,
          "actual adapter did not preserve a selected reservation after a zero delta");
  bool mismatch = false;
  try {
    (void)bind_reference_provider_initialization(
        request, {true, true}, ProviderInitializationEigenFamily::jacobi_batched, 3, observation);
  } catch (const std::invalid_argument&) {
    mismatch = true;
  }
  require(mismatch, "another eigen family was accepted by the actual adapter");
  observation.status = runtime::ContextIdentityStatus::query_error;
  observation.error_domain = runtime::IdentityErrorDomain::driver;
  observation.error_status = 700;
  bool driver_error = false;
  try {
    (void)bind_reference_provider_initialization(
        request, {true, true}, ProviderInitializationEigenFamily::xsyevd, 3, observation);
  } catch (const ProviderIdentityDriverError& error) {
    driver_error = error.status() == 700;
  }
  require(driver_error, "driver observation error became an unsupported fallback");
  observation.error_domain = runtime::IdentityErrorDomain::cuda;
  observation.error_status = cudaErrorMemoryAllocation;
  bool allocation_error = false;
  try {
    (void)bind_reference_provider_initialization(
        request, {true, true}, ProviderInitializationEigenFamily::xsyevd, 3, observation);
  } catch (const generativeqc_tensor::DeviceAllocationError&) {
    allocation_error = true;
  }
  require(allocation_error, "CUDA observation allocation error lost its type");
  require(bind_reference_provider_initialization({}, {true, true}, {}, 3, observation)
                  .allowance_bytes == (192ULL << 20),
          "legacy request consumed the new error/identity observation");
}
}  // namespace

int main() {
  try {
    legacy();
    unknown_ledger_basis();
    scalar_identity_and_capacity();
    native_observation_adapter();
    std::cout << "Provider initialization scalar admission and unknown-ledger refusal passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
