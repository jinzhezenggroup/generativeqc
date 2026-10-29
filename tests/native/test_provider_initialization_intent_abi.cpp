#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

extern "C" int intent_c_header_available(void);

#if __has_include("generativeqc/provider_initialization.h")
#include "api/provider_initialization_intent.hpp"
#include "generativeqc/provider_initialization.h"

extern "C" std::size_t intent_c_size(void);
extern "C" std::size_t intent_c_allowance_offset(void);
extern "C" generativeqc_status intent_c_prepare(
    generativeqc_context*, const generativeqc_system*, const generativeqc_method_descriptor*,
    const generativeqc_provider_initialization_intent_v1*, generativeqc_calculation**);

namespace {
int live_calculations{}, legacy_calls{}, pending_calls{};
bool fail_candidate{};
generativeqc_provider_initialization_intent_v1* mutate_input{};
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
}  // namespace

// This fixture implements only the private transport port. It performs no native
// preparation, CUDA observation, geometry, numerical work or ABI publication test.
struct generativeqc_context {};
struct generativeqc_system {};
struct generativeqc_calculation {
  const std::optional<generativeqc::scf::ProviderInitializationIntent> intent;
  explicit generativeqc_calculation(
      std::optional<generativeqc::scf::ProviderInitializationIntent> value = {})
      : intent(value) {
    ++live_calculations;
  }
  ~generativeqc_calculation() { --live_calculations; }
};

extern "C" generativeqc_status generativeqc_calculation_prepare(
    generativeqc_context* context, const generativeqc_system* system,
    const generativeqc_method_descriptor* method, generativeqc_calculation** output) {
  if (!context || !system || !method || !output) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  ++legacy_calls;
  *output = new generativeqc_calculation;
  return GENERATIVEQC_STATUS_SUCCESS;
}

namespace generativeqc::api {
std::uint32_t provider_initialization_intent_capabilities() noexcept {
  return GENERATIVEQC_PROVIDER_INITIALIZATION_MP2_CUDA |
         GENERATIVEQC_PROVIDER_INITIALIZATION_RCCSD_CUDA;
}
generativeqc_status prepare_calculation_with_provider_initialization_intent(
    generativeqc_context*, const generativeqc_system*, const generativeqc_method_descriptor*,
    scf::ProviderInitializationIntent intent, generativeqc_calculation** output) noexcept {
  ++pending_calls;
  if (mutate_input) mutate_input->allowance_bytes = 7;
  auto candidate = std::make_unique<generativeqc_calculation>(intent);
  if (fail_candidate) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  *output = candidate.release();
  return GENERATIVEQC_STATUS_SUCCESS;
}
}  // namespace generativeqc::api

int run(const std::string& mode) {
  generativeqc_context context;
  generativeqc_system system;
  generativeqc_method_descriptor method{};
  method.struct_size = sizeof(method);
  method.abi_version = GENERATIVEQC_ABI_VERSION;
  method.method = GENERATIVEQC_METHOD_MP2;
  generativeqc_provider_initialization_intent_v1 request{
      sizeof(request), 1, GENERATIVEQC_PROVIDER_INITIALIZATION_PENDING, 0, 512ULL << 20};
  generativeqc_calculation* output = reinterpret_cast<generativeqc_calculation*>(1);
  const auto prepare = [&](const auto* intent) {
    return intent_c_prepare(&context, &system, &method, intent, &output);
  };
  require(intent_c_header_available() && intent_c_size() == sizeof(request), "C/C++ size");
  require(intent_c_allowance_offset() == offsetof(decltype(request), allowance_bytes),
          "C/C++ offset");
  if (mode == "capture") {
    require(generativeqc_provider_initialization_intent_version() == 1, "intent ABI version");
    require(generativeqc_provider_initialization_capabilities_v1() ==
                (GENERATIVEQC_PROVIDER_INITIALIZATION_MP2_CUDA |
                 GENERATIVEQC_PROVIDER_INITIALIZATION_RCCSD_CUDA),
            "software capability");
    mutate_input = &request;
    require(prepare(&request) == GENERATIVEQC_STATUS_SUCCESS, "pending transport");
    require(request.allowance_bytes == 7 && output->intent->allowance_bytes() == 512ULL << 20,
            "intent was not copied before the private port");
    request = {};
    require(output->intent->allowance_bytes() == 512ULL << 20, "caller storage was retained");
    delete output;
    require(live_calculations == 0 && pending_calls == 1 && legacy_calls == 0,
            "transport lifetime");
  } else if (mode == "overlap") {
    union Storage {
      generativeqc_provider_initialization_intent_v1 intent;
      generativeqc_calculation* output;
    } storage{request};
    require(intent_c_prepare(&context, &system, &method, &storage.intent, &storage.output) ==
                GENERATIVEQC_STATUS_SUCCESS,
            "intent was changed before its by-value snapshot");
    require(storage.output->intent->allowance_bytes() == 512ULL << 20,
            "overlapping output changed copied intent");
    delete storage.output;
    require(live_calculations == 0, "overlapping output lifetime");
  } else if (mode == "legacy") {
    method.method = GENERATIVEQC_METHOD_RHF;
    require(prepare(static_cast<const decltype(request)*>(nullptr)) == GENERATIVEQC_STATUS_SUCCESS,
            "absent intent legacy path");
    require(!output->intent, "absent intent acquired pending state");
    delete output;
    request.mode = GENERATIVEQC_PROVIDER_INITIALIZATION_LEGACY_DEFAULT;
    request.allowance_bytes = 0;
    require(prepare(&request) == GENERATIVEQC_STATUS_SUCCESS && !output->intent,
            "explicit legacy intent path");
    delete output;
    require(legacy_calls == 2 && pending_calls == 0 && live_calculations == 0,
            "legacy path used the pending port");
  } else if (mode == "negative") {
    const auto valid = request;
    const auto refuses = [&](generativeqc_status expected) {
      output = reinterpret_cast<generativeqc_calculation*>(1);
      require(prepare(&request) == expected && !output, "invalid intent/status/output");
      request = valid;
    };
    request.struct_size = sizeof(request) - 1;
    refuses(GENERATIVEQC_STATUS_ABI_MISMATCH);
    request.abi_version = 2;
    refuses(GENERATIVEQC_STATUS_ABI_MISMATCH);
    request.mode = 77;
    refuses(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    request.reserved = 1;
    refuses(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    request.allowance_bytes = 0;
    refuses(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    request.allowance_bytes = std::numeric_limits<std::uint64_t>::max();
    refuses(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    request.mode = GENERATIVEQC_PROVIDER_INITIALIZATION_LEGACY_DEFAULT;
    refuses(GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    method.method = GENERATIVEQC_METHOD_RHF;
    refuses(GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    method.method = GENERATIVEQC_METHOD_MP2;
    method.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_AUTO;
    refuses(GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    require(pending_calls == 0 && legacy_calls == 0 && live_calculations == 0,
            "rejected request reached preparation");
  } else if (mode == "lifetime") {
    fail_candidate = true;
    require(prepare(&request) == GENERATIVEQC_STATUS_OUT_OF_MEMORY && !output,
            "private candidate failure");
    require(live_calculations == 0, "failed candidate leaked");
    fail_candidate = false;
    require(prepare(&request) == GENERATIVEQC_STATUS_SUCCESS && live_calculations == 1,
            "fresh candidate after failure");
    delete output;
    require(live_calculations == 0, "successful candidate leaked");
  } else {
    throw std::runtime_error("unknown ABI probe mode");
  }
  return 0;
}
#endif

int main(int argc, char** argv) {
  if (!intent_c_header_available()) {
    std::cerr << "provider initialization intent C ABI is unavailable\n";
    return 7;
  }
#if __has_include("generativeqc/provider_initialization.h")
  try {
    if (argc != 2) throw std::runtime_error("one ABI probe mode required");
    return run(argv[1]);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
#else
  (void)argc;
  (void)argv;
  return 7;
#endif
}
