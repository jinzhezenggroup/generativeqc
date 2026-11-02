#include <cmath>
#include <cstring>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>

#include "generativeqc/precision_admission.h"

/** Always evaluate executable contract checks, including Release/NDEBUG builds.
 * Assertions must not erase the API calls whose effects a fixture validates. */
inline void require_test(bool condition,
                         std::source_location location = std::source_location::current()) {
  if (!condition)
    throw std::runtime_error(std::string("native contract check failed: ") + location.file_name() +
                             ":" + std::to_string(location.line()));
}

int main() {
  generativeqc_precision_controls_context_v1 input{sizeof(input), GENERATIVEQC_ABI_VERSION};
  input.modules_available = input.ks_model = input.restricted_spin = 1;
  input.self_consistent = input.device_fused = 1;
  input.energy_abs = 1e-5;
  input.force_max_abs = 1e-4;
  input.force_rms = 3e-5;
  generativeqc_precision_controls_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION};
  const auto query = [&] { return generativeqc_precision_controls_query_v1(&input, &output); };
  require_test(query() == 0);
  require_test(output.policy_version == 1 && output.selected == 1);
  require_test(output.controls[0] == 1e-6 && output.controls[1] == 1e-4 &&
               output.controls[2] == 1e-10);
  // A one-ULP tighter allowance is outside this qualified policy's envelope.
  for (double* allowance : {&input.energy_abs, &input.force_max_abs, &input.force_rms}) {
    const auto exact = *allowance;
    *allowance = std::nextafter(exact, 0.0);
    require_test(query() == 0 && output.selected == 0);
    require_test(output.controls[0] == 0 && output.controls[1] == 0 && output.controls[2] == 0);
    *allowance = exact;
  }
  input.force_rms = 0;
  input.override_mask = 7;
  input.requested_controls[0] = 1e-12;
  input.requested_controls[1] = 1e-3;
  input.requested_controls[2] = 1e-14;
  require_test(query() == 0 && output.selected == 1);
  require_test(output.controls[0] == 1e-12 && output.controls[1] == 1e-4 &&
               output.controls[2] == 1e-14);
  for (auto* flag : {&input.modules_available, &input.ks_model, &input.restricted_spin,
                     &input.self_consistent, &input.device_fused}) {
    *flag = 0;
    require_test(query() == 0 && output.selected == 0);
    *flag = 1;
  }
  require_test(query() == 0);
  const auto before = output;
  input.requested_controls[0] = std::numeric_limits<double>::quiet_NaN();
  require_test(query() == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  input.requested_controls[0] = 1e-12;
  input.override_mask = 8;
  require_test(query() == GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  input.override_mask = 7;
  input.abi_version += 1;
  require_test(query() == GENERATIVEQC_STATUS_ABI_MISMATCH);
  require_test(std::memcmp(&before, &output, sizeof(output)) == 0);
  require_test(generativeqc_precision_controls_query_v1(nullptr, &output) ==
               GENERATIVEQC_STATUS_INVALID_ARGUMENT);
}
