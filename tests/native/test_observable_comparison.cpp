#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "generativeqc/precision_admission.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void contract() {
  generativeqc_observable_comparison_context_v1 context{
      sizeof(context), GENERATIVEQC_ABI_VERSION, 7, 0, 2, 1, 0, {1, 1, 1}, {}};
  std::array<double, 6> forces{1, 1, 1, 1, 1, 1}, reference{};
  generativeqc_observable_comparison_v1 result{sizeof(result), GENERATIVEQC_ABI_VERSION};
  require(generativeqc_observable_comparison_query_v1(&context, forces.data(), reference.data(),
                                                      &result) == GENERATIVEQC_STATUS_SUCCESS,
          "valid observable request failed");
  require(result.policy_version == 1 && result.measured_mask == 7 && result.met_mask == 7,
          "inclusive independent observable thresholds changed");
  for (unsigned slot = 0; slot != 3; ++slot)
    require(result.differences[slot] == 1 && result.reference_norms[slot] == 0 &&
                result.allowances[slot] == 1,
            "full-force norm or allowance changed");
  const auto before = result;
  forces[0] = std::numeric_limits<double>::infinity();
  require(
      generativeqc_observable_comparison_query_v1(&context, forces.data(), reference.data(),
                                                  &result) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
      "nonfinite observable accepted");
  require(std::memcmp(&before, &result, sizeof(result)) == 0, "failure modified output");
}
}  // namespace

int main() {
  try {
    contract();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
