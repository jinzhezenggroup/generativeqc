#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "generativeqc/generativeqc.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc_hf_plan_descriptor_v1 request() {
  return {sizeof(generativeqc_hf_plan_descriptor_v1),
          GENERATIVEQC_ABI_VERSION,
          7,
          0,
          20,
          100,
          20,
          120,
          1e-6,
          1e-5,
          1e-10,
          1e-8,
          1.0,
          4.0,
          5.0};
}

std::uint32_t failures(const generativeqc_hf_plan_descriptor_v1& input) {
  generativeqc_hf_plan_result_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION, 99, 99};
  require(generativeqc_hf_plan_query_v1(&input, &output) == GENERATIVEQC_STATUS_SUCCESS,
          "valid numerical plan query failed");
  require(!output.reserved, "reserved output was not cleared");
  return output.failures;
}

void run() {
  auto input = request();
  require(!failures(input), "exact budgets should pass");
  input.source_iteration_limit = 19;
  input.total_iteration_limit = 119;
  input.source_energy_tolerance = 1e-11;
  input.cost_limit = 4.9;
  require(failures(input) == 23, "simultaneous planning failures were lost");
  input.flags &= ~GENERATIVEQC_HF_PLAN_TARGET_COST_PRESENT;
  require(failures(input) == 15, "missing cost must supersede cost comparison");
  input = request();
  input.source_iterations = input.source_iteration_limit = input.total_iteration_limit =
      std::numeric_limits<std::uint64_t>::max();
  require(failures(input) == GENERATIVEQC_HF_PLAN_TOTAL_ITERATIONS,
          "iteration sum wrapped into budget");
  input.source_iterations -= input.target_iterations;
  require(!failures(input), "full-width exact sum failed");
  input = request();
  input.source_cost = input.target_cost = input.cost_limit = std::numeric_limits<double>::max();
  require(failures(input) == GENERATIVEQC_HF_PLAN_TOTAL_COST,
          "overflowing cost estimate was admitted");
  input.flags = 0;
  input.source_cost = input.target_cost = input.cost_limit =
      std::numeric_limits<double>::quiet_NaN();
  require(!failures(input), "absent costs were inspected");

  for (int test = 0; test < 7; ++test) {
    auto invalid = request();
    if (test == 0) invalid.flags |= 8;
    if (test == 1) invalid.reserved = 1;
    if (test == 2) invalid.source_iterations = 0;
    if (test == 3) invalid.source_cost = std::numeric_limits<double>::infinity();
    if (test == 4) invalid.target_density_tolerance = 0;
    if (test == 5) invalid.abi_version += 1;
    if (test == 6) invalid.struct_size -= 1;
    generativeqc_hf_plan_result_v1 output{sizeof(output), GENERATIVEQC_ABI_VERSION, 123, 456};
    const auto preserved = output;
    const auto status = generativeqc_hf_plan_query_v1(&invalid, &output);
    require(status == (test >= 5 ? GENERATIVEQC_STATUS_ABI_MISMATCH
                                 : GENERATIVEQC_STATUS_INVALID_ARGUMENT),
            "malformed plan query was admitted");
    require(!std::memcmp(&output, &preserved, sizeof(output)), "failure published output");
  }
  // Input and output may overlap: all input must be read before publication.
  input = request();
  auto* output = reinterpret_cast<generativeqc_hf_plan_result_v1*>(&input);
  require(generativeqc_hf_plan_query_v1(&input, output) == GENERATIVEQC_STATUS_SUCCESS,
          "overlapping valid buffers failed");
  require(!output->failures && !output->reserved, "overlap changed plan admission");
  require(generativeqc_hf_plan_query_v1(nullptr, output) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null input was admitted");
  require(generativeqc_hf_plan_query_v1(&input, nullptr) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "null output was admitted");
}
}  // namespace

int main() {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
