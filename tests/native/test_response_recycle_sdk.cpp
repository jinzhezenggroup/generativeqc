#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

#include "generativeqc/generativeqc.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void check(generativeqc_status status) {
  if (status != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error(generativeqc_status_message(status));
}

template <class Descriptor>
Descriptor versioned() {
  Descriptor result{};
  result.struct_size = sizeof(result);
  result.abi_version = GENERATIVEQC_ABI_VERSION;
  return result;
}

generativeqc_response_columns_v1 columns(std::span<const double> storage, std::size_t count = 1) {
  auto result = versioned<generativeqc_response_columns_v1>();
  result.storage = storage.data();
  result.element_count = storage.size();
  result.rows = 3;
  result.columns = count;
  result.row_stride = 1;
  result.column_stride = 3;
  return result;
}

auto state(generativeqc_response_recycle* owner) {
  auto result = versioned<generativeqc_response_recycle_state_v1>();
  check(generativeqc_response_recycle_state_v1_get(owner, &result));
  return result;
}

void run() {
  auto descriptor = versioned<generativeqc_response_recycle_descriptor_v1>();
  descriptor.dimension = 3;
  descriptor.max_vectors = 2;
  descriptor.max_retained_bytes = 48;
  auto capacity = versioned<generativeqc_response_recycle_capacity_v1>();
  check(generativeqc_response_recycle_capacity_v1_get(&descriptor, &capacity));
  require(capacity.vector_capacity == 2 && capacity.retained_capacity_bytes == 48,
          "shape-only retained capacity mismatch");
  generativeqc_response_recycle* raw = nullptr;
  check(generativeqc_response_recycle_create_v1(&descriptor, &raw));
  std::unique_ptr<generativeqc_response_recycle, decltype(&generativeqc_response_recycle_destroy)>
      owner(raw, generativeqc_response_recycle_destroy);
  require(state(raw).measured_live_bytes == 0 && state(raw).allocation_count == 0,
          "empty owner allocated dimension-sized payload");
  const std::array<double, 3> solved{{3, 0, 0}}, rhs{{1, 2, 3}};
  const std::array<double, 9> basis{{9, 0, 0, 0, 4, 0, 0, 0, 5}};
  auto solution = columns(solved), candidates = columns(basis, 3), input = columns(rhs);
  require(generativeqc_response_recycle_update_v1(raw, 0, &solution, &candidates, 2, 48,
                                                  capacity.update_workspace_bytes - 1) ==
                  GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
              state(raw).generation == 0 && state(raw).allocation_count == 0,
          "short update budget mutated or allocated vectors");
  check(generativeqc_response_recycle_update_v1(raw, 0, &solution, &candidates, 2, 48,
                                                capacity.update_workspace_bytes));
  auto measured = state(raw);
  require(measured.vector_count == 2 && measured.generation == 1 && measured.storage_bytes == 48 &&
              measured.measured_live_bytes == measured.storage_bytes &&
              measured.measured_peak_bytes <= capacity.update_workspace_bytes,
          "replacement ownership/peak accounting mismatch");
  std::array<double, 6> detached{};
  check(generativeqc_response_recycle_copy_v1(raw, 1, 0, 2, detached.data(), detached.size()));
  const std::array<double, 6> expected{{1, 0, 0, 0, 1, 0}};
  require(detached == expected, "independent MGS directions mismatch");
  std::array<double, 3> guess{{-17, -17, -17}};
  require(generativeqc_response_recycle_project_v1(raw, 1, &input, guess.data(), guess.size(),
                                                   capacity.projection_workspace_bytes - 1) ==
                  GENERATIVEQC_STATUS_OUT_OF_MEMORY &&
              guess[0] == -17,
          "short projection budget published data");
  check(generativeqc_response_recycle_project_v1(raw, 1, &input, guess.data(), guess.size(),
                                                 capacity.projection_workspace_bytes));
  require(guess == std::array<double, 3>{{1, 2, 0}}, "independent projection oracle mismatch");
  input.origin = 2;
  input.row_stride = -1;
  check(generativeqc_response_recycle_project_v1(raw, 1, &input, guess.data(), guess.size(),
                                                 capacity.projection_workspace_bytes));
  require(guess == std::array<double, 3>{{3, 2, 0}}, "signed RHS strides changed projection");
  input.row_stride = std::numeric_limits<std::int64_t>::min();
  guess.fill(-17);
  require(generativeqc_response_recycle_project_v1(raw, 1, &input, guess.data(), guess.size(),
                                                   capacity.projection_workspace_bytes) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              guess[0] == -17,
          "overflowing stride read or published data");
  measured = state(raw);
  require(generativeqc_response_recycle_update_v1(raw, 0, &solution, &candidates, 2, 48,
                                                  capacity.update_workspace_bytes) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              state(raw).generation == measured.generation &&
              state(raw).allocation_count == measured.allocation_count,
          "stale generation allocated or mutated vectors");
  check(generativeqc_response_recycle_update_v1(raw, 1, &solution, &candidates, 1, 24,
                                                capacity.update_workspace_bytes));
  require(state(raw).vector_count == 1 && state(raw).generation == 2,
          "changed retained limits did not trim atomically");
  owner.reset();
  require(detached == expected, "detached publication depended on destroyed owner");
  check(generativeqc_response_recycle_create_v1(&descriptor, &raw));
  owner.reset(raw);
  auto invalid = solved;
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  auto bad = columns(invalid);
  require(generativeqc_response_recycle_update_v1(raw, 0, &bad, &candidates, 2, 48,
                                                  capacity.update_workspace_bytes) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              state(raw).vector_count == 0 && state(raw).generation == 0 &&
              state(raw).measured_live_bytes == 0,
          "failed candidate leaked or published partial state");
  check(generativeqc_response_recycle_update_v1(raw, 0, &bad, &candidates, 2, 1,
                                                capacity.update_workspace_bytes));
  require(state(raw).vector_count == 0 && state(raw).generation == 1,
          "zero retained capacity visited unused invalid candidates");
  capacity.vector_capacity = 777;
  descriptor.dimension = std::numeric_limits<std::uint64_t>::max();
  require(generativeqc_response_recycle_capacity_v1_get(&descriptor, &capacity) ==
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
              capacity.vector_capacity == 777,
          "overflowing shape query published partial capacity");
  descriptor.struct_size = 0;
  require(generativeqc_response_recycle_create_v1(&descriptor, &raw) ==
                  GENERATIVEQC_STATUS_ABI_MISMATCH &&
              raw == nullptr,
          "bad create header published an owner");
  generativeqc_response_recycle_destroy(nullptr);
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
