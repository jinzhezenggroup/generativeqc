#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "dft/cuda_stationary_capacity.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_ledger.hpp"

int main() {
  using generativeqc::runtime::AsyncGeneration;
  using generativeqc::runtime::BorrowedWorkspace;
  using generativeqc::runtime::checked_add;
  using generativeqc::runtime::checked_multiply;
  using generativeqc::runtime::ranges_overlap;
  using generativeqc::runtime::WorkspaceLayout;

  std::size_t value = 0;
  assert(checked_add(7, 9, value) && value == 16);
  assert(!checked_add(std::numeric_limits<std::size_t>::max(), 1, value));
  assert(checked_multiply(7, 9, value) && value == 63);
  assert(!checked_multiply(std::numeric_limits<std::size_t>::max(), 2, value));

  WorkspaceLayout layout;
  std::size_t byte_offset = 0;
  std::size_t double_offset = 0;
  assert(layout.append<std::uint8_t>(3, byte_offset));
  assert(layout.append<double>(2, double_offset));
  assert(byte_offset == 0);
  assert(double_offset % alignof(double) == 0);
  assert(layout.bytes() == double_offset + 2 * sizeof(double));

  alignas(double) std::array<std::byte, 64> storage{};
  BorrowedWorkspace workspace(storage.data(), storage.size());
  const auto doubles = workspace.view<double>(0, 2);
  assert(doubles.data == reinterpret_cast<double*>(storage.data()));
  assert(doubles.elements == 2);
  BorrowedWorkspace empty(nullptr, 0);
  const auto empty_view = empty.view<std::byte>(0, 0);
  assert(empty_view.data == nullptr && empty_view.elements == 0);

  generativeqc::runtime::ResourcePlan plan{64, 32, 16, alignof(double)};
  assert(plan.valid());
  plan.workspace_bytes = 65;
  assert(!plan.valid());

  bool bounds_failed = false;
  try {
    (void)workspace.view<double>(60, 1);
  } catch (const std::out_of_range&) {
    bounds_failed = true;
  }
  assert(bounds_failed);

  assert(ranges_overlap(storage.data(), 16, storage.data() + 8, 16));
  assert(!ranges_overlap(storage.data(), 8, storage.data() + 8, 8));

  AsyncGeneration generation;
  generation.begin(1);
  assert(generation.published() == 0);
  generation.commit(1);
  generation.require(1);

  generation.begin(2);
  assert(generation.published() == 0);
  bool stale_failed = false;
  try {
    generation.require(1);
  } catch (const std::invalid_argument&) {
    stale_failed = true;
  }
  assert(stale_failed);

  // Simulate an enqueue failure: without commit(), no result is publishable.
  bool failed_result_hidden = false;
  try {
    generation.require(2);
  } catch (const std::invalid_argument&) {
    failed_result_hidden = true;
  }
  assert(failed_result_hidden);
  generation.begin(3);
  generation.commit(3);
  generation.require(3);

  // Provider allowances compose with existing numeric ownership but never
  // invent numeric allocations, journal identities or requested-byte samples.
  using namespace generativeqc::runtime;
  const auto previous_ledger = active_device_resource_ledger;
  auto ledger = std::make_shared<DeviceResourceLedger>();
  ledger->device = 3;
  ledger->limit = 100;
  ledger->live = ledger->peak = 20;
  active_device_resource_ledger = ledger;
  assert(device_resource_available(3) == 80);
  {
    DeviceProviderAllowance first(3, 60);
    assert(device_resource_available(3) == 20 && ledger->provider_reserved == 60);
    bool rejected = false;
    try {
      DeviceProviderAllowance too_large(3, 21);
    } catch (const std::bad_alloc&) {
      rejected = true;
    }
    assert(rejected && ledger->provider_reserved == 60 && ledger->rejected == 1);
    {
      DeviceProviderAllowance exact(3, 20);
      assert(device_resource_available(3) == 0 && ledger->provider_reserved == 80 &&
             ledger->provider_reservation_peak == 80 && ledger->admission_peak == 100);
    }
    assert(device_resource_available(3) == 20 && ledger->provider_reserved == 60);
    assert(ledger->live == 20 && ledger->peak == 20 && ledger->allocations == 0 &&
           ledger->requested_bytes == 0 && device_allocation_owners.empty());
    bool wrong_device = false;
    try {
      DeviceProviderAllowance incompatible(4, 1);
    } catch (const std::invalid_argument&) {
      wrong_device = true;
    }
    assert(wrong_device && ledger->provider_reserved == 60);
    active_device_resource_ledger.reset();
  }
  assert(ledger->provider_reserved == 0 && ledger->live == 20);
  active_device_resource_ledger = previous_ledger;

  // A stub emitted polynomial isolates the portable shape/overflow contract;
  // actual profile bytes versus allocations are checked by real CUDA owners.
  generativeqc::dft::CudaStationaryBinding profile;
  profile.spin_blocks = 1;
  profile.ao_jets = 10;
  profile.geometry_max_lanes = 128;
  profile.geometry_max_threads = 16;
  profile.geometry_max_scratch_bytes = 18 * 2 * 64 * sizeof(double);
  profile.numeric_bytes = [](std::size_t, std::size_t, std::size_t, std::size_t, std::size_t,
                             std::size_t, std::size_t lanes, bool cached) {
    if (!lanes) throw std::invalid_argument("missing geometry lane");
    return std::size_t{cached ? 176U : 128U};
  };
  const auto capacity = generativeqc::dft::cuda_stationary_capacity(profile, 2, 2, 2, 100, 128);
  assert(capacity.packed_elements == 42 && capacity.tile_points == 100 &&
         capacity.geometry_lanes == 64 && capacity.geometry_threads == 16 &&
         capacity.minimum_device_bytes ==
             capacity.grid.allocation_bytes + capacity.grid.provider_allowance + 128 &&
         capacity.cached_device_bytes == capacity.minimum_device_bytes + 48);
  bool overflow_failed = false;
  try {
    (void)generativeqc::dft::cuda_stationary_capacity(
        profile, 2, 2, std::numeric_limits<std::size_t>::max(), 100, 128);
  } catch (const std::overflow_error&) {
    overflow_failed = true;
  }
  assert(overflow_failed);
  for (const auto shape : {std::array<std::size_t, 3>{0, 100, 128}, {2, 99, 128}, {2, 100, 0}}) {
    bool invalid = false;
    try {
      (void)generativeqc::dft::cuda_stationary_capacity(profile, shape[0], 2, 2, shape[1],
                                                        shape[2]);
    } catch (const std::invalid_argument&) {
      invalid = true;
    }
    assert(invalid);
  }

  return 0;
}
