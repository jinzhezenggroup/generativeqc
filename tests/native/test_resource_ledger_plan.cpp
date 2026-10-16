#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "runtime/resource_ledger.hpp"
#include "runtime/resource_ledger_plan.hpp"

extern "C" {
int generativeqc_resource_ledger_create_from_buffers_v1(const std::uint64_t*, std::size_t, void**,
                                                        std::uint64_t*, std::size_t);
void generativeqc_resource_ledger_destroy_v1(void*);
int generativeqc_resource_ledger_bind_v1(void*);
int generativeqc_resource_ledger_read_v3(void*, std::uint64_t*, std::size_t);
int generativeqc_resource_tracking_begin_v1(unsigned);
int generativeqc_resource_tracking_end_v1(std::uint64_t*, std::uint64_t*);
}

namespace {
using namespace generativeqc::runtime;
using namespace generativeqc::runtime::resources;
constexpr auto maximum = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void test_sparse_allowance_accounting() {
  std::vector<DeviceLedgerBuffer> buffers{{64, 4, 7, 0, 0, false},
                                          {32, 4, 7, 1, 1, false},
                                          {100, 4, 7, 1, 1, true},
                                          {maximum, 3, 0, maximum, maximum, false}};
  const auto plan = device_ledger_admission(buffers);
  require(plan && plan->device == 7 && plan->numeric_limit == 64 && plan->declared_peak == 132,
          "runtime allowance funded numeric storage or separate peaks were subtracted");
  buffers = {{maximum, 4, std::numeric_limits<int>::max(), 0, 0, true}};
  const auto allowance_only = device_ledger_admission(buffers);
  require(allowance_only && allowance_only->numeric_limit == 0 &&
              allowance_only->declared_peak == maximum,
          "allowance-only inventory fabricated a numeric limit");
  buffers.push_back({0, 4, 0, 0, 0, false});
  require(!device_ledger_admission(buffers), "multiple visible devices were silently merged");
  buffers = {{1, 3, 0, 0, 0, false}};
  require(!device_ledger_admission(buffers), "missing device invented an ordinal");
}

void test_transactional_wire_and_lifecycle() {
  std::vector<std::uint64_t> rows{64, 4, 7, 0, 0, 0, 32, 4, 7, 1, 1, 0, 100, 4, 7, 1, 1, 1};
  void* handle = nullptr;
  std::array<std::uint64_t, 3> receipt{41, 42, 43};
  const auto query = [&] {
    return generativeqc_resource_ledger_create_from_buffers_v1(rows.data(), rows.size(), &handle,
                                                               receipt.data(), receipt.size());
  };
  const auto rejected = [&](int status) {
    const auto before = receipt;
    require(query() == status && !handle && receipt == before,
            "rejected inventory changed the handle or receipt");
  };
  const auto valid = rows;
  for (const auto index : {std::size_t{1}, std::size_t{5}}) {
    rows[index] = 9;
    rejected(1);
    rows = valid;
  }
  rows[2] = static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1;
  rejected(1);
  rows = valid;
  rows[3] = 1;
  rejected(1);
  rows = valid;
  rows[13] = 3;
  rejected(1);
  rows = valid;
  rows[8] = 8;
  rejected(4);
  rows = valid;
  rows[0] = maximum;
  rows[9] = 0;
  rejected(2);
  rows = valid;
  const auto before = rows;
  require(generativeqc_resource_ledger_create_from_buffers_v1(rows.data(), rows.size(), &handle,
                                                              rows.data() + 1, 3) == 1 &&
              rows == before && !handle,
          "input/output alias was accepted");
  require(generativeqc_resource_ledger_create_from_buffers_v1(
              rows.data(), std::numeric_limits<std::size_t>::max() - 3, &handle, receipt.data(),
              receipt.size()) != 0 &&
              !handle && receipt == std::array<std::uint64_t, 3>{41, 42, 43},
          "overflowing input extent was dereferenced or changed output");
  require(
      generativeqc_resource_ledger_create_from_buffers_v1(
          rows.data(), rows.size(), &handle, reinterpret_cast<std::uint64_t*>(&handle), 3) == 1 &&
          !handle,
      "overlapping output ranges were accepted");
  auto adjacent = rows;
  adjacent.resize(rows.size() + 3, 991);
  require(generativeqc_resource_ledger_create_from_buffers_v1(
              adjacent.data(), rows.size(), &handle, adjacent.data() + rows.size(), 3) == 0 &&
              handle && std::equal(rows.begin(), rows.end(), adjacent.begin()) &&
              adjacent[rows.size() + 1] == 64,
          "adjacent disjoint input/output ranges were rejected or mutated");
  generativeqc_resource_ledger_destroy_v1(handle);
  handle = nullptr;
  require(query() == 0 && handle && receipt == std::array<std::uint64_t, 3>{7, 64, 132},
          "native selected-inventory receipt changed");
  const auto original = handle;
  require(query() == 1 && handle == original, "existing ledger handle was replaced");
  std::weak_ptr<DeviceResourceLedger> lifetime =
      *static_cast<std::shared_ptr<DeviceResourceLedger>*>(handle);
  require(generativeqc_resource_tracking_begin_v1(1) == 0 &&
              generativeqc_resource_ledger_bind_v1(handle) == 0,
          "new ledger is not compatible with existing observation binding");
  auto retained = std::make_unique<DeviceProviderAllowance>(7, 64);
  bool shortage = false;
  try {
    DeviceProviderAllowance excess(7, 1);
  } catch (const std::bad_alloc&) {
    shortage = true;
  }
  std::array<std::uint64_t, 6> observation{};
  require(shortage && generativeqc_resource_ledger_read_v3(handle, observation.data(), 6) == 0 &&
              observation[0] == 64 && observation[1] == 0 && observation[3] == 64 &&
              observation[5] == 64,
          "numeric admission exceeded its cap or fabricated an allocation");
  std::uint64_t peak = 0, samples = 0;
  require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0,
          "observation did not retire its binding");
  generativeqc_resource_ledger_destroy_v1(handle);
  handle = nullptr;
  require(!lifetime.expired(), "handle close released a retained provider's ledger");
  std::thread release([lease = std::move(retained)]() mutable { lease.reset(); });
  release.join();
  require(lifetime.expired(), "cross-thread final release retained ledger metadata");
}
}  // namespace

int main() {
  try {
    test_sparse_allowance_accounting();
    test_transactional_wire_and_lifecycle();
    std::cout << "Native device-ledger admission, rollback and retained lifecycle passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
