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

void test_native_scope_without_observation() {
  auto ledger = std::make_shared<DeviceResourceLedger>();
  ledger->limit = 64;
  ledger->device = 7;
  std::unique_ptr<DeviceProviderAllowance> retained;
  {
    DeviceResourceScope operation(ledger);
    require(ledger->active && active_device_resource_ledger == ledger,
            "native scope depended on a CPU/Python observation");
    retained = std::make_unique<DeviceProviderAllowance>(7, 64);
    sample_cpu_capacity(17);
    sample_cuda_arena_capacity(11);
    const auto available = device_resource_available(7);
    {
      DeviceResourceScope nested(ledger);
      require(device_resource_available(7) == available && ledger->provider_reserved == 64,
              "same-owner nesting reset or detached retained charges");
    }
    auto foreign = std::make_shared<DeviceResourceLedger>();
    bool rejected = false;
    try {
      DeviceResourceScope wrong_owner(foreign);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected && !foreign->active && active_device_resource_ledger == ledger,
            "foreign owner replaced the native prepared ledger");
    bool cross_thread_rejected = false;
    std::thread concurrent([&] {
      try {
        DeviceResourceScope duplicate(ledger);
      } catch (const std::invalid_argument&) {
        cross_thread_rejected = true;
      }
      require(!active_device_resource_ledger, "failed worker binding leaked thread-local state");
    });
    concurrent.join();
    require(cross_thread_rejected, "concurrent thread borrowed another operation's owner");
    require(cpu_resource_observation.active && generativeqc_resource_tracking_begin_v1(1) == 1,
            "native operation did not own its observation or replaced an active observer");
    std::uint64_t peak = 0, samples = 0;
    require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0 &&
                active_device_resource_ledger == ledger && ledger->active &&
                resource_cpu_worker_limit() == 1,
            "ending observation detached an executing native resource owner");
  }
  require(!ledger->active && !active_device_resource_ledger &&
              native_device_resource_scope_depth == 0 && ledger->provider_reserved == 64,
          "native scope did not unbind or erased retained provider charges");
  require(!cpu_resource_observation.active && ledger->native_host_peak == 17 &&
              ledger->native_host_samples == 1 && ledger->native_cuda_arena_peak == 11 &&
              ledger->native_cuda_arena_samples == 1,
          "native operation did not preserve samples or restore its caller's observation state");
  try {
    DeviceResourceScope replay(ledger);
    require(ledger->provider_reservation_peak == 64 && ledger->admission_peak == 64,
            "replay reset did not include retained live reservations");
    DeviceProviderAllowance excess(7, 1);
    throw std::runtime_error("over-cap provider reservation was accepted");
  } catch (const std::bad_alloc&) {
  }
  require(!ledger->active && !active_device_resource_ledger && ledger->rejected == 1,
          "exception did not restore the native resource binding");
  require(generativeqc_resource_tracking_begin_v1(1) == 0 && bind_device_resource_ledger(ledger),
          "legacy outer binding was rejected");
  sample_cpu_capacity(33);
  {
    DeviceResourceScope borrowed(ledger);
    require(ledger->rejected == 0, "outer scope did not reset operation counters");
  }
  require(ledger->active && active_device_resource_ledger == ledger,
          "native borrowed scope detached its legacy outer owner");
  require(ledger->native_host_samples == 0 && cpu_resource_observation.active,
          "native scope mislabeled a borrowed outer observer as an isolated owner sample");
  unbind_device_resource_ledger();
  std::uint64_t peak = 0, samples = 0;
  require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0 && peak == 33,
          "native scope erased its caller's outer samples");
  require(generativeqc_resource_tracking_begin_v1(1) == 0 && bind_device_resource_ledger(ledger),
          "legacy observer could not bind before borrowed-scope teardown");
  {
    DeviceResourceScope borrowed(ledger);
    {
      DeviceResourceScope nested(ledger);
      require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0,
              "legacy observer could not end inside a borrowed native scope");
      require(ledger->active && active_device_resource_ledger == ledger &&
                  resource_cpu_worker_limit() == 1,
              "observer teardown detached a nested borrowed native scope");
    }
    require(ledger->active && active_device_resource_ledger == ledger &&
                resource_cpu_worker_limit() == 1,
            "nested teardown detached the outer borrowed native scope");
  }
  require(
      !ledger->active && !active_device_resource_ledger && native_device_resource_scope_depth == 0,
      "borrowed native scope retained an ended observer's binding");
  require(bind_device_resource_ledger(ledger),
          "ended observer prevented the next operation from binding");
  {
    DeviceResourceScope borrowed_without_observer(ledger);
  }
  require(
      ledger->active && active_device_resource_ledger == ledger && !cpu_resource_observation.active,
      "native scope inferred ownership of an explicit non-observer binding");
  unbind_device_resource_ledger();
  require(generativeqc_resource_tracking_begin_v1(0) == 0 && bind_device_resource_ledger(ledger),
          "legacy observer could not bind before exception teardown");
  struct ExpectedFailure {};
  try {
    DeviceResourceScope borrowed(ledger);
    DeviceResourceScope nested(ledger);
    require(generativeqc_resource_tracking_end_v1(&peak, &samples) == 0,
            "legacy observer could not end before exception teardown");
    throw ExpectedFailure{};
  } catch (const ExpectedFailure&) {
  }
  require(!ledger->active && !active_device_resource_ledger &&
              native_device_resource_scope_depth == 0 && !device_resource_unbind_pending,
          "exception teardown retained an ended observer's binding or pending request");
  std::weak_ptr<DeviceResourceLedger> lifetime = ledger;
  ledger.reset();
  require(!lifetime.expired(), "metadata close released a retained provider owner");
  std::thread release([lease = std::move(retained)]() mutable { lease.reset(); });
  release.join();
  require(lifetime.expired(), "cross-thread release retained the closed native owner");
}
}  // namespace

int main() {
  try {
    test_sparse_allowance_accounting();
    test_transactional_wire_and_lifecycle();
    test_native_scope_without_observation();
    std::cout << "Native device-ledger admission, rollback and retained lifecycle passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
