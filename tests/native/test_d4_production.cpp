#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "d4_eeq_oracle_fixtures.hpp"
#include "dft/dispersion/d4_runtime.hpp"

#if GENERATIVEQC_HAS_CUDA
#include <cuda_runtime.h>

#include "runtime/resource_ledger.hpp"
extern "C" {
void* generativeqc_resource_ledger_create_v1(std::size_t, int);
void generativeqc_resource_ledger_destroy_v1(void*);
int generativeqc_resource_ledger_bind_v1(void*);
int generativeqc_resource_ledger_read_v1(void*, std::uint64_t*);
int generativeqc_resource_tracking_begin_v1(unsigned);
int generativeqc_resource_tracking_end_v1(std::uint64_t*, std::uint64_t*);
void d4_cuda_fail_after_allocations_for_test_v1();
}
extern "C" void d4_cuda_fail_after_coordinate_upload_for_test_v1();
#endif

using namespace d4_eeq_tests;
using namespace generativeqc::dft::dispersion;

namespace {

bool near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

bool cuda_available() {
#if GENERATIVEQC_HAS_CUDA
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
#else
  return false;
#endif
}

std::unique_ptr<D4Plan> plan_for(const EEQOracleFixture& f, generativeqc_backend backend) {
  std::vector<std::uint32_t> offsets{0, static_cast<std::uint32_t>(f.atoms)};
  std::vector<std::int32_t> z(f.z.begin(), f.z.begin() + f.atoms);
  std::vector<double> xyz(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
  std::string detail;
  generativeqc_status status{};
  auto plan = D4Plan::prepare(backend, 0, std::move(offsets), std::move(z),
                              std::vector<double>{f.total_charge}, std::move(xyz), f.parameters,
                              f.profile, 64u << 20, detail, status);
  if (!plan || status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
  return plan;
}

void execute_one(D4Plan& plan, const std::vector<double>& xyz, bool gradient,
                 std::vector<D4Status>& statuses, std::vector<double>& components,
                 std::vector<double>& gradients, std::vector<double>& charges) {
  const std::uint8_t active = 1;
  const std::uint8_t requested = gradient ? 1 : 0;
  std::string detail;
  const auto status = plan.execute(xyz, std::span(&active, 1), std::span(&requested, 1), statuses,
                                   components, gradients, charges, detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}

void independent_oracles(generativeqc_backend backend) {
  for (const auto& f : kEEQOracleFixtures) {
    auto plan = plan_for(f, backend);
    std::vector<double> xyz(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
    std::vector<D4Status> statuses;
    std::vector<double> components, gradients, charges;
    execute_one(*plan, xyz, true, statuses, components, gradients, charges);
    if (statuses != std::vector<D4Status>{D4Status::success})
      throw std::runtime_error(std::string(f.name) + ": production status failed");
    if (components.size() != 2 || !near(components[0] + components[1], f.energy, 2e-13))
      throw std::runtime_error(std::string(f.name) + ": production energy mismatch");
    for (int i = 0; i < f.atoms; ++i)
      if (!near(charges[i], f.charges[i], 1e-10))
        throw std::runtime_error(std::string(f.name) + ": production charge mismatch");
    for (int i = 0; i < 3 * f.atoms; ++i)
      if (!near(gradients[i], f.gradient[i], 2e-12))
        throw std::runtime_error(std::string(f.name) + ": production gradient mismatch");
  }
}

void finite_difference(generativeqc_backend backend) {
  const auto& f = kEEQOracleFixtures[1];
  auto plan = plan_for(f, backend);
  std::vector<double> xyz(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
  std::vector<D4Status> statuses;
  std::vector<double> components, gradient, charges;
  execute_one(*plan, xyz, true, statuses, components, gradient, charges);
  for (double h : {1e-4, 2e-5}) {
    double maximum = 0.0;
    for (int coordinate = 0; coordinate < 3 * f.atoms; ++coordinate) {
      auto plus = xyz, minus = xyz;
      plus[coordinate] += h;
      minus[coordinate] -= h;
      std::vector<double> ep, em, ignored_gradient, ignored_charges;
      execute_one(*plan, plus, false, statuses, ep, ignored_gradient, ignored_charges);
      execute_one(*plan, minus, false, statuses, em, ignored_gradient, ignored_charges);
      const double numerical = ((ep[0] + ep[1]) - (em[0] + em[1])) / (2 * h);
      maximum = std::max(maximum, std::fabs(numerical - gradient[coordinate]));
    }
    if (maximum > 2e-9)
      throw std::runtime_error("production D4 finite-difference gradient gate failed");
  }
}

#if GENERATIVEQC_HAS_CUDA
// Use the library binding ABI so the test exercises the same allocation
// authority as a prepared frontend, including detached owner destruction.
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Ledger {
  void* handle{};
  explicit Ledger(std::uint64_t limit) : handle(generativeqc_resource_ledger_create_v1(limit, 0)) {
    require(handle != nullptr, "D4 ledger creation failed");
  }
  ~Ledger() { generativeqc_resource_ledger_destroy_v1(handle); }
  std::array<std::uint64_t, 4> read() const {
    std::array<std::uint64_t, 4> values{};
    require(generativeqc_resource_ledger_read_v1(handle, values.data()) == 0,
            "D4 ledger read failed");
    return values;
  }
};

struct LedgerScope {
  explicit LedgerScope(const Ledger& ledger) {
    require(generativeqc_resource_tracking_begin_v1(1) == 0, "D4 observation begin failed");
    require(generativeqc_resource_ledger_bind_v1(ledger.handle) == 0, "D4 ledger bind failed");
  }
  ~LedgerScope() {
    std::uint64_t peak{}, samples{};
    generativeqc_resource_tracking_end_v1(&peak, &samples);
  }
};

void bounded_device_lifetime() {
  const auto& f = kEEQOracleFixtures[0];
  auto unbounded = plan_for(f, GENERATIVEQC_BACKEND_CUDA);
  const auto required = unbounded->resources().device_bytes;
  unbounded.reset();
  require(required > 1, "D4 resource diagnostic omitted CUDA storage");

  // First-allocation, partial-construction and final-allocation shortages must
  // leave no live charge; the independent diagnostic fixes the exact boundary.
  for (const auto limit : {std::uint64_t{0}, required / 2, required - 1}) {
    Ledger ledger(limit);
    bool refused = false;
    {
      LedgerScope scope(ledger);
      try {
        auto plan = plan_for(f, GENERATIVEQC_BACKEND_CUDA);
      } catch (const std::runtime_error& error) {
        refused = std::string(error.what()).find("allocation failed") != std::string::npos;
      }
    }
    const auto receipt = ledger.read();
    require(refused && receipt[0] == 0 && receipt[1] <= limit && receipt[3] > 0,
            "D4 allocation escaped its cap or leaked a partial owner");
  }

  Ledger ledger(required);
  std::unique_ptr<D4Plan> plan;
  {
    LedgerScope scope(ledger);
    plan = plan_for(f, GENERATIVEQC_BACKEND_CUDA);
    require(ledger.read()[0] == required, "D4 live bytes disagree with the native diagnostic");
    std::vector<double> xyz(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
    std::vector<D4Status> statuses;
    std::vector<double> components, gradients, charges;
    execute_one(*plan, xyz, true, statuses, components, gradients, charges);
    require(statuses == std::vector<D4Status>{D4Status::success} &&
                near(components[0] + components[1], f.energy, 2e-13),
            "budgeted D4 disagrees with the independent energy oracle");
    for (int i = 0; i < 3 * f.atoms; ++i)
      require(near(gradients[i], f.gradient[i], 2e-12),
              "budgeted D4 disagrees with the independent gradient oracle");
    require(ledger.read()[0] == required && ledger.read()[3] == 0,
            "D4 execution exceeded the exact retained capacity");
  }
  // The ledger binding is thread-local; allocation ownership must outlive it.
  std::thread release([owner = std::move(plan)]() mutable { owner.reset(); });
  release.join();
  require(ledger.read()[0] == 0, "D4 cross-thread destruction retained live device bytes");

  {
    LedgerScope scope(ledger);
    d4_cuda_fail_after_allocations_for_test_v1();
    bool refused = false;
    try {
      auto failed = plan_for(f, GENERATIVEQC_BACKEND_CUDA);
    } catch (const std::runtime_error& error) {
      refused = std::string(error.what()).find("host allocation failed") != std::string::npos;
    }
    require(refused && ledger.read()[0] == 0,
            "D4 host exception leaked partially constructed CUDA storage");
    plan = plan_for(f, GENERATIVEQC_BACKEND_CUDA);
  }
  // Closing observation metadata is legal while the prepared owner is live.
  // A weak witness checks final retirement without retaining the ledger itself.
  std::weak_ptr<generativeqc::runtime::DeviceResourceLedger> lifetime =
      *static_cast<std::shared_ptr<generativeqc::runtime::DeviceResourceLedger>*>(ledger.handle);
  generativeqc_resource_ledger_destroy_v1(ledger.handle);
  ledger.handle = nullptr;
  require(!lifetime.expired(), "D4 detached owner lost its allocation authority");
  plan.reset();
  require(lifetime.expired(), "D4 final owner retained ledger metadata");
}

void changed_geometry_failure_recovery() {
  const auto& f = kEEQOracleFixtures[0];
  auto plan = plan_for(f, GENERATIVEQC_BACKEND_CUDA);
  std::vector<double> original(f.xyz.begin(), f.xyz.begin() + 3 * f.atoms);
  std::vector<D4Status> statuses;
  std::vector<double> baseline, gradients, charges;
  execute_one(*plan, original, false, statuses, baseline, gradients, charges);

  auto changed = original;
  changed[0] += 0.037;
  const std::uint8_t active = 1;
  const std::uint8_t requested = 0;
  std::string detail;
  std::vector<double> failed_components, failed_gradient, failed_charges;
  d4_cuda_fail_after_coordinate_upload_for_test_v1();
  const auto failed =
      plan->execute(changed, std::span(&active, 1), std::span(&requested, 1), statuses,
                    failed_components, failed_gradient, failed_charges, detail);
  if (failed != GENERATIVEQC_STATUS_CUDA_ERROR ||
      detail.find("after coordinate upload") == std::string::npos)
    throw std::runtime_error("D4 post-upload CUDA failure injection did not trigger");

  std::vector<double> recovered_components, recovered_gradient, recovered_charges;
  execute_one(*plan, original, false, statuses, recovered_components, recovered_gradient,
              recovered_charges);
  if (recovered_components.size() != baseline.size())
    throw std::runtime_error("D4 recovery changed the result shape");
  for (std::size_t i = 0; i < baseline.size(); ++i)
    if (!near(recovered_components[i], baseline[i], 2e-13))
      throw std::runtime_error("D4 failed-upload recovery reused poisoned device coordinates");

  const auto bytes = original.size() * sizeof(double);
  const auto& counters = plan->counters();
  if (counters.execution_count != 3 || counters.unchanged_geometry_replays != 1 ||
      counters.changed_geometry_replays != 2 || counters.coordinate_h2d_bytes != 2 * bytes)
    throw std::runtime_error("D4 failed-upload recovery did not force coordinate re-upload");
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    const bool device = argc > 1 && std::string(argv[1]) == "cuda";
    if (device && !cuda_available()) return 77;
    const auto backend = device ? GENERATIVEQC_BACKEND_CUDA : GENERATIVEQC_BACKEND_CPU_REFERENCE;
    independent_oracles(backend);
    finite_difference(backend);
#if GENERATIVEQC_HAS_CUDA
    if (device) {
      changed_geometry_failure_recovery();
      bounded_device_lifetime();
    }
#endif
    std::puts(device ? "production D4 CUDA oracle/FD gates passed"
                     : "production D4 CPU oracle/FD gates passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
