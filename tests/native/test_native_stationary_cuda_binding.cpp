#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/cuda_stationary_binding.hpp"
#include "dft/cuda_stationary_controller.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"

extern "C" {
int generativeqc_resource_ledger_read_v1(void*, std::uint64_t*);
int generativeqc_resource_ledger_read_v2(void*, std::uint64_t*, std::size_t);
int generativeqc_resource_ledger_read_v3(void*, std::uint64_t*, std::size_t);
int generativeqc_resource_stationary_cuda_v1(const char*, std::size_t,
                                             std::size_t, std::size_t,
                                             std::size_t, std::size_t,
                                             std::uint64_t*, std::size_t);
void grid_cuda_fail_next_allocation_for_test_v1();
void grid_cuda_fail_next_host_allocation_for_test_v1();
}

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/** These transport/lifetime gates are not complete molecular force oracles.
 * Public prepared admission additionally requires the enclosing controller's
 * actual SCF, full-grid coverage and independent force acceptance gates. */
void profile_sources(const generativeqc::dft::CudaStationaryBinding& binding) {
  using namespace generativeqc::scf;
  cudaDeviceProp property{};
  require(cudaGetDeviceProperties(&property, 0) == cudaSuccess,
          "device properties failed");
  std::array<char, 512> error{};
  const auto minimum =
      binding.numeric_bytes(2, 2, 2, 64, 1, binding.spin_blocks, 64, false);
  void* bounded_owner = reinterpret_cast<void*>(1);
  require(binding.create(0, property.major, property.minor, 2, 2, 2, 64, 1,
                         binding.spin_blocks, 1, minimum - 1, 64, 16,
                         &bounded_owner, error.data(), error.size()) != 0 &&
              !bounded_owner,
          "native profile admitted a below-floor numeric arena");
  require(binding.create(0, property.major, property.minor, 2, 2, 2, 64, 1,
                         binding.spin_blocks, 1, minimum, 64, 16,
                         &bounded_owner, error.data(), error.size()) == 0,
          error.data());
  binding.destroy(bounded_owner);
  // An out-of-range visible ordinal is a validated argument failure, not a
  // driver call. Preserve this distinction without allocating an owner.
  int visible_devices{};
  require(cudaGetDeviceCount(&visible_devices) == cudaSuccess,
          "device count failed");
  bounded_owner = reinterpret_cast<void*>(1);
  require(binding.create(visible_devices, property.major, property.minor, 2, 2,
                         2, 64, 1, binding.spin_blocks, 1, minimum, 64, 16,
                         &bounded_owner, error.data(), error.size()) == 1 &&
              !bounded_owner,
          "native creation misclassified an invalid device ordinal");
  require(cudaPeekAtLastError() == cudaSuccess,
          "invalid ordinal validation invoked a failing driver call");
  // The generated arena must participate in the actual native ledger, not
  // merely report a polynomial while allocating outside the prepared budget.
  for (const auto allowance : {minimum - 1, minimum}) {
    auto ledger =
        std::make_shared<generativeqc::runtime::DeviceResourceLedger>();
    ledger->limit = allowance;
    ledger->device = 0;
    const auto previous = generativeqc::runtime::active_device_resource_ledger;
    generativeqc::runtime::active_device_resource_ledger = ledger;
    bounded_owner = nullptr;
    const auto status = binding.create(
        0, property.major, property.minor, 2, 2, 2, 64, 1, binding.spin_blocks,
        1, minimum, 64, 16, &bounded_owner, error.data(), error.size());
    generativeqc::runtime::active_device_resource_ledger = previous;
    require((status == 0) == (allowance == minimum),
            "generated arena bypassed the real native allocation limit");
    if (!status)
      require(ledger->live == minimum && ledger->peak == minimum &&
                  ledger->allocations == 1,
              "generated arena numeric receipt differs from actual requested "
              "storage");
    else
      require(
          status == 2 && !bounded_owner && !ledger->live &&
              ledger->rejected == 1,
          "generated arena failed allocation retained an owner or reservation");
    binding.destroy(bounded_owner);
    require(!ledger->live,
            "generated arena escaped its retained ledger after scope exit");
  }
  std::vector<double> input(24);
  struct ProviderLease {
    cudaStream_t stream{};
    double* values{};
    ~ProviderLease() {
      if (stream) (void)cudaStreamSynchronize(stream);
      if (values) (void)cudaFree(values);
      if (stream) (void)cudaStreamDestroy(stream);
    }
  } provider;
  struct OwnerLease {
    const generativeqc::dft::CudaStationaryBinding& binding;
    void* owner{};
    ~OwnerLease() { binding.destroy(owner); }
  } lease{binding};
  require(binding.create(0, property.major, property.minor, 2, 2, 2, 64, 1,
                         binding.spin_blocks, 1, 8U << 20, 64, 16, &lease.owner,
                         error.data(), error.size()) == 0,
          error.data());
  const std::array<double, 4> primitives{0.8, 0.7, 0.6, 1.0};
  const std::array<std::int64_t, 4> ranges{0, 1, 1, 1};
  const std::array<double, 2> norms{1.0, 1.0}, charges{1.0, 1.0};
  const std::array<std::int64_t, 2> ao_atoms{0, 1};
  const std::array<double, 6> centers{0, 0, -0.7, 0, 0, 0.7};
  require(binding.topology(lease.owner, primitives.data(), ranges.data(),
                           norms.data(), ao_atoms.data(), error.data(),
                           error.size()) == 0,
          error.data());
  const auto reset = [&] {
    require(binding.reset_geometry(lease.owner, centers.data(), 1e-12,
                                   error.data(), error.size()) == 0,
            error.data());
  };
  reset();
  require(cudaStreamCreateWithFlags(&provider.stream, cudaStreamNonBlocking) ==
                  cudaSuccess &&
              cudaMalloc(reinterpret_cast<void**>(&provider.values),
                         24 * sizeof(double)) == cudaSuccess,
          "provider test storage failed");
  for (std::size_t coordinate = 0; coordinate < 6; ++coordinate) {
    input[coordinate] = 0.1 * (coordinate + 1);
    input[6 + coordinate] = -0.2 * (coordinate + 1);
    input[12 + coordinate] = 0.3 * (coordinate + 1);
    input[18 + coordinate] =
        binding.full_range_exchange ? -0.4 * (coordinate + 1) : 0.0;
  }
  require(cudaMemcpyAsync(provider.values, input.data(),
                          input.size() * sizeof(double), cudaMemcpyHostToDevice,
                          provider.stream) == cudaSuccess,
          "provider test upload failed");
  const auto receive = [&](CudaDerivativeSourceKind kind, const double* values,
                           std::size_t channels,
                           CudaDerivativeConvention convention) {
    const CudaDerivativeSourceView view{
        kind, convention, 0, provider.stream, values, 6, channels};
    return binding.consume_integrals(lease.owner, &view, error.data(),
                                     error.size());
  };
  const auto submit = [&] {
    require(
        receive(CudaDerivativeSourceKind::OneElectron, provider.values, 1,
                CudaDerivativeConvention::EnergyGradient) == 0 &&
            receive(CudaDerivativeSourceKind::OverlapPulay, provider.values + 6,
                    1, CudaDerivativeConvention::EnergyGradient) == 0 &&
            receive(CudaDerivativeSourceKind::FullRangeJk, provider.values + 12,
                    2, CudaDerivativeConvention::Force) == 0,
        error.data());
  };
  std::array<std::uint64_t, 24> before{}, after{};
  require(binding.metrics(lease.owner, before.data(), before.size()) == 0,
          "initial native metrics failed");
  submit();
  require(binding.metrics(lease.owner, after.data(), after.size()) == 0 &&
              before[1] == after[1] && before[2] == after[2] &&
              before[12] == after[12] && after[3] == before[3] + 3,
          "native integral receiver allocated transfers or ordinary drains");
  require(binding.nuclear(lease.owner, charges.data(), charges.size(),
                          error.data(), error.size()) == 0,
          error.data());
  std::vector<double> actual(binding.source_count * 6);
  require(
      binding.source_span(lease.owner, 0, binding.source_count, actual.data(),
                          actual.size(), error.data(), error.size()) == 0,
      error.data());
  for (std::size_t source = 0; source < binding.source_count; ++source) {
    const std::string name(binding.source_names[source]);
    for (std::size_t coordinate = 0; coordinate < 6; ++coordinate) {
      double expected = 0;
      if (name == "one_electron") expected = input[coordinate];
      if (name == "overlap_pulay") expected = input[6 + coordinate];
      if (name == "coulomb") expected = -input[12 + coordinate];
      if (name == "exact_exchange") expected = -input[18 + coordinate];
      if (name != "nuclear")
        require(actual[source * 6 + coordinate] == expected,
                "native source slot/sign ownership changed");
    }
  }
  std::vector<double> unpublished(6, 42.0);
  require(binding.finish_gradient(lease.owner, 64, unpublished.data(),
                                  unpublished.size(), error.data(),
                                  error.size()) != 0 &&
              unpublished == std::vector<double>(6, 42.0),
          "missing grid sources partially published a native force");
  reset();
  require(receive(CudaDerivativeSourceKind::OneElectron, provider.values, 1,
                  CudaDerivativeConvention::EnergyGradient) == 0 &&
              receive(CudaDerivativeSourceKind::OneElectron, provider.values, 1,
                      CudaDerivativeConvention::EnergyGradient) != 0,
          "duplicate native source was admitted");
  reset();
  require(receive(CudaDerivativeSourceKind::FullRangeJk, provider.values + 12,
                  1, CudaDerivativeConvention::EnergyGradient) != 0,
          "native J/K convention mismatch was admitted");
  reset();
  require(
      cudaStreamSynchronize(provider.stream) == cudaSuccess &&
          cudaStreamBeginCapture(
              provider.stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
      "native capture fixture failed");
  require(receive(CudaDerivativeSourceKind::OneElectron, provider.values, 1,
                  CudaDerivativeConvention::EnergyGradient) != 0,
          "native receiver admitted a captured producer");
  cudaGraph_t graph{};
  require(cudaStreamEndCapture(provider.stream, &graph) == cudaSuccess && graph,
          "native receiver rejection invalidated producer capture");
  require(cudaGraphDestroy(graph) == cudaSuccess,
          "native capture graph destruction failed");
  reset();
  submit();
  // Destroy the receiver before its producer: a queued transport must complete
  // before the receiver's generated source storage can be reclaimed.
  binding.destroy(lease.owner);
  lease.owner = nullptr;
  std::cout << binding.profile
            << " native source inventory/lifetime gates PASS\n";
}

/** Exercise actual arenas and provider owners, not just a capacity polynomial.
 * The quadrature/SCF owners have separate already-admitted lifetimes; only the
 * additional stationary consumer is charged in this observation. */
void controller_admission(
    const generativeqc::dft::CudaStationaryBinding& binding) {
  using namespace generativeqc;
  core::System system;
  system.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.15, 0.13, 1.5}}};
  system.shells = {{0, 0, {{0.6, 1.0}}}, {1, 0, {{0.6, 1.0}}}};
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) ==
              GENERATIVEQC_STATUS_SUCCESS,
          detail.c_str());
  const dft::AoBasis basis(system);
  const auto grid =
      dft::MolecularGrid::from_cuda(system, {1, 24, 12, 24, 3, 1e-12}, 0);
  const auto prepare = [&](std::size_t capacity) {
    return std::make_unique<dft::CudaStationaryController>(
        system, basis, grid, binding.profile, 0, 64, capacity);
  };
  auto prototype = prepare(128U << 20);
  const auto sizing = prototype->resources();
  prototype.reset();
  const auto minimum = sizing.minimum_device_bytes;
  const auto upper = sizing.device_upper_bytes;
  require(upper > minimum,
          "fixture omitted the optional center-geometry storage");
  std::array<std::uint64_t, 9> queried{};
  const auto query = [&](const char* profile, std::size_t atoms,
                         std::size_t points, std::size_t tile,
                         std::size_t count) {
    return generativeqc_resource_stationary_cuda_v1(
        profile, atoms, basis.nao, basis.nprimitive, points, tile,
        queried.data(), count);
  };
  struct RestoreLedger {
    std::shared_ptr<runtime::DeviceResourceLedger> previous{
        runtime::active_device_resource_ledger};
    ~RestoreLedger() { runtime::active_device_resource_ledger = previous; }
  } restore;
  auto query_ledger = std::make_shared<runtime::DeviceResourceLedger>();
  query_ledger->device = 0;
  runtime::active_device_resource_ledger = query_ledger;
  require(query(binding.profile, basis.natom, grid.point_count(), 64,
                queried.size()) == GENERATIVEQC_STATUS_SUCCESS &&
              queried[0] == sizing.grid_numeric_bytes &&
              queried[1] == sizing.grid_provider_allowance_bytes &&
              queried[4] == minimum && queried[5] == upper &&
              queried[6] == 64 && !query_ledger->live &&
              !query_ledger->provider_reserved && !query_ledger->allocations &&
              !query_ledger->rejected,
          "native shape query diverged from actual controller or allocated "
          "CUDA storage");
  const auto unchanged_query = queried;
  for (const auto status : {
           query("unbuilt_profile", basis.natom, grid.point_count(), 64,
                 queried.size()),
           query(nullptr, basis.natom, grid.point_count(), 64, queried.size()),
           query(binding.profile, 0, grid.point_count(), 64, queried.size()),
           query(binding.profile, basis.natom, grid.point_count() - 1, 64,
                 queried.size()),
           query(binding.profile, basis.natom, grid.point_count(), 0,
                 queried.size()),
           query(binding.profile, basis.natom, grid.point_count(), 64, 8),
       })
    require(status != GENERATIVEQC_STATUS_SUCCESS && queried == unchanged_query,
            "invalid/unbuilt shape query partially published a capacity plan");
  runtime::active_device_resource_ledger.reset();
  for (const auto capacity : {minimum - 1, minimum, upper}) {
    auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
    ledger->device = 0;
    ledger->limit = capacity;
    runtime::active_device_resource_ledger = ledger;
    std::unique_ptr<dft::CudaStationaryController> controller;
    bool rejected = false;
    try {
      controller = prepare(upper);
    } catch (const std::bad_alloc&) {
      rejected = true;
    }
    require(rejected == (capacity < minimum),
            "controller exact/below admission disagrees");
    if (controller) {
      const auto& resources = controller->resources();
      const auto numeric =
          resources.grid_numeric_bytes + resources.stationary_numeric_bytes;
      require(
          resources.device_upper_bytes == capacity && ledger->live == numeric &&
              ledger->peak == numeric &&
              ledger->provider_reserved ==
                  resources.grid_provider_allowance_bytes &&
              ledger->provider_reservation_peak == ledger->provider_reserved &&
              ledger->admission_peak == capacity && ledger->allocations == 2 &&
              ledger->requested_bytes == numeric && !ledger->rejected,
          "real controller arenas/provider allowance do not compose exactly");
    } else {
      // Shape-only preflight never calls CUDA or fabricates an allocator event.
      require(!ledger->live && !ledger->provider_reserved &&
                  !ledger->allocations && !ledger->rejected,
              "below-minimum preflight performed unadmitted allocation work");
    }
    runtime::active_device_resource_ledger.reset();
    controller.reset();
    require(!ledger->live && !ledger->provider_reserved,
            "controller release outside observation retained a charge");
  }
  for (const auto fail : {grid_cuda_fail_next_allocation_for_test_v1,
                          grid_cuda_fail_next_host_allocation_for_test_v1}) {
    auto failed_ledger = std::make_shared<runtime::DeviceResourceLedger>();
    failed_ledger->device = 0;
    failed_ledger->limit = minimum;
    runtime::active_device_resource_ledger = failed_ledger;
    fail();
    bool allocation_failure = false;
    try {
      auto incomplete = prepare(minimum);
    } catch (const std::bad_alloc&) {
      allocation_failure = true;
    }
    require(allocation_failure && !failed_ledger->live &&
                !failed_ledger->provider_reserved &&
                failed_ledger->allocations == 1,
            "partial controller preparation lost OOM typing or retained its "
            "first arena");
    auto recovered = prepare(minimum);
    require(failed_ledger->live + failed_ledger->provider_reserved == minimum,
            "failed controller preparation prevented clean recovery");
    runtime::active_device_resource_ledger.reset();
    recovered.reset();
    require(!failed_ledger->live && !failed_ledger->provider_reserved,
            "recovered controller release retained a charge");
  }
  auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
  ledger->device = 0;
  ledger->limit = 2 * minimum;
  runtime::active_device_resource_ledger = ledger;
  auto first = prepare(minimum);
  auto second = prepare(minimum);
  bool rejected = false;
  try {
    auto third = prepare(minimum);
  } catch (const std::bad_alloc&) {
    rejected = true;
  }
  require(rejected && ledger->allocations == 4 &&
              ledger->admission_peak == 2 * minimum,
          "retained controllers overspent the shared prepared request");
  std::array<std::uint64_t, 4> numeric_v1{};
  std::array<std::uint64_t, 5> numeric_v2{};
  std::array<std::uint64_t, 6> admission{};
  require(
      generativeqc_resource_ledger_read_v1(&ledger, numeric_v1.data()) == 0 &&
          generativeqc_resource_ledger_read_v2(&ledger, numeric_v2.data(),
                                               numeric_v2.size()) == 0 &&
          generativeqc_resource_ledger_read_v3(&ledger, admission.data(),
                                               admission.size()) == 0,
      "ledger ABI receipts failed");
  require(
      std::equal(numeric_v1.begin(), numeric_v1.end(), numeric_v2.begin()) &&
          numeric_v2[0] == ledger->live && numeric_v2[4] == ledger->live &&
          admission ==
              std::array<std::uint64_t, 6>{
                  ledger->limit, ledger->live, ledger->peak,
                  ledger->provider_reserved, ledger->provider_reservation_peak,
                  2 * minimum},
      "v1/v2 numeric receipts were inflated by provider reservations");
  const auto unchanged = admission;
  require(
      generativeqc_resource_ledger_read_v3(&ledger, admission.data(), 5) != 0 &&
          admission == unchanged,
      "short v3 receipt partially published admission data");
  runtime::active_device_resource_ledger.reset();
  first.reset();
  require(ledger->live + ledger->provider_reserved == minimum,
          "one controller released another owner's retained capacity");
  const std::weak_ptr<runtime::DeviceResourceLedger> surviving = ledger;
  ledger.reset();
  require(!surviving.expired(), "ledger died before its retained controller");
  second.reset();
  require(surviving.expired(),
          "released controller retained its observation ledger");
  std::cout << binding.profile
            << " controller numeric/provider admission/lifetime PASS\n";
}
}  // namespace

int main() {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    require(cudaSetDevice(0) == cudaSuccess, "device selection failed");
    bool available = false;
    for (const auto* profile : {"pbe_rks", "pbe_uks", "pbe0_rks", "pbe0_uks"}) {
      const auto* binding = generativeqc::dft::cuda_stationary_binding(profile);
      if (!binding) continue;
      available = true;
      profile_sources(*binding);
      controller_admission(*binding);
    }
    require(!generativeqc::dft::cuda_stationary_binding("unqualified_profile"),
            "unbuilt native profile was admitted");
    return available ? 0 : 77;
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
