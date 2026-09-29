#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "api/handles.hpp"
#include "integrals/ecp_cuda.hpp"
#include "runtime/resource_cuda.cuh"

// Private resource-observation ABI used by prepared Python requests as well.
extern "C" {
void* generativeqc_resource_ledger_create_v1(std::size_t bytes, int device);
void generativeqc_resource_ledger_destroy_v1(void* handle);
int generativeqc_resource_ledger_bind_v1(void* handle);
int generativeqc_resource_ledger_read_v1(void* handle, std::uint64_t* values);
int generativeqc_resource_tracking_begin_v1(unsigned workers);
int generativeqc_resource_tracking_end_v1(std::uint64_t* peak, std::uint64_t* samples);
}

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Observation {
  std::unique_ptr<void, decltype(&generativeqc_resource_ledger_destroy_v1)> ledger{
      nullptr, generativeqc_resource_ledger_destroy_v1};
  Observation(std::size_t bytes, int device)
      : ledger(generativeqc_resource_ledger_create_v1(bytes, device),
               generativeqc_resource_ledger_destroy_v1) {
    require(ledger != nullptr, "ledger creation failed");
    require(generativeqc_resource_tracking_begin_v1(0) == 0, "observation failed");
    if (generativeqc_resource_ledger_bind_v1(ledger.get()) != 0) {
      std::uint64_t peak{}, samples{};
      generativeqc_resource_tracking_end_v1(&peak, &samples);
      throw std::runtime_error("ledger binding failed");
    }
  }
  ~Observation() {
    std::uint64_t peak{}, samples{};
    generativeqc_resource_tracking_end_v1(&peak, &samples);
  }
  std::array<std::uint64_t, 4> read() const {
    std::array<std::uint64_t, 4> values{};
    require(generativeqc_resource_ledger_read_v1(ledger.get(), values.data()) == 0,
            "ledger read failed");
    return values;
  }
};

void allocation_failure_origin() {
  // CUDA status alone cannot distinguish host registry OOM. Synthetic allocator
  // results exercise both origins without exhausting host or device memory.
  void* pointer = nullptr;
  bool host_oom = true;
  auto device_oom = [] { return cudaErrorMemoryAllocation; };
  auto unused_release = [] { return cudaSuccess; };
  require(generativeqc::runtime::resource_cuda_allocate(&pointer, 8, device_oom, unused_release,
                                                        &host_oom) == cudaErrorMemoryAllocation &&
              !host_oom,
          "untracked device OOM misclassified as host failure");
  {
    Observation observation(8, 0);
    require(generativeqc::runtime::resource_cuda_allocate(&pointer, 8, device_oom, unused_release,
                                                          &host_oom) == cudaErrorMemoryAllocation &&
                !host_oom,
            "tracked device OOM misclassified as host failure");
    require(observation.read()[0] == 0 && observation.read()[3] == 1,
            "device OOM lost rejection or reservation cleanup");
  }
  {
    Observation observation(8, 0);
    // Generation exhaustion takes the same bad_alloc path as registry insertion
    // failure. Restore it even on assertion failure so subsequent tests recover.
    struct RestoreGeneration {
      std::uint64_t value = generativeqc::runtime::device_allocation_generation;
      ~RestoreGeneration() { generativeqc::runtime::device_allocation_generation = value; }
    } restore;
    generativeqc::runtime::device_allocation_generation = std::numeric_limits<std::uint64_t>::max();
    double storage{};
    int releases = 0;
    auto allocate = [&] {
      pointer = &storage;
      return cudaSuccess;
    };
    auto release = [&] {
      ++releases;
      return cudaSuccess;
    };
    require(generativeqc::runtime::resource_cuda_allocate(&pointer, 8, allocate, release,
                                                          &host_oom) == cudaErrorMemoryAllocation &&
                host_oom,
            "host registry OOM incorrectly authorizes device fallback");
    require(pointer == nullptr && releases == 1, "registry failure leaked allocation");
    require(observation.read()[0] == 0 && observation.read()[3] == 0,
            "registry failure leaked reservation or reported device rejection");
  }
}

void errors_and_recovery() {
  const generativeqc_context_descriptor context_desc{sizeof(context_desc), GENERATIVEQC_ABI_VERSION,
                                                     0, GENERATIVEQC_BACKEND_CUDA};
  generativeqc_context* raw_context{};
  require(generativeqc_context_create(&context_desc, &raw_context) == GENERATIVEQC_STATUS_SUCCESS,
          "CUDA context creation failed");
  std::unique_ptr<generativeqc_context, decltype(&generativeqc_context_destroy)> context(
      raw_context, generativeqc_context_destroy);
  const generativeqc_atom atom{11, 0, 0, 0};
  const generativeqc_primitive primitive{0.7, 1.0};
  const generativeqc_shell shell{0, 0, 0, 1};
  const generativeqc_system_descriptor descriptor{sizeof(descriptor),
                                                  GENERATIVEQC_ABI_VERSION,
                                                  &atom,
                                                  1,
                                                  &shell,
                                                  1,
                                                  &primitive,
                                                  1,
                                                  0,
                                                  2,
                                                  GENERATIVEQC_BASIS_CARTESIAN};
  const int32_t core = 10;
  const generativeqc_ecp_term term{0, -1, 2, 0.8, -2.0};
  generativeqc_system* raw_system{};
  require(generativeqc_system_create_ecp(context.get(), &descriptor, &core, &term, 1,
                                         &raw_system) == GENERATIVEQC_STATUS_SUCCESS,
          "ECP system creation failed");
  std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)> system(
      raw_system, generativeqc_system_destroy);

  int device_count = 0;
  require(cudaGetDeviceCount(&device_count) == cudaSuccess, "device inventory failed");
  // A multi-device allocation exercises restoration to a distinct caller
  // device. A single-device allocation still checks every success/error exit.
  const int caller_device = device_count > 1 ? 1 : 0;
  for (bool device_consumer : {false, true}) {
    std::array<double, 8> output;
    output.fill(123.0);
    std::string detail;
    const auto execute = [&] {
      require(cudaSetDevice(caller_device) == cudaSuccess, "caller device selection failed");
      const auto status =
          device_consumer ? generativeqc::integrals::add_ecp_cuda(0, system->data, nullptr, nullptr,
                                                                  nullptr, nullptr, detail)
                          : generativeqc_system_ecp_integrals(context.get(), system.get(), 160, 32,
                                                              1, output.data(), output.size());
      int after = -1;
      require(cudaGetDevice(&after) == cudaSuccess && after == caller_device,
              "ECP entry point changed the caller's current CUDA device");
      if (!device_consumer) detail = generativeqc_context_get_last_detail(context.get());
      return status;
    };
    {
      // Enough for several real allocations, but not the angular grid. This
      // deterministic budget rejection exercises unwinding after partial upload.
      Observation observation(256, 0);
      require(execute() == GENERATIVEQC_STATUS_OUT_OF_MEMORY, "CUDA OOM lost its status");
      require(!detail.empty(), "OOM detail missing");
      const auto values = observation.read();
      require(values[0] == 0 && values[1] > 0 && values[2] > 1 && values[3] > 0,
              "partial ECP allocation leaked or rejection was not exercised");
      require(std::all_of(output.begin(), output.end(), [](double x) { return x == 123.0; }),
              "failed C ABI call published partial output");
    }
    {
      // Device mismatch is a non-OOM CUDA error without poisoning the GPU.
      Observation observation(8 << 20, 1);
      require(execute() == GENERATIVEQC_STATUS_CUDA_ERROR, "CUDA runtime error lost its status");
      require(!detail.empty(), "CUDA error detail missing");
      require(observation.read()[0] == 0, "CUDA error leaked device allocations");
    }
    {
      // One radial layer fits, four do not. Optional batching must fall back
      // without publishing partial output, leaking, or masking non-OOM errors.
      const std::size_t polar = device_consumer ? 44 : 32;
      const auto budget =
          2 * polar * polar *
              (sizeof(generativeqc::integrals::EcpSpherePoint) + 4 * sizeof(double)) +
          (16 << 10);
      Observation observation(budget, 0);
      require(execute() == GENERATIVEQC_STATUS_SUCCESS, "single-layer OOM fallback failed");
      const auto values = observation.read();
      require(values[0] == 0 && values[3] > 0, "radial fallback was not exercised or leaked");
    }
    {
      Observation observation(8 << 20, 0);
      require(execute() == GENERATIVEQC_STATUS_SUCCESS, "ECP execution did not recover");
      const auto values = observation.read();
      require(values[0] == 0 && values[2] > 1 && values[3] == 0,
              "successful ECP execution leaked device allocations");
      if (!device_consumer)
        require(std::isfinite(output[0]) && output[0] < 0,
                "recovered execution did not publish ECP output");
    }
  }

  std::string detail;
  double force{};
  require(generativeqc::integrals::add_ecp_cuda(0, system->data, nullptr, nullptr, nullptr, &force,
                                                detail) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "non-CUDA argument error changed category");
  system->data.ecp_terms[0].coefficient = std::numeric_limits<double>::infinity();
  Observation observation(8 << 20, 0);
  require(generativeqc::integrals::add_ecp_cuda(0, system->data, nullptr, nullptr, nullptr, nullptr,
                                                detail) == GENERATIVEQC_STATUS_NUMERICAL_FAILURE,
          "non-CUDA convergence error changed category");
  require(observation.read()[0] == 0, "numerical failure leaked allocations");
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "CUDA device unavailable\n";
    return 77;
  }
  try {
    allocation_failure_origin();
    errors_and_recovery();
    std::cout << "ECP CUDA error categories, partial cleanup and recovery PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
