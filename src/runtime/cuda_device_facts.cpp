#include "runtime/cuda_device_facts.hpp"

#include <cstring>

#if defined(__linux__) && !GENERATIVEQC_CUDA_PROVIDER_CUMETAL
#include <dlfcn.h>
#endif

namespace generativeqc::runtime {
namespace {

#if defined(__linux__) && !GENERATIVEQC_CUDA_PROVIDER_CUMETAL
// These optional, stable driver ABI functions supply the two facts absent
// from cudaDeviceGetAttribute. Retain only their library/function lifetime,
// never a device handle, ordinal mapping, memory size or capability record.
struct DriverMetadata {
  int (*device)(int*, int) = nullptr;
  int (*name)(char*, int, int) = nullptr;
  int (*memory)(std::size_t*, int) = nullptr;
};

const DriverMetadata& driver_metadata() noexcept {
  static const DriverMetadata functions = [] {
    DriverMetadata result;
    // Attribute queries have already initialized the selected CUDA provider.
    // Do not introduce an NVIDIA driver into another provider's process. Keep
    // this reference alive with the function pointers until process exit.
    void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (library) {
      result.device = reinterpret_cast<decltype(result.device)>(dlsym(library, "cuDeviceGet"));
      result.name = reinterpret_cast<decltype(result.name)>(dlsym(library, "cuDeviceGetName"));
      result.memory =
          reinterpret_cast<decltype(result.memory)>(dlsym(library, "cuDeviceTotalMem_v2"));
    }
    return result;
  }();
  return functions;
}

cudaError_t selected_facts(int device, CudaTargetInfo& target, char (&name)[256],
                           bool& qualified) noexcept {
  // A fallback may consume only its own unsupported-attribute diagnostic.
  // Preserve an earlier runtime/launch failure before issuing a new probe.
  const auto pending = cudaPeekAtLastError();
  if (pending != cudaSuccess) return pending;
  const cudaDeviceAttr attributes[] = {cudaDevAttrComputeCapabilityMajor,
                                       cudaDevAttrComputeCapabilityMinor,
                                       cudaDevAttrWarpSize,
                                       cudaDevAttrMaxThreadsPerBlock,
                                       cudaDevAttrMaxThreadsPerMultiProcessor,
                                       cudaDevAttrMaxBlocksPerMultiprocessor,
                                       cudaDevAttrMaxRegistersPerMultiprocessor,
                                       cudaDevAttrMaxSharedMemoryPerBlock,
                                       cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                       cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                       cudaDevAttrMultiProcessorCount};
  int values[11]{};
  for (unsigned i = 0; i < 11; ++i) {
    const auto error = cudaDeviceGetAttribute(&values[i], attributes[i], device);
    if (error == cudaErrorInvalidValue || error == cudaErrorNotSupported) {
      // Successful property queries do not reset the runtime's last-error slot.
      // Clear the handled probe error so it cannot poison a later launch check,
      // but propagate any different asynchronous failure observed during cleanup.
      const auto cleared = cudaGetLastError();
      return cleared == cudaSuccess || cleared == error ? cudaSuccess : cleared;
    }
    if (error != cudaSuccess) return error;
  }
  const auto& driver = driver_metadata();
  int driver_device = 0;
  std::size_t memory = 0;
  if (!driver.device || !driver.name || !driver.memory || driver.device(&driver_device, device) ||
      driver.name(name, sizeof(name), driver_device) || driver.memory(&memory, driver_device))
    return cudaSuccess;
  target.compute_capability_major = values[0];
  target.compute_capability_minor = values[1];
  target.warp_size = static_cast<unsigned>(values[2]);
  target.maximum_threads_per_block = static_cast<unsigned>(values[3]);
  target.maximum_threads_per_sm = static_cast<unsigned>(values[4]);
  target.maximum_blocks_per_sm = static_cast<unsigned>(values[5]);
  target.registers_per_sm = static_cast<std::size_t>(values[6]);
  target.shared_memory_per_block = static_cast<std::size_t>(values[7]);
  target.shared_memory_per_block_optin = static_cast<std::size_t>(values[8]);
  target.shared_memory_per_sm = static_cast<std::size_t>(values[9]);
  target.multiprocessor_count = static_cast<unsigned>(values[10]);
  target.total_global_memory = memory;
  qualified = true;
  return cudaSuccess;
}
#else
cudaError_t selected_facts(int, CudaTargetInfo&, char (&)[256], bool&) noexcept {
  return cudaSuccess;
}
#endif

}  // namespace

cudaError_t cuda_device_facts(int device, CudaTargetInfo& target, char (&name)[256]) noexcept {
  target = {};
  std::memset(name, 0, sizeof(name));
  CudaTargetInfo candidate{};
  char candidate_name[256]{};
  bool qualified = false;
  const auto status = selected_facts(device, candidate, candidate_name, qualified);
  if (status != cudaSuccess) return status;
  if (!qualified) {
    // Unsupported attributes and unavailable driver metadata
    // all retain the provider's complete, authoritative property query.
    cudaDeviceProp properties{};
    const auto error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return error;
    candidate = cuda_target_info_from_properties(properties);
    std::memcpy(candidate_name, properties.name, sizeof(candidate_name));
  }
  candidate_name[sizeof(candidate_name) - 1] = '\0';
  target = candidate;
  std::memcpy(name, candidate_name, sizeof(name));
  return cudaSuccess;
}

}  // namespace generativeqc::runtime
