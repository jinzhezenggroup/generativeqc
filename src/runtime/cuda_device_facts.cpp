#include "runtime/cuda_device_facts.hpp"

#include <cstring>
#include <mutex>
#include <utility>

#include "runtime/cuda_provider_identity.hpp"
#include "runtime/loaded_provider_identity.hpp"

#if defined(__linux__) && !GENERATIVEQC_CUDA_PROVIDER_CUMETAL
#include <cuda.h>
#include <dlfcn.h>
#endif

namespace generativeqc::runtime {

CudaProviderIdentityObservation observe_cuda_provider_identity(
    std::uint8_t providers, const std::array<const void*, 3>& execution_symbols) {
  CudaProviderIdentityObservation result;
#if defined(__linux__) && !GENERATIVEQC_CUDA_PROVIDER_CUMETAL
  if (providers > 3) return result;
  constexpr const char* libraries[]{"libcudart.so.12", "libcublas.so.12", "libcusolver.so.11",
                                    "libcuda.so.1"};
  constexpr const char* symbols[]{"cudaRuntimeGetVersion", "cublasCreate_v2", "cusolverDnCreate",
                                  "cuCtxGetId"};
  CudaProviderIdentity identity;
  identity.provider = CudaProviderKind::Nvidia;
  identity.providers = providers;
  for (unsigned index = 0; index != 4; ++index) {
    if ((index == 1 && !(providers & 1)) || (index == 2 && !(providers & 2))) continue;
    if (index != 3 && !execution_symbols[index]) return result;
    auto loaded = loaded_provider_identity(libraries[index], symbols[index],
                                           index == 3 ? nullptr : execution_symbols[index]);
    result.loaded_status = loaded.status;
    if (loaded.status != LoadedIdentityStatus::available) return result;
    identity.build_ids[index] = std::move(loaded.build_id);
    result.library_pins[index] = std::move(loaded.library_pin);
  }
  // Compile against the selected SDK's real declarations, but keep these driver
  // capabilities optional/lazy. Taking a decltype creates no static driver link.
  struct Queries {
    decltype(&cuCtxGetCurrent) current{};
    decltype(&cuCtxGetId) id{};
    decltype(&cuCtxGetDevice) device{};
    decltype(&cuDeviceGetUuid_v2) uuid{};
    decltype(&cuDriverGetVersion) driver_version{};
    decltype(&cudaRuntimeGetVersion) runtime_version{};
  } query;
  void* driver = result.library_pins[3].get();
  query.current = reinterpret_cast<decltype(query.current)>(dlsym(driver, "cuCtxGetCurrent"));
  query.id = reinterpret_cast<decltype(query.id)>(dlsym(driver, "cuCtxGetId"));
  query.device = reinterpret_cast<decltype(query.device)>(dlsym(driver, "cuCtxGetDevice"));
  query.uuid = reinterpret_cast<decltype(query.uuid)>(dlsym(driver, "cuDeviceGetUuid_v2"));
  query.driver_version =
      reinterpret_cast<decltype(query.driver_version)>(dlsym(driver, "cuDriverGetVersion"));
  query.runtime_version = reinterpret_cast<decltype(query.runtime_version)>(
      dlsym(result.library_pins[0].get(), "cudaRuntimeGetVersion"));
  if (!query.current || !query.id || !query.device || !query.uuid || !query.driver_version ||
      !query.runtime_version) {
    result.loaded_status = LoadedIdentityStatus::missing_symbol;
    return result;
  }
  const std::pair<const char*, const void*> driver_bindings[]{
      {"cuCtxGetCurrent", reinterpret_cast<const void*>(query.current)},
      {"cuCtxGetId", reinterpret_cast<const void*>(query.id)},
      {"cuCtxGetDevice", reinterpret_cast<const void*>(query.device)},
      {"cuDeviceGetUuid_v2", reinterpret_cast<const void*>(query.uuid)},
      {"cuDriverGetVersion", reinterpret_cast<const void*>(query.driver_version)}};
  for (const auto& [name, address] : driver_bindings) {
    const auto owner = loaded_provider_identity("libcuda.so.1", name, address);
    if (owner.status != LoadedIdentityStatus::available || owner.library_pin.get() != driver ||
        owner.build_id != *identity.build_ids[3]) {
      result.loaded_status = owner.status == LoadedIdentityStatus::available
                                 ? LoadedIdentityStatus::wrong_symbol_owner
                                 : owner.status;
      return result;
    }
  }
  const CurrentContextQueries access{
      &query,
      [](void* state, std::uintptr_t& context) {
        CUcontext value{};
        const auto status = static_cast<Queries*>(state)->current(&value);
        context = reinterpret_cast<std::uintptr_t>(value);
        return static_cast<std::uint32_t>(status);
      },
      [](void* state, std::uintptr_t context, std::uint64_t& id) {
        unsigned long long value{};
        const auto status =
            static_cast<Queries*>(state)->id(reinterpret_cast<CUcontext>(context), &value);
        id = value;
        return static_cast<std::uint32_t>(status);
      },
      [](void* state, int& device) {
        CUdevice value{};
        const auto status = static_cast<Queries*>(state)->device(&value);
        device = value;
        return static_cast<std::uint32_t>(status);
      },
      [](void* state, int device, std::array<unsigned char, 16>& uuid) {
        CUuuid value{};
        const auto status = static_cast<Queries*>(state)->uuid(&value, device);
        std::memcpy(uuid.data(), value.bytes, uuid.size());
        return static_cast<std::uint32_t>(status);
      },
      [](void* state, int& version) {
        return static_cast<std::uint32_t>(static_cast<Queries*>(state)->driver_version(&version));
      },
      [](void* state, int& version) {
        return static_cast<std::uint32_t>(static_cast<Queries*>(state)->runtime_version(&version));
      }};
  const auto before = observe_current_context(access);
  result.status = before.status;
  result.error_domain = before.error_domain;
  result.error_status = before.error_status;
  if (!before.identity) return result;
  const auto after = observe_current_context(access);
  result.status = after.status;
  result.error_domain = after.error_domain;
  result.error_status = after.error_status;
  if (!after.identity) return result;
  if (*before.identity != *after.identity) {
    result.status = ContextIdentityStatus::changed;
    return result;
  }
  identity.context = *after.identity;
  result.identity = std::move(identity);
#else
  (void)providers;
  (void)execution_symbols;
#endif
  return result;
}

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

cudaError_t cuda_ensure_stack_limit(std::size_t minimum_bytes) {
  // A dedicated lock must not hold the allocation-ledger mutex across driver
  // calls, whose callbacks may retire registered resources. Context creation
  // is infrequent; serializing across devices avoids a stale ordinal cache.
  static std::mutex limit_mutex;
  std::lock_guard<std::mutex> lock(limit_mutex);
  std::size_t current{};
  const auto status = cudaDeviceGetLimit(&current, cudaLimitStackSize);
  if (status != cudaSuccess || current >= minimum_bytes) return status;
  return cudaDeviceSetLimit(cudaLimitStackSize, minimum_bytes);
}

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
