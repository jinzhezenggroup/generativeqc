// Validation-only interposer: fail owned tensor allocations after VRAM preflight,
// without consuming shared GPU memory or changing the production allocator.
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <cstdlib>
#include <cstring>

extern "C" cudaError_t cudaMalloc(void** output, std::size_t bytes) {
  if (const auto* limit = std::getenv("GENERATIVEQC_TEST_TRIPLES_DEVICE_LIMIT")) {
    Dl_info caller{};
    if (dladdr(__builtin_return_address(0), &caller) && caller.dli_fname &&
        std::strstr(caller.dli_fname, "libgenerativeqc") &&
        bytes > std::strtoull(limit, nullptr, 10)) {
      *output = nullptr;
      return cudaErrorMemoryAllocation;
    }
  }
  using Allocate = cudaError_t (*)(void**, std::size_t);
  // ctypes loads the native library locally, so RTLD_NEXT cannot see its CUDA
  // dependency from this global interposer. Bind the already loaded vendor DSO.
  static const auto allocate = [] {
    const auto runtime = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
    return runtime ? reinterpret_cast<Allocate>(dlsym(runtime, "cudaMalloc")) : nullptr;
  }();
  return allocate ? allocate(output, bytes) : cudaErrorUnknown;
}
