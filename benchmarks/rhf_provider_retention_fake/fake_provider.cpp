/** Independent fake CUDA/cuBLAS/cuSOLVER provider for the RHF observer.
 *
 * Every intercepted symbol has the production import version and a deliberately
 * different default version. The versioned implementations validate arguments,
 * retain deterministic output/status state, and expose a compact audit snapshot.
 * The default implementations return different sentinels so an unversioned
 * RTLD_NEXT lookup cannot accidentally pass the comparison.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <dlfcn.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

enum Target : std::size_t {
  kCudaMemGetInfo,
  kCublasCreate,
  kCublasSetStream,
  kCublasSetPointerMode,
  kCusolverCreate,
  kCusolverSetStream,
  kCusolverCreateParams,
  kTargetCount,
};

std::array<std::atomic<unsigned>, kTargetCount> expected_calls{};
std::array<std::atomic<unsigned>, kTargetCount> default_calls{};
std::atomic<unsigned> argument_failures{};

constexpr std::uintptr_t kCublasHandle = 0xC0B1A5U;
constexpr std::uintptr_t kDefaultCublasHandle = 0xD0B1A5U;
constexpr std::uintptr_t kCusolverHandle = 0xC0501U;
constexpr std::uintptr_t kDefaultCusolverHandle = 0xD0501U;
constexpr std::uintptr_t kCusolverParams = 0xCA4AU;
constexpr std::uintptr_t kDefaultCusolverParams = 0xDA4AU;
constexpr std::uintptr_t kStream = 0x5157EAU;

template <class Pointer>
Pointer pointer(std::uintptr_t value) noexcept {
  return reinterpret_cast<Pointer>(value);
}

bool provider_failure_requested() noexcept {
  const char* value = std::getenv("FAKE_PROVIDER_CASE");
  return value && std::strcmp(value, "provider_failure") == 0;
}

void bad_argument() noexcept { argument_failures.fetch_add(1, std::memory_order_relaxed); }

}  // namespace

extern "C" cudaError_t fake_cuda_mem_get_info_expected(std::size_t* free_bytes,
                                                         std::size_t* total_bytes) {
  const unsigned call =
      expected_calls[kCudaMemGetInfo].fetch_add(1, std::memory_order_relaxed) + 1;
  if (!free_bytes || !total_bytes) {
    bad_argument();
    return cudaErrorInvalidValue;
  }
  *total_bytes = 2000;
  *free_bytes = call == 1 ? 1000 : (call == 2 ? 950 : 800);
  return cudaSuccess;
}

extern "C" cudaError_t fake_cuda_mem_get_info_default(std::size_t* free_bytes,
                                                        std::size_t* total_bytes) {
  default_calls[kCudaMemGetInfo].fetch_add(1, std::memory_order_relaxed);
  if (!free_bytes || !total_bytes) {
    bad_argument();
    return cudaErrorInvalidValue;
  }
  *free_bytes = 111;
  *total_bytes = 222;
  return cudaSuccess;
}

extern "C" cublasStatus_t fake_cublas_create_expected(cublasHandle_t* handle) {
  expected_calls[kCublasCreate].fetch_add(1, std::memory_order_relaxed);
  if (!handle) {
    bad_argument();
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  using MemInfo = cudaError_t (*)(std::size_t*, std::size_t*);
  auto nested = reinterpret_cast<MemInfo>(
      ::dlvsym(RTLD_DEFAULT, "cudaMemGetInfo", "libcudart.so.12"));
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (!nested || nested(&free_bytes, &total_bytes) != cudaSuccess || free_bytes != 950 ||
      total_bytes != 2000) {
    bad_argument();
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
  *handle = pointer<cublasHandle_t>(kCublasHandle);
  return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t fake_cublas_create_default(cublasHandle_t* handle) {
  default_calls[kCublasCreate].fetch_add(1, std::memory_order_relaxed);
  if (!handle) {
    bad_argument();
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  *handle = pointer<cublasHandle_t>(kDefaultCublasHandle);
  return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t fake_cublas_set_stream_expected(cublasHandle_t handle,
                                                            cudaStream_t stream) {
  expected_calls[kCublasSetStream].fetch_add(1, std::memory_order_relaxed);
  if (handle != pointer<cublasHandle_t>(kCublasHandle) ||
      stream != pointer<cudaStream_t>(kStream)) {
    bad_argument();
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t fake_cublas_set_stream_default(cublasHandle_t handle,
                                                           cudaStream_t stream) {
  default_calls[kCublasSetStream].fetch_add(1, std::memory_order_relaxed);
  if (handle != pointer<cublasHandle_t>(kDefaultCublasHandle) ||
      stream != pointer<cudaStream_t>(kStream)) {
    bad_argument();
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t fake_cublas_set_pointer_mode_expected(
    cublasHandle_t handle, cublasPointerMode_t mode) {
  expected_calls[kCublasSetPointerMode].fetch_add(1, std::memory_order_relaxed);
  if (handle != pointer<cublasHandle_t>(kCublasHandle) ||
      mode != CUBLAS_POINTER_MODE_DEVICE) {
    bad_argument();
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return provider_failure_requested() ? CUBLAS_STATUS_EXECUTION_FAILED
                                      : CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t fake_cublas_set_pointer_mode_default(
    cublasHandle_t handle, cublasPointerMode_t mode) {
  default_calls[kCublasSetPointerMode].fetch_add(1, std::memory_order_relaxed);
  if (handle != pointer<cublasHandle_t>(kDefaultCublasHandle) ||
      mode != CUBLAS_POINTER_MODE_DEVICE) {
    bad_argument();
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return provider_failure_requested() ? CUBLAS_STATUS_EXECUTION_FAILED
                                      : CUBLAS_STATUS_SUCCESS;
}

extern "C" cusolverStatus_t fake_cusolver_create_expected(cusolverDnHandle_t* handle) {
  expected_calls[kCusolverCreate].fetch_add(1, std::memory_order_relaxed);
  if (!handle) {
    bad_argument();
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  *handle = pointer<cusolverDnHandle_t>(kCusolverHandle);
  return CUSOLVER_STATUS_SUCCESS;
}

extern "C" cusolverStatus_t fake_cusolver_create_default(cusolverDnHandle_t* handle) {
  default_calls[kCusolverCreate].fetch_add(1, std::memory_order_relaxed);
  if (!handle) {
    bad_argument();
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  *handle = pointer<cusolverDnHandle_t>(kDefaultCusolverHandle);
  return CUSOLVER_STATUS_SUCCESS;
}

extern "C" cusolverStatus_t fake_cusolver_set_stream_expected(cusolverDnHandle_t handle,
                                                                cudaStream_t stream) {
  expected_calls[kCusolverSetStream].fetch_add(1, std::memory_order_relaxed);
  if (handle != pointer<cusolverDnHandle_t>(kCusolverHandle) ||
      stream != pointer<cudaStream_t>(kStream)) {
    bad_argument();
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  return CUSOLVER_STATUS_SUCCESS;
}

extern "C" cusolverStatus_t fake_cusolver_set_stream_default(cusolverDnHandle_t handle,
                                                               cudaStream_t stream) {
  default_calls[kCusolverSetStream].fetch_add(1, std::memory_order_relaxed);
  if (handle != pointer<cusolverDnHandle_t>(kDefaultCusolverHandle) ||
      stream != pointer<cudaStream_t>(kStream)) {
    bad_argument();
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  return CUSOLVER_STATUS_SUCCESS;
}

extern "C" cusolverStatus_t fake_cusolver_create_params_expected(
    cusolverDnParams_t* parameters) {
  expected_calls[kCusolverCreateParams].fetch_add(1, std::memory_order_relaxed);
  if (!parameters) {
    bad_argument();
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  *parameters = pointer<cusolverDnParams_t>(kCusolverParams);
  return CUSOLVER_STATUS_SUCCESS;
}

extern "C" cusolverStatus_t fake_cusolver_create_params_default(
    cusolverDnParams_t* parameters) {
  default_calls[kCusolverCreateParams].fetch_add(1, std::memory_order_relaxed);
  if (!parameters) {
    bad_argument();
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  *parameters = pointer<cusolverDnParams_t>(kDefaultCusolverParams);
  return CUSOLVER_STATUS_SUCCESS;
}

extern "C" void fake_provider_reset() {
  for (auto& count : expected_calls) count.store(0, std::memory_order_relaxed);
  for (auto& count : default_calls) count.store(0, std::memory_order_relaxed);
  argument_failures.store(0, std::memory_order_relaxed);
}

extern "C" const char* fake_provider_snapshot() {
  static thread_local char output[768]{};
  std::snprintf(
      output, sizeof(output),
      "{\"expected_calls\":[%u,%u,%u,%u,%u,%u,%u],"
      "\"default_calls\":[%u,%u,%u,%u,%u,%u,%u],\"argument_failures\":%u}",
      expected_calls[0].load(std::memory_order_relaxed),
      expected_calls[1].load(std::memory_order_relaxed),
      expected_calls[2].load(std::memory_order_relaxed),
      expected_calls[3].load(std::memory_order_relaxed),
      expected_calls[4].load(std::memory_order_relaxed),
      expected_calls[5].load(std::memory_order_relaxed),
      expected_calls[6].load(std::memory_order_relaxed),
      default_calls[0].load(std::memory_order_relaxed),
      default_calls[1].load(std::memory_order_relaxed),
      default_calls[2].load(std::memory_order_relaxed),
      default_calls[3].load(std::memory_order_relaxed),
      default_calls[4].load(std::memory_order_relaxed),
      default_calls[5].load(std::memory_order_relaxed),
      default_calls[6].load(std::memory_order_relaxed),
      argument_failures.load(std::memory_order_relaxed));
  return output;
}

__asm__(".symver fake_cuda_mem_get_info_expected,cudaMemGetInfo@libcudart.so.12");
__asm__(".symver fake_cuda_mem_get_info_default,cudaMemGetInfo@@FAKE_PROVIDER_9.9");
__asm__(".symver fake_cublas_create_expected,cublasCreate_v2@libcublas.so.12");
__asm__(".symver fake_cublas_create_default,cublasCreate_v2@@FAKE_PROVIDER_9.9");
__asm__(".symver fake_cublas_set_stream_expected,cublasSetStream_v2@libcublas.so.12");
__asm__(".symver fake_cublas_set_stream_default,cublasSetStream_v2@@FAKE_PROVIDER_9.9");
__asm__(
    ".symver fake_cublas_set_pointer_mode_expected,cublasSetPointerMode_v2@libcublas.so.12");
__asm__(
    ".symver fake_cublas_set_pointer_mode_default,cublasSetPointerMode_v2@@FAKE_PROVIDER_9.9");
__asm__(".symver fake_cusolver_create_expected,cusolverDnCreate@libcusolver.so.11");
__asm__(".symver fake_cusolver_create_default,cusolverDnCreate@@FAKE_PROVIDER_9.9");
__asm__(".symver fake_cusolver_set_stream_expected,cusolverDnSetStream@libcusolver.so.11");
__asm__(".symver fake_cusolver_set_stream_default,cusolverDnSetStream@@FAKE_PROVIDER_9.9");
__asm__(
    ".symver fake_cusolver_create_params_expected,cusolverDnCreateParams@libcusolver.so.11");
__asm__(
    ".symver fake_cusolver_create_params_default,cusolverDnCreateParams@@FAKE_PROVIDER_9.9");
