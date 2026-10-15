/** Version-pinned fake provider driver.
 *
 * The undefined references below carry the same versions observed in the a308
 * production library. Stable JSON output makes the baseline/interposed result a
 * byte-for-byte forwarding oracle without using a real CUDA device.
 */

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" cudaError_t import_cuda_mem_get_info(std::size_t*, std::size_t*);
extern "C" cublasStatus_t import_cublas_create(cublasHandle_t*);
extern "C" cublasStatus_t import_cublas_set_stream(cublasHandle_t, cudaStream_t);
extern "C" cublasStatus_t import_cublas_set_pointer_mode(cublasHandle_t,
                                                          cublasPointerMode_t);
extern "C" cusolverStatus_t import_cusolver_create(cusolverDnHandle_t*);
extern "C" cusolverStatus_t import_cusolver_set_stream(cusolverDnHandle_t,
                                                        cudaStream_t);
extern "C" cusolverStatus_t import_cusolver_create_params(cusolverDnParams_t*);
extern "C" void fake_provider_reset();
extern "C" const char* fake_provider_snapshot();

__asm__(".symver import_cuda_mem_get_info,cudaMemGetInfo@libcudart.so.12");
__asm__(".symver import_cublas_create,cublasCreate_v2@libcublas.so.12");
__asm__(".symver import_cublas_set_stream,cublasSetStream_v2@libcublas.so.12");
__asm__(
    ".symver import_cublas_set_pointer_mode,cublasSetPointerMode_v2@libcublas.so.12");
__asm__(".symver import_cusolver_create,cusolverDnCreate@libcusolver.so.11");
__asm__(".symver import_cusolver_set_stream,cusolverDnSetStream@libcusolver.so.11");
__asm__(
    ".symver import_cusolver_create_params,cusolverDnCreateParams@libcusolver.so.11");

namespace {

constexpr std::uintptr_t kStream = 0x5157EAU;

template <class Pointer>
unsigned long long address(Pointer value) noexcept {
  return static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(value));
}

int run_provider_case(const char* name) {
  fake_provider_reset();
  std::size_t before_free = 0;
  std::size_t before_total = 0;
  std::size_t after_free = 0;
  std::size_t after_total = 0;
  cublasHandle_t cublas_handle{};
  cusolverDnHandle_t cusolver_handle{};
  cusolverDnParams_t cusolver_parameters{};
  const auto stream = reinterpret_cast<cudaStream_t>(kStream);

  const auto before_status = import_cuda_mem_get_info(&before_free, &before_total);
  const auto cublas_create_status = import_cublas_create(&cublas_handle);
  const auto cublas_stream_status = import_cublas_set_stream(cublas_handle, stream);
  const auto cublas_pointer_status =
      import_cublas_set_pointer_mode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
  const auto cusolver_create_status = import_cusolver_create(&cusolver_handle);
  const auto cusolver_stream_status = import_cusolver_set_stream(cusolver_handle, stream);
  const auto cusolver_params_status = import_cusolver_create_params(&cusolver_parameters);
  const auto after_status = import_cuda_mem_get_info(&after_free, &after_total);

  std::printf(
      "{\"case\":\"%s\",\"before_status\":%d,\"before_free\":%zu,"
      "\"before_total\":%zu,\"cublas_create_status\":%d,"
      "\"cublas_handle\":%llu,\"cublas_stream_status\":%d,"
      "\"cublas_pointer_status\":%d,\"cusolver_create_status\":%d,"
      "\"cusolver_handle\":%llu,\"cusolver_stream_status\":%d,"
      "\"cusolver_params_status\":%d,\"cusolver_params\":%llu,"
      "\"after_status\":%d,\"after_free\":%zu,\"after_total\":%zu,"
      "\"audit\":%s}\n",
      name, static_cast<int>(before_status), before_free, before_total,
      static_cast<int>(cublas_create_status), address(cublas_handle),
      static_cast<int>(cublas_stream_status), static_cast<int>(cublas_pointer_status),
      static_cast<int>(cusolver_create_status), address(cusolver_handle),
      static_cast<int>(cusolver_stream_status), static_cast<int>(cusolver_params_status),
      address(cusolver_parameters), static_cast<int>(after_status), after_free,
      after_total, fake_provider_snapshot());
  return 0;
}

int run_overflow_case() {
  fake_provider_reset();
  cudaError_t status = cudaSuccess;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  for (int iteration = 0; iteration < 520; ++iteration) {
    status = import_cuda_mem_get_info(&free_bytes, &total_bytes);
  }
  std::printf(
      "{\"case\":\"overflow\",\"iterations\":520,\"status\":%d,"
      "\"free_bytes\":%zu,\"total_bytes\":%zu,\"audit\":%s}\n",
      static_cast<int>(status), free_bytes, total_bytes, fake_provider_snapshot());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--overflow") == 0) {
    return run_overflow_case();
  }
  const char* provider_case = std::getenv("FAKE_PROVIDER_CASE");
  return run_provider_case(provider_case && *provider_case ? provider_case : "positive");
}
